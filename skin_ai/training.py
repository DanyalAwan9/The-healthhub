"""
training.py - train the multi-task skin model and export skin_model.pth.

DATA
----
Provide a CSV (--labels) with one row per image. Header, case-insensitive:

  filepath            required  image path (absolute, or relative to --images-root)
  skin_type           required  Normal | Dry | Oily | Combination
  redness             optional  clear|mild|moderate|severe   (or 0..3)
  hyperpigmentation   optional  clear|mild|moderate|severe   (or 0..3)
  blemish_count       optional  integer >= 0
  texture             optional  float 0..1   (0 smooth .. 1 rough)

Empty optional cells are ignored by the loss for that row, so datasets that only
label some attributes can be mixed.

Recommended public sources (map their native labels onto the columns above):

  * Fitzpatrick 17k  - ~17k images spanning 6 skin-tone types. Use it for TONE
                       DIVERSITY so the model is not biased toward light skin,
                       and for hyperpigmentation.
  * ISIC Archive     - large set of clinical / dermoscopic lesion images. Good
                       source of blemish_count and lesion presence.
  * DermNet NZ       - many named conditions; a source of redness / texture labels.
  * A plain labelled selfie set for skin_type (Normal/Dry/Oily/Combination).

If you only have skin-type folders, skip --labels and pass
--imagefolder DIR  with subfolders  Normal/ Dry/ Oily/ Combination/.

RUN
---
  python training.py --labels labels.csv --images-root images/ \
      --arch efficientnet_b0 --epochs 60 --batch 32 --out skin_model.pth

  # split 70/15/15 stratified on skin_type · Adam · CrossEntropy + SmoothL1 ·
  # ReduceLROnPlateau · early stopping · temperature calibration on val ·
  # writes skin_model_reports/{confusion_matrix.png, training_curve.png, metrics.json}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate

try:
    from .preprocessing import (IMAGENET_MEAN, IMAGENET_STD, eval_transform,
                                preprocess_for_model, training_augmentation)
    from .model import SKIN_TYPES, SEVERITY, SkinLoss, build_model
except ImportError:
    from preprocessing import (IMAGENET_MEAN, IMAGENET_STD, eval_transform,
                               preprocess_for_model, training_augmentation)
    from model import SKIN_TYPES, SEVERITY, SkinLoss, build_model

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_SEV = {**{w: i for i, w in enumerate(SEVERITY)}, "0": 0, "1": 1, "2": 2, "3": 3}


# ----------------------------------------------------------------- data plumbing
def _isnan(v) -> bool:
    try:
        if v is None:
            return True
        if isinstance(v, float):
            return math.isnan(v)
        return str(v).strip() == "" or str(v).strip().lower() == "nan"
    except Exception:
        return True


def _load_rows(csv_path: str) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "filepath" not in df.columns or "skin_type" not in df.columns:
        sys.exit("labels CSV needs at least 'filepath' and 'skin_type' columns")
    rows = []
    for _, d in df.iterrows():
        st = str(d["skin_type"]).strip().capitalize()
        if st not in SKIN_TYPES:
            continue
        r = {"filepath": str(d["filepath"]).strip(), "skin_type": SKIN_TYPES.index(st)}
        for col in ("redness", "hyperpigmentation"):
            if col in df.columns and not _isnan(d[col]):
                v = _SEV.get(str(d[col]).strip().lower())
                if v is not None:
                    r[col] = v
        if "blemish_count" in df.columns and not _isnan(d["blemish_count"]):
            r["blemish_count"] = float(d["blemish_count"])
        if "texture" in df.columns and not _isnan(d["texture"]):
            r["texture"] = float(d["texture"])
        rows.append(r)
    if not rows:
        sys.exit("no usable rows in labels CSV (check skin_type values)")
    return rows


def _rows_from_imagefolder(root: str) -> list[dict]:
    rows = []
    for st in SKIN_TYPES:
        d = Path(root) / st
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in _IMG_EXT:
                rows.append({"filepath": str(f), "skin_type": SKIN_TYPES.index(st)})
    if not rows:
        sys.exit(f"no class subfolders with images under {root!r} "
                 f"(expected {'/ '.join(SKIN_TYPES)}/)")
    return rows


class SkinDataset(Dataset):
    def __init__(self, rows, images_root, transform, preprocess=False):
        self.rows = rows
        self.root = images_root
        self.tf = transform
        self.preprocess = preprocess     # route through preprocessing.preprocess_for_model

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = r["filepath"]
        if self.root and not os.path.isabs(p):
            p = os.path.join(self.root, p)
        img = Image.open(p).convert("RGB")
        if self.preprocess:
            crop = preprocess_for_model(img)
            if crop is not None:
                img = crop
        x = self.tf(img)
        y = {
            "skin_type": torch.tensor(int(r["skin_type"]), dtype=torch.long),
            "redness": torch.tensor(int(r.get("redness", -1)), dtype=torch.long),
            "hyperpigmentation": torch.tensor(int(r.get("hyperpigmentation", -1)),
                                              dtype=torch.long),
            "blemish": torch.tensor(float(r.get("blemish_count", math.nan)),
                                    dtype=torch.float),
            "texture": torch.tensor(float(r.get("texture", math.nan)), dtype=torch.float),
        }
        return x, y


def _split(rows, seed):
    from sklearn.model_selection import train_test_split

    y = [r["skin_type"] for r in rows]
    tr, tmp = train_test_split(rows, test_size=0.30, random_state=seed, stratify=y)
    ytmp = [r["skin_type"] for r in tmp]
    strat = ytmp if len(set(ytmp)) > 1 else None
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=strat)
    return tr, va, te


# ------------------------------------------------------------------ train / eval
def _run_epoch(model, loader, crit, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot, seen, agg = 0.0, 0, {}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
        with torch.set_grad_enabled(train), torch.autocast(
                device_type=device.type, enabled=scaler is not None):
            out = model(x)
            loss, parts = crit(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        bs = x.size(0)
        tot += float(loss.detach()) * bs
        seen += bs
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v * bs
    return tot / max(seen, 1), {k: v / max(seen, 1) for k, v in agg.items()}


@torch.no_grad()
def _evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

    model.eval()
    st_t, st_p, logits, ys = [], [], [], []
    red_t, red_p, hp_t, hp_p = [], [], [], []
    bl_t, bl_p = [], []
    for x, y in loader:
        out = model(x.to(device))
        lg = out["skin_type"].cpu()
        logits.append(lg)
        ys.append(y["skin_type"])
        st_p += lg.argmax(1).tolist()
        st_t += y["skin_type"].tolist()
        for name, tt, pp in (("redness", red_t, red_p), ("hyperpigmentation", hp_t, hp_p)):
            yy = y[name]
            mask = yy >= 0
            if mask.any():
                tt += yy[mask].tolist()
                pp += out[name].cpu().argmax(1)[mask].tolist()
        b = y["blemish"]
        mb = ~torch.isnan(b)
        if mb.any():
            bl_t += b[mb].tolist()
            bl_p += torch.expm1(out["blemish"].squeeze(1).cpu())[mb].clamp(min=0).tolist()

    m = {"skin_type_accuracy": float(accuracy_score(st_t, st_p)) if st_t else None,
         "confusion_matrix": confusion_matrix(
             st_t, st_p, labels=list(range(len(SKIN_TYPES)))).tolist() if st_t else None,
         "redness_accuracy": float(accuracy_score(red_t, red_p)) if red_t else None,
         "hyperpigmentation_accuracy": float(accuracy_score(hp_t, hp_p)) if hp_t else None,
         "blemish_mae": float(mean_absolute_error(bl_t, bl_p)) if bl_t else None,
         "n": len(st_t)}
    return m, torch.cat(logits), torch.cat(ys)


def _fit_temperature(logits, labels):
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=80)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def _plots(history, confmat, outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ep = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, history["train"], label="train")
    plt.plot(ep, history["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "training_curve.png", dpi=120); plt.close()

    if confmat is not None:
        cm = np.array(confmat, float)
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        plt.figure(figsize=(5, 4.5))
        plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        plt.xticks(range(len(SKIN_TYPES)), SKIN_TYPES, rotation=45, ha="right")
        plt.yticks(range(len(SKIN_TYPES)), SKIN_TYPES)
        for i in range(len(SKIN_TYPES)):
            for j in range(len(SKIN_TYPES)):
                plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                         color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
        plt.ylabel("true"); plt.xlabel("predicted"); plt.title("skin type (row-normalised)")
        plt.tight_layout(); plt.savefig(outdir / "confusion_matrix.png", dpi=120); plt.close()


# ------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels CSV (see module docstring)")
    ap.add_argument("--imagefolder", help="dir with Normal/ Dry/ Oily/ Combination/ subfolders")
    ap.add_argument("--images-root", default="", help="prefix for relative filepaths in the CSV")
    ap.add_argument("--arch", choices=["efficientnet_b0", "resnet50"], default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=60, help="50-100 recommended")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience")
    ap.add_argument("--freeze-epochs", type=int, default=3, help="train heads only for first N")
    ap.add_argument("--no-contrast", action="store_true", help="disable the 4th contrast channel")
    ap.add_argument("--preprocess", action="store_true",
                    help="route every image through preprocessing.preprocess_for_model "
                         "(face crop + colour constancy + occlusion mask + glare clip) so "
                         "training matches inference. Slow - pre-process to disk for real runs.")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="skin_model.pth")
    a = ap.parse_args(argv)

    if not a.labels and not a.imagefolder:
        sys.exit(__doc__ + "\n\nNothing to train on: pass --labels CSV or --imagefolder DIR.")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")

    rows = _load_rows(a.labels) if a.labels else _rows_from_imagefolder(a.imagefolder)
    tr, va, te = _split(rows, a.seed)
    counts = np.bincount([r["skin_type"] for r in rows], minlength=len(SKIN_TYPES))
    print(f"{len(rows)} images  |  train {len(tr)}  val {len(va)}  test {len(te)}")
    print("skin_type counts: " + ", ".join(f"{k}={int(v)}" for k, v in zip(SKIN_TYPES, counts)))

    root = a.images_root or None
    pp = a.preprocess
    dl_tr = DataLoader(SkinDataset(tr, root, training_augmentation(a.img_size), preprocess=pp),
                       batch_size=a.batch, shuffle=True, num_workers=a.workers,
                       drop_last=True, pin_memory=device.type == "cuda", collate_fn=default_collate)
    dl_va = DataLoader(SkinDataset(va, root, eval_transform(a.img_size), preprocess=pp),
                       batch_size=a.batch, shuffle=False, num_workers=a.workers)
    dl_te = DataLoader(SkinDataset(te, root, eval_transform(a.img_size), preprocess=pp),
                       batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model = build_model(a.arch, pretrained=True, contrast_channel=not a.no_contrast).to(device)
    crit = SkinLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.3, patience=4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    hist = {"train": [], "val": []}
    best, best_state, bad = math.inf, None, 0
    for ep in range(1, a.epochs + 1):
        model.freeze_backbone(ep <= a.freeze_epochs)
        tl, tparts = _run_epoch(model, dl_tr, crit, device, opt, scaler)
        vl, _ = _run_epoch(model, dl_va, crit, device)
        sched.step(vl)
        hist["train"].append(tl)
        hist["val"].append(vl)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep:3d}/{a.epochs}  train {tl:.4f}  val {vl:.4f}  lr {lr:.2e}  "
              + " ".join(f"{k}:{v:.3f}" for k, v in tparts.items()))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep} (no val improvement for {a.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, val_logits, val_labels = _evaluate(model, dl_va, device)
    temperature = _fit_temperature(val_logits, val_labels)
    test_metrics, _, _ = _evaluate(model, dl_te, device)
    print("\n== test ==")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  temperature (from val): {temperature:.3f}")

    outdir = Path("skin_model_reports")
    outdir.mkdir(exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(
        {"val": metrics, "test": test_metrics, "temperature": temperature,
         "arch": a.arch, "contrast_channel": not a.no_contrast,
         "class_counts": {k: int(v) for k, v in zip(SKIN_TYPES, counts)}}, indent=2))
    _plots(hist, test_metrics.get("confusion_matrix"), outdir)

    torch.save({
        "state_dict": model.state_dict(),
        "arch": a.arch,
        "contrast_channel": not a.no_contrast,
        "skin_types": SKIN_TYPES,
        "severity": SEVERITY,
        "img_size": a.img_size,
        "norm_mean": list(IMAGENET_MEAN),
        "norm_std": list(IMAGENET_STD),
        "temperature": temperature,
        "metrics": {"skin_type_accuracy": test_metrics.get("skin_type_accuracy"),
                    "blemish_mae": test_metrics.get("blemish_mae")},
        "format": 1,
    }, a.out)
    print(f"\nsaved {a.out}  ·  reports in {outdir}/")
    print("SkinAnalyzer(model_path=...) loads it; app.py uses skin_ai/skin_model.pth.")


if __name__ == "__main__":
    main()
