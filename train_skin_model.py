"""Train + calibrate a skin-condition classifier for HealthHub.

This is the FULL pipeline (Phases 2-4 of the plan). It is deliberately data-
agnostic: point it at a folder of labelled images and it produces a trained,
temperature-calibrated checkpoint that `skin_analyzer.py` loads automatically.

------------------------------------------------------------------------------
PHASE 1 - you supply the data (this script cannot download datasets for you):

    data/
      clear_skin/       (>= ~100 images)
      acne/
      oily/
      dry/
      sensitive/
      redness/
      hyperpigmentation/

  Where to get images:
    * HAM10000 (Kaggle: kmader/skin-cancer-mnist-ham10000)     - lesions
    * ISIC Archive  https://www.isic-archive.com/               - dermoscopy
    * Kaggle "acne", "acne04", "face skin diseases" datasets
    * Roboflow Universe "acne", "skin type" projects
  `prepare_skin_data.py` turns a flat folder + labels.csv into the layout above.

------------------------------------------------------------------------------
PHASE 2-4 - run:

    python train_skin_model.py --data data --arch resnet50 --epochs 60
    # -> skin_model_v2.pth  +  skin_model_reports/{confusion_matrix,roc}.png, metrics.json

Then just restart HealthHub - skin_analyzer.py picks up skin_model_v2.pth.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ------------------------------------------------------------------ model factory
# Imported by skin_analyzer.py too, so the architecture has ONE definition.
def build_model(arch: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    if arch == "resnet50":
        try:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            net = models.resnet50(weights=None)
        # resnet's own avgpool == Global Average Pooling; replace the classifier
        net.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        return net

    if arch == "scratch":
        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                      nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        return nn.Sequential(
            *block(3, 32, False),
            *block(32, 64, True),
            *block(64, 128, True),
            *block(128, 256, True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    raise ValueError(f"unknown arch {arch!r} (use resnet50 or scratch)")


ARCH_CFG = {
    "resnet50": dict(img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "scratch":  dict(img_size=128, mean=(0.5, 0.5, 0.5),        std=(0.5, 0.5, 0.5)),
}


# ------------------------------------------------------------------ training
def _transforms(cfg, train: bool):
    from torchvision import transforms

    n = cfg["img_size"]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(n, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
    return transforms.Compose([
        transforms.Resize(int(n * 1.15)),
        transforms.CenterCrop(n),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def _stratified_splits(targets, seed):
    from sklearn.model_selection import train_test_split

    idx = list(range(len(targets)))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=targets)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed,
                              stratify=[targets[i] for i in tmp])
    return tr, va, te


def _fit_temperature(logits, labels):
    """Temperature scaling (Guo et al. 2017): learn one scalar T that minimises
    NLL on the validation set. Fixes soft-max over-confidence (the '99 %' bug)."""
    import torch
    import torch.nn as nn

    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=80)
    nll = nn.CrossEntropyLoss()

    def _closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(T.detach().clamp(min=0.05).item())


def _plots(y_true, y_pred, y_prob, classes, outdir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.0 * len(classes) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=120); plt.close(fig)

    yb = label_binarize(y_true, classes=list(range(len(classes))))
    if yb.shape[1] == 1:  # binary edge-case
        yb = np.hstack([1 - yb, yb])
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, name in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(yb[:, k], y_prob[:, k])
            ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.2f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC (one-vs-rest)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(outdir / "roc.png", dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="ImageFolder root (subfolder per class)")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "scratch"])
    ap.add_argument("--out", default="skin_model_v2.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    ap.add_argument("--freeze-epochs", type=int, default=3,
                    help="resnet50: train only the head for the first N epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset
        from torchvision.datasets import ImageFolder
        from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
    except Exception as exc:  # pragma: no cover
        sys.exit(f"Missing a dependency ({exc}). Run: pip install -r requirements.txt")

    root = Path(args.data)
    if not root.is_dir() or not any(p.is_dir() for p in root.iterdir()):
        sys.exit(
            f"No training data in '{root}/'.\n"
            "Create one subfolder of images per class, e.g.:\n"
            "  data/clear_skin/  data/acne/  data/oily/  data/dry/\n"
            "  data/sensitive/   data/redness/  data/hyperpigmentation/\n"
            "See prepare_skin_data.py to sort a flat folder + labels.csv."
        )

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = ARCH_CFG[args.arch]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_train = ImageFolder(str(root), transform=_transforms(cfg, True))
    full_eval = ImageFolder(str(root), transform=_transforms(cfg, False))
    classes = full_train.classes
    if len(classes) < 2:
        sys.exit(f"Need >= 2 class folders, found {classes}.")
    counts = np.bincount([t for _, t in full_train.samples], minlength=len(classes))
    print(f"classes ({len(classes)}): " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
    if counts.min() < 10:
        print("WARNING: some classes have < 10 images - results will be weak. "
              "Aim for 100+ per class.")

    tr_i, va_i, te_i = _stratified_splits([t for _, t in full_train.samples], args.seed)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh,
                                   num_workers=args.workers, pin_memory=(device == "cuda"))
    train_dl = dl(Subset(full_train, tr_i), True)
    val_dl = dl(Subset(full_eval, va_i), False)
    test_dl = dl(Subset(full_eval, te_i), False)
    print(f"split: train {len(tr_i)} / val {len(va_i)} / test {len(te_i)}  · device {device}")

    model = build_model(args.arch, len(classes)).to(device)
    w = torch.tensor((counts.sum() / (len(classes) * np.maximum(counts, 1))),
                     dtype=torch.float32, device=device)          # class-balanced
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=4)

    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")] \
        if args.arch == "resnet50" else []

    def _run(loader, train: bool):
        model.train(train)
        tot, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if train:
                    opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                if train:
                    loss.backward(); opt.step()
                loss_sum += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                tot += xb.size(0)
        return loss_sum / tot, correct / tot

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        if args.arch == "resnet50":
            frozen = ep <= args.freeze_epochs
            for p in backbone:
                p.requires_grad_(not frozen)
        tl, ta = _run(train_dl, True)
        vl, vaacc = _run(val_dl, False)
        sched.step(vl)
        print(f"epoch {ep:3d}  train {tl:.3f}/{ta:.3f}   val {vl:.3f}/{vaacc:.3f}"
              + ("  *" if vl < best_val - 1e-4 else ""))
        if vl < best_val - 1e-4:
            best_val, best_state, bad = vl, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping at epoch {ep} (no val improvement for {bad})")
                break

    model.load_state_dict(best_state)

    # ---- temperature calibration on the validation set ----
    model.eval()
    with torch.no_grad():
        vlogits = torch.cat([model(xb.to(device)).cpu() for xb, _ in val_dl])
        vlabels = torch.cat([yb for _, yb in val_dl])
    temperature = _fit_temperature(vlogits, vlabels)
    print(f"calibrated temperature T = {temperature:.3f}")

    # ---- test-set evaluation ----
    with torch.no_grad():
        tlogits = torch.cat([model(xb.to(device)).cpu() for xb, _ in test_dl])
        tlabels = torch.cat([yb for _, yb in test_dl]).numpy()
    tprob = torch.softmax(tlogits / temperature, dim=1).numpy()
    tpred = tprob.argmax(1)
    acc = accuracy_score(tlabels, tpred)
    print(f"\nTEST accuracy: {acc:.3f}  (target >= 0.85)")
    print(classification_report(tlabels, tpred, target_names=classes, zero_division=0))
    try:
        auc_ovr = roc_auc_score(tlabels, tprob, multi_class="ovr", average="macro")
    except Exception:
        auc_ovr = float("nan")
    print(f"macro ROC-AUC (OvR): {auc_ovr:.3f}")

    outdir = Path("skin_model_reports"); outdir.mkdir(exist_ok=True)
    try:
        _plots(tlabels, tpred, tprob, classes, outdir)
    except Exception as exc:
        print(f"(plot step skipped: {exc})")

    rep = classification_report(tlabels, tpred, target_names=classes, zero_division=0,
                                output_dict=True)
    metrics = {"accuracy": acc, "macro_roc_auc_ovr": auc_ovr,
               "per_class": rep, "class_counts": {c: int(n) for c, n in zip(classes, counts)},
               "n_test": int(len(tlabels))}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "classes": classes,
        "img_size": cfg["img_size"],
        "norm_mean": list(cfg["mean"]),
        "norm_std": list(cfg["std"]),
        "temperature": temperature,
        "metrics": {"accuracy": acc, "macro_roc_auc_ovr": auc_ovr},
        "format": 2,
    }, args.out)
    print(f"\nsaved {args.out}  ·  reports in {outdir}/")
    print("Restart HealthHub - skin_analyzer.py will load it automatically.")


if __name__ == "__main__":
    main()
