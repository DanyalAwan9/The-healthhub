"""Phase 1 helper: turn a downloaded dataset into the ImageFolder layout that
train_skin_model.py expects.

Two ways to run it:

  # 1. a flat folder of images + a CSV  (filename,label)  - the common Kaggle shape
  python prepare_skin_data.py --images raw/images --labels raw/labels.csv --out data

  # 2. pull a Kaggle dataset first (needs the `kaggle` CLI + ~/.kaggle/kaggle.json)
  python prepare_skin_data.py --from-kaggle kmader/skin-cancer-mnist-ham10000 --out data_raw
  #   ...then inspect data_raw/, write a labels.csv, and run form (1).

TARGET CLASSES (folders it creates): clear_skin, acne, oily, dry, sensitive,
redness, hyperpigmentation.  Anything else in the CSV is mapped via LABEL_MAP
below (edit it for your dataset) or copied under its own name.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

TARGET = ["clear_skin", "acne", "oily", "dry", "sensitive", "redness",
          "hyperpigmentation"]

# dataset-label -> HealthHub class.  Extend for whatever you downloaded.
LABEL_MAP = {
    # HAM10000 dx codes (mostly lesions - use sparingly / as 'hyperpigmentation')
    "nv": "hyperpigmentation", "bkl": "hyperpigmentation", "mel": "hyperpigmentation",
    "df": "hyperpigmentation", "vasc": "redness",
    # common acne-dataset labels
    "acne": "acne", "pimple": "acne", "acne_vulgaris": "acne", "level1": "acne",
    "level2": "acne", "level3": "acne", "comedone": "acne", "papule": "acne",
    "pustule": "acne", "clear": "clear_skin", "clear_skin": "clear_skin",
    "normal": "clear_skin", "healthy": "clear_skin", "no_acne": "clear_skin",
    "oily": "oily", "oily_skin": "oily", "dry": "dry", "dry_skin": "dry",
    "sensitive": "sensitive", "rosacea": "redness", "redness": "redness",
    "erythema": "redness", "melasma": "hyperpigmentation",
    "hyperpigmentation": "hyperpigmentation", "pih": "hyperpigmentation",
    "dark_spots": "hyperpigmentation", "freckles": "hyperpigmentation",
}
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def from_kaggle(slug: str, out: Path) -> None:
    if shutil.which("kaggle") is None:
        sys.exit("`kaggle` CLI not found. `pip install kaggle`, then put your token "
                 "at ~/.kaggle/kaggle.json (chmod 600). See kaggle.com/settings.")
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {slug} -> {out}/ ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out),
                    "--unzip"], check=True)
    n = sum(1 for p in out.rglob("*") if p.suffix.lower() in _IMG_EXT)
    print(f"done - {n} image files under {out}/. Now build a labels.csv and re-run "
          "with --images/--labels.")


def organise(images: Path, labels_csv: Path, out: Path, move: bool) -> None:
    if not images.is_dir():
        sys.exit(f"--images '{images}' is not a folder")
    if not labels_csv.is_file():
        sys.exit(f"--labels '{labels_csv}' not found")

    # index every image by basename and by stem so the CSV can use either
    by_name = {}
    for p in images.rglob("*"):
        if p.suffix.lower() in _IMG_EXT:
            by_name.setdefault(p.name.lower(), p)
            by_name.setdefault(p.stem.lower(), p)

    rows = list(csv.DictReader(labels_csv.open(newline="", encoding="utf-8-sig")))
    if not rows:
        sys.exit("labels.csv is empty")
    cols = {c.lower(): c for c in rows[0]}
    fcol = cols.get("filename") or cols.get("file") or cols.get("image") or cols.get("id")
    lcol = cols.get("label") or cols.get("class") or cols.get("dx") or cols.get("category")
    if not fcol or not lcol:
        sys.exit(f"labels.csv needs a filename and a label column; got {list(rows[0])}")

    out.mkdir(parents=True, exist_ok=True)
    done, missing, mapped = 0, 0, {}
    for r in rows:
        raw = str(r[fcol]).strip()
        src = by_name.get(raw.lower()) or by_name.get(Path(raw).name.lower()) \
            or by_name.get(Path(raw).stem.lower())
        if src is None:
            missing += 1
            continue
        lab = str(r[lcol]).strip().lower().replace(" ", "_").replace("-", "_")
        cls = LABEL_MAP.get(lab, lab if lab in TARGET else None)
        if cls is None:
            mapped[lab] = mapped.get(lab, 0) + 1
            continue
        dst = out / cls
        dst.mkdir(exist_ok=True)
        target = dst / src.name
        if not target.exists():
            (shutil.move if move else shutil.copy2)(str(src), str(target))
        done += 1

    print(f"placed {done} images into {out}/")
    if missing:
        print(f"  {missing} rows had no matching image file")
    if mapped:
        print("  unmapped labels (add them to LABEL_MAP): "
              + ", ".join(f"{k}×{v}" for k, v in sorted(mapped.items(), key=lambda x: -x[1])))
    for c in TARGET:
        d = out / c
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        print(f"    {c:20} {n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-kaggle", metavar="SLUG",
                    help="e.g. kmader/skin-cancer-mnist-ham10000")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    a = ap.parse_args(argv)

    if a.from_kaggle:
        from_kaggle(a.from_kaggle, a.out)
    elif a.images and a.labels:
        organise(a.images, a.labels, a.out, a.move)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
