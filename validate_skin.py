"""Calibration check for skin_analyzer.

Confidence must scale with real SEVERITY, not "over a low threshold":
  * clear skin            -> care_level basic, is_clear, low %
  * a few tiny spots      -> < 50 %, NOT actionable, still "maintain routine"
  * visible acne          -> 50-75 %, care_level "light" (basic + optional serum)
  * widespread / severe   -> >= 75 %, care_level "treat"
  * bad photo / not-face  -> assessed = False
"""
import os, warnings
warnings.simplefilter("ignore")
os.environ["HEALTHHUB_DB"] = os.path.join(os.environ["TEMP"], "hh_val.db")
# the CV-path tests must not be swayed by a real skin_model_v2.pth lying around
os.environ["HEALTHHUB_SKIN_MODEL"] = os.path.join(os.environ["TEMP"], "no_such_model.pth")
import numpy as np
from PIL import Image, ImageDraw
import database as db
db.init_db()
import skin_analyzer as sa

rng = np.random.default_rng(7)
N = 320


def base_skin(tone=(214, 165, 136), noise=4.0):
    a = np.zeros((N, N, 3), np.float32) + np.array(tone, np.float32)
    a += rng.normal(0, noise, a.shape)
    a += np.linspace(-10, 10, N)[:, None]
    return np.clip(a, 0, 255)


def img(a):
    return Image.fromarray(a.astype(np.uint8))


def spots(a, n, dr, radius):
    im = img(a); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rng.integers(28, N - 28, 2)
        r = int(rng.integers(*radius))
        col = tuple(int(np.clip(c + o, 0, 255)) for c, o in zip((214, 165, 136), dr))
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return np.asarray(im).astype(np.float32)


FACE = dict(top_label="mask", top_confidence=0.11, entropy=6.2, embedding_norm=9.0,
            face_ish=True, non_face=False, agreement=0.95)
LOWAGREE = {**FACE, "agreement": 0.15, "entropy": 2.0}
NOTFACE = dict(top_label="orange", top_confidence=0.93, entropy=0.6, embedding_norm=8.0,
               face_ish=False, non_face=True, agreement=0.2)

results = []


def check(name, image, want, patch=FACE):
    old = sa._resnet_readout
    sa._resnet_readout = lambda _i: patch
    try:
        r = sa.analyze(image, mode="csv")
    finally:
        sa._resnet_readout = old
    bad = []
    for key in ("care_level", "is_clear", "assessed"):
        if key in want and r.get(key) != want[key]:
            bad.append(f"{key}={r.get(key)!s} != {want[key]!s}")
    if "conf_lt" in want and r["confidence"] >= want["conf_lt"]:
        bad.append(f"confidence {r['confidence']} not < {want['conf_lt']}")
    if "conf_ge" in want and r["confidence"] < want["conf_ge"]:
        bad.append(f"confidence {r['confidence']} not >= {want['conf_ge']}")
    if want.get("no_findings") and r["findings"]:
        bad.append(f"expected no actionable findings, got {[f['name'][:16] for f in r['findings']]}")
    if want.get("flag") and not any(want["flag"] in f["name"] for f in r["findings"]):
        bad.append(f"missing finding {want['flag']}")
    if want.get("skin_type") and r["skin_type"] != want["skin_type"]:
        bad.append(f"skin_type {r['skin_type']}")
    if want.get("severity") and r.get("severity") != want["severity"]:
        bad.append(f"severity {r.get('severity')} != {want['severity']}")
    if want.get("no_treatment_cat"):
        cats = [g["category"] for g in r["recommendations"].get("product_options", [])]
        if any("Treatment" in c for c in cats):
            bad.append(f"treatment category present: {cats}")
    fs = ",".join(f"{f['name'].split(' (')[0][:10]}:{f['severity']}:{f['confidence']:.0f}"
                  for f in r["findings"]) or "-"
    obs = ",".join(f"{o['name'].split(' (')[0][:10]}:{o['confidence']:.0f}"
                   for o in r["observations"]) or "-"
    print(("  OK  " if not bad else "  XX  ")
          + f"{name:26} level={r.get('care_level'):5} conf={r['confidence']:>4} "
            f"clear={r['is_clear']!s:5} find=[{fs}] obs=[{obs}]")
    for x in bad:
        print("        -> " + x)
    results.append(not bad)


check("CLEAR smooth skin", img(base_skin()),
      dict(care_level="basic", is_clear=True, conf_lt=50, no_findings=True))
check("CLEAR darker tone", img(base_skin((150, 110, 88))),
      dict(care_level="basic", is_clear=True, conf_lt=50, no_findings=True))
check("A FEW TINY SPOTS (minor)", img(spots(base_skin(), 12, (-55, -48, -44), (2, 4))),
      dict(care_level="basic", is_clear=True, conf_lt=50, no_findings=True))
check("VISIBLE ACNE (moderate)", img(spots(base_skin(), 66, (-66, -55, -50), (3, 8))),
      dict(care_level="light", is_clear=False, conf_ge=50, conf_lt=75,
           flag="Hyperpigmentation", no_treatment_cat=True))
check("WIDESPREAD (severe)", img(spots(base_skin(), 150, (-78, -64, -58), (4, 10))),
      dict(care_level="treat", is_clear=False, conf_ge=75, flag="Hyperpigmentation",
           severity="high"))
check("STRONG redness (severe)", img(spots(base_skin(), 110, (32, -13, -11), (6, 12))),
      dict(care_level="treat", is_clear=False, conf_ge=75, flag="Redness"))
check("OILY specular shine", img(spots(base_skin(), 14, (31, 78, 104), (16, 34))),
      dict(skin_type="Oily"))
check("moderate signal + low ResNet agreement",
      img(spots(base_skin(), 46, (-72, -60, -55), (4, 9))),
      dict(care_level="basic", is_clear=True, conf_lt=50, no_findings=True), patch=LOWAGREE)
check("ResNet says NOT a face", img(base_skin()),
      dict(assessed=False), patch=NOTFACE)
check("BAD PHOTO (noise)", img(rng.integers(0, 255, (N, N, 3)).astype(np.float32)),
      dict(assessed=False))
check("BAD PHOTO (dark)", img(np.zeros((N, N, 3), np.float32) + 12),
      dict(assessed=False))


# ---- trained-model path: fabricate a checkpoint, confirm skin_analyzer loads it
#      and still returns the full output contract --------------------------------
def check_trained_roundtrip():
    try:
        import torch
        from train_skin_model import build_model, ARCH_CFG
    except Exception as e:  # torch / module not importable - not a calibration failure
        print(f"  ..  TRAINED model round-trip     SKIPPED ({e})")
        return
    classes = ["clear_skin", "acne", "oily", "dry", "redness", "hyperpigmentation"]
    arch, cfg = "scratch", ARCH_CFG["scratch"]
    ckpt_path = os.path.join(os.environ["TEMP"], "hh_val_model.pth")
    torch.save({
        "state_dict": build_model(arch, len(classes)).state_dict(),
        "arch": arch, "classes": classes, "img_size": cfg["img_size"],
        "norm_mean": list(cfg["mean"]), "norm_std": list(cfg["std"]),
        "temperature": 1.7, "metrics": {"accuracy": 0.88, "macro_roc_auc_ovr": 0.95},
        "format": 2,
    }, ckpt_path)

    old_path, old_readout = sa._TRAINED_PATH, sa._resnet_readout
    sa._TRAINED_PATH = ckpt_path
    sa._trained, sa._trained_tried = None, False
    sa._resnet_readout = lambda _i: FACE
    try:
        r = sa.analyze(img(base_skin()), mode="csv")
    finally:
        sa._resnet_readout = old_readout
        sa._TRAINED_PATH = old_path
        sa._trained, sa._trained_tried = None, False
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    need = {"skin_type", "condition", "conditions", "findings", "observations",
            "assessed", "is_clear", "severity", "advice_band", "care_level",
            "summary", "confidence", "metrics", "model", "backbone",
            "recommendations", "reco_source", "disclaimer"}
    mdl = r.get("model", {}) or {}
    bad = []
    if need - set(r):
        bad.append(f"missing keys: {sorted(need - set(r))}")
    if mdl.get("trained") is not True:
        bad.append(f"model.trained != True ({mdl})")
    if mdl.get("top_class") not in classes:
        bad.append(f"top_class {mdl.get('top_class')!r} not one of the fabricated classes")
    if mdl.get("test_accuracy") != 0.88:
        bad.append(f"test_accuracy not carried through ({mdl.get('test_accuracy')})")
    if r.get("severity") not in {"none", "low", "medium", "high"}:
        bad.append(f"severity {r.get('severity')!r}")
    if r.get("care_level") not in {"basic", "light", "treat"}:
        bad.append(f"care_level {r.get('care_level')!r}")
    if not str(r.get("backbone", "")).startswith("skin_model_v2"):
        bad.append(f"backbone {r.get('backbone')!r}")
    print(("  OK  " if not bad else "  XX  ")
          + f"{'TRAINED model round-trip':26} level={r.get('care_level'):5} "
            f"conf={r['confidence']:>4} top={mdl.get('top_class')}")
    for x in bad:
        print("        -> " + x)
    results.append(not bad)


check_trained_roundtrip()

os.remove(os.environ["HEALTHHUB_DB"])
n = sum(results)
print(f"\n{n}/{len(results)} " + ("ALL PASS" if n == len(results) else "-- SOME FAILED"))
raise SystemExit(0 if n == len(results) else 1)
