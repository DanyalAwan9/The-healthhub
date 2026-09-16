"""Skin analysis.

Two paths, chosen automatically:

  * If **skin_model_v2.pth** exists (train it with `train_skin_model.py` on a
    labelled dataset), its temperature-calibrated class probabilities are the
    primary signal - `condition`, `confidence`, `severity` come from the model.
  * Otherwise: ImageNet-ResNet50 gives a photo-sanity read-out (is this even a
    face?) and calibrated skin-region CV metrics drive a conservative,
    severity-scaled read-out that defaults to "clear".

Either way the output contract is identical (see analyze()'s return dict) and a
concern is only reported at >= 50% confidence. Cosmetic wellness helper, NOT a
medical diagnosis.
"""
from __future__ import annotations

import io
import math
import re

import numpy as np
from PIL import Image

try:  # torch is heavy; degrade gracefully if it is missing
    import torch
    from torchvision import models

    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False

_model = None
_feature_model = None
_preprocess = None

# ---- trained classifier (skin_model_v2.pth) - optional, produced by train_skin_model.py
import os as _os

_TRAINED_PATH = _os.environ.get(
    "HEALTHHUB_SKIN_MODEL",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "skin_model_v2.pth"),
)
_trained = None
_trained_tried = False

# trained class name -> HealthHub finding / skin-type
_FINDING_FOR_CLASS = {
    "acne": "Uneven texture (possible active acne or scarring)",
    "redness": "Redness / inflammation (possible acne or sensitivity)",
    "sensitive": "Redness / inflammation (possible acne or sensitivity)",
    "rosacea": "Redness / inflammation (possible acne or sensitivity)",
    "hyperpigmentation": "Hyperpigmentation / dark spots / uneven tone",
    "pigmentation": "Hyperpigmentation / dark spots / uneven tone",
    "melasma": "Hyperpigmentation / dark spots / uneven tone",
    "oily": "Excess oil / enlarged pores",
}
_CLEAR_CLASSES = {"clear_skin", "clear", "normal", "healthy", "no_acne"}
_SKINTYPE_CLASS = {"oily": "Oily", "dry": "Dry", "combination": "Combination",
                   "normal": "Normal", "clear_skin": "Normal"}


def torch_available() -> bool:
    return _TORCH_OK


def _load_trained():
    """Load skin_model_v2.pth if present. Returns a dict or None (never raises)."""
    global _trained, _trained_tried
    if _trained_tried:
        return _trained
    _trained_tried = True
    if not _TORCH_OK or not _os.path.isfile(_TRAINED_PATH):
        return None
    try:
        import torch
        from torchvision import transforms
        from train_skin_model import build_model

        ckpt = torch.load(_TRAINED_PATH, map_location="cpu")
        net = build_model(ckpt["arch"], len(ckpt["classes"]))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        n = int(ckpt.get("img_size", 224))
        tfm = transforms.Compose([
            transforms.Resize(int(n * 1.15)), transforms.CenterCrop(n),
            transforms.ToTensor(),
            transforms.Normalize(ckpt.get("norm_mean", (0.485, 0.456, 0.406)),
                                 ckpt.get("norm_std", (0.229, 0.224, 0.225))),
        ])
        _trained = {"net": net, "tfm": tfm, "arch": ckpt.get("arch", "resnet50"),
                    "classes": [c.lower() for c in ckpt["classes"]],
                    "temperature": float(ckpt.get("temperature", 1.0)),
                    "accuracy": (ckpt.get("metrics") or {}).get("accuracy")}
    except Exception:
        _trained = None
    return _trained


def _trained_predict(img: Image.Image) -> dict | None:
    t = _load_trained()
    if t is None:
        return None
    try:
        import torch

        with torch.no_grad():
            x = t["tfm"](img).unsqueeze(0)
            logits = t["net"](x)
            probs = torch.softmax(logits / max(t["temperature"], 0.05), dim=1)[0]
        p = {c: float(probs[i].item()) for i, c in enumerate(t["classes"])}
        top = max(p, key=p.get)
        ent = float(-(probs * (probs + 1e-9).log()).sum().item())
        return {"present": True, "probs": p, "top_class": top, "top_conf": p[top],
                "entropy": round(ent, 3), "accuracy": t["accuracy"]}
    except Exception:
        return None


def _load_model():
    global _model, _feature_model, _preprocess
    if not _TORCH_OK:
        return None
    if _model is None:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        _model = models.resnet50(weights=weights)
        _model.eval()
        _preprocess = weights.transforms()
        _feature_model = torch.nn.Sequential(*list(_model.children())[:-1])
        _feature_model.eval()
    return _model


# ImageNet-1k classes that a face / head-and-shoulders selfie legitimately fires
# on (there is no "face" class, so a real portrait produces a diffuse, low top-1
# probability, often landing on one of these). If ResNet is *confident* the image
# is something else entirely, we abstain instead of inventing a skin diagnosis.
_FACE_ISH_LABELS = {
    "mask", "ski mask", "bathing cap", "swimming cap", "wig", "hair spray",
    "sunscreen", "lipstick", "lip balm", "band aid", "neck brace", "stole",
    "cardigan", "sweatshirt", "jersey", "T-shirt", "suit", "bow tie", "Windsor tie",
    "necklace", "sunglasses", "sunglass", "hat", "cowboy hat", "sombrero",
}
_IMAGENET_CATS = None


def _resnet_readout(img: Image.Image) -> dict | None:
    """ResNet50's own read-out, used as a *photo-sanity / abstention* signal.

    ImageNet-pretrained ResNet50 is not a dermatology model, but its output tells
    us whether the picture even looks like a face close-up:
      * a real portrait -> diffuse softmax, low top-1 prob (< ~0.45), label often
        a garment / cosmetic / headwear (_FACE_ISH_LABELS);
      * a photo of a peach / wall / pet -> confident, non-face label -> abstain.
    """
    global _IMAGENET_CATS
    model = _load_model()
    if model is None:
        return None
    if _IMAGENET_CATS is None:
        try:
            _IMAGENET_CATS = list(models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
        except Exception:
            _IMAGENET_CATS = []
    with torch.no_grad():
        x = _preprocess(img).unsqueeze(0)
        emb = _feature_model(x).flatten()
        probs = torch.softmax(model(x), dim=1)[0]
        top_p, top_i = torch.topk(probs, 3)
    top_conf = float(top_p[0].item())
    top_idx = int(top_i[0].item())
    label = _IMAGENET_CATS[top_idx] if 0 <= top_idx < len(_IMAGENET_CATS) else ""
    entropy = float(-(probs * (probs + 1e-9).log()).sum().item())   # 0..~6.9
    face_ish = (top_conf < 0.45) or (label in _FACE_ISH_LABELS)
    # confident + clearly-not-a-face -> the photo is probably not skin at all
    non_face = (top_conf >= 0.55) and (label not in _FACE_ISH_LABELS)
    return {
        "top_label": label,
        "top_confidence": round(top_conf, 3),
        "entropy": round(entropy, 3),
        "embedding_norm": round(float(emb.norm().item()), 2),
        "face_ish": bool(face_ish),
        "non_face": bool(non_face),
        # 1.0 when the backbone clearly agrees this is a face close-up, tapering
        # to ~0.6 when it is unsure - used to down-weight borderline CV findings.
        "agreement": round(float(np.clip(0.6 + 0.9 * (entropy / 6.9)
                                         - 1.2 * max(0.0, top_conf - 0.45), 0.0, 1.0)), 3),
    }


# --------------------------------------------------------------------------- #
# Calibrated CV metrics.
#
# The old version measured brightness / redness / gradient variance over the
# WHOLE frame, so warm indoor light, hair, background, a soft shadow or JPEG
# noise all read as "redness / texture / pigmentation" on perfectly clear skin.
#
# This version:
#   1. keeps only skin-tone pixels inside a centre crop (drops hair / background),
#   2. measures LOCAL, patchy deviations relative to *this face's own* skin
#      (a uniformly warm face is not "inflamed"; a uniform tan is not "spots"),
#   3. is resolution / contrast normalised,
#   4. returns `reliable=False` when it cannot find a face-sized skin region.
# Thresholds default to CLEAR unless the evidence is strong (see _THRESH).
# --------------------------------------------------------------------------- #
_RESIZE = 320


def _skin_mask(arr: np.ndarray) -> np.ndarray:
    """Permissive skin-tone mask (YCbCr ∩ a warm-RGB rule). Works across the range
    of South-Asian skin tones; rejects hair, cloth, background, blown highlights."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # cr upper bound kept generous - inflamed / flushed skin can sit near 185
    ycc = (cb >= 72) & (cb <= 138) & (cr >= 132) & (cr <= 186) & (y > 45) & (y < 248)
    mn = arr.min(axis=-1)
    warm = (r > 55) & (g > 28) & (b > 12) & (r >= g - 6) & (g >= b - 6) & (r - mn > 5)
    return ycc & warm


def _box(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Cheap blur: shrink then grow with PIL (area-ish average)."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.resize(out_hw[::-1]).resize((a.shape[1], a.shape[0]))).astype(np.float32)


def _cv_metrics(img: Image.Image) -> dict:
    full = np.asarray(img.convert("RGB").resize((_RESIZE, _RESIZE))).astype(np.float32)
    h, w, _ = full.shape
    cen = full[int(h * 0.16):int(h * 0.93), int(w * 0.14):int(w * 0.86)]   # centre crop
    mask = _skin_mask(cen)
    frac = float(mask.mean())

    if frac < 0.14 or int(mask.sum()) < 1500:
        return {"skin_fraction": round(frac, 3), "reliable": False, "brightness": 0.5,
                "redness_score": 0.0, "spot_score": 0.0,
                "texture_score": 0.0, "shine_score": 0.0}

    sk = cen[mask]                                   # (N, 3) skin pixels only
    r, g, b = sk[:, 0], sk[:, 1], sk[:, 2]
    v = (sk.max(1) / 255.0)
    lum = cen.mean(axis=-1)

    # shine / oil: bright, DESATURATED pixels (specular highlights read as white,
    # so they fall outside the skin mask) that sit on/near skin.
    cmx, cmn = cen.max(axis=-1), cen.min(axis=-1)
    bright = (cmx / 255.0 > 0.86) & ((cmx - cmn) / np.maximum(cmx, 1e-5) < 0.18)
    near_skin = _box(mask.astype(np.float32) * 255.0, (24, 24)) > 45
    shine_score = float((bright & near_skin).sum() / max(mask.sum(), 1))

    # redness: fraction of skin markedly redder than THIS face's own median
    redness = r - (g + b) / 2.0
    red_hot = redness > (np.median(redness) + 15.0)
    redness_score = float(red_hot.mean())

    # pigmentation: localized dark blobs (darker than the neighbourhood by >24/255)
    local_mean = _box(lum, (18, 18))
    dark = (lum < local_mean - 24.0) & mask
    spot_score = float(dark.sum() / max(mask.sum(), 1))

    # texture: high-frequency energy at pore/blemish scale, contrast-normalised
    hi = lum - _box(lum, (h // 5, w // 5))
    contrast = float(lum[mask].std()) + 1e-3
    texture_score = float(np.abs(hi[mask]).mean() / contrast)

    return {
        "skin_fraction": round(frac, 3),
        "reliable": True,
        "brightness": round(float(v.mean()), 3),
        "shine_score": round(shine_score, 4),
        "redness_score": round(redness_score, 4),
        "spot_score": round(spot_score, 4),
        "texture_score": round(texture_score, 4),
    }


# Evidence must clearly exceed these before a concern is even a candidate.
# Tuned so a smooth, evenly-toned face reads as CLEAR (see validate_skin.py).
_THRESH = {
    "redness": 0.17,
    "spots": 0.045,
    "texture": 0.62,
    "shine": 0.055,
}

# Confidence bands (step 3 of the brief):
#   < 50   -> "Skin looks clear, maintain routine"   -> basic recs only
#   50-75  -> "Minor concerns, basic care"           -> basic + one optional niacinamide
#   >= 75  -> "Visible concerns, consider treatment" -> full (gentle) recs
_BAND_CLEAR = 50.0
_BAND_TREAT = 75.0
_MIN_OBSERVATION_CONF = 22.0    # below this a metric bump is just noise, not shown

# Per-metric SEVERITY scale (score at which the concern is: barely-there / mild /
# moderate / severe). Confidence is interpolated clear->5%, mild->30%,
# moderate->62%, severe->88%. A few tiny spots land ~12-22% (not 99%); genuine
# visible acne lands in the 50-75% band; widespread lands 75%+.
_FLAGS = [
    #  label,                                                 metric key,     clear  mild   mod    severe
    ("Redness / inflammation (possible acne or sensitivity)", "redness_score", 0.050, 0.110, 0.190, 0.320),
    ("Hyperpigmentation / dark spots / uneven tone",          "spot_score",    0.015, 0.035, 0.065, 0.140),
    ("Uneven texture (possible active acne or scarring)",     "texture_score", 0.350, 0.520, 0.750, 1.100),
    ("Excess oil / enlarged pores",                           "shine_score",   0.030, 0.075, 0.140, 0.260),
]


def _pw(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolation, gently extrapolated past the last knot."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / max(xs[i] - xs[i - 1], 1e-9)
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    t = (x - xs[-1]) / max(xs[-1] - xs[-2], 1e-9)
    return min(97.0, ys[-1] + t * (ys[-1] - ys[-2]))


def _finding_confidence(score: float, cb: float, mi: float, mo: float, se: float,
                        agreement: float) -> float:
    """0..97, scaled to SEVERITY (magnitude), not just 'over a low threshold'.
    Down-weighted when the ResNet50 read-out is only weakly sure it's a face."""
    base = _pw(score, [cb, mi, mo, se], [5.0, 30.0, 62.0, 88.0])
    return round(float(np.clip(base * (0.6 + 0.4 * agreement), 0.0, 97.0)), 1)


def _severity_of(conf: float) -> str:
    """low (keep routine) / medium (basic care) / high (consider treatment)."""
    if conf >= _BAND_TREAT:
        return "high"
    if conf >= _BAND_CLEAR:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# Product / diet knowledge base (Pakistan, approx PKR)
# --------------------------------------------------------------------------- #
_CLEANSERS = {
    "Oily": [("Garnier Bright Complete Face Wash", 550),
             ("CeraVe Foaming Cleanser 236 ml", 3200),
             ("Neutrogena Oil-Free Acne Wash", 2500)],
    "Dry": [("Cetaphil Gentle Skin Cleanser 125 ml", 1500),
            ("CeraVe Hydrating Cleanser 236 ml", 3100),
            ("QYASO / Conatural Cream Cleanser", 900)],
    "Combination": [("Cetaphil Gentle Skin Cleanser 125 ml", 1500),
                    ("CeraVe Foaming Cleanser 236 ml", 3200)],
    "Normal": [("Garnier Micellar Water 125 ml", 700),
               ("Cetaphil Gentle Skin Cleanser 125 ml", 1500)],
}
_MOISTURISERS = {
    "Oily": [("Ponds Light Moisturiser 100 ml", 650),
             ("Neutrogena Hydro Boost Gel", 3500)],
    "Dry": [("CeraVe Moisturising Lotion 236 ml", 3200),
            ("Nivea Soft Cream 100 ml", 700),
            ("QYASO Ceramide Moisturiser", 1200)],
    "Combination": [("Cetaphil Daily Oil-Free Moisturiser", 2200),
                    ("Ponds Light Moisturiser 100 ml", 650)],
    "Normal": [("Nivea Soft Cream 100 ml", 700),
               ("Ponds Light Moisturiser 100 ml", 650)],
}
_SUNSCREEN = [("Face Guard SPF 30 (local)", 1200),
              ("Rivaj UK Sunblock SPF 60", 900),
              ("Neutrogena Ultra Sheer SPF 50+", 3500)]

# Gentle, non-prescription only. Supplements are Pakistani brands.
_CONDITION_KB = {
    "Redness / inflammation (possible acne or sensitivity)": {
        "actives": [("Conatural Pure Aloe Vera Gel 100 ml", 700),
                    ("La Roche-Posay Toleriane Soothing Fluid", 5500),
                    ("Jenpharm Niacin Sheer 4% Niacinamide", 950)],
        "supplements": [("Nutrifactor Zinc Plus 20 mg", 650),
                        ("Nutrifactor Vitamin E 400 IU", 700)],
        "diet": ["Cut deep-fried and very spicy food for 4-6 weeks",
                 "More fish, walnuts and seeds for omega-3",
                 "Green tea 1-2 cups/day; 3 L water/day"],
    },
    "Uneven texture (possible active acne or scarring)": {
        "actives": [("The Ordinary Azelaic Acid 10% (well-tolerated)", 3300),
                    ("Jenpharm Niacin Sheer 4% Niacinamide", 950),
                    ("QYASO Centella (Cica) Soothing Gel", 1300)],
        "supplements": [("Nutrifactor Zinc Plus 20 mg", 650),
                        ("Nutrifactor Once Daily Multivitamin", 950)],
        "diet": ["Lower high-glycaemic foods: white bread, sugary chai, bakery items",
                 "Reduce flavoured/sweetened dairy for 6 weeks and reassess",
                 "Be patient - texture improves over months, not days"],
    },
    "Hyperpigmentation / dark spots / uneven tone": {
        "actives": [("Conatural Vitamin C Facial Serum (mild)", 2200),
                    ("The Ordinary Ascorbic Acid 8% + Alpha Arbutin 2%", 3500),
                    ("The Ordinary Azelaic Acid 10%", 3300)],
        "supplements": [("Nutrifactor Vitamin C 500 mg", 550),
                        ("Nutrifactor Vitamin E 400 IU", 700)],
        "diet": ["Daily vitamin C foods: amrood, citrus, tomato, bell pepper",
                 "Strict daily SPF and re-apply - pigmentation will not fade without it",
                 "Colourful vegetables for antioxidants"],
    },
    "Excess oil / enlarged pores": {
        "actives": [("The Ordinary Niacinamide 10% + Zinc 1%", 3200),
                    ("Simple Soothing Alcohol-free Toner", 1500)],
        "supplements": [("Nutrifactor Zinc Plus 20 mg", 650),
                        ("Nutrifactor Once Daily Multivitamin", 950)],
        "diet": ["Reduce fried food, ghee-heavy salan and sugary drinks",
                 "More fibre: sabzi, salad, whole daal",
                 "Don't over-wash - a gentle cleanser twice a day is plenty"],
    },
    "Clear / generally healthy skin": {
        "actives": [("The Ordinary Niacinamide 10% (maintenance, optional)", 3200)],
        "supplements": [("Nutrifactor Once Daily Multivitamin", 950)],
        "diet": ["Keep it up: balanced plate, seasonal fruit and veg, 2.5-3 L water",
                 "Daily sunscreen to prevent future pigmentation"],
    },
}


def _routine(skin_type: str) -> dict:
    cl = _CLEANSERS.get(skin_type, _CLEANSERS["Normal"])[0]
    mo = _MOISTURISERS.get(skin_type, _MOISTURISERS["Normal"])[0]
    ss = _SUNSCREEN[1]
    return {
        "AM": [f"Gentle cleanser — {cl[0]} (PKR {cl[1]:,})",
               f"Moisturiser — {mo[0]} (PKR {mo[1]:,})",
               f"Sunscreen SPF 30+ — {ss[0]} (PKR {ss[1]:,})  (most important step)"],
        "PM": [f"Gentle cleanser — {cl[0]}",
               "Optional gentle serum (niacinamide or azelaic), 2-3 nights/week, "
               "patch-test first",
               f"Moisturiser — {mo[0]}"],
    }


def _recommend_builtin(skin_type: str, conditions: list[str]) -> dict:
    products, supplements, diet = [], [], []
    seen_p, seen_s, seen_d = set(), set(), set()

    for name, price in (_CLEANSERS.get(skin_type, _CLEANSERS["Normal"])
                        + _MOISTURISERS.get(skin_type, _MOISTURISERS["Normal"])
                        + _SUNSCREEN):
        if name not in seen_p:
            products.append({"category": "Skincare", "name": name, "price_pkr": price})
            seen_p.add(name)

    for cond in conditions:
        kb = _CONDITION_KB.get(cond, {})
        for name, price in kb.get("actives", []):
            if name not in seen_p:
                products.append({"category": "Active / treatment", "name": name, "price_pkr": price})
                seen_p.add(name)
        for name, price in kb.get("supplements", []):
            if name not in seen_s:
                supplements.append({"name": name, "price_pkr": price})
                seen_s.add(name)
        for tip in kb.get("diet", []):
            if tip not in seen_d:
                diet.append(tip)
                seen_d.add(tip)

    return {
        "skincare_routine": _routine(skin_type),
        "products": products,
        "supplements": supplements,
        "diet_changes": diet,
        "safety_note": _safety_note(),
    }


_AI_RECO_SCHEMA = (
    '{"skincare_routine":{"AM":["step - Product Name (PKR 000)"],'
    '"PM":["step - Product Name (PKR 000)"]},'
    '"products":[{"category":"Cleanser","name":"Brand Product","price_pkr":0}],'
    '"supplements":[{"name":"Supplement","price_pkr":0}],'
    '"diet_changes":["short actionable tip"]}'
)

# Prepended to every Gemini skincare prompt.
_SAFE_CONSTRAINT = (
    "SAFETY RULES - follow strictly:\n"
    "- Only recommend SAFE, GENTLE skincare. Better safe than sorry.\n"
    "- Do NOT recommend tretinoin, adapalene (Differin), any strong retinoid or "
    "retinol, or benzoyl peroxide.\n"
    "- For sensitive or dry skin do NOT recommend salicylic / glycolic / AHA / BHA "
    "leave-on exfoliants or peels.\n"
    "- Focus on: cleanse, moisturise, sunscreen, done consistently. Niacinamide or "
    "azelaic acid are the strongest actives you may suggest, and only as optional, "
    "patch-tested, 2-3 nights a week.\n"
)

# Product-name fragments that must never be recommended (harsh / prescription /
# skin-bleaching).
_HARSH_TERMS = (
    "tretinoin", "retin-a", "retino", "retinaldehyde", "adapalene", "differin",
    "isotretinoin", "accutane", "tazarotene", "retinol", "retinyl",
    "benzoyl peroxide", "benzoyl", "bpo", "on-the-spot", "acne control",
    "salicylic", "glycolic", "lactic acid", " aha", " bha", "peel",
    "hydroquinone", "melacare", "kojic", "mercury",
    "beauty cream", "gluta", "glutathione", "bleach",
)


def _is_harsh(name: str) -> bool:
    n = f" {name.lower()} "
    return any(h in n for h in _HARSH_TERMS)


_SAFETY_NOTE_FALLBACK = (
    "Cautious approach: gentle products only - no tretinoin, adapalene, strong "
    "retinoids or benzoyl peroxide. Routine = cleanse, moisturise, sunscreen, "
    "consistently. Add at most one new product a week and patch-test first."
)


def _safety_note() -> str:
    try:
        import price_data
        return price_data.SAFETY_NOTE
    except Exception:
        return _SAFETY_NOTE_FALLBACK


def _recommend_ai(skin_type: str, conditions: list[str]) -> dict | None:
    try:
        import gemini_client

        if not gemini_client.is_available():
            return None
        prompt = (
            f"A user in Pakistan has {skin_type.lower()} skin with these observations: "
            f"{'; '.join(conditions)}.\n"
            + _SAFE_CONSTRAINT +
            "Recommend a simple AM/PM routine and gentle products actually sold in "
            "Pakistan (CeraVe, Cetaphil, Simple, QYASO, Physiogel, Conatural, "
            "Saeed Ghani, Jenpharm; niacinamide or azelaic acid are the strongest "
            "allowed) with REAL approximate PKR prices. Supplements: Pakistani "
            "brands only (Nutrifactor, Hilton Pharma, CCL, or a pharmacy generic) - "
            "safe vitamins / minerals / biotin, no prescription items.\n"
            "This is cosmetic guidance only, not medical treatment.\n"
            "Return ONLY valid minified JSON (no markdown) with this exact schema:\n"
            + _AI_RECO_SCHEMA
        )
        raw = gemini_client.generate(prompt, temperature=0.5)
        data = gemini_client.extract_json(raw)
        if not isinstance(data, dict):
            return None

        routine = data.get("skincare_routine") or {}
        am = [str(x) for x in routine.get("AM", []) if str(x).strip()]
        pm = [str(x) for x in routine.get("PM", []) if str(x).strip()]

        products = []
        for p in data.get("products", []):
            if not isinstance(p, dict) or not p.get("name"):
                continue
            nm = str(p["name"]).strip()
            if _is_harsh(nm):            # drop anything harsh Gemini slipped in
                continue
            products.append({
                "category": str(p.get("category") or "Skincare"),
                "name": nm,
                "price_pkr": _price(p.get("price_pkr") or p.get("price")),
            })
        supplements = []
        for s in data.get("supplements", []):
            if not isinstance(s, dict) or not s.get("name"):
                continue
            supplements.append({
                "name": str(s["name"]).strip(),
                "price_pkr": _price(s.get("price_pkr") or s.get("price")),
            })
        diet = [str(x).strip() for x in data.get("diet_changes", []) if str(x).strip()]

        if not (am and pm and products):
            return None
        return {
            "skincare_routine": {"AM": am, "PM": pm},
            "products": products,
            "supplements": supplements,
            "diet_changes": diet,
            "safety_note": _safety_note(),
        }
    except Exception:
        return None


def _price(value) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# CSV-priced recommendations - every trusted brand in skincare_brands.csv
# --------------------------------------------------------------------------- #
VERIFIED_SKINCARE_BRANDS = ("cerave", "neutrogena", "cetaphil", "la roche-posay",
                            "the ordinary", "olay", "nivea", "dove", "garnier",
                            "simple", "ponds", "golden pearl", "jenpharm", "conatural",
                            "physiogel", "qyaso", "saeed ghani")
VERIFIED_SUPPLEMENT_BRANDS = ("nutrifactor", "hilton pharma", "hilton", "ccl",
                              "jenpharm", "local pharmacy")

_CATEGORY_LABEL = {
    "cleanser": "Gentle cleanser", "toner": "Alcohol-free toner",
    "soothing": "Soothing / barrier", "serum": "Gentle serum",
    "treatment": "Gentle treatment (azelaic / niacinamide)",
    "moisturizer": "Moisturizer", "sunscreen": "Sunscreen",
}
_AM_ORDER = ["cleanser", "toner", "serum", "moisturizer", "sunscreen"]
_PM_ORDER = ["cleanser", "toner", "soothing", "treatment", "serum", "moisturizer"]

# Overwhelm control: show only the TOP options per category (rest -> "See all").
_TOP_N = {"cleanser": 2, "moisturizer": 2, "sunscreen": 2, "toner": 1,
         "soothing": 1, "serum": 1, "treatment": 2}
_TOP_N_DEFAULT = 2

_USAGE = {
    "cleanser": "AM & PM - massage 30s on damp skin, rinse. Nothing harsh.",
    "toner": "Optional. Alcohol-free only - sweep over with hands after cleansing.",
    "soothing": "Any time skin feels tight, red or irritated - a thin layer.",
    "serum": "Optional. Thin layer after cleansing, before moisturiser.",
    "treatment": "Optional. Azelaic acid or niacinamide only, 2-3 nights/week, "
                 "patch-test first. No retinoids, no benzoyl peroxide.",
    "moisturizer": "AM & PM - the step most people skip. Last step before sunscreen.",
    "sunscreen": "AM, every day - 2 finger-lengths, reapply every 3-4 h outdoors. "
                 "The single most important product.",
}


def _brand_ok(name: str, brands=VERIFIED_SKINCARE_BRANDS) -> bool:
    n = name.lower()
    return any(b in n for b in brands) or "azelaic acid" in n or "niacinamide" in n


def _rank_key(r: dict, fragments: set[str]):
    """Sort key for 'best first': relevance to the condition DESC, verified
    brand DESC, price ASC (cheaper wins a tie on quality)."""
    name = f"{r['brand']} {r['product']}".lower()
    relevance = sum(1 for f in fragments if f in name)
    trusted = 1 if _brand_ok(r["brand"]) else 0
    price = r["price_pkr"] if r["price_pkr"] else 10 ** 9
    return (-relevance, -trusted, price)


def _pick_reason(idx: int, cap: int, r: dict, fragments: set[str]) -> tuple[str, str]:
    """Return (label, why) for the option shown at position `idx` (0-based)."""
    name = f"{r['brand']} {r['product']}".lower()
    matched = sorted({f for f in fragments if f in name})
    if idx == 0:
        bits = [f"targets {', '.join(matched[:2])}"] if matched else ["gentle, widely suitable"]
        if _brand_ok(r["brand"]):
            bits.append("verified brand")
        return "Best Match", "; ".join(bits).capitalize()
    if cap > 1 and idx == cap - 1:
        price_bit = f"Rs {r['price_pkr']:,}" if r["price_pkr"] else "no listed price"
        return "Budget Pick", f"Cheapest gentle option here - {price_bit}."
    return "Also Good", "Another gentle, well-suited option."


def _recommend_csv(skin_type: str, conditions: list[str]) -> tuple[dict, str] | None:
    """Pull every relevant product (all brands) from skincare_brands.csv."""
    import price_data

    opts = price_data.skincare_options(skin_type, conditions)
    if opts["error"] or not opts["flat"]:
        return None
    by_cat = opts["by_category"]

    def _step(cat):
        picks = by_cat.get(cat) or []
        if not picks:
            return None
        cheap = min(picks, key=lambda r: r["price_pkr"] or 10 ** 9)
        n = len(picks)
        return (f"{_CATEGORY_LABEL.get(cat, cat.title())} - e.g. {cheap['brand']} "
                f"{cheap['product']} (Rs {cheap['price_pkr']:,})  "
                f"[{n} option{'s' if n != 1 else ''} below]")

    am = [s for c in _AM_ORDER if (s := _step(c))]
    pm = [s for c in _PM_ORDER if (s := _step(c))]

    fragments = price_data.condition_fragments(conditions)
    products, option_groups = [], []
    ordered_cats = _PM_ORDER + [c for c in by_cat if c not in _PM_ORDER]
    for cat in ordered_cats:
        rows = by_cat.get(cat)
        if not rows:
            continue
        ranked = sorted(rows, key=lambda r: _rank_key(r, fragments))
        cap = _TOP_N.get(cat, _TOP_N_DEFAULT)
        top, rest = ranked[:cap], ranked[cap:]
        priced = [r["price_pkr"] for r in rows if r["price_pkr"]]

        shown = []
        for i, r in enumerate(top):
            label, why = _pick_reason(i, len(top), r, fragments)
            shown.append({"brand": r["brand"], "product": r["product"],
                         "price_pkr": r["price_pkr"], "where_to_buy": r["shop"],
                         "authentic_source": r["authentic_source"],
                         "label": label, "why": why})

        option_groups.append({
            "category": _CATEGORY_LABEL.get(cat, cat.title()),
            "usage": _USAGE.get(cat, ""),
            "price_low": min(priced) if priced else None,
            "price_high": max(priced) if priced else None,
            "n_total": len(rows),
            "options": shown,
            "more_options": [
                {"brand": r["brand"], "product": r["product"], "price_pkr": r["price_pkr"],
                 "where_to_buy": r["shop"], "authentic_source": r["authentic_source"]}
                for r in rest
            ],
        })
        for r in rows:
            products.append({
                "category": _CATEGORY_LABEL.get(cat, cat.title()),
                "brand": r["brand"], "product": r["product"],
                "name": f"{r['brand']} {r['product']}",
                "price_pkr": r["price_pkr"],
                "where_to_buy": r["shop"], "authentic_source": r["authentic_source"],
            })

    sup = price_data.supplement_options(conditions)
    supplements = [
        {"name": f"{r['brand']} {r['product']}", "brand": r["brand"], "product": r["product"],
         "price_pkr": r["price_pkr"], "dosage": r["dosage"], "timing": r["timing"],
         "where_to_buy": r["shop"], "authentic_source": r["authentic_source"],
         "suggested": r["relevant"]}
        for r in sup["rows"]
    ]

    diet = _CONDITION_KB.get(conditions[0], {}).get("diet", []) if conditions else []
    if not diet:
        diet = ["2.5-3 L water/day",
                "More seasonal fruit, leafy greens, dahi, nuts",
                "Less deep-fried food, sugary chai and bakery items"]

    recs = {
        "skincare_routine": {"AM": am or ["Gentle cleanser, moisturiser, sunscreen"],
                             "PM": pm or ["Gentle cleanser, moisturiser"]},
        "products": products,
        "product_options": option_groups,
        "supplements": supplements,
        "diet_changes": list(diet),
        "authenticity_tips": price_data.AUTHENTICITY_TIPS,
        "safety_note": price_data.SAFETY_NOTE,
        "brands_note": ("Gentle products only - no retinoids, no benzoyl peroxide; "
                        "leave-on acids are skipped for sensitive / dry skin. "
                        "Supplements are Pakistani brands only (Nutrifactor, Hilton "
                        "Pharma, CCL, or a pharmacy generic) - safe vitamins, minerals "
                        "and biotin, nothing that needs a prescription."),
    }
    return recs, "csv"


# --------------------------------------------------------------------------- #
# Live internet product search (SerpAPI) - BROAD condition-based queries.
# We do NOT search for named products; we ask "what exists for this condition"
# and let Gemini pick ONLY from whatever the search actually returned.
# --------------------------------------------------------------------------- #
_MAX_QUERIES = 4    # 3 for the primary condition + 1 for a distinct secondary one


def _condition_label(skin_type: str, conditions: list[str]) -> str:
    text = " ".join(conditions).lower()
    tags = []
    if "acne" in text or "texture" in text:
        tags.append("acne")
    if "redness" in text or "inflammation" in text:
        tags.append("sensitivity / redness")
    if "pigment" in text or "spot" in text:
        tags.append("dark spots")
    if "oil" in text or "pore" in text:
        tags.append("excess oil")
    label = skin_type
    if tags:
        label += " skin with " + " + ".join(tags)
    else:
        label += " skin"
    return label


def _skin_categories(skin_type: str, conditions: list[str]) -> list[str]:
    """Every condition category this skin shows, most prominent first. Order
    matters: index 0 drives the primary (3-query) search, index 1 (if any and
    distinct) adds one extra targeted query.

    Parenthetical asides are stripped first - a finding's descriptive text (e.g.
    "Redness / inflammation (possible acne or sensitivity)") must not make a
    pure-redness case match the "acne" keyword."""
    text = re.sub(r"\([^)]*\)", " ", " ".join(conditions)).lower()
    cats = []
    if "acne" in text or "texture" in text:
        cats.append("acne")
    if "redness" in text or "inflammation" in text:
        cats.append("sensitive")
    if "pigment" in text or "spot" in text:
        cats.append("pigmentation")
    if "oil" in text or "pore" in text:
        cats.append("oily")
    if not cats:
        cats.append("dry" if skin_type.lower() == "dry" else "general")
    return cats


# Short, SINGLE-concept queries (3-4 words) per condition category. Never chain
# two conditions into one query string - that is what was returning zero
# matches. Country + a named marketplace keep results Pakistan-relevant.
_QUERY_TEMPLATES: dict[str, list[str]] = {
    "dry": ["dry skin moisturizer Pakistan", "face cream dry skin Daraz",
            "hydrating serum Pakistan price"],
    "oily": ["oily skin treatment Pakistan", "oil control cream Daraz",
             "cleanser oily skin Pakistan"],
    "acne": ["acne treatment Pakistan", "acne cream Daraz price",
             "salicylic acid cleanser Pakistan"],
    "sensitive": ["sensitive skin cream Pakistan", "redness cream Daraz",
                  "soothing moisturizer Pakistan price"],
    "pigmentation": ["dark spot cream Pakistan", "pigmentation cream Daraz",
                     "brightening serum Pakistan price"],
    "general": ["gentle moisturizer Pakistan", "sunscreen Daraz price",
                "gentle cleanser Pakistan"],
}


def _search_queries(skin_type: str, conditions: list[str]) -> list[str]:
    """3-4 short, single-concept queries. Primary category gets its 3 templates;
    a distinct secondary category (e.g. acne + redness) adds one more query."""
    cats = _skin_categories(skin_type, conditions)
    queries = list(_QUERY_TEMPLATES.get(cats[0], _QUERY_TEMPLATES["general"]))
    if len(cats) > 1:
        queries.append(_QUERY_TEMPLATES.get(cats[1], _QUERY_TEMPLATES["general"])[0])
    return queries[:_MAX_QUERIES]


_CONSTRAINT = (
    "RULES:\n"
    "- The numbered list below is ALL the products an internet search of the "
    "Pakistan market returned. Treat it as the only products that exist.\n"
    "- Do NOT suggest any product that is not in this list.\n"
    "- Do NOT invent product names, brands, or ingredients.\n"
    "- Recommend ONLY by choosing numbers from the list, best-suited first.\n"
)


def _gemini_pick(skin_type, conditions, products):
    """Gemini chooses indices from the real-product list only. Returns
    {"picks": {index: why}, "summary": str}. Out-of-range indices are dropped."""
    try:
        import gemini_client

        if not gemini_client.is_available() or not products:
            return {}
        listing = "\n".join(
            f"{i}. {p['name']}"
            + (f"  (~Rs {p['price_pkr']:,})" if p["price_pkr"] else "  (price not listed)")
            + (f"  [{p['n_sources']} sites]" if p.get("n_sources", 0) > 1 else "")
            for i, p in enumerate(products, 1)
        )
        prompt = (
            f"A person in Pakistan has {skin_type.lower()} skin with these observations: "
            f"{'; '.join(conditions)}.\n\n" + _SAFE_CONSTRAINT + "\n" + _CONSTRAINT +
            "\nProducts found:\n" + listing +
            "\n\nPick the 5-10 SAFEST, gentlest options (skip anything harsh even if "
            "it is in the list). Return ONLY minified JSON:\n"
            '{"picks":[{"n":<number>,"why":"<one sentence: why it suits THIS skin>"}],'
            '"summary":"<1-2 sentence overview of the gentle routine>"}'
        )
        data = gemini_client.extract_json(gemini_client.generate(prompt, temperature=0.3))
        picks = {}
        for item in (data.get("picks", []) if isinstance(data, dict) else []):
            try:
                n = int(item["n"])
            except (KeyError, ValueError, TypeError):
                continue
            if 1 <= n <= len(products):
                picks[n] = str(item.get("why", "")).strip()
        return {"picks": picks,
                "summary": str(data.get("summary", "")).strip()
                if isinstance(data, dict) else ""}
    except Exception:
        return {}


_WHY_BY_CONDITION = {
    "oil": "helps control oil and shine",
    "pore": "helps refine and unclog pores",
    "acne": "targets acne-causing bacteria / clogged pores",
    "texture": "smooths uneven texture and post-acne marks over time",
    "redness": "calms redness and visible inflammation",
    "inflammation": "soothes inflamed, reactive skin",
    "pigment": "fades dark spots and evens tone (use with daily SPF)",
    "spot": "fades post-acne / sun spots",
    "dry": "restores moisture and repairs the skin barrier",
}


def _fallback_why(name: str, conditions: list[str]) -> str:
    text = " ".join(conditions).lower()
    for key, why in _WHY_BY_CONDITION.items():
        if key in text:
            return f"Relevant to your concern - {why}."
    return "A gentle staple suitable for daily use."


def _mk_product(p: dict, why: str = "") -> dict:
    return {
        "name": p["name"],
        "price_pkr": p["price_pkr"],
        "price_range": (f"Rs {p['price_low']:,} - {p['price_high']:,}"
                        if p.get("price_low") and p.get("price_high")
                        and p["price_low"] != p["price_high"] else ""),
        "shop": ", ".join(p.get("sources", [])[:3]) or "see link",
        "link": p.get("link", ""),
        "verified": p.get("verified", False),
        "n_sources": p.get("n_sources", 1),
        "why": why,
    }


def _recommend_search(skin_type: str, conditions: list[str]) -> tuple[dict, str] | None:
    """Broad SerpAPI search -> Gemini picks from the real results only.

    Fallback cascade - Level 1 (SerpAPI) -> Level 2 (meals..skincare CSV, via the
    None return here) -> Level 3 (built-in list, inside _recommend_csv). Returns
    None (never an error the user sees) when: no API key, the quota is
    exhausted, or every query genuinely found nothing.
    """
    import product_search

    if not product_search.is_available():
        return None                                    # Level 1 unavailable

    queries = _search_queries(skin_type, conditions)
    raw, listings, errors, quota_hit = [], [], [], False
    for q in queries:
        res = product_search.search_products(q)
        if res.get("quota_exhausted"):
            quota_hit = True
        if res.get("error") and not res.get("products"):
            errors.append(f"{q}: {res['error']}")
        raw.extend(res.get("products", []))
        listings.extend(res.get("listings", []))

    if quota_hit:
        return None                                    # Level 1 quota exhausted -> CSV

    merged = [m for m in product_search.verify_products(raw) if not _is_harsh(m["name"])]
    if not merged:
        return None                                    # Level 1 found nothing real -> CSV

    diet = _CONDITION_KB.get(conditions[0], {}).get("diet", []) if conditions else []
    if not diet:
        diet = ["2.5-3 L water/day", "More seasonal fruit, leafy greens, dahi, nuts",
                "Less deep-fried food, sugary chai and bakery items"]
    supplements, auth_tips = [], []
    try:
        import price_data

        supplements = [
            {"name": f"{r['brand']} {r['product']}", "price_pkr": r["price_pkr"],
             "dosage": r["dosage"], "timing": r["timing"], "where_to_buy": r["shop"],
             "suggested": r["relevant"]}
            for r in price_data.supplement_options(conditions).get("rows", [])
        ]
        auth_tips = list(price_data.AUTHENTICITY_TIPS)
    except Exception:
        pass

    label = _condition_label(skin_type, conditions)
    seen_shops = []
    for x in listings:
        s = (x.get("source") or "").strip()
        if s and s not in seen_shops:
            seen_shops.append(s)

    base = {
        "condition_label": label,
        "skincare_routine": {
            "AM": ["Gentle cleanser", "Optional gentle serum (niacinamide)",
                   "Moisturiser", "Sunscreen (most important)"],
            "PM": ["Gentle cleanser", "Optional gentle serum, 2-3 nights/week",
                   "Moisturiser"],
        },
        "diet_changes": list(diet),
        "supplements": supplements,
        "authenticity_tips": auth_tips,
        "safety_note": _safety_note(),
        "searched_queries": queries,
        "search_errors": errors,
        "browse_shops": seen_shops[:8],
    }

    picked = _gemini_pick(skin_type, conditions, merged)
    pick_map = picked.get("picks", {}) if isinstance(picked, dict) else {}
    chosen = sorted(pick_map) if pick_map else list(range(1, min(len(merged), 10) + 1))

    products = [_mk_product(merged[n - 1], pick_map.get(n)
                            or _fallback_why(merged[n - 1]["name"], conditions))
                for n in chosen]
    picked_names = {p["name"] for p in products}
    other = [_mk_product(m) for m in merged if m["name"] not in picked_names][:25]

    base.update({
        "products": products,
        "other_products": other,
        "search_summary": picked.get("summary", "") if isinstance(picked, dict) else "",
        "found_note": (f"{len(merged)} real products found across {len(queries)} broad "
                       f"searches; Gemini picked {len(products)} for your skin. "
                       f"✅ = seen on 2+ sites."),
    })
    return base, "search"


_ACTIVE_CATS = {"Serum", "Treatment", "Gentle treatment (azelaic / niacinamide)",
                "Exfoliant (BHA/AHA)", "Toner", "Alcohol-free toner", "Acne spot treatment"}


def _limit_to_level(recs: dict, level: str) -> dict:
    """level 'light' -> keep only cleanser/moisturiser/sunscreen/soothing + ONE
    optional niacinamide serum; drop treatments. 'treat' -> unchanged."""
    if level != "light":
        return recs

    def _is_niac(name: str) -> bool:
        return "niacinamide" in name.lower()

    keep_prefix = ("Gentle cleanser", "Cleanser", "Moisturizer", "Moisturiser",
                   "Sunscreen", "Soothing")
    groups = []
    for g in recs.get("product_options", []):
        cat = g.get("category", "")
        if any(cat.startswith(p) for p in keep_prefix):
            groups.append(g)
        elif "serum" in cat.lower():
            opts = [o for o in g["options"] if _is_niac(o.get("product", ""))][:2]
            if opts:
                groups.append({**g, "category": "Optional niacinamide serum",
                               "usage": "Optional - AM, gentle. Only if you want to; "
                                        "not required for minor concerns.",
                               "options": opts})
    recs["product_options"] = groups
    keep_names = {o["product"] for g in groups for o in g["options"]}
    recs["products"] = [p for p in recs.get("products", [])
                        if p.get("product") in keep_names or p.get("name", "") in keep_names]
    recs["level_note"] = ("Minor concern - basic care is enough. Cleanser, "
                          "moisturiser and daily sunscreen; a niacinamide serum is "
                          "optional. No strong actives or spot treatments for small "
                          "blemishes - prevention beats treatment here.")
    return recs


def _recommend(skin_type: str, conditions: list[str], mode: str = "csv",
               level: str = "treat") -> tuple[dict, str]:
    """Return (recommendations, source).

    level="basic" -> maintenance routine only (no actives)
    level="light" -> maintenance + one optional niacinamide serum
    level="treat" -> full gentle recommendations
    mode ("search"/"csv") only matters for "light"/"treat".
    """
    if level == "basic":
        return _recommend_maintenance(skin_type, "")

    if mode == "search":
        res = _recommend_search(skin_type, conditions)
        if res is not None:
            recs, src = res
            return _limit_to_level(recs, level), src
    csv_res = _recommend_csv(skin_type, conditions)
    if csv_res:
        recs, src = csv_res
        return _limit_to_level(recs, level), src
    return _limit_to_level(_recommend_builtin(skin_type, conditions), level), "builtin"


_CLEAR = "Clear / generally healthy skin"
_UNSURE = "Not assessed - photo not clear enough"


def _recommend_maintenance(skin_type: str, note: str) -> tuple[dict, str]:
    """Clear skin (or an unreliable photo) -> a minimal maintenance routine only.
    No treatments, no actives, no supplements, no pressure to buy."""
    groups = {"cleanser": [], "moisturizer": [], "sunscreen": []}
    src = "builtin"
    try:
        import price_data

        opts = price_data.skincare_options(skin_type, [])
        if not opts["error"]:
            by = opts["by_category"]
            for c in groups:
                groups[c] = [
                    {"brand": r["brand"], "product": r["product"],
                     "price_pkr": r["price_pkr"], "where_to_buy": r["shop"],
                     "authentic_source": r["authentic_source"]}
                    for r in sorted(by.get(c, []),
                                    key=lambda x: x["price_pkr"] or 10 ** 9)[:3]
                ]
            if all(groups.values()):
                src = "csv"
        safety = price_data.SAFETY_NOTE
    except Exception:
        safety = _safety_note()

    if not all(groups.values()):
        def _fb(rows):
            return [{"brand": "", "product": n, "price_pkr": p,
                     "where_to_buy": "Daraz / pharmacy", "authentic_source": ""}
                    for n, p in rows[:3]]
        groups["cleanser"] = _fb(_CLEANSERS.get(skin_type, _CLEANSERS["Normal"]))
        groups["moisturizer"] = _fb(_MOISTURISERS.get(skin_type, _MOISTURISERS["Normal"]))
        groups["sunscreen"] = _fb(_SUNSCREEN)

    _CAT = {"cleanser": "Gentle cleanser", "moisturizer": "Moisturizer",
            "sunscreen": "Sunscreen"}
    _USE = {"cleanser": "AM & PM (or just water in the AM if skin feels fine)",
            "moisturizer": "AM & PM", "sunscreen": "every morning - the one step worth doing daily"}
    products, option_groups = [], []
    for c in ("cleanser", "moisturizer", "sunscreen"):
        for o in groups[c]:
            products.append({"category": _CAT[c], "brand": o["brand"],
                             "product": o["product"],
                             "name": f"{o['brand']} {o['product']}".strip(),
                             "price_pkr": o["price_pkr"],
                             "where_to_buy": o["where_to_buy"],
                             "authentic_source": o["authentic_source"]})
        option_groups.append({"category": _CAT[c], "usage": _USE[c],
                              "options": groups[c]})

    recs = {
        "is_clear_message": note,
        "skincare_routine": {
            "AM": ["Gentle cleanser (or just water if skin feels fine)",
                   "Light moisturiser", "Sunscreen SPF 30+ - the one step worth doing daily"],
            "PM": ["Gentle cleanser", "Moisturiser"],
        },
        "products": products,
        "product_options": option_groups,
        "supplements": [],
        "diet_changes": ["Keep it up: balanced plate, seasonal fruit & veg, 2.5-3 L water/day",
                         "Daily sunscreen is the best long-term investment for skin"],
        "safety_note": safety,
        "brands_note": ("Maintenance only - nothing to treat. A cleanser, a "
                        "moisturiser and daily sunscreen are all that's needed. "
                        "No serums, actives or treatments recommended."),
    }
    return recs, src


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def analyze(image, mode: str = "csv") -> dict:
    """mode: "csv" (offline brand database) or "search" (live SerpAPI + Gemini).

    Defaults to CLEAR skin unless the evidence is strong. An unclear photo is
    reported as such instead of inventing conditions.
    """
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    img = image.convert("RGB")

    m = _cv_metrics(img)
    rr = _resnet_readout(img)                      # ResNet50's own photo-sanity read-out
    agreement = float(rr["agreement"]) if rr else 0.85

    # ---- is this even a usable face photo? (CV skin region AND ResNet agree) ----
    reasons = []
    if not m.get("reliable"):
        reasons.append("not enough clearly-lit, front-facing skin is visible")
    if rr and rr["non_face"]:
        reasons.append(f"the model reads the picture as '{rr['top_label']}', not a face")
    assessed = not reasons

    pred = _trained_predict(img) if assessed else None      # skin_model_v2.pth, if trained

    # ---- skin type (a descriptor, not a 'finding') ----
    sh = m.get("shine_score", 0.0)
    if not assessed:
        skin_type = "Normal"
    elif pred and any(k in pred["probs"] for k in ("oily", "dry", "combination")):
        st_probs = {k: pred["probs"].get(k, 0.0) for k in ("oily", "dry", "combination", "normal")}
        st_top = max(st_probs, key=st_probs.get)
        skin_type = _SKINTYPE_CLASS.get(st_top, "Normal") if st_probs[st_top] >= 0.40 else "Normal"
    elif sh >= _THRESH["shine"]:
        skin_type = "Oily"
    elif sh <= 0.004 and m["brightness"] < 0.55:
        skin_type = "Dry"
    elif 0.004 < sh < _THRESH["shine"]:
        skin_type = "Combination"
    else:
        skin_type = "Normal"

    # ---- candidate concerns -------------------------------------------------
    # trained model probabilities (temperature-calibrated) are the primary
    # signal when skin_model_v2.pth exists; otherwise the calibrated CV metrics.
    candidates: list[dict] = []
    if not assessed:
        pass
    elif pred is not None:
        by_label: dict[str, dict] = {}
        for cls, p in pred["probs"].items():
            if cls in _CLEAR_CLASSES:
                continue
            label = _FINDING_FOR_CLASS.get(cls)
            if not label:
                continue
            conf = round(float(p) * 100.0, 1)
            if conf < _MIN_OBSERVATION_CONF:
                continue
            cur = by_label.get(label)
            if cur is None or conf > cur["confidence"]:
                by_label[label] = {"name": label, "metric": f"model:{cls}",
                                   "score": round(float(p), 4), "confidence": conf,
                                   "severity": _severity_of(conf)}
        candidates = list(by_label.values())
    else:
        for label, mkey, cb, mi, mo, se in _FLAGS:
            score = float(m.get(mkey, 0.0))
            conf = _finding_confidence(score, cb, mi, mo, se, agreement)
            if conf >= _MIN_OBSERVATION_CONF:
                candidates.append({"name": label, "metric": mkey,
                                   "score": round(score, 4), "confidence": conf,
                                   "severity": _severity_of(conf)})
    candidates.sort(key=lambda c: -c["confidence"])
    overall = candidates[0]["confidence"] if candidates else 0.0

    findings = [c for c in candidates if c["confidence"] >= _BAND_CLEAR]   # actionable
    observations = [c for c in candidates if c["confidence"] < _BAND_CLEAR]  # minor / FYI

    if not assessed:
        conditions, is_clear = [_UNSURE], None
        confidence, level = 20.0, "basic"
        advice = "Not assessed"
        summary = ("Could not assess reliably - " + "; ".join(reasons)
                   + ". Please retake a clear, front-facing face photo in daylight, "
                     "no filters.")
    elif overall < _BAND_CLEAR:
        conditions, is_clear = [_CLEAR], True
        confidence, level = round(overall, 1), "basic"
        advice = "Skin looks clear - maintain your routine"
        if observations:
            summary = (f"{advice}. Very minor variation seen "
                       f"({observations[0]['name'].split(' (')[0].lower()}, "
                       f"~{observations[0]['confidence']:.0f}% - not actionable). "
                       "Cleanse, moisturise, sunscreen.")
        else:
            summary = advice + ". No concerns detected."
    elif overall < _BAND_TREAT:
        conditions = [f["name"] for f in findings]
        is_clear = False
        confidence, level = round(overall, 1), "light"
        advice = "Minor concerns - basic care"
        summary = (advice + ": " + "; ".join(
            f"{f['name'].split(' (')[0]} ({f['severity']}, {f['confidence']:.0f}%)"
            for f in findings) + ". Consistent basics; a niacinamide serum is optional.")
    else:
        conditions = [f["name"] for f in findings]
        is_clear = False
        confidence, level = round(overall, 1), "treat"
        advice = "Visible concerns - consider treatment"
        summary = (advice + ": " + "; ".join(
            f"{f['name'].split(' (')[0]} ({f['severity']}, {f['confidence']:.0f}%)"
            for f in findings) + ".")

    recommendations, reco_source = _recommend(skin_type, conditions, mode=mode, level=level)
    if level == "basic":
        recommendations["is_clear_message"] = summary

    return {
        "skin_type": skin_type,
        "condition": "; ".join(conditions),
        "conditions": conditions,
        "findings": findings,                       # actionable (>= 50% confidence)
        "observations": observations,               # minor / not-actionable (< 50%)
        "assessed": assessed,
        "is_clear": is_clear,
        # low / medium / high (per confidence band); "none" when clear / not assessed
        "severity": ("none" if (is_clear or not assessed)
                     else (findings[0]["severity"] if findings else "none")),
        "advice_band": advice,
        "care_level": level,                        # basic | light | treat
        "summary": summary,
        "confidence": confidence,                   # 0..97, scaled to real severity
        "metrics": m,
        "resnet": rr,
        "cnn": rr,                                  # back-compat alias
        "model": ({"trained": True, "file": _os.path.basename(_TRAINED_PATH),
                   "classes": (_load_trained() or {}).get("classes", []),
                   "probs": {k: round(v, 3) for k, v in pred["probs"].items()},
                   "top_class": pred["top_class"], "top_conf": round(pred["top_conf"], 3),
                   "test_accuracy": pred.get("accuracy")}
                  if pred else {"trained": False}),
        "backbone": (
            f"skin_model_v2 ({(_load_trained() or {}).get('arch', 'trained')}, "
            "temperature-calibrated) + CV cross-check" if pred else
            "ResNet50 read-out (photo sanity) + calibrated skin-region CV metrics"
            if rr else "calibrated skin-region CV metrics (torch unavailable)"),
        "recommendations": recommendations,
        "reco_source": reco_source,
        "disclaimer": (
            "Cosmetic wellness estimate only - not a medical diagnosis. Confidence "
            "reflects how visible/severe a concern is; minor blemishes score low and "
            "get maintenance advice, not treatment. See a dermatologist for "
            "persistent acne, sudden changes, or any lesion you are worried about."
        ),
    }
