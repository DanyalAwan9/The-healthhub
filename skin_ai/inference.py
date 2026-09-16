"""
inference.py - end-to-end skin analysis: image in, JSON out.

  from skin_ai.inference import SkinAnalyzer
  SkinAnalyzer("skin_model.pth").analyze("selfie.jpg")

  python inference.py --image selfie.jpg --model skin_model.pth

Output JSON
-----------
  assessed                       bool
  skin_type                      Normal | Dry | Oily | Combination   (Gemini Vision)
  skin_type_confidence           0..1   Gemini's own confidence (None if it was
                                        unavailable and the default was used)
  skin_type_source               UI string, e.g. "Gemini Vision (gemini-flash-latest)
                                 - visible T-zone shine with matte cheeks"
  skin_type_reasoning            str    Gemini's brief explanation
  skin_type_signals              list[str]  extra signals passed to / weighed by Gemini
  confidence                     0..0.97   (of the CONCERN read-out, tempered by
                                            model<->CV agreement and visible skin)
  blemish_count                  int    (model estimate fused with a CV blob count)
  redness_severity               clear | mild | moderate | severe
  hyperpigmentation_severity     clear | mild | moderate | severe
  texture_severity               clear | mild | moderate | severe
  primary_concerns               list[str]  - only concerns at moderate+ / conf >= 0.55
  concern_details                list[{concern, severity, confidence}]
  zones                          per-zone oiliness / redness quick read
  quality, skin_fraction, model, disclaimer

Reasoning rules (false-positive filter)
---------------------------------------
  * No face / too little skin / blur / bad light  -> assessed = False, nothing guessed.
  * A concern is only reported if BOTH the CNN head and an independent CV metric
    measured inside the skin mask (specular highlights and shadows removed) agree;
    a CNN spike with no CV support is capped at "clear".
  * Uniform ruddiness (high a* with low spatial variance) is treated as skin tone
    / warm lighting, not inflammation -> redness downgraded one level.
  * Blemish count is the CV blob count fused with the CNN estimate; if CV sees
    none, the CNN cannot invent more than 2.
  * Confidence scales with severity and with how much skin was actually visible.

Skin-type decision - Google Gemini Vision
-----------------------------------------
  The EfficientNet skin_type head (~61% accurate, and blind to oiliness because
  the crop it sees has specular shine preserved but no way to reason about glare
  vs sebum) has been replaced by gemini_vision.classify_skin_type_gemini():

    preprocessed face crop (colour-constant, shine kept) + a 0..1 blemish hint
      -> Gemini -> {skin_type, confidence, reasoning}

  The CNN still runs - its blemish / texture / redness heads are fused with the
  CV metrics exactly as before. If Gemini is unavailable (no GEMINI_API_KEY, SDK
  missing, or every fallback model failed) skin_type falls back to the
  conservative default "Combination" and skin_type_confidence is None.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

try:                                   # package import
    from .preprocessing import Preprocessor, eval_transform, local_contrast
    from .model import build_model, postprocess, SEVERITY, SKIN_TYPES
    from .gemini_vision import classify_skin_type_gemini, last_model as _gemini_model
except ImportError:                     # run as a script from inside skin_ai/
    from preprocessing import Preprocessor, eval_transform, local_contrast
    from model import build_model, postprocess, SEVERITY, SKIN_TYPES
    from gemini_vision import classify_skin_type_gemini, last_model as _gemini_model

log = logging.getLogger("healthhub.skin_ai")

_SIDX = {w: i for i, w in enumerate(SEVERITY)}

# conservative default when Gemini Vision is unavailable (no key / SDK / all
# fallback models failed) - "Combination" routines are the safest generic pick.
_SKIN_TYPE_DEFAULT = "Combination"


def _sev_from_index(x: float, t=(0.15, 0.35, 0.60)) -> int:
    return 3 if x >= t[2] else 2 if x >= t[1] else 1 if x >= t[0] else 0


class SkinAnalyzer:
    def __init__(self, weights: str = "skin_model.pth", device: str = "auto",
                 use_mediapipe: bool = True, model_path: str | None = None,
                 use_gemini_skin_type: bool = True):
        self.use_gemini_skin_type = bool(use_gemini_skin_type)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights = model_path or weights           # `model_path=` alias
        self.weights_path = weights
        ck = None
        if weights and os.path.isfile(weights):
            ck = torch.load(weights, map_location="cpu")
        c = ck or {}
        self.trained = ck is not None
        self.arch = c.get("arch", "efficientnet_b0")
        self.contrast_channel = bool(c.get("contrast_channel", True))
        self.img_size = int(c.get("img_size", 224))
        self.temperature = float(c.get("temperature", 1.0))
        self.skin_types = list(c.get("skin_types", SKIN_TYPES))
        self.severity = list(c.get("severity", SEVERITY))
        self.metrics = c.get("metrics", {})

        self.model = build_model(self.arch, pretrained=False,
                                 contrast_channel=self.contrast_channel).to(self.device).eval()
        if ck is not None:
            self.model.load_state_dict(ck["state_dict"])

        self.pre = Preprocessor(work_size=512, use_mediapipe=use_mediapipe)
        self._tf = eval_transform(self.img_size)

    # -- public ---------------------------------------------------------------
    @torch.no_grad()
    def analyze(self, image) -> dict:
        fr = self.pre(image)

        reasons = []
        if not fr.face_found:
            reasons.append("no face detected")
        if fr.skin_fraction < 0.10:
            reasons.append("not enough visible skin")
        if not fr.quality.get("sharp", False):
            reasons.append("image too blurry")
        b = fr.quality.get("brightness", 0.5)
        if not 0.12 <= b <= 0.96:
            reasons.append("uneven or poor lighting")
        if reasons:
            return self._not_assessed(reasons, fr)

        if self.trained:
            # model sees the occlusion-masked, colour-normalised, glare-clipped crop
            x = self._tf(Image.fromarray(fr.model_input)).unsqueeze(0).to(self.device)
            pp = postprocess(self.model(x), self.temperature)
            st = pp["skin_type"][0].cpu().numpy()
            red = pp["redness"][0].cpu().numpy()
            hp = pp["hyperpigmentation"][0].cpu().numpy()
            blem_model = float(pp["blemish_count"][0])
            tex_model = float(pp["texture"][0])
        else:
            # no checkpoint -> the CNN heads are random; run on CV signals alone.
            st = np.array([1.0, 0.0, 0.0, 0.0], np.float32)   # -> "Normal", low conf
            red = hp = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
            blem_model = tex_model = 0.0

        cv = self._cv_signals(fr)

        # ---- skin type: Google Gemini Vision (replaces the EfficientNet head) --
        gem = None
        if self.use_gemini_skin_type:
            cvc = int(cv.get("blemish_count", 0) or 0)
            if self.trained:
                cnt = (min(int(round(blem_model)), 2) if cvc == 0
                       else int(round(0.35 * blem_model + 0.65 * cvc)))
            else:
                cnt = cvc
            blem_hint = (0.0 if cnt == 0 else 0.33 if cnt <= 4
                         else 0.66 if cnt <= 12 else 1.0)
            try:
                gem = classify_skin_type_gemini(
                    Image.fromarray(fr.face_cc), blemish_severity=blem_hint)
            except Exception as exc:                       # never fail the analysis
                log.warning("skin-type: Gemini call errored: %s", exc)
                gem = None

        res = self._reason(st, red, hp, blem_model, tex_model, cv, fr,
                           trained=self.trained, gem=gem)
        res["model"] = {"trained": self.trained, "arch": self.arch,
                        "contrast_channel": self.contrast_channel,
                        "test_metrics": self.metrics or None}
        res["disclaimer"] = (
            "Automated cosmetic estimate, not a medical diagnosis. See a "
            "dermatologist for persistent, painful or changing skin concerns."
        )
        return res

    # -- CV cross-checks ----------------------------------------------------
    @staticmethod
    def _cv_signals(fr) -> dict:
        rgb = fr.rgb
        m = fr.skin_mask & ~fr.specular_mask & ~fr.shadow_mask
        # pull the mask ~20 px in from the skin/background boundary: the rim
        # (hairline, jaw, lips, nostrils) is a strong-gradient band that is not a
        # blemish, and it poisons the local-contrast statistics if left in.
        m = cv2.erode(m.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))).astype(bool)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[..., 0] * (100.0 / 255.0)
        A = lab[..., 1] - 128.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lc_fine = local_contrast(gray, 8)          # blemish scale
        lc_tex = local_contrast(gray, 20)          # the "20 px" texture signal

        if int(m.sum()) < 500:
            return {"blemish_count": 0, "redness_index": 0.0, "redness_uniform": True,
                    "hyperpig_index": 0.0, "texture_index": 0.0, "zones": {}}

        px = int(m.sum())
        a_sk, l_sk = A[m], L[m]
        a_base = float(np.median(a_sk))
        l_med = float(np.median(l_sk))
        l_mad = float(np.median(np.abs(l_sk - l_med))) + 1e-6

        # blemishes: fine local-contrast spikes that are ALSO meaningfully darker
        # or redder than the person's own median skin (relative test + absolute
        # floor so noise / soft gradients don't count) -> size-banded components
        v = lc_fine[m]
        c_med = float(np.median(v))
        c_mad = float(np.median(np.abs(v - c_med))) + 1e-6
        hot = (lc_fine > c_med + 3.5 * c_mad) & (lc_fine > 0.045) & m
        hot &= (L < l_med - 5.0) | (A > a_base + 6.0)
        n, _, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
        lo, hi = max(5.0, px * 4e-5), px * 4e-3
        blem = sum(1 for i in range(1, n) if lo <= stats[i, cv2.CC_STAT_AREA] <= hi)

        # redness: high a* percentile inside skin; uniform high a* -> tone / light
        p85 = float(np.percentile(a_sk, 85))
        redness_index = float(np.clip((p85 - a_base - 4.0) / 18.0, 0.0, 1.0))
        redness_uniform = bool(np.std(a_sk) < 4.0 and a_base > 8.0)

        # hyperpigmentation: SOFT dark patches (low local contrast rules out sharp
        # blemishes), >= min_area px, >= 5 L* units darker than median
        dark = ((L < l_med - 3.0 * l_mad) & (L < l_med - 5.0)
                & (lc_fine < 0.05) & m)
        n2, _, s2, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
        min_area = max(50.0, px * 3e-4)
        patch = sum(s2[i, cv2.CC_STAT_AREA] for i in range(1, n2)
                    if s2[i, cv2.CC_STAT_AREA] >= min_area)
        hyperpig_index = float(np.clip((patch / px) / 0.05, 0.0, 1.0))

        texture_index = float(np.clip(float(np.mean(lc_tex[m])) / 0.10, 0.0, 1.0))

        zones = {}
        for z, zm in fr.zones.items():
            zzm = zm & m
            if int(zzm.sum()) < 150:
                zones[z] = {"oiliness": "n/a", "redness": "n/a"}
                continue
            shine = float((fr.specular_mask & zm).sum()) / float(max(int(zm.sum()), 1))
            za = float(np.median(A[zzm])) - a_base
            zones[z] = {
                "oiliness": "high" if shine > 0.06 else "moderate" if shine > 0.02 else "low",
                "redness": "high" if za > 6 else "moderate" if za > 2.5 else "low",
            }

        return {"blemish_count": int(min(blem, 80)), "redness_index": redness_index,
                "redness_uniform": redness_uniform, "hyperpig_index": hyperpig_index,
                "texture_index": texture_index, "zones": zones}

    # -- reasoning ---------------------------------------------------------
    def _reason(self, st, red, hp, blem_model, tex_model, cv, fr,
                trained=True, gem=None) -> dict:
        sev = self.severity
        cv_red = _sev_from_index(cv["redness_index"])
        cv_hp = _sev_from_index(cv["hyperpig_index"])
        cvc = int(cv["blemish_count"])

        if trained:
            st_i = int(np.argmax(st))
            st_conf = float(st[st_i])                  # CNN's own read confidence
            m_red, m_hp = int(np.argmax(red)), int(np.argmax(hp))
            # CNN head and CV must agree: take the lower, allow CV +1 headroom
            red_sev = min(m_red, cv_red + 1)
            if cv_red == 0 and red[m_red] < 0.80:
                red_sev = 0
            hp_sev = min(m_hp, cv_hp + 1)
            if cv_hp == 0 and hp[m_hp] < 0.80:
                hp_sev = 0
            if cvc == 0:
                count = 0 if blem_model < 3.0 else min(int(round(blem_model)), 2)
            else:
                count = int(round(0.35 * blem_model + 0.65 * cvc))
            tex_val = 0.5 * tex_model + 0.5 * cv["texture_index"]
            agree = float(np.mean([
                1.0 - abs(m_red - cv_red) / 3.0,
                1.0 - abs(m_hp - cv_hp) / 3.0,
                1.0 - min(abs(blem_model - cvc), 10.0) / 10.0,
            ]))
            conf = 0.30 + 0.45 * st_conf + 0.25 * agree
        else:
            # no trained checkpoint: CV signals only, skin type undetermined,
            # confidence capped. Unsupervised CV on a real face reads normal
            # anatomy (folds, under-eye, nasolabial) as texture/spots, so this
            # path uses HIGHER severity thresholds and only surfaces strong,
            # unambiguous signal - it is a screen, not a classifier.
            st_conf = 0.0
            red_sev = _sev_from_index(cv["redness_index"], (0.35, 0.60, 0.82))
            hp_sev = _sev_from_index(cv["hyperpig_index"], (0.40, 0.65, 0.85))
            count = cvc
            tex_val = cv["texture_index"]
            conf = min(0.55, 0.30 + 0.10 * max(red_sev, hp_sev,
                                               0 if count == 0 else 2))

        if cv.get("redness_uniform") and red_sev > 0:
            red_sev -= 1
        count = int(np.clip(count, 0, 80))
        if trained:
            blem_sev = 0 if count == 0 else 1 if count <= 4 else 2 if count <= 12 else 3
            tex_sev = _sev_from_index(tex_val, (0.30, 0.50, 0.72))
        else:
            blem_sev = 0 if count <= 4 else 1 if count <= 15 else 2 if count <= 35 else 3
            tex_sev = _sev_from_index(tex_val, (0.45, 0.68, 0.88))

        conf *= float(np.clip(fr.skin_fraction / 0.25, 0.5, 1.0))
        conf = round(float(np.clip(conf, 0.0, 0.97)), 2)

        # skin type: Google Gemini Vision (gem), else a conservative default
        if gem is not None:
            skin_type = gem.skin_type
            st_conf_out = round(float(gem.confidence) / 100.0, 3)
            st_reasoning = gem.reasoning or ""
            st_source = f"Gemini Vision ({_gemini_model() or 'gemini'}) - {st_reasoning}"
            st_signals = ([f"local blemish severity ~{blem_sev / 3.0:.2f}"]
                          if blem_sev else [])
        else:
            skin_type = _SKIN_TYPE_DEFAULT
            st_conf_out = None
            st_reasoning = ""
            st_source = f"Gemini unavailable - conservative default ({_SKIN_TYPE_DEFAULT})"
            st_signals = []

        details = []

        def add(name, s):
            if s > 0:
                details.append({"concern": name, "severity": sev[s],
                                "confidence": round(min(0.97, conf * (0.7 + 0.1 * s)), 2)})

        add("redness", red_sev)
        add("hyperpigmentation", hp_sev)
        add("blemishes", blem_sev)
        add("uneven texture", tex_sev)

        primary = [d["concern"] for d in details
                   if _SIDX[d["severity"]] >= 2 and d["confidence"] >= 0.55]

        return {
            "assessed": True,
            "skin_type": skin_type,
            "skin_type_confidence": st_conf_out,
            "skin_type_source": st_source,
            "skin_type_reasoning": st_reasoning,
            "skin_type_signals": st_signals,
            "skin_type_note": ("Skin type: Google Gemini Vision. "
                               "Blemish / texture / redness: local CV + CNN."),
            "confidence": conf,
            "blemish_count": count,
            "redness_severity": sev[red_sev],
            "hyperpigmentation_severity": sev[hp_sev],
            "texture_severity": sev[tex_sev],
            "primary_concerns": primary,
            "concern_details": details,
            "zones": cv.get("zones", {}),
            "quality": {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in fr.quality.items()},
            "skin_fraction": round(float(fr.skin_fraction), 3),
        }

    def _not_assessed(self, reasons, fr) -> dict:
        return {
            "assessed": False,
            "reason": "; ".join(reasons),
            "message": ("Could not analyse reliably. Retake a clear, front-facing "
                        "photo in even daylight, no filter, whole face visible."),
            "skin_type": "Unknown",
            "skin_type_confidence": None,
            "skin_type_source": "not assessed",
            "skin_type_reasoning": "",
            "skin_type_signals": [],
            "skin_type_note": "",
            "confidence": 0.0,
            "blemish_count": 0,
            "redness_severity": "unknown",
            "hyperpigmentation_severity": "unknown",
            "texture_severity": "unknown",
            "primary_concerns": [],
            "concern_details": [],
            "zones": {},
            "quality": {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in fr.quality.items()},
            "skin_fraction": round(float(fr.skin_fraction), 3),
            "model": {"trained": self.trained, "arch": self.arch},
        }


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Skin analysis: image -> JSON")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="skin_model.pth")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-mediapipe", action="store_true")
    a = ap.parse_args(argv)

    az = SkinAnalyzer(a.model, device=a.device, use_mediapipe=not a.no_mediapipe)
    if not az.trained:
        print(f"# note: no trained weights at {a.model!r} - running on ImageNet "
              f"init + CV reasoning rules only", file=sys.stderr)
    print(json.dumps(az.analyze(a.image), indent=2))


if __name__ == "__main__":
    _main()
