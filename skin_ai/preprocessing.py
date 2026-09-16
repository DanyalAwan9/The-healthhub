"""
preprocessing.py - face-aware skin-image preprocessing for the HealthHub Skin
Analysis AI.

Pipeline (every stage degrades gracefully if an optional dependency is missing):

  1. Face detection & landmarks   MediaPipe Face Mesh / Face Landmarker
                                  (468-478 pts; legacy `solutions` API or the
                                  1.0+ Tasks API, auto-detected) -> Haar-cascade
                                  bounding box -> whole-frame fallback.
  2. Colour constancy             gray-world WB (cv2.xphoto SimpleWB / Grayworld /
                                  manual) - strips the warm/cool lighting cast.
                                  Applied on BOTH output paths.
  3. THREE crops                  * face_cc      (GEMINI): colour-constant full
                                    face crop, nothing erased or filled - Gemini
                                    Vision judges glare vs sebum itself.
                                  * model_input  (CNN): colour-constant, specular
                                    shine PRESERVED, non-skin filled with the skin
                                    median (blemish / texture / redness heads).
                                  * rgb          (DISPLAY / CV): additionally
                                    inpaints specular glare (HSV V>235 & S<60) and
                                    deep shadow (Lab L<60) with cv2.inpaint TELEA.
  4. Occlusion masking            skin-region polygon from the 468/478 landmarks
                                  (forehead + cheeks + nose + chin), minus eyes /
                                  brows / lips / nostrils; hair, glasses, neck and
                                  background fall outside it.
  5. Zones + soft shadow          forehead / cheek / nose / chin sub-masks; a
                                  relative directional-shade mask further trims the
                                  analysable skin.

Pipeline:  face_crop -> colour_constancy -> { face_cc | model_input | rgb (inpainted) }

Public API
----------
  Preprocessor(work_size=512, use_mediapipe=True)(image) -> FaceRegions
  local_contrast(gray, radius=20)      -> np.float32   (local standard deviation)
  training_augmentation(img_size=224)  -> torchvision transform
  eval_transform(img_size=224)         -> torchvision transform
  IMAGENET_MEAN, IMAGENET_STD
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# MediaPipe Face Mesh landmark clusters used as ellipse seeds for each face zone.
# (Indices < 468 so they are valid with or without refine_landmarks.)
_ZONE_SEEDS = {
    "forehead":    [10, 151, 9, 107, 336, 66, 296, 69, 299, 108, 337, 67, 297],
    "left_cheek":  [50, 101, 118, 117, 123, 116, 205, 207, 187, 147],
    "right_cheek": [280, 330, 347, 346, 352, 345, 425, 427, 411, 376],
    "nose":        [1, 4, 5, 195, 197, 6, 19, 94, 2, 98, 327],
    "chin":        [152, 175, 199, 200, 18, 83, 313, 421, 201, 208, 428],
}
_ZONES = tuple(_ZONE_SEEDS)

# non-skin features punched out of the skin mask (landmark rings, MediaPipe topology)
_FEATURE_RINGS = {
    "left_eye":  [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "right_eye": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    "lips":      [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
                  0, 37, 39, 40, 185],
    "l_brow":    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "r_brow":    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "nostrils":  [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64],
}

# MediaPipe FACEMESH_FACE_OVAL, ordered - the outer skin boundary (no hair / ears /
# neck / background outside it).
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
              379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
              234, 127, 162, 21, 54, 103, 67, 109]

# Per-region skin polygons (occlusion mask = union of these, hulled). Seeds follow
# the request: forehead 10/21/52/103..., cheeks 50/101/205/410..., nose
# 94/188/122/351..., chin 152/176/149/150... - plus neighbours for a smooth hull.
_SKIN_REGION_POLYS = {
    "forehead":    [10, 21, 52, 103, 67, 109, 338, 297, 332, 284, 251, 301, 293,
                    108, 69, 104, 68, 71, 9, 336, 296, 334, 105, 66, 107],
    "left_cheek":  [50, 101, 205, 410, 187, 147, 123, 116, 117, 118, 119, 100,
                    142, 129, 203, 206, 216, 92, 165, 98],
    "right_cheek": [280, 330, 425, 186, 411, 376, 352, 345, 346, 347, 348, 329,
                    371, 358, 423, 426, 436, 322, 391, 327],
    "nose":        [94, 188, 122, 351, 6, 197, 195, 5, 4, 1, 19, 2, 98, 327, 168,
                    114, 343, 45, 275, 236, 456],
    "chin":        [152, 176, 149, 150, 148, 136, 377, 400, 378, 379, 365, 397,
                    175, 199, 200, 208, 428, 171, 396, 18, 83, 313],
}


@dataclass
class FaceRegions:
    rgb: np.ndarray                       # HxWx3 uint8  - DISPLAY / CV crop: colour-constant, glare- & deep-shadow-inpainted
    model_input: np.ndarray              # HxWx3 uint8  - CLASSIFIER crop: colour-constant ONLY (natural shine kept), non-skin filled
    face_cc: np.ndarray                  # HxWx3 uint8  - GEMINI crop: colour-constant full face crop, nothing erased/filled
    normalized: np.ndarray               # HxWx3 float32 - model_input, luminance standardised, 0..1
    skin_mask: np.ndarray                # HxW bool  - analysable skin (occlusion - specular - shadow)
    occlusion_mask: np.ndarray           # HxW bool  - skin-region polygon (no hair/glasses/neck/bg)
    specular_mask: np.ndarray            # HxW bool  - shine / highlights (incl. V>235 glare)
    glare_mask: np.ndarray               # HxW bool  - pixels inpainted for the DISPLAY crop (HSV V > 235)
    shadow_mask: np.ndarray              # HxW bool  - deep shadow inpainted (Lab L<60) + soft directional shade
    face_mask: np.ndarray                # HxW bool  - alias of occlusion_mask (back-compat)
    zones: dict                          # name -> HxW bool  (region INTERSECT skin_mask)
    landmarks: np.ndarray | None         # Nx2 float32 in crop pixel coords, or None
    face_found: bool
    skin_fraction: float
    quality: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- I/O
def _load_image(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, (bytes, bytearray)):
        buf = np.frombuffer(image, np.uint8)
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if dec is None:
            raise ValueError("could not decode image bytes")
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    if isinstance(image, np.ndarray):
        a = image
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        elif a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(a[..., :3]).astype(np.uint8)
    a = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(f"image not found: {image}")
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- local contrast
def local_contrast(gray: np.ndarray, radius: int = 20) -> np.ndarray:
    """20-px (default) local standard deviation - the texture / blemish signal."""
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g /= 255.0
    k = int(radius) * 2 + 1
    mean = cv2.boxFilter(g, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.clip(sq - mean * mean, 0.0, None))


# official Tasks-API model, used when MediaPipe >= 1.0 (no `mp.solutions`)
_FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                        "face_landmarker/float16/1/face_landmarker.task")


def _face_landmarker_model() -> str | None:
    """Path to face_landmarker.task: env override -> user cache -> download once."""
    import os

    env = os.environ.get("HEALTHHUB_FACE_LANDMARKER")
    if env and os.path.isfile(env):
        return env
    cache = os.path.join(os.path.expanduser("~"), ".cache", "healthhub")
    dst = os.path.join(cache, "face_landmarker.task")
    if os.path.isfile(dst):
        return dst
    try:
        import urllib.request

        os.makedirs(cache, exist_ok=True)
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, dst)
        return dst
    except Exception:
        return None


# ------------------------------------------------------------------- Preprocessor
class Preprocessor:
    def __init__(self, work_size: int = 512, use_mediapipe: bool = True):
        import os

        self.work = int(work_size)
        self._mp = None
        self._mesh = None          # legacy  mp.solutions.face_mesh.FaceMesh  (<= 0.10.x)
        self._landmarker = None    # Tasks   vision.FaceLandmarker            (>= 1.0)
        self._haar = None

        if use_mediapipe:
            try:
                import mediapipe as mp

                self._mp = mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                    self._mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True, max_num_faces=1,
                        refine_landmarks=True, min_detection_confidence=0.5,
                    )
                else:                                    # MediaPipe 1.0+: Tasks API only
                    from mediapipe.tasks import python as _mtp
                    from mediapipe.tasks.python import vision as _mvision

                    model = _face_landmarker_model()
                    if model:
                        self._landmarker = _mvision.FaceLandmarker.create_from_options(
                            _mvision.FaceLandmarkerOptions(
                                base_options=_mtp.BaseOptions(model_asset_path=model),
                                num_faces=1, output_face_blendshapes=False,
                                output_facial_transformation_matrixes=False,
                            )
                        )
            except Exception:
                self._mesh = self._landmarker = None

        try:                                             # Haar fallback (needs the XML)
            xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.isfile(xml):
                self._haar = cv2.CascadeClassifier(xml)
        except Exception:
            self._haar = None

    # -- detection ---------------------------------------------------------------
    def _landmarks(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        if self._mesh is not None:
            try:
                res = self._mesh.process(rgb)
            except Exception:
                return None
            if not res.multi_face_landmarks:
                return None
            return np.array([[p.x * w, p.y * h]
                             for p in res.multi_face_landmarks[0].landmark], np.float32)
        if self._landmarker is not None:
            try:
                mp = self._mp
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb.astype(np.uint8)))
                res = self._landmarker.detect(img)
            except Exception:
                return None
            if not res.face_landmarks:
                return None
            return np.array([[p.x * w, p.y * h]
                             for p in res.face_landmarks[0]], np.float32)
        return None

    def _face_box(self, rgb: np.ndarray):
        if self._haar is None or self._haar.empty():
            return None
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.15, 5, minSize=(80, 80))
        except cv2.error:
            return None
        if len(faces) == 0:
            return None
        return max(faces, key=lambda b: b[2] * b[3]).astype(np.float32)

    # -- geometry -------------------------------------------------------------
    def _crop(self, rgb, pts, box):
        H, W = rgb.shape[:2]
        if pts is not None:
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 0.62 * max(x1 - x0, y1 - y0)
        elif box is not None:
            x, y, w, h = box
            cx, cy, half = x + w / 2, y + h / 2, 0.62 * max(w, h)
        else:
            cx, cy, half = W / 2, H / 2, 0.5 * min(H, W)
        half = max(float(half), 32.0)
        ox0, oy0 = cx - half, cy - half
        side = int(round(2 * half))
        pad_l, pad_t = max(0, int(np.ceil(-ox0))), max(0, int(np.ceil(-oy0)))
        pad_r = max(0, int(np.ceil(ox0 + side - W)))
        pad_b = max(0, int(np.ceil(oy0 + side - H)))
        canvas = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        X0, Y0 = int(round(ox0)) + pad_l, int(round(oy0)) + pad_t
        crop = canvas[Y0:Y0 + side, X0:X0 + side]
        crop = cv2.resize(crop, (self.work, self.work), interpolation=cv2.INTER_AREA)
        s = self.work / side
        p2 = None if pts is None else (pts - np.array([ox0, oy0], np.float32)) * s
        b2 = None if box is None else (box - np.array([ox0, oy0, 0, 0], np.float32)) * s
        return crop, p2, b2

    def _zone_masks(self, pts, box):
        n = self.work
        out = {}
        if pts is not None:
            fw = float(pts[:, 0].max() - pts[:, 0].min())
            fh = float(pts[:, 1].max() - pts[:, 1].min())
            axes = {
                "forehead": (0.34 * fw, 0.15 * fh),
                "left_cheek": (0.15 * fw, 0.17 * fh),
                "right_cheek": (0.15 * fw, 0.17 * fh),
                "nose": (0.09 * fw, 0.20 * fh),
                "chin": (0.20 * fw, 0.11 * fh),
            }
            for z, seeds in _ZONE_SEEDS.items():
                c = pts[seeds].mean(0)
                m = np.zeros((n, n), np.uint8)
                ax = tuple(max(3, int(v)) for v in axes[z])
                cv2.ellipse(m, (int(c[0]), int(c[1])), ax, 0, 0, 360, 1, -1)
                out[z] = m.astype(bool)
            return out
        if box is not None:
            x, y, w, h = box
        else:
            x, y, w, h = n * 0.15, n * 0.12, n * 0.70, n * 0.78
        spec = {
            "forehead": (x + 0.50 * w, y + 0.16 * h, 0.40 * w, 0.13 * h),
            "left_cheek": (x + 0.74 * w, y + 0.60 * h, 0.17 * w, 0.18 * h),
            "right_cheek": (x + 0.26 * w, y + 0.60 * h, 0.17 * w, 0.18 * h),
            "nose": (x + 0.50 * w, y + 0.52 * h, 0.10 * w, 0.19 * h),
            "chin": (x + 0.50 * w, y + 0.86 * h, 0.22 * w, 0.11 * h),
        }
        for z, (cx, cy, ax, ay) in spec.items():
            m = np.zeros((n, n), np.uint8)
            cv2.ellipse(m, (int(cx), int(cy)), (max(3, int(ax)), max(3, int(ay))),
                        0, 0, 360, 1, -1)
            out[z] = m.astype(bool)
        return out

    def _face_mask(self, pts, box):
        n = self.work
        if pts is not None:
            m = np.zeros((n, n), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
            return cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool)
        if box is not None:
            x, y, w, h = box
            m = np.zeros((n, n), np.uint8)
            cv2.ellipse(m, (int(x + w / 2), int(y + h / 2)),
                        (int(w * 0.62), int(h * 0.74)), 0, 0, 360, 1, -1)
            return m.astype(bool)
        return np.ones((n, n), bool)

    def _feature_holes(self, pts):
        """Eyes / eyebrows / lips / nostrils - filled, dilated - to remove from skin."""
        n = self.work
        holes = np.zeros((n, n), np.uint8)
        if pts is None:
            return holes.astype(bool)
        for idx in _FEATURE_RINGS.values():
            poly = pts[idx].astype(np.int32)
            cv2.fillConvexPoly(holes, cv2.convexHull(poly), 1)
        return cv2.dilate(holes, np.ones((9, 9), np.uint8), iterations=1).astype(bool)

    def _occlusion_mask(self, pts, box):
        """Skin-region polygon (forehead + cheeks + nose + chin) minus eyes /
        brows / lips / nostrils. Hair, glasses, neck and background lie outside
        it. Falls back to the face hull / ellipse when there are no landmarks."""
        n = self.work
        if pts is None:
            return self._face_mask(None, box)
        oval = np.zeros((n, n), np.uint8)
        cv2.fillPoly(oval, [pts[_FACE_OVAL].astype(np.int32)], 1)
        oval = cv2.erode(oval, np.ones((7, 7), np.uint8))          # drop hairline / jaw rim
        regions = np.zeros((n, n), np.uint8)
        for idx in _SKIN_REGION_POLYS.values():
            cv2.fillConvexPoly(regions, cv2.convexHull(pts[idx].astype(np.int32)), 1)
        regions = cv2.dilate(regions, np.ones((5, 5), np.uint8))
        skin = regions.astype(bool) & oval.astype(bool)
        if skin.mean() < 0.05:                                     # degenerate hulls
            skin = oval.astype(bool)
        return skin & ~self._feature_holes(pts)

    @staticmethod
    def _inpaint_artefacts(cc_crop, glare_v: int = 235, shadow_l: int = 60):
        """Given a colour-constant crop, inpaint specular glare + deep shadow for
        the DISPLAY / CV crop. NOTE: this is deliberately NOT used for the
        classifier input - erasing shine hides oiliness from the model, which
        then can never output "Oily". Returns (clean, glare_mask, shadow_mask).

            cc_crop -> (a) specular masking  HSV V > 235 & S < 60 inpainted (TELEA)
                    -> (b) luminance thresh. Lab L < 60 inpainted (TELEA)
        HSV V and Lab L are channel-order independent, so RGB input is fine.
        """
        hsv = cv2.cvtColor(cc_crop, cv2.COLOR_RGB2HSV)
        glare_mask = (hsv[:, :, 2] > glare_v) & (hsv[:, :, 1] < 60)
        if glare_mask.any():
            glare_masked = cv2.inpaint(np.ascontiguousarray(cc_crop),
                                       (glare_mask.astype(np.uint8) * 255), 3,
                                       cv2.INPAINT_TELEA)
        else:
            glare_masked = cc_crop

        lab = cv2.cvtColor(glare_masked, cv2.COLOR_RGB2LAB)
        shadow_mask = lab[:, :, 0] < shadow_l             # 0..255 (L* * 255/100)
        if shadow_mask.any():
            shadow_masked = cv2.inpaint(np.ascontiguousarray(glare_masked),
                                        (shadow_mask.astype(np.uint8) * 255), 3,
                                        cv2.INPAINT_TELEA)
        else:
            shadow_masked = glare_masked
        return shadow_masked, glare_mask, shadow_mask

    # -- colour ------------------------------------------------------------------
    @staticmethod
    def _ycbcr_skin(rgb):
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        Hh, Ss = hsv[..., 0], hsv[..., 1]
        skin = ((Cr >= 133) & (Cr <= 183) & (Cb >= 77) & (Cb <= 128) &
                (Y > 40) & (Y < 250) & (Ss > 20) & ((Hh <= 25) | (Hh >= 165)))
        skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return skin.astype(bool)

    @staticmethod
    def _color_constancy(rgb, ref=None):
        """Gray-world colour constancy - strips the warm/cool lighting cast.

            wb = cv2.xphoto.createSimpleWB()
            normalized_face = wb.balanceWhite(face_crop)

        SimpleWB (per-channel percentile stretch) is tried first, then GrayworldWB,
        then a manual gray-world on the reference (skin) pixels. Full strength -
        the corrected image is returned as-is (100% Gray World, no blend).
        """
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = None
        xp = getattr(cv2, "xphoto", None)
        if xp is not None and hasattr(xp, "createSimpleWB"):
            try:
                wb = xp.createSimpleWB()
                wb.setP(0.5)
                out = wb.balanceWhite(bgr)
            except Exception:
                out = None
        if out is None and xp is not None and hasattr(xp, "createGrayworldWB"):
            try:
                wb = xp.createGrayworldWB()
                wb.setSaturationThreshold(0.95)
                out = wb.balanceWhite(bgr)
            except Exception:
                out = None
        if out is None:
            px = (rgb[ref].reshape(-1, 3) if (ref is not None and ref.any())
                  else rgb.reshape(-1, 3))
            means = px.mean(0)
            gain = np.clip(means.mean() / np.clip(means, 1.0, None), 0.6, 1.7)
            corrected = np.clip(rgb.astype(np.float32) * gain, 0, 255)
        else:
            corrected = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32)
        return np.clip(corrected, 0, 255).astype(np.uint8)   # 100% Gray World, no blend

    @staticmethod
    def _normalize_luma(rgb, ref):
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y = ycc[..., 0]
        sel = Y[ref] if ref.any() else Y.reshape(-1)
        mu, sd = float(sel.mean()), float(sel.std()) + 1e-6
        ycc[..., 0] = np.clip((Y - mu) / sd * 42.0 + 150.0, 0, 255)
        out = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return out.astype(np.float32) / 255.0

    @staticmethod
    def _specular(rgb, skin):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        v, s = hsv[..., 2] / 255.0, hsv[..., 1] / 255.0
        Y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) / 255.0
        spec = (v > 0.90) & (s < 0.22)
        if skin.any():
            spec |= (Y >= np.percentile(Y[skin], 99.0)) & skin
        return cv2.dilate(spec.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    @staticmethod
    def _shadow(rgb, skin):
        Y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) / 255.0
        if skin.any():
            return Y < 0.55 * float(np.median(Y[skin]))
        return Y < 0.18

    # -- entry point -----------------------------------------------------------
    def __call__(self, image) -> FaceRegions:
        src = _load_image(image)
        pts = self._landmarks(src)
        box = None if pts is not None else self._face_box(src)
        face_found = pts is not None or box is not None

        # (1) face_crop ------------------------------------------------------
        crop, p2, b2 = self._crop(src, pts, box)

        # (2) COLOUR CONSTANCY (gray world) - applied on BOTH paths
        cc = self._color_constancy(crop)

        # skin detection - unioned across raw + colour-constant so a borderline pixel is kept
        prov = self._ycbcr_skin(crop) | self._ycbcr_skin(cc)

        # occlusion: skin-region polygon; hair / glasses / neck / background out
        occ = self._occlusion_mask(p2, b2)

        # ---- CLASSIFIER path: colour-constant ONLY, specular shine PRESERVED ----
        # (no inpainting - erasing shine stops the model ever detecting oiliness)
        model_input = cc.copy()
        sel = occ & prov
        fill = (np.median(cc[sel].reshape(-1, 3), axis=0)
                if int(sel.sum()) > 50 else np.array([200, 160, 135], np.float32))
        model_input[~occ] = fill
        normalized = self._normalize_luma(model_input, occ & prov)

        # ---- GEMINI path: colour-constant full crop, nothing erased or filled
        # (Gemini judges glare vs real sebum itself, so shine is left in place) --
        face_cc = cc

        # ---- DISPLAY / CV path: specular + deep-shadow inpainted ----
        clean, glare, shadow_dark = self._inpaint_artefacts(cc)

        # masks for analysis / reporting
        specular = (self._specular(clean, prov & occ) | glare) & occ
        soft_shadow = self._shadow(clean, prov & occ)          # relative directional shade
        shadow = (shadow_dark | soft_shadow) & occ
        skin_mask = prov & occ & ~specular & ~shadow
        zones = {z: (m & skin_mask) for z, m in self._zone_masks(p2, b2).items()}

        gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        Y = cv2.cvtColor(clean, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) / 255.0
        bright = float(Y[skin_mask].mean()) if skin_mask.any() else float(Y.mean())
        skin_frac = float(skin_mask.mean())
        quality = {
            "brightness": bright,
            "sharpness": sharp,
            "sharp": bool(sharp > 40.0),
            "face_found": bool(face_found),
            "glare_fraction": float((glare & occ).mean()),
            "shadow_fraction": float((shadow_dark & occ).mean()),
            "reliable": bool(face_found and skin_frac > 0.12 and sharp > 40.0
                             and 0.12 <= bright <= 0.97),
        }

        return FaceRegions(
            rgb=clean, model_input=model_input, face_cc=face_cc, normalized=normalized,
            skin_mask=skin_mask, occlusion_mask=occ, specular_mask=specular,
            glare_mask=(glare & occ), shadow_mask=shadow, face_mask=occ, zones=zones,
            landmarks=p2, face_found=bool(face_found), skin_fraction=skin_frac,
            quality=quality,
        )


_SHARED_PRE = None


def preprocess_for_model(image, work_size: int = 512, use_mediapipe: bool = True):
    """Run the full pipeline and return the model-ready crop as a PIL RGB image
    (face_crop -> gray-world colour constancy -> occlusion fill; specular shine
    is NOT inpainted). Use this from training so the classifier sees the same
    representation it gets at inference. Returns None if no face is found."""
    global _SHARED_PRE
    if _SHARED_PRE is None or _SHARED_PRE.work != int(work_size):
        _SHARED_PRE = Preprocessor(work_size=work_size, use_mediapipe=use_mediapipe)
    fr = _SHARED_PRE(image)
    if not fr.face_found:
        return None
    return Image.fromarray(fr.model_input)


# ------------------------------------------------------- data augmentation (train)
class _RandomGamma:
    def __init__(self, lo=0.7, hi=1.6):
        self.lo, self.hi = lo, hi

    def __call__(self, img: Image.Image) -> Image.Image:
        g = random.uniform(self.lo, self.hi)
        a = np.asarray(img).astype(np.float32) / 255.0
        return Image.fromarray((np.power(a, g) * 255).clip(0, 255).astype(np.uint8))


class _RandomShadow:
    """Darken a soft random polygon - simulates window / hair / hand shadows."""

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img).astype(np.float32)
        h, w = a.shape[:2]
        pts = np.array([[random.randint(0, w), random.randint(0, h)]
                        for _ in range(random.randint(3, 5))], np.int32)
        mask = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, w * 0.05))
        factor = 1.0 - mask[..., None] * (1.0 - random.uniform(0.35, 0.8))
        return Image.fromarray((a * factor).clip(0, 255).astype(np.uint8))


class _RandomLightBlob:
    """Add a soft radial highlight - simulates flash / bright window / screen glow."""

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img).astype(np.float32)
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.uniform(0.20, 0.50) * max(h, w)
        d = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r * r)))
        return Image.fromarray((a + d[..., None] * random.uniform(20, 70))
                               .clip(0, 255).astype(np.uint8))


def training_augmentation(img_size: int = 224):
    """Photometric-heavy pipeline: resilient to real selfie lighting."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.25, 0.03)], p=0.9),
        transforms.RandomApply([_RandomGamma(0.7, 1.6)], p=0.5),
        transforms.RandomApply([_RandomShadow()], p=0.4),
        transforms.RandomApply([_RandomLightBlob()], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), value="random"),
    ])


def eval_transform(img_size: int = 224):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(int(round(img_size * 1.15))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
