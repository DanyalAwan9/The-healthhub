# skin_ai — production-grade Skin Analysis AI

A self-contained face-skin analysis pipeline: **image in → JSON out**, with a
false-positive filter that only reports a concern when a CNN head and an
independent CV measurement (taken inside a face-zone skin mask, specular
highlights and shadows removed) agree.

```
preprocessing.py   face_crop → gray-world colour constancy (cv2.xphoto SimpleWB)
                   → occlusion mask (skin polygon from the 468/478 landmarks,
                   minus eyes/brows/lips/nostrils) → THREE crops:
                     • face_cc (GEMINI) — colour-constant full face crop, nothing
                       erased or filled (Gemini judges glare vs sebum itself)
                     • model_input (CNN) — colour-constant, specular shine kept,
                       non-skin filled with the skin median
                     • rgb (DISPLAY / CV) — additionally inpaints glare (HSV
                       V>235 & S<60) and deep shadow (Lab L<60), cv2.inpaint TELEA
                   Also: soft directional-shadow mask, zones,
                   training_augmentation / eval_transform / local_contrast,
                   preprocess_for_model(image) for training parity.
model.py           EfficientNet-B0 or ResNet-50 transfer learning, optional 4th
                   input channel = 20-px local contrast, 5 task heads. The
                   skin_type head is no longer used at inference (see below); the
                   redness / hyperpigmentation / blemish / texture heads are.
gemini_vision.py   classify_skin_type_gemini(pil_image) → {skin_type, confidence,
                   reasoning} via the Google Gemini Vision API (new google-genai
                   SDK, pydantic response_schema, model-fallback chain). Needs
                   GEMINI_API_KEY. Returns None when unavailable → caller uses the
                   conservative default "Combination".
inference.py       end-to-end: SkinAnalyzer("skin_model.pth").analyze(img) → dict.
                   Skin type comes from Gemini Vision (face_cc crop + a 0..1
                   blemish hint); blemish / texture / redness come from the CNN
                   heads fused with the CV metrics. skin_type_source /
                   skin_type_reasoning / skin_type_confidence expose the call.
recommendations.py supplement de-duplication by ingredient class + overdose caps
training.py        70/15/15 stratified split, Adam + CrossEntropy/SmoothL1,
                   ReduceLROnPlateau, early stop, temperature calibration,
                   confusion matrix + ROC-style report
requirements.txt
```

## Install

```bash
pip install -r skin_ai/requirements.txt
```

`preprocessing.py` auto-detects the MediaPipe API:

- **MediaPipe ≤ 0.10.x** → `solutions.face_mesh.FaceMesh` (468 pts).
- **MediaPipe ≥ 1.0** → Tasks API `FaceLandmarker` (478 pts). The `.task` model
  auto-downloads to `~/.cache/healthhub/face_landmarker.task` on first use — or
  set `HEALTHHUB_FACE_LANDMARKER` to a local copy.
- **No MediaPipe** → OpenCV Haar face box → whole-frame fallback.

OpenCV, torch and torchvision are required. Use `opencv-python` (not
`-headless`) so the Haar fallback has its cascade XML.

## Inference

Skin type needs `GEMINI_API_KEY` (Gemini Vision); the concern heads work with a
trained checkpoint, or fall back to CV rules only if the `.pth` is missing.

```bash
export GEMINI_API_KEY=...        # skin type; without it → "Combination" default
python skin_ai/inference.py --image selfie.jpg --model skin_model.pth
```

```python
from skin_ai.inference import SkinAnalyzer

az = SkinAnalyzer("skin_model.pth")          # missing file → ImageNet init + CV rules
print(az.analyze("selfie.jpg"))
# skin type only, no CNN/CV:
from skin_ai.gemini_vision import classify_skin_type_gemini
from PIL import Image
print(classify_skin_type_gemini(Image.open("selfie.jpg").convert("RGB")))
```

Output:

```json
{
  "assessed": true,
  "skin_type": "Combination",
  "skin_type_confidence": 0.78,
  "skin_type_source": "Gemini Vision (gemini-flash-latest) - visible T-zone shine, matte cheeks",
  "skin_type_reasoning": "Visible T-zone shine, matte cheeks",
  "confidence": 0.71,
  "blemish_count": 3,
  "redness_severity": "mild",
  "hyperpigmentation_severity": "clear",
  "texture_severity": "mild",
  "primary_concerns": [],
  "concern_details": [{"concern": "blemishes", "severity": "mild", "confidence": 0.58}],
  "zones": {"forehead": {"oiliness": "high", "redness": "low"}, "...": {}},
  "quality": {"brightness": 0.52, "sharpness": 180.4, "sharp": true, "reliable": true},
  "model": {"trained": true, "arch": "efficientnet_b0", "contrast_channel": true},
  "disclaimer": "Automated cosmetic estimate, not a medical diagnosis. ..."
}
```

If Gemini is unavailable (no key, SDK missing, every fallback model failed)
`skin_type` is the conservative default `"Combination"` and
`skin_type_confidence` is `null`.

A poor photo (no face / too little skin / blur / bad light) returns
`{"assessed": false, "reason": "...", "message": "Retake ..."}` — nothing is guessed.

## Training

You supply the data — this repo ships no clinical images.

1. **Assemble labels.** One CSV row per image:

   | column | required | values |
   |---|---|---|
   | `filepath` | yes | path (abs or relative to `--images-root`) |
   | `skin_type` | yes | `Normal` `Dry` `Oily` `Combination` |
   | `redness` | no | `clear` `mild` `moderate` `severe` (or `0..3`) |
   | `hyperpigmentation` | no | `clear` `mild` `moderate` `severe` (or `0..3`) |
   | `blemish_count` | no | integer ≥ 0 |
   | `texture` | no | float `0..1` (0 smooth → 1 rough) |

   Empty optional cells are skipped by the loss for that row, so you can mix
   datasets that only label some attributes.

   Suggested sources (map their native labels onto the columns above):
   - **Fitzpatrick 17k** — 6 skin-tone types, ~17k images → tone diversity (avoids
     light-skin bias) + hyperpigmentation.
   - **ISIC Archive** — clinical/dermoscopic lesion images → `blemish_count`.
   - **DermNet NZ** — many named conditions → redness / texture.
   - A plain labelled selfie set for `skin_type`.

   Or, for skin-type only, skip the CSV and use
   `--imagefolder DIR` with `Normal/ Dry/ Oily/ Combination/` subfolders.

2. **Train.**

   ```bash
   python skin_ai/training.py --labels labels.csv --images-root images/ \
       --arch efficientnet_b0 --epochs 60 --batch 32 --out skin_model.pth
   ```

   Produces `skin_model.pth` and `skin_model_reports/{confusion_matrix.png,
   training_curve.png, metrics.json}`. `inference.py` loads the checkpoint
   automatically on next run.

## Relation to the rest of HealthHub

`app.py` uses this package for the analysis: `page_skin()` →
`analyze_skin_image()` → `_skin_ai()` (a cached `SkinAnalyzer(model_path=
"skin_ai/skin_model.pth")`, `use_gemini_skin_type=True`) → `_adapt_skin_ai()`
maps the result onto the schema the rest of the page expects. Skin type is a
**Google Gemini Vision** call (`gemini_vision.classify_skin_type_gemini`); the
EfficientNet `skin_type` head is retired. The **product recommendations** still come
from `skin_analyzer._recommend()` (gentle CSV / SerpAPI engine) — `skin_ai` has
no product database of its own — but the **supplement list is de-duplicated** by
`recommendations.deduplicate_supplements()` before it is shown.

### `recommendations.py` — supplement de-dup + overdose safety

`deduplicate_supplements(list, context="basic"|"acne"|"pigment"|"general")`:

- classifies each product by active-ingredient class (`PRODUCT_INGREDIENTS`
  exact table + `INGREDIENT_KEYWORDS` fallback; a multivitamin expands to
  `MULTIVITAMIN_COVERS`)
- keeps **one product per class**, preferring the combo that covers the most
  (Surbex-Z → Zinc + Vitamin C + B-Complex, so it alone is recommended)
- drops classes irrelevant to the routine (Alpha-Lipoic Acid / Sensolin for a
  basic routine); `acne` → 1 targeted product (Zinc **or** Vitamin C, not both)
- caps at 2 (`basic`/`acne`/`pigment`) or 3 (`general`); a Vitamin D / Calcium
  add-on only when `detected={"low_vitamin_d"}` is passed
- sums each ingredient across the final picks and **flags** a routine total over
  the ceiling (`DAILY_LIMITS`: Zinc 25 mg, Vitamin C 750 mg, Calcium 1200 mg,
  Vitamin D 4000 IU) — a lone label-dose combo is fine, a *stack* is flagged
- returns `{supplements:[…+label/covers/note], primary, optional, flags,
  safety_note, dropped}`; `safety_note` always ends "don't add other supplements"

`python skin_ai/recommendations.py` prints the before/after for the 8-item
example in the brief.

`skin_ai/skin_model.pth` shipped here is a **synthetic bootstrap** (trained by
`training.py` on procedurally-generated faces) so the model path is live and the
app works end-to-end. Its numbers are not clinically meaningful — retrain on
DermNet / ISIC / Fitzpatrick 17k (see *Training* below) and drop the new
checkpoint in at the same path.
