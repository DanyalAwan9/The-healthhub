# HealthHub — Health & Fitness AI Assistant

Streamlit app with five modules. AI features are powered by the **Google Gemini API**.

| Page | What it does |
|------|--------------|
| 🏠 Home / Profile | Create/select a profile (height, weight, age, gender, goal, budget, gym access, activity). Shows BMI. |
| 💪 Fitness Assessment | BMR (Mifflin–St Jeor), TDEE, calorie target, macro split, **7-day Pakistani meal plan** priced **only** from `meals_prices.csv` (real Pakistan prices you maintain by hand). Gemini **arranges the foods and quantities**; it cannot change a price. Each meal shows `Item (qty × price = Rs total)` and the arithmetic is validated. No Gemini key → the plan is arranged by a local calculator using the same CSV prices. No SerpAPI, no scraping, no price hallucinations. Respects the profile's dietary preference. **Workout:** a "Personalise" panel (experience · location · equipment · days/week · injuries) → Gemini generates a schema-validated weekly split using **only** the listed equipment and an experience-appropriate volume; anything unavailable / slow / invalid falls back to the static PPL/Upper-Lower routines. Identical profiles are cached (no repeat API call). |
| 🥗 AI Nutritionist | Streaming chat backed by **Google Gemini** (`gemini-flash-latest`). Remembers history (SQLite). Pakistan-specific food + price advice. Offline keyword fallback when no API key. |
| 🔬 Skin Analysis | Upload a face photo → if you have trained `skin_model_v2.pth` (see *Train your own classifier* below) its **temperature-calibrated probabilities** drive the read-out; otherwise a **calibrated skin-region CV read-out** (see below). Either path **defaults to "clear"** and only flags redness / spots / texture / oil on strong evidence; a poor photo is reported as *"not assessed"* rather than guessed. Clear skin → a **maintenance routine only** (cleanse → moisturise → sunscreen), no products pushed. Otherwise **cautious, gentle** recommendations from `skincare_brands.csv` or SerpAPI live search — **no retinoids, no benzoyl peroxide, no leave-on acids for sensitive/dry skin**. Supplements: **Pakistani brands only** (Nutrifactor · Hilton Pharma · CCL · pharmacy generics). *Not a medical diagnosis.* |
| 📈 Progress Tracking | Log weight + measurements + photo → SQLite. Weight graph, measurements graph + table, photo timeline, rule-based (optionally Gemini-enriched) progress analysis. |
| 📋 Reports | Compiles profile + latest assessment + progress + skin analysis into a downloadable Markdown report. |

## File structure

```
app.py               main UI + sidebar navigation + router
price_data.py        loads meals_prices.csv / skincare_brands.csv; item lookup; price lines for Gemini
product_search.py    SerpAPI live product search (SKINCARE only) + PKR parsing + multi-source verification
meals_prices.csv     Item, Unit, Price_PKR, Notes   <- REAL Pakistan prices, the ONLY meal price source
skincare_brands.csv  brand, product, price_pkr, shop, authentic_source, category
gemini_client.py     shared Gemini wrapper: generate / stream / chat + JSON extract + model fallback + timeouts
fitness_engine.py    BMR/TDEE/macros · budget-hard-constraint meal optimiser · hybrid Gemini/static workout generator
nutritionist.py      Gemini chat + streaming + offline keyword fallback
skin_analyzer.py     loads trained skin_model_v2.pth if present (primary), else calibrated CV read-out; gentle CSV brand DB / SerpAPI recs
train_skin_model.py  PyTorch training/eval/calibration pipeline -> skin_model_v2.pth (ResNet50 transfer or from-scratch CNN); metrics + confusion matrix + ROC
prepare_skin_data.py Phase-1 helper: turn a downloaded/Kaggle dataset into the data/<class>/ ImageFolder layout train_skin_model.py wants
validate_skin.py     calibration check (12/12): clear->"clear", real problems->flagged, bad photo->"not assessed", + trained-model round-trip
progress_tracker.py  matplotlib charts, tables, progress analysis
database.py          SQLite schema + all CRUD + photo storage + gemplan cache
requirements.txt     + scikit-learn (metrics for train_skin_model.py)
```

Runtime data: `healthhub.db` (SQLite) and `data/photos/` are created automatically.
`skin_model_v2.pth` and `skin_model_reports/` are produced only if you run the training.

## Meal-plan pricing — how it works

**`meals_prices.csv`** is the single source of truth for meal prices. Columns:

```
Item,Unit,Price_PKR,Notes
Roti,Per piece,25,Freshly made
Rice,Per kg,430,Regular basmati
Chicken,Per kg,800,Fresh meat
Egg,Per piece,26,Chicken egg
...
```

It is loaded once at startup (`price_data.load_meal_prices()`).

### Budget is a HARD constraint

`build_meal_plan_csv()` first calls `validate_budget()`:

| Daily budget | Result |
|---|---|
| `< PKR 500` (`< PKR 15,000/month`) | `{"error": …}` — no plan. "Your budget is too low. Minimum viable: PKR 500/day. Try PKR 800-1200/day for balanced nutrition." |
| `>= PKR 500` | **cost-per-macro optimiser** — the budget is a spending **ceiling** |
| not set (`<= 0`) | Gemini arranges the foods (prices fixed from the CSV), else the template assembler |
| `> PKR 5,000` | clamped to PKR 5,000 for the optimiser (noted in the plan) |

**Optimiser** (`source: csv-optimized`) — `fitness_engine.build_meal_plan_optimized()`:

1. **Efficiency ranking** — every CSV food scored `(protein_g + kcal/100) / PKR`
   from `_FOOD_NUTRITION` (per-unit macros). Highest efficiency wins each slot;
   the top choices are rotated across the 7 days for variety.
2. **Portion caps** (`_portion_cap`) — per meal: roti ≤ 3, rice ≤ 200 g, egg ≤ 3,
   meat/fish ≤ 200 g, banana ≤ 2, legume ≤ 180 g dry, oil ≤ 20 ml … and **no
   single item > 25 % of the day's calories** (`MAX_ITEM_KCAL_SHARE`).
3. **Per-meal budget share** — each of breakfast/lunch/dinner/snack gets its slice
   of the *still-unspent* budget, so a tight budget stays legume-forward and a
   larger one unlocks egg / dairy / fish / chicken (`_protein_tier`).
4. **Feasibility** — a day must reach ≥ 82 % of the calorie target and ≥ 60 % of
   the protein target inside budget. If any day can't, the whole call returns
   `{"error", "min_budget", …}` with the cheapest budget that *would* work
   (`_min_feasible_budget`) — **no half-baked plan**.
5. **Validation block** — `plan["validation"] = {daily_budget, daily_cost,
   within_budget, feasible}`; the Fitness page shows *Daily budget PKR Y · Daily
   cost PKR Z · ✅/❌*.

Each meal still reads `qty × item (qty × price = Rs cost)`, e.g.
`0.15 x Chicken (0.15 x 800 = Rs 120) + 0.17 x Rice (Sella) (0.17 x 380 = Rs 65)`.

**Fallback:** if `meals_prices.csv` is missing or malformed the page shows the
**built-in static plan** (`source: csv-missing`) — it never displays a wrong
price. Override the file path with `HEALTHHUB_MEALS_CSV`.

**Caching:** validated Gemini plans are cached 2 days (in-process + SQLite) keyed
by price list + calorie target + budget (bucketed to Rs 25) + goal + diet, with
Streamlit `@st.cache_data` on top — a repeat visit is instant. `GEMINI_MEAL_TIMEOUT`
(18 s) caps the call; on timeout the local calculator is used with no retry.

## Workout generation — hybrid (Gemini + static)

The **Personalise my workout** panel collects a profile:

| field | source |
|---|---|
| Experience (Beginner / Intermediate / Advanced) | picker · default inferred from activity level |
| Location (Home / Gym / Both) · Focus (Strength / Hypertrophy / Endurance / Fat loss / General) | pickers · defaults from `gym_access` + goal |
| Days per week (3–6) | slider · **capped by experience** (Beginner ≤ 4, Intermediate ≤ 5, Advanced ≤ 6) |
| Available equipment | multiselect (`EQUIPMENT_OPTIONS`); empty ⇒ bodyweight only |
| Injuries / limitations | free text |

`fitness_engine.generate_workout_dynamic(profile)`:

1. **Cache** — 7-day SQLite lookup keyed by the whole profile. Identical inputs
   never re-call Gemini.
2. **Gemini** — the prompt states the profile and hard rules (use ONLY the listed
   equipment; work around the limitations; per-`_EXP_VOLUME` sets/reps/exercise
   count/rest days) and asks for strict JSON:
   `{location, goal_focus, experience_level, week:{monday:{focus, exercises:[…],
   sets, reps}, … sunday}}`.
3. **Validation** (`_validate_workout`) — training-day count within ±1 of the
   request; rest days ≥ the experience minimum; 2…`ex_max+3` exercises per day;
   total weekly sets ≤ the experience ceiling ×1.15; **exercises checked against
   the equipment list** (`_equipment_violations`) — a bodyweight-only profile
   tolerates zero gear references. Any failure ⇒ fallback.
4. **Fallback** — the static `GYM_WORKOUTS` / `HOME_WORKOUTS` PPL / Upper-Lower
   dicts, trimmed to the requested day count. Used on: no API key, timeout,
   non-JSON reply, or a plan that fails validation. Fast, free, always works.

`result["workout"]` carries `source` (`gemini` / `gemini-cached` / `static`),
`experience_level`, `goal_focus`, and per-day `sets` / `reps`. The page shows the
source badge and the personalisation note.

**`skincare_brands.csv`** — `brand, product, price_pkr, shop, authentic_source, category`
`category` ∈ `cleanser, toner, soothing, serum, treatment, moisturizer,
sunscreen, supplement`. For the detected skin type + conditions the app shows
matching products across all brands as Option 1 / 2 / 3 …, sorted by price, plus
supplement rows with dosage/timing and authenticity tips.

**Cautious / gentle by design:**
- **No harsh actives, ever.** `HARSH_TERMS` in `price_data.py` (and
  `_HARSH_TERMS` in `skin_analyzer.py`) filter out tretinoin, adapalene, any
  retinoid/retinol, benzoyl peroxide, and (for **sensitive or dry** skin) all
  leave-on salicylic / glycolic / AHA / BHA / peels — from the CSV, from the
  built-in KB, and from anything Gemini or a web search returns.
- **Routine = cleanse → moisturise → sunscreen, done consistently.** Niacinamide
  and azelaic acid are the strongest options offered, and only as optional,
  patch-tested, 2–3 nights a week. A `SAFETY_NOTE` banner explains this on every
  result.
- **Supplements: Pakistani brands only** — Nutrifactor, Hilton Pharma, CCL, or a
  pharmacy generic (`VERIFIED_SUPPLEMENT_BRANDS`). Safe OTC vitamins, minerals
  and biotin; nothing that needs a prescription. No NOW Foods / GNC / Optimum
  Nutrition.
- Every Gemini skincare prompt is prefixed with `_SAFE_CONSTRAINT`
  (*"Only recommend SAFE, GENTLE skincare … Better safe than sorry."*).

**Error handling:** a missing or malformed CSV never crashes the app — the meal
page shows the built-in plan, the skin page falls back to its built-in list.
Override paths with `HEALTHHUB_MEALS_CSV`, `HEALTHHUB_SKINCARE_CSV`.

## Skin read-out — trained classifier (primary) or severity-calibrated CV

`skin_analyzer.analyze()` picks its signal automatically:

* **`skin_model_v2.pth` present** → the trained CNN's **temperature-calibrated
  softmax** is the read-out. Each non-clear class above `_MIN_OBSERVATION_CONF`
  becomes a candidate finding at `confidence = P(class) × 100`; `oily / dry /
  combination` probabilities set the skin type. `backbone` reports
  `skin_model_v2 (<arch>, temperature-calibrated) + CV cross-check`, and the
  return dict carries a `model` block (`trained, file, classes, probs,
  top_class, top_conf, test_accuracy`).
* **no `skin_model_v2.pth`** → the calibrated CV read-out below, unchanged.
  `model` is `{"trained": false}`.

Point `HEALTHHUB_SKIN_MODEL` at a checkpoint elsewhere, or set it to a
non-existent path to force the CV path (that is what `validate_skin.py` does).

### Train your own classifier (`train_skin_model.py`)

The bundled ResNet50 is ImageNet-pretrained, **not** a dermatology model — a real
classifier needs labelled skin images, which are **not** shipped here. Four steps:

**Phase 1 — data.** Collect a few hundred images per class into
`data/<class>/*.jpg`. Classes: `clear_skin, acne, oily, dry, sensitive, redness,
hyperpigmentation` (100+ each, 200+ preferred). Sources: HAM10000
(`kmader/skin-cancer-mnist-ham10000`), the ISIC archive, Kaggle acne-grading
datasets. `prepare_skin_data.py` turns the common *flat folder + labels.csv*
shape into that layout:

```bash
python prepare_skin_data.py --from-kaggle kmader/skin-cancer-mnist-ham10000 --out data_raw
python prepare_skin_data.py --images data_raw/images --labels data_raw/labels.csv --out data
#   edit LABEL_MAP in the script for your dataset's label names
```

**Phase 2 — train.**

```bash
python train_skin_model.py --data data --arch resnet50 --epochs 60
#   --arch scratch   for the from-scratch Conv(32/64/128/256)+BN CNN (img 128)
#   --batch 32 --lr 1e-3 --patience 8 --freeze-epochs 3 --seed 42
```

70 / 15 / 15 stratified split; augmentation (RandomResizedCrop 0.8–1.2, ±20°
rotation, h-flip, colour jitter); class-balanced `CrossEntropyLoss`; Adam +
`ReduceLROnPlateau`; best-val checkpointing + early stop; backbone frozen for the
first `--freeze-epochs`.

**Phase 3 — evaluate + calibrate.** On the held-out test set it prints accuracy,
a per-class `classification_report`, and macro OvR ROC-AUC, and writes
`skin_model_reports/{confusion_matrix.png, roc.png, metrics.json}`. A single
**temperature** scalar is fitted on the validation set (Guo et al., LBFGS on NLL)
so the deployed probabilities are honest — a "60 %" means roughly 60 %.

**Phase 4 — deploy.** It saves `skin_model_v2.pth` =
`{state_dict, arch, classes, img_size, norm_mean, norm_std, temperature,
metrics{accuracy, macro_roc_auc_ovr}, format:2}`. Drop it next to `app.py` (or
set `HEALTHHUB_SKIN_MODEL`) and restart — `skin_analyzer.py` loads it with no
code change. On Streamlit Cloud, commit the `.pth` or fetch it at startup.

### Calibrated CV read-out (fallback)

The old read-out measured brightness / redness / gradient-variance over the
**whole image** and reported 99 % confidence for a couple of freckles. Two
signals now decide, and confidence tracks how *visible* the concern is:

**1. `_resnet_readout()` — the ResNet50 model's own read-out (photo sanity).**
ImageNet-pretrained ResNet50 is not a dermatology model, but its output says
whether the picture is even a face close-up: a real portrait gives a *diffuse*
softmax (low top-1 probability, high entropy, label often a garment/cosmetic);
a photo of a peach / wall / pet gives a *confident, non-face* label. If it is
confidently **non-face → the analysis abstains** (`assessed=False`). Its
`agreement` term (0–1, from entropy + top-1 probability) **down-weights** every
CV finding — a strong-looking CV signal on a photo the backbone barely reads as a
face will not be reported.

**2. `_cv_metrics()` — calibrated, skin-region only.** Keeps skin-tone pixels
inside a centre crop (YCbCr ∩ warm-RGB), measures **local, patchy deviations vs
*this face's own* skin** (a uniformly warm face is not "inflamed", an even tan is
not "dark spots"), resolution / contrast normalised. Returns `reliable=False`
when no face-sized skin region is found.

**Confidence = severity, not "over a low threshold".** Each metric has a
`(clear / mild / moderate / severe)` score scale; `_finding_confidence()`
interpolates it to `clear→5 %, mild→30 %, moderate→62 %, severe→88 %` and blends
in the ResNet `agreement`. **A few tiny spots now score ~10-20 %, not 99 %.**

**Confidence bands → care level** (`analyze()` returns `care_level`, `severity`,
`advice_band`):

| Concern level | Meaning | Recommendation |
|---|---|---|
| **< 50 %** | *Skin looks clear — maintain your routine* (`severity: none`) | `basic`: cleanser + moisturiser + sunscreen only; sub-50 % bumps are listed as non-actionable `observations` |
| **50–75 %** | *Minor concerns — basic care* (`severity: medium`) | `light`: the basics **plus one optional niacinamide serum**; the `Treatment` category is removed — no strong actives / spot treatments for small blemishes |
| **≥ 75 %** | *Visible concerns — consider treatment* (`severity: high`) | `treat`: full **gentle** recommendations (azelaic / niacinamide; still no retinoids or benzoyl peroxide) |

`analyze()` **always** returns these keys (no more `KeyError`): `skin_type,
condition, conditions, findings, observations, assessed, is_clear, severity`
(`low`/`medium`/`high`/`none`), `advice_band, care_level` (`basic`/`light`/`treat`),
`summary, confidence, metrics, resnet, cnn, model, backbone, recommendations,
reco_source, disclaimer`. Each item in `findings` / `observations` has
`name, metric, score, confidence, severity`.

A bad photo returns *"Not assessed — retake in daylight, no filters"*, never a
guessed condition.

**`python validate_skin.py`** — 12/12: clear skin (two tones) → 0 %, `basic`;
**a few tiny spots → < 50 %, `basic`, no actionable findings**; visible acne →
50-75 %, `light`, no `Treatment` category; widespread / strong redness →
≥ 75 %, `treat`, `severity: high`; a moderate signal with **low ResNet
agreement → suppressed to `basic`**; "not a face" / noise / dark → `assessed=False`;
**a fabricated `skin_model_v2.pth` round-trips** — loaded, used, full contract
returned. The 11 CV cases force the CV path via `HEALTHHUB_SKIN_MODEL`.

> Note: ImageNet-pretrained ResNet50 is **not** a dermatology model. Until you run
> `train_skin_model.py` on a labelled dataset, the read-out is the conservative CV
> path: when unsure, it says *clear*.

## Live internet search — Skin Analysis only (optional, SerpAPI)

Meal pricing no longer uses SerpAPI (it was unreliable for grocery prices — see
the Daraz section above). **Skin Analysis** still offers **🔎 Live internet
search** via SerpAPI (`google-search-results`), with **broad, condition-based
queries — never named products**:

**Skincare** — from the CNN read-out (e.g. *oily, acne*) the app builds ~3 broad
queries: `acne oily skin treatment Pakistan price`, `acne oily skin skincare
products Pakistan Daraz`, `best acne oily skin face products Pakistan price
list`. SerpAPI runs them (`gl=pk`, `google_domain=google.com.pk`); each result
**snippet is mined for every `<name> … Rs <price>` pair** (listing pages pack
several), merged by token-overlap. A product **seen on 2+ sites → ✅ verified**;
single-source → "verify authenticity". The numbered list of *real* results is
handed to Gemini with hard constraints ("this is ALL that exists; do NOT invent
products/ingredients; pick only by number"); any index Gemini invents is dropped
in code. Output:

```
For your skin condition: Oily skin with acne + excess oil
Searched the Pakistan market for available products…

Product 1: <name from search> — Rs <price> — <shops>
  ✅ seen on 2+ sites
> Why: <Gemini reason for THIS skin>
```

plus an "Other real products found" list and a "Browse" list of the shops.
Empty search → *"No products found — try a different search."*

No `SERPAPI_API_KEY` → skincare uses `skincare_brands.csv`. Results are cached
7 days in SQLite. Free tier = 100 searches/month; SerpAPI's `google_shopping`
engine does **not** support Pakistan, so only normal Google results are used.

## Setup

```bash
pip install -r requirements.txt          # includes google-generativeai
```

Get a Gemini API key from https://aistudio.google.com/apikey and expose it as an
environment variable:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key"
# macOS / Linux
export GEMINI_API_KEY="your-key"

streamlit run app.py
```

or put it in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL   = "gemini-flash-latest"   # optional override
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (required for AI features) |
| `GEMINI_MODEL` | `gemini-flash-latest` | primary model. The request asked for `gemini-pro`; that plus `gemini-1.5-flash` and `gemini-2.5-flash` are retired for new keys, so this defaults to the `-latest` alias. Pin `gemini-3.6-flash` etc. if you prefer. |
| `GEMINI_FALLBACK_MODELS` | built-in list | comma-separated models to try when the primary hits a 429/not-found. Each model has its own free-tier quota bucket, so this multiplies effective quota. |
| `SERPAPI_API_KEY` / `SERP_API_KEY` | — | SerpAPI key for **Skin Analysis** live search only (free tier = 100/month). Optional. |
| `SERPAPI_TIMEOUT` | `5` | seconds per SerpAPI query |
| `GEMINI_TIMEOUT` | `25` | seconds per Gemini `generate()` |
| `GEMINI_MEAL_TIMEOUT` | `18` | seconds for the meal-plan Gemini call |
| `HEALTHHUB_DB` | `./healthhub.db` | SQLite path |
| `HEALTHHUB_MEALS_CSV` / `HEALTHHUB_SKINCARE_CSV` | `./*.csv` | override CSV file locations |
| `HEALTHHUB_SKIN_MODEL` | `./skin_model_v2.pth` | trained skin classifier checkpoint; missing file → CV read-out |

Without a key the app still runs: the nutritionist uses a keyword fallback, the
meal plan uses the built-in food database, and skincare uses the built-in KB.

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo.
2. On share.streamlit.io, point to `app.py`.
3. Add `GEMINI_API_KEY` under **App → Settings → Secrets**.
4. First skin analysis downloads ResNet50 weights (~100 MB) — expect a slow first run.
   SQLite/photos on Streamlit Cloud are ephemeral (reset on redeploy).
5. To ship a trained classifier, commit `skin_model_v2.pth` to the repo (or have
   the app download it to the working dir at startup, or set `HEALTHHUB_SKIN_MODEL`).
   Without it the app runs fine on the calibrated CV read-out.

## Notes

- `google-generativeai` is Google's older SDK (deprecated in favour of `google-genai`)
  but is what this build targets, per the request. `gemini_client.py` isolates it so a
  future swap only touches one file.
- The Fitness page caches the Gemini meal plan per (calorie target, budget, goal) to
  avoid repeat API calls on every rerun.

## Disclaimer

Educational/wellness tool only. Not medical, dietetic, or dermatological advice.
Consult a qualified professional for medical conditions, pregnancy, or medication.
AI-generated prices are estimates — verify locally or on Daraz before purchasing.
