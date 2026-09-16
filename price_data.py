"""CSV-backed price data for HealthHub.

Files, edited manually:
    meals_prices.csv     Item, Unit, Price_PKR, Notes   (real Pakistan prices - the ONLY meal price source)
    skincare_brands.csv  brand, product, price_pkr, shop, authentic_source, category

Everything is loaded once into memory. Nothing here raises: a missing or broken
file yields an empty dataset plus an `error` string the UI can show.
"""
from __future__ import annotations

import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEALS_CSV = os.environ.get("HEALTHHUB_MEALS_CSV", os.path.join(BASE_DIR, "meals_prices.csv"))
SKINCARE_CSV = os.environ.get("HEALTHHUB_SKINCARE_CSV",
                              os.path.join(BASE_DIR, "skincare_brands.csv"))

_meals_cache: dict | None = None
_skincare_cache: dict | None = None


def _to_int(v):
    try:
        return int(round(float(str(v).replace(",", "").replace("Rs", "").strip())))
    except (TypeError, ValueError):
        return None


def _read_csv(path: str, required: set[str]) -> tuple[list[dict], str | None]:
    name = os.path.basename(path)
    if not os.path.exists(path):
        return [], f"{name} not found - add it to the app folder."
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
    except Exception as exc:  # noqa: BLE001
        return [], f"Could not read {name}: {exc}"
    if not rows:
        return [], f"{name} is empty."
    missing = required - {k.strip() for k in rows[0].keys()}
    if missing:
        return [], f"{name} is missing column(s): {', '.join(sorted(missing))}"
    return rows, None


# --------------------------------------------------------------------------- #
# Meals - meals_prices.csv is the single source of truth for meal prices.
# Columns: Item, Unit, Price_PKR, Notes
# --------------------------------------------------------------------------- #
def load_meal_prices(force: bool = False) -> dict:
    """Return {'rows': [ {item, unit, price_pkr, notes} ], 'error'}."""
    global _meals_cache
    if _meals_cache is not None and not force:
        return _meals_cache
    raw, err = _read_csv(MEALS_CSV, {"Item", "Unit", "Price_PKR"})
    rows = []
    for r in raw:
        item = (r.get("Item") or "").strip()
        price = _to_int(r.get("Price_PKR"))
        if not item or price is None:
            continue
        rows.append({
            "item": item,
            "unit": (r.get("Unit") or "").strip() or "each",
            "price_pkr": price,
            "notes": (r.get("Notes") or "").strip(),
        })
    if not rows and not err:
        err = f"{os.path.basename(MEALS_CSV)} has no usable rows."
    _meals_cache = {"rows": rows, "error": err, "path": MEALS_CSV}
    return _meals_cache


# Bare template keys -> preferred item name when the CSV uses fuller names.
_MEAL_ALIASES = {
    "rice": "rice (basmati)", "chicken": "chicken (whole cut)",
    "milk": "milk (fresh)", "daal": "daal masoor", "oil": "cooking oil",
    "yogurt": "yogurt (dahi)", "bread": "bread (large)",
    "tea": "tea (prepared)", "vegetables": "spinach", "beans": "rajma",
    "roti": "roti", "egg": "egg", "sugar": "sugar", "butter": "butter",
    "paneer": "paneer", "fish": "fish", "mutton": "mutton",
}


def find_row(key: str, default: dict | None = None) -> dict | None:
    """Look up a meal item row by (loose) name, honouring _MEAL_ALIASES."""
    k = key.lower().strip()
    rows = load_meal_prices()["rows"]
    target = _MEAL_ALIASES.get(k, k)
    for r in rows:                       # exact match on the alias / key
        if r["item"].lower() == target:
            return r
    for r in rows:                       # substring on the alias
        if target in r["item"].lower():
            return r
    for r in rows:                       # substring on the raw key
        name = r["item"].lower()
        if k in name or name in k:
            return r
    return default


def find_price(key: str, default: int | None = None) -> int | None:
    row = find_row(key)
    return row["price_pkr"] if row else default


def meal_price_lines() -> list[str]:
    """['Roti: Rs 25 per piece', 'Rice: Rs 430 per kg', ...] - for the Gemini prompt."""
    return [f"{r['item']}: Rs {r['price_pkr']} {r['unit'].lower()}"
            + (f" ({r['notes']})" if r["notes"] else "")
            for r in load_meal_prices()["rows"]]


def meal_status() -> dict:
    d = load_meal_prices()
    return {"rows": len(d["rows"]), "error": d["error"], "path": d["path"]}


# Back-compat aliases (older callers / tests).
def reference_price_lines() -> list[str]:  # noqa: D401
    return meal_price_lines()


def meal_price_table() -> str:
    return "\n".join(f"- {ln}" for ln in meal_price_lines())


# --------------------------------------------------------------------------- #
# Skincare + supplements
# --------------------------------------------------------------------------- #
def load_skincare(force: bool = False) -> dict:
    global _skincare_cache
    if _skincare_cache is not None and not force:
        return _skincare_cache
    raw, err = _read_csv(
        SKINCARE_CSV,
        {"brand", "product", "price_pkr", "shop", "authentic_source", "category"},
    )
    rows = []
    for r in raw:
        price = _to_int(r.get("price_pkr"))
        brand = (r.get("brand") or "").strip()
        product = (r.get("product") or "").strip()
        if not brand or not product:
            continue
        rows.append({
            "brand": brand,
            "product": product,
            "price_pkr": price if price is not None else 0,
            "shop": (r.get("shop") or "").strip(),
            "authentic_source": (r.get("authentic_source") or "").strip(),
            "category": (r.get("category") or "").strip().lower(),
        })
    if not rows and not err:
        err = f"{os.path.basename(SKINCARE_CSV)} has no usable rows."
    _skincare_cache = {
        "rows": rows,
        "error": err,
        "brands": sorted({r["brand"] for r in rows}),
        "path": SKINCARE_CSV,
    }
    return _skincare_cache


# Cautious / gentle only. Harsh actives (retinoids, benzoyl peroxide, leave-on
# acid exfoliants) are NEVER sought and are filtered out in skincare_options().
_CONDITION_KEYWORDS = {
    "oil": ["niacinamide", "oil-free", "light", "gel", "clay"],
    "pore": ["niacinamide", "oil-free", "light", "gel"],
    "acne": ["azelaic", "niacinamide", "centella", "cica", "gentle", "aloe"],
    "texture": ["azelaic", "niacinamide"],
    "redness": ["azelaic", "niacinamide", "toleriane", "cica", "centella",
                "soothing", "sensitive", "aloe", "physiogel", "hydrating", "barrier"],
    "inflammation": ["azelaic", "toleriane", "soothing", "sensitive", "aloe",
                     "cica", "centella", "physiogel", "barrier"],
    "pigment": ["vitamin c", "ascorbic", "arbutin", "niacinamide", "azelaic", "bright"],
    "spot": ["vitamin c", "arbutin", "azelaic", "niacinamide"],
    "dry": ["hydrating", "hyaluronic", "cream", "ceramide", "soft", "moistur",
            "physiogel", "barrier", "rich"],
}

# Product-name fragments that are ALWAYS excluded - too harsh, prescription, or
# skin-bleaching (mercury/hydroquinone/glutathione risk).
HARSH_TERMS = (
    "tretinoin", "retin-a", "retino", "retinaldehyde", "adapalene", "differin",
    "isotretinoin", "accutane", "tazarotene", "retinol", "retinyl",
    "benzoyl peroxide", "benzoyl", "bpo", "on-the-spot", "acne control",
    "salicylic", "glycolic", "lactic acid", "aha", "bha", "peel", "peeling",
    "hydroquinone", "melacare", "kojic", "mercury",
    "beauty cream", "gluta", "glutathione", "skin lightening", "bleach",
)

# Sensitive / dry skin: also drop even the milder acids.
_EXTRA_GENTLE_EXCLUDE = ("acid", "exfoliant", "resurfac", "10%", "20%")

_SUPP_HINT = {
    "zinc": ("1 tablet (~20 mg elemental zinc)", "with a meal; a few weeks, not long-term",
             ["acne", "texture", "oil", "inflammation"]),
    "vitamin c": ("1 tablet (500 mg)", "morning, with food",
                  ["pigment", "spot", "healthy"]),
    "vitamin e": ("1 softgel (200-400 IU)", "with a meal containing fat",
                  ["redness", "healthy", "pigment"]),
    "vitamin d": ("1 tablet (1000-2000 IU)", "with a fatty meal", ["healthy"]),
    "biotin": ("1 tablet (up to 10000 mcg / 10 mg)", "morning, with food",
               ["healthy", "texture"]),
    "b-complex": ("1 tablet daily", "with breakfast", ["healthy", "redness"]),
    "collagen": ("1 scoop (~10 g)", "any time, in water or tea", ["healthy", "texture"]),
    "multivitamin": ("1 tablet daily", "with breakfast", ["healthy"]),
}


def _condition_tokens(conditions: list[str]) -> set[str]:
    text = " ".join(conditions).lower()
    toks = set()
    for key in _CONDITION_KEYWORDS:
        if key in text:
            toks.add(key)
    return toks


def condition_fragments(conditions: list[str]) -> set[str]:
    """Ingredient/keyword fragments relevant to these conditions (e.g. ['acne',
    'oil'] -> {'niacinamide','azelaic','oil-free',...}). Used both to select
    products (skincare_options) and to rank/explain them (skin_analyzer)."""
    frags = set()
    for t in _condition_tokens(conditions):
        frags.update(_CONDITION_KEYWORDS[t])
    return frags


def skincare_options(skin_type: str, conditions: list[str]) -> dict:
    """Return {'by_category': {cat: [rows]}, 'flat': [rows], 'error': str|None}.

    Includes EVERY relevant product from EVERY brand in the CSV - not capped.
    """
    data = load_skincare()
    rows = data["rows"]
    if not rows:
        return {"by_category": {}, "flat": [], "error": data["error"]}

    dry = skin_type.lower() in ("dry", "normal")
    toks = _condition_tokens(conditions)
    sensitive = bool(toks & {"redness", "inflammation"})
    cond_fragments = condition_fragments(conditions)

    base_cats = ["cleanser", "moisturizer", "sunscreen"]
    active_cats = ["serum", "treatment", "toner", "soothing"]

    def keep(r) -> bool:
        cat, name = r["category"], r["product"].lower()
        if cat == "supplement":
            return False
        if any(h in name for h in HARSH_TERMS):          # never, for anyone
            return False
        if (sensitive or dry) and any(x in name for x in _EXTRA_GENTLE_EXCLUDE):
            return False                                  # skip even mild acids
        if cat in base_cats:
            if dry and any(x in name for x in ("oil-free", "oily skin", "charcoal")):
                return False
            return True
        if cat in active_cats:
            if cat == "soothing":                         # always welcome
                return True
            if not cond_fragments:                        # no concern -> light niacinamide only
                return "niacinamide" in name
            return any(f in name for f in cond_fragments)
        return False

    chosen = [r for r in rows if keep(r)]
    order = {c: i for i, c in enumerate(
        ["cleanser", "toner", "soothing", "serum", "treatment",
         "moisturizer", "sunscreen"])}
    chosen.sort(key=lambda r: (order.get(r["category"], 99), r["price_pkr"]))

    by_cat: dict[str, list] = {}
    for r in chosen:
        by_cat.setdefault(r["category"], []).append(r)
    return {"by_category": by_cat, "flat": chosen, "error": None}


def supplement_options(conditions: list[str]) -> dict:
    data = load_skincare()
    rows = [r for r in data["rows"] if r["category"] == "supplement"
            and not any(h in r["product"].lower() for h in HARSH_TERMS)]  # no gluta/whitening
    if not rows:
        return {"rows": [], "error": data["error"]}
    toks = _condition_tokens(conditions)
    out = []
    for r in rows:
        name = r["product"].lower()
        dosage, timing, relevant_for = "as per label", "with a meal", []
        for key, (d, t, conds) in _SUPP_HINT.items():
            if key in name:
                dosage, timing, relevant_for = d, t, conds
                break
        relevant = bool(toks & set(relevant_for)) or not toks
        out.append({**r, "dosage": dosage, "timing": timing, "relevant": relevant})
    out.sort(key=lambda r: (not r["relevant"], r["price_pkr"]))
    return {"rows": out, "error": None}


AUTHENTICITY_TIPS = [
    "Buy from the brand's official store on Daraz or a registered pharmacy "
    "(dvago.pk, Naheed, Al-Fatah) - not unbranded resellers.",
    "Check the batch code and expiry printed on the carton and tube match.",
    "International brands (CeraVe, Cetaphil, La Roche-Posay, The Ordinary) should "
    "have an importer sticker with a local company name and contact.",
    "Supplements: prefer Pakistani-registered brands (Nutrifactor, Hilton Pharma, "
    "CCL) or a pharmacy's own generic - no unregistered iHerb imports.",
    "Be wary of prices far below the ranges here - counterfeits are usually cheap.",
]

# Shown on every skincare result. Cautious, gentle, non-prescription.
SAFETY_NOTE = (
    "Cautious approach: gentle products only. No tretinoin, adapalene, strong "
    "retinoids or benzoyl peroxide; leave-on acid exfoliants are skipped for "
    "sensitive or dry skin. The routine is cleanse - moisturise - sunscreen, "
    "done consistently. Introduce at most one new product a week, patch-test "
    "first, and see a dermatologist for persistent or worsening acne."
)


def status() -> dict:
    m, s = load_meal_prices(), load_skincare()
    return {
        "meals_rows": len(m["rows"]), "meals_error": m["error"],
        "skincare_rows": len(s["rows"]), "skincare_error": s["error"],
        "brands": s.get("brands", []),
    }
