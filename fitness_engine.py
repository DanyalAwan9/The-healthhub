"""Fitness math + Pakistani meal-plan / workout generation.

Pure logic, no Streamlit imports so it can be unit-tested or reused.
Prices are approximate 2024-2025 Pakistan retail prices in PKR.
"""
from __future__ import annotations

import concurrent.futures
import os

ACTIVITY_MULTIPLIERS = {
    "Sedentary (little to no exercise, desk job)": 1.2,
    "Lightly active (exercise 1-3 days/week)": 1.375,
    "Moderately active (exercise 4-5 days/week)": 1.55,
    "Very active (exercise 6-7 days/week)": 1.725,
    "Extremely active (intense daily training + physical job)": 1.9,
}

GOALS = [
    "Lose weight",
    "Mild weight loss",
    "Maintain weight",
    "Mild muscle gain",
    "Build muscle",
]

GOAL_ADJUSTMENTS = {
    "Lose weight": -0.20,
    "Mild weight loss": -0.10,
    "Maintain weight": 0.0,
    "Mild muscle gain": 0.10,
    "Build muscle": 0.15,
}


# --------------------------------------------------------------------------- #
# Energy calculations
# --------------------------------------------------------------------------- #
def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor equation."""
    s = 5 if str(gender).lower().startswith("m") else -161
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s


def calculate_tdee(bmr: float, activity_level: str) -> float:
    return bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.375)


def calculate_calorie_target(tdee: float, goal: str) -> int:
    adj = GOAL_ADJUSTMENTS.get(goal, 0.0)
    target = tdee * (1 + adj)
    return int(max(1200, round(target / 10) * 10))


def macro_split(calorie_target: int, goal: str) -> dict:
    if goal in ("Build muscle", "Mild muscle gain"):
        p, c, f = 0.30, 0.45, 0.25
    elif goal in ("Lose weight", "Mild weight loss"):
        p, c, f = 0.35, 0.35, 0.30
    else:
        p, c, f = 0.25, 0.50, 0.25
    return {
        "protein_g": round(calorie_target * p / 4),
        "carbs_g": round(calorie_target * c / 4),
        "fat_g": round(calorie_target * f / 9),
    }


# --------------------------------------------------------------------------- #
# Pakistani food database  (kcal + PKR price per serving, approx protein g)
# --------------------------------------------------------------------------- #
FOODS = {
    "Breakfast": [
        {"name": "Anda paratha (2 eggs + 1 paratha)", "kcal": 450, "price": 130, "protein": 20},
        {"name": "Omelette (3 eggs) + 2 roti", "kcal": 500, "price": 150, "protein": 24},
        {"name": "Dahi + 1 paratha", "kcal": 400, "price": 120, "protein": 12},
        {"name": "Oats with milk + banana", "kcal": 350, "price": 160, "protein": 14},
        {"name": "Chana chaat", "kcal": 320, "price": 100, "protein": 13},
        {"name": "Halwa puri (small)", "kcal": 650, "price": 200, "protein": 10},
        {"name": "Nihari (small) + 1 naan", "kcal": 620, "price": 350, "protein": 28},
    ],
    "Lunch": [
        {"name": "Daal chawal", "kcal": 550, "price": 150, "protein": 18},
        {"name": "Chicken karahi (1 serving) + 2 roti", "kcal": 700, "price": 450, "protein": 42},
        {"name": "Chicken biryani (1 plate)", "kcal": 800, "price": 300, "protein": 30},
        {"name": "Aloo gosht + 2 roti", "kcal": 650, "price": 350, "protein": 30},
        {"name": "Chana pulao", "kcal": 600, "price": 180, "protein": 16},
        {"name": "Fish fry + salad + 1 roti", "kcal": 560, "price": 400, "protein": 34},
        {"name": "Rajma chawal", "kcal": 560, "price": 160, "protein": 17},
    ],
    "Dinner": [
        {"name": "Grilled chicken breast + veg + 1 roti", "kcal": 500, "price": 420, "protein": 45},
        {"name": "Mixed sabzi + 2 roti", "kcal": 420, "price": 150, "protein": 12},
        {"name": "Chicken shorba + 2 roti", "kcal": 460, "price": 320, "protein": 28},
        {"name": "Beef seekh kabab (2) + salad + 1 roti", "kcal": 550, "price": 350, "protein": 30},
        {"name": "Palak paneer + 2 roti", "kcal": 520, "price": 260, "protein": 20},
        {"name": "Egg curry (2 eggs) + rice", "kcal": 560, "price": 180, "protein": 20},
        {"name": "Masoor daal + 2 roti", "kcal": 430, "price": 130, "protein": 18},
    ],
    "Snack": [
        {"name": "Fruit chaat", "kcal": 150, "price": 100, "protein": 3},
        {"name": "Handful of almonds / peanuts", "kcal": 200, "price": 120, "protein": 7},
        {"name": "Dahi bowl (250 g)", "kcal": 150, "price": 90, "protein": 9},
        {"name": "Whey protein shake", "kcal": 150, "price": 150, "protein": 24},
        {"name": "Boiled eggs (2)", "kcal": 140, "price": 70, "protein": 12},
        {"name": "Banana + 1 tbsp peanut butter", "kcal": 250, "price": 120, "protein": 8},
        {"name": "Roasted chana (50 g)", "kcal": 180, "price": 70, "protein": 10},
    ],
}

DAYS = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]


def _scaled(item: dict, factor: float) -> dict:
    factor = max(0.5, min(2.0, factor))
    return {
        "slot": item.get("slot", ""),
        "name": item["name"],
        "portion": round(factor, 2),
        "kcal": int(round(item["kcal"] * factor)),
        "price": int(round(item["price"] * factor)),
        "protein": int(round(item["protein"] * factor)),
    }


def build_meal_plan(calorie_target: int, daily_budget_pkr: float, days: int = 7) -> dict:
    """Return {'days': [...], 'weekly_cost': int, 'avg_daily_cost': int, 'notes': [...]}."""
    plan_days = []
    notes = []
    for i in range(days):
        b = dict(FOODS["Breakfast"][i % len(FOODS["Breakfast"])], slot="Breakfast")
        l = dict(FOODS["Lunch"][i % len(FOODS["Lunch"])], slot="Lunch")
        d = dict(FOODS["Dinner"][i % len(FOODS["Dinner"])], slot="Dinner")
        s1 = dict(FOODS["Snack"][i % len(FOODS["Snack"])], slot="Snack")
        s2 = dict(FOODS["Snack"][(i + 3) % len(FOODS["Snack"])], slot="Snack")

        base_items = [b, l, d, s1, s2]
        base_kcal = sum(x["kcal"] for x in base_items)
        factor = calorie_target / base_kcal if base_kcal else 1.0

        meals = [_scaled(x, factor) for x in base_items]
        total_kcal = sum(m["kcal"] for m in meals)
        total_price = sum(m["price"] for m in meals)
        total_protein = sum(m["protein"] for m in meals)

        over_budget = daily_budget_pkr and total_price > daily_budget_pkr
        plan_days.append({
            "day": DAYS[i] if i < len(DAYS) else f"Day {i + 1}",
            "meals": meals,
            "total_kcal": total_kcal,
            "total_price": total_price,
            "total_protein": total_protein,
            "over_budget": bool(over_budget),
        })

    weekly_cost = sum(x["total_price"] for x in plan_days)
    avg_daily = round(weekly_cost / max(1, len(plan_days)))
    if daily_budget_pkr:
        if avg_daily > daily_budget_pkr:
            notes.append(
                f"Average daily cost (PKR {avg_daily:,}) exceeds your daily budget "
                f"(PKR {int(daily_budget_pkr):,}). Swap chicken/beef days for daal, "
                f"chana or egg-based meals to cut cost."
            )
        else:
            notes.append(
                f"Plan fits your budget: avg PKR {avg_daily:,}/day vs limit "
                f"PKR {int(daily_budget_pkr):,}/day."
            )
    notes.append("Prices are approximate home-cooked / dhaba estimates and vary by city.")
    return {
        "days": plan_days,
        "weekly_cost": weekly_cost,
        "avg_daily_cost": avg_daily,
        "notes": notes,
        "source": "builtin",
    }


# --------------------------------------------------------------------------- #
# JSON parsing helpers for Gemini meal-plan output
# --------------------------------------------------------------------------- #
def _coerce_num(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _normalise_ai_days(data, days: int, daily_budget_pkr: float) -> list[dict]:
    raw_days = data.get("days") if isinstance(data, dict) else data
    if not isinstance(raw_days, list) or not raw_days:
        return []
    out = []
    for i, rd in enumerate(raw_days[:days]):
        meals_in = rd.get("meals", []) if isinstance(rd, dict) else []
        meals = []
        for meal in meals_in:
            if not isinstance(meal, dict):
                continue
            meals.append({
                "slot": str(meal.get("slot") or meal.get("meal") or "Meal"),
                "name": str(meal.get("name") or meal.get("food") or "").strip() or "-",
                "portion": 1.0,
                "kcal": _coerce_num(meal.get("kcal") or meal.get("calories")),
                "protein": _coerce_num(meal.get("protein") or meal.get("protein_g")),
                "price": _coerce_num(meal.get("price") or meal.get("price_pkr")
                                     or meal.get("pkr")),
                "items": str(meal.get("items") or meal.get("breakdown") or "").strip(),
            })
        if not meals:
            continue
        total_kcal = _coerce_num(rd.get("total_kcal")) or sum(m["kcal"] for m in meals)
        total_protein = _coerce_num(rd.get("total_protein")) or sum(m["protein"] for m in meals)
        total_price = _coerce_num(rd.get("total_price")) or sum(m["price"] for m in meals)
        out.append({
            "day": str(rd.get("day") or (DAYS[i] if i < len(DAYS) else f"Day {i + 1}")),
            "meals": meals,
            "total_kcal": total_kcal,
            "total_protein": total_protein,
            "total_price": total_price,
            "over_budget": bool(daily_budget_pkr and total_price > daily_budget_pkr),
        })
    return out


# --------------------------------------------------------------------------- #
# CSV-priced meal plan. meals_prices.csv (Item, Unit, Price_PKR, Notes) is the
# ONLY source of prices. Gemini arranges FOODS + QUANTITIES; prices are fixed.
# A deterministic assembler runs with zero API calls when Gemini is unavailable.
# --------------------------------------------------------------------------- #
DIET_PREFERENCES = [
    "No preference",
    "Vegetarian",
    "Vegan",
    "Halal only",
    "Eggetarian (vegetarian + eggs)",
    "High protein focus",
    "Low carb",
    "Dairy-free",
    "Low budget",
]

# Templates reference meals_prices.csv items by name. `qty` is in the item's own
# unit (piece -> count, kg/litre/500ml/500g/loaf -> fraction of that unit).
# cost = qty x Price_PKR. Portions are fixed (no calorie scaling) so the
# "N x price = Rs total" breakdown stays clean.
_MEAL_TEMPLATES = {
    "breakfast": [
        {"name": "Eggs, roti & chai", "kcal": 480, "protein": 24, "tags": {"egg"},
         "ing": [("Egg", 2, "2 eggs"), ("Roti", 2, "2 roti"), ("Tea", 1, "tea"),
                 ("Oil", 0.01, "fry oil")]},
        {"name": "Daal with roti", "kcal": 470, "protein": 20, "tags": {"veg"},
         "ing": [("Daal", 0.12, "120g daal"), ("Roti", 2, "2 roti"),
                 ("Oil", 0.02, "tarka oil"), ("Tea", 1, "tea")]},
        {"name": "Paneer bhurji & roti", "kcal": 520, "protein": 24, "tags": {"veg"},
         "ing": [("Paneer", 0.1, "100g paneer"), ("Roti", 2, "2 roti"),
                 ("Vegetables", 0.05, "onion/tomato"), ("Oil", 0.02, "oil")]},
        {"name": "Bread, butter & milk", "kcal": 430, "protein": 12, "tags": {"veg"},
         "ing": [("Bread", 0.2, "3 slices"), ("Butter", 0.03, "butter"),
                 ("Milk", 0.25, "250ml milk"), ("Sugar", 0.01, "sugar")]},
        {"name": "Yogurt, bread & tea", "kcal": 380, "protein": 14, "tags": {"veg"},
         "ing": [("Yogurt", 0.5, "250ml yogurt"), ("Bread", 0.2, "3 slices"),
                 ("Tea", 1, "tea")]},
        {"name": "Egg & bread breakfast", "kcal": 450, "protein": 18, "tags": {"egg"},
         "ing": [("Egg", 2, "2 eggs"), ("Bread", 0.2, "3 slices"),
                 ("Butter", 0.02, "butter"), ("Tea", 1, "tea")]},
        {"name": "Beans on toast", "kcal": 460, "protein": 18, "tags": {"veg"},
         "ing": [("Beans", 0.12, "120g beans"), ("Bread", 0.2, "3 slices"),
                 ("Oil", 0.01, "oil"), ("Tea", 1, "tea")]},
    ],
    "lunch": [
        {"name": "Chicken curry & rice", "kcal": 680, "protein": 42, "tags": {"meat"},
         "ing": [("Chicken", 0.2, "200g chicken"), ("Rice", 0.15, "150g rice"),
                 ("Vegetables", 0.1, "onion/tomato"), ("Oil", 0.03, "oil")]},
        {"name": "Daal chawal", "kcal": 560, "protein": 18, "tags": {"veg"},
         "ing": [("Daal", 0.15, "150g daal"), ("Rice", 0.15, "150g rice"),
                 ("Vegetables", 0.05, "onion"), ("Oil", 0.03, "oil")]},
        {"name": "Fish & rice", "kcal": 600, "protein": 38, "tags": {"meat"},
         "ing": [("Fish", 0.2, "200g fish"), ("Rice", 0.15, "150g rice"),
                 ("Vegetables", 0.1, "salad"), ("Oil", 0.03, "oil")]},
        {"name": "Mutton salan & roti", "kcal": 680, "protein": 36, "tags": {"meat"},
         "ing": [("Mutton", 0.12, "120g mutton"), ("Roti", 2, "2 roti"),
                 ("Vegetables", 0.1, "onion/tomato"), ("Oil", 0.03, "oil")]},
        {"name": "Rajma (beans) & rice", "kcal": 600, "protein": 20, "tags": {"veg"},
         "ing": [("Beans", 0.15, "150g beans"), ("Rice", 0.15, "150g rice"),
                 ("Vegetables", 0.05, "onion"), ("Oil", 0.03, "oil")]},
        {"name": "Chicken & roti", "kcal": 620, "protein": 40, "tags": {"meat"},
         "ing": [("Chicken", 0.18, "180g chicken"), ("Roti", 2, "2 roti"),
                 ("Vegetables", 0.1, "onion/tomato"), ("Oil", 0.02, "oil")]},
        {"name": "Vegetable pulao", "kcal": 560, "protein": 12, "tags": {"veg"},
         "ing": [("Rice", 0.18, "180g rice"), ("Vegetables", 0.2, "200g veg"),
                 ("Daal", 0.05, "50g daal"), ("Oil", 0.03, "oil")]},
    ],
    "dinner": [
        {"name": "Grilled chicken & veg", "kcal": 520, "protein": 46, "tags": {"meat"},
         "ing": [("Chicken", 0.2, "200g chicken"), ("Vegetables", 0.25, "250g veg"),
                 ("Roti", 1, "1 roti"), ("Oil", 0.02, "oil")]},
        {"name": "Mixed vegetable & roti", "kcal": 400, "protein": 12, "tags": {"veg"},
         "ing": [("Vegetables", 0.3, "300g veg"), ("Roti", 2, "2 roti"),
                 ("Oil", 0.03, "oil")]},
        {"name": "Egg curry & rice", "kcal": 500, "protein": 20, "tags": {"egg"},
         "ing": [("Egg", 2, "2 eggs"), ("Rice", 0.12, "120g rice"),
                 ("Vegetables", 0.1, "onion/tomato"), ("Oil", 0.02, "oil")]},
        {"name": "Daal & rice", "kcal": 500, "protein": 18, "tags": {"veg"},
         "ing": [("Daal", 0.14, "140g daal"), ("Rice", 0.12, "120g rice"),
                 ("Vegetables", 0.05, "salad"), ("Oil", 0.02, "oil")]},
        {"name": "Fish & vegetables", "kcal": 480, "protein": 36, "tags": {"meat"},
         "ing": [("Fish", 0.18, "180g fish"), ("Vegetables", 0.2, "200g veg"),
                 ("Roti", 1, "1 roti"), ("Oil", 0.02, "oil")]},
        {"name": "Paneer & roti", "kcal": 520, "protein": 22, "tags": {"veg"},
         "ing": [("Paneer", 0.1, "100g paneer"), ("Roti", 2, "2 roti"),
                 ("Vegetables", 0.1, "onion/tomato"), ("Oil", 0.02, "oil")]},
        {"name": "Chicken shorba & roti", "kcal": 470, "protein": 30, "tags": {"meat"},
         "ing": [("Chicken", 0.15, "150g chicken"), ("Roti", 2, "2 roti"),
                 ("Vegetables", 0.05, "onion"), ("Oil", 0.02, "oil")]},
    ],
    "snack": [
        {"name": "2 boiled eggs", "kcal": 140, "protein": 12, "tags": {"egg"},
         "ing": [("Egg", 2, "2 eggs")]},
        {"name": "Yogurt bowl", "kcal": 130, "protein": 9, "tags": {"veg"},
         "ing": [("Yogurt", 0.5, "250ml yogurt")]},
        {"name": "Milk & sugar", "kcal": 170, "protein": 8, "tags": {"veg"},
         "ing": [("Milk", 0.25, "250ml milk"), ("Sugar", 0.01, "sugar")]},
        {"name": "Tea & bread", "kcal": 180, "protein": 6, "tags": {"veg"},
         "ing": [("Tea", 1, "tea"), ("Bread", 0.13, "2 slices"), ("Butter", 0.01, "butter")]},
        {"name": "Bread & butter", "kcal": 200, "protein": 6, "tags": {"veg"},
         "ing": [("Bread", 0.13, "2 slices"), ("Butter", 0.02, "butter")]},
        {"name": "Paneer cubes", "kcal": 110, "protein": 8, "tags": {"veg"},
         "ing": [("Paneer", 0.04, "40g paneer")]},
        {"name": "Roti & butter", "kcal": 150, "protein": 4, "tags": {"veg"},
         "ing": [("Roti", 1, "1 roti"), ("Butter", 0.015, "butter")]},
    ],
}

_CSV_MEAL_SCHEMA = (
    '{"days":[{"day":"Day 1","meals":[{"slot":"Breakfast",'
    '"name":"dish name",'
    '"items":"2 roti (2 x 25 = Rs 50) + 1 egg (1 x 26 = Rs 26) + tea (1 x 70 = Rs 70)",'
    '"kcal":0,"protein":0,"price":0}],"total_kcal":0,"total_price":0}]}'
)

# Gemini arranges the foods; it is NOT allowed to touch prices.
_MEAL_CALC_RULES = (
    "PRICING RULES - follow EXACTLY:\n"
    "- The price list above is the ONLY source of prices. Do NOT invent, guess or "
    "change any price.\n"
    "- YOU choose the foods and quantities. The PRICES are fixed by the list.\n"
    "- Cost of a food = quantity x its listed unit price. Respect the unit:\n"
    "    'per piece' -> a count  (2 roti = 2 x 25 = Rs 50)\n"
    "    'per cup'   -> a count  (1 tea  = 1 x 70 = Rs 70)\n"
    "    'per loaf'  -> a fraction of the loaf (3 slices ~ 0.2 loaf)\n"
    "    'per kg'    -> a fraction of a kg     (200 g chicken = 0.2 x 800 = Rs 160)\n"
    "    'per liter' / 'per 500ml' / 'per 500g' -> a fraction of that pack\n"
    "- 'items' MUST show every food with its arithmetic, e.g. "
    "'2 roti (2 x 25 = Rs 50) + 1 egg (1 x 26 = Rs 26) + tea (1 x 70 = Rs 70)'.\n"
    "- meal 'price' = sum of those amounts. day 'total_price' = sum of its meals.\n"
    "WORKED EXAMPLE (Breakfast): "
    "2 roti (2 x 25 = Rs 50) + 1 egg (1 x 26 = Rs 26) + tea (1 x 70 = Rs 70) = Rs 146\n"
)


def _strict_meal_prompt(price_lines: list[str], calorie_target: int, daily_budget_pkr: float,
                        goal: str, preference: str, days: int, extra: str = "") -> str:
    listing = "\n".join(f"- {ln}" for ln in price_lines)
    return (
        f"Create a {days}-day home-cooked Pakistani meal plan.\n"
        f"Goal: {goal}. Dietary preference: {preference}. "
        f"Daily calorie target: {calorie_target} kcal. Daily food budget: "
        f"Rs {daily_budget_pkr:.0f}.\n\n"
        f"AVAILABLE FOODS AND FIXED PRICES (from meals_prices.csv - the ONLY foods "
        f"and prices you may use):\n{listing}\n\n"
        + _MEAL_CALC_RULES
        + (f"\n{extra}\n" if extra else "")
        + "\nEach day: Breakfast, Lunch, Dinner + 1-2 Snacks. Keep each day's total "
        "kcal within ~150 of the target.\n"
        "Return ONLY minified JSON, no markdown:\n" + _CSV_MEAL_SCHEMA
    )


def _validate_meal_days(plan_days: list[dict], daily_budget_pkr: float) -> bool:
    """Reject obviously-wrong Gemini output (bad units / made-up prices)."""
    if not plan_days or len(plan_days) < 3:
        return False
    for d in plan_days:
        if not (2 <= len(d["meals"]) <= 6):
            return False
        for m in d["meals"]:
            if not (5 <= m["price"] <= 1600):
                return False
            if not (50 <= m["kcal"] <= 1800):
                return False
        if not (100 <= d["total_price"] <= 5000):
            return False
        if not (800 <= d["total_kcal"] <= 4500):
            return False
    return True


def _gemini_meal_plan_days(price_lines, calorie_target, daily_budget_pkr, goal,
                           preference, days, extra="", timeout=None):
    """Run the strict prompt, parse, validate. Returns plan_days or None.

    The validated result is cached (in-process + SQLite, 2 days) keyed by the
    inputs, so a repeat request with the same budget/goal/diet is instant and
    never calls Gemini again. `timeout` (default GEMINI_MEAL_TIMEOUT, 18s) keeps
    a slow model from stalling the Fitness page.
    """
    import hashlib

    ckey = "gemplan:" + hashlib.sha1(
        ("|".join(price_lines)
         + f"|{calorie_target}|{round(daily_budget_pkr)}|{goal}|{preference}|{days}"
         + f"|{extra[:120]}").encode()).hexdigest()[:20]
    try:
        import database as _db

        hit = _db.get_cached_price(ckey, max_age_days=2)
        if hit:
            return hit
    except Exception:
        pass

    try:
        import gemini_client

        if not gemini_client.is_available() or not price_lines:
            return None
        to = float(timeout or os.environ.get("GEMINI_MEAL_TIMEOUT", "18"))
        prompt = _strict_meal_prompt(price_lines, calorie_target, daily_budget_pkr,
                                     goal, preference, days, extra)
        raw = gemini_client.generate(prompt, temperature=0.4, timeout=to)
        plan_days = _normalise_ai_days(gemini_client.extract_json(raw), days, daily_budget_pkr)
        if not _validate_meal_days(plan_days, daily_budget_pkr):
            return None
        try:
            import database as _db

            _db.set_cached_price(ckey, "gemplan", plan_days)
        except Exception:
            pass
        return plan_days
    except Exception:
        return None


def _template_ok(t: dict, pref: str) -> bool:
    pref = pref.lower()
    # vegan has no separate "dairy" tag on these templates, so it's treated as
    # vegetarian here (meat/egg excluded) - the CSV-priced optimiser below
    # (_food_pool) is where vegan/dairy-free/low-carb get precise, item-level
    # filtering; this deterministic assembler is only reached when no budget
    # is set at all.
    if "vegan" in pref or ("vegetarian" in pref and "egg" not in pref):
        return t["tags"] <= {"veg"}
    if "eggetarian" in pref:
        return t["tags"] <= {"veg", "egg"}
    return True


def _fmt_qty(q: float) -> str:
    return f"{q:g}"


def _assemble_meal_plan_from_csv(calorie_target, daily_budget_pkr, preference, days):
    """Deterministic - no API. Fixed portions, prices straight from meals_prices.csv.

    Returns (plan_days, used_rows dict by csv item). Breakdown text is
    'label (qty x price = Rs cost)'.
    """
    import price_data

    pref = (preference or "").lower()
    pools = {}
    for slot, temps in _MEAL_TEMPLATES.items():
        pool = [t for t in temps if _template_ok(t, pref)] or list(temps)
        if "low budget" in pref:
            pool = [t for t in pool if not (t["tags"] & {"meat"})] or pool
        if "high protein" in pref or "high-protein" in pref:
            pool = sorted(pool, key=lambda t: -t["protein"])
        pools[slot] = pool

    used: dict[str, dict] = {}
    plan_days = []
    for i in range(days):
        picks = [
            dict(pools["breakfast"][i % len(pools["breakfast"])], slot="Breakfast"),
            dict(pools["lunch"][i % len(pools["lunch"])], slot="Lunch"),
            dict(pools["dinner"][i % len(pools["dinner"])], slot="Dinner"),
            dict(pools["snack"][i % len(pools["snack"])], slot="Snack"),
            dict(pools["snack"][(i + 3) % len(pools["snack"])], slot="Snack"),
        ]
        meals = []
        for p in picks:
            parts, cost = [], 0.0
            for key, qty, label in p["ing"]:
                row = price_data.find_row(key)
                if not row:
                    continue
                used[row["item"]] = row
                price = row["price_pkr"]
                amount = price * qty
                cost += amount
                parts.append(f"{label} ({_fmt_qty(qty)} x {price} = Rs {int(round(amount))})")
            meals.append({
                "slot": p["slot"], "name": p["name"], "portion": 1.0,
                "kcal": p["kcal"], "protein": p["protein"],
                "price": int(round(cost)),
                "items": " + ".join(parts),
            })
        tk = sum(m["kcal"] for m in meals)
        tp = sum(m["protein"] for m in meals)
        tpr = sum(m["price"] for m in meals)
        plan_days.append({
            "day": DAYS[i] if i < len(DAYS) else f"Day {i + 1}",
            "meals": meals, "total_kcal": tk, "total_protein": tp, "total_price": tpr,
            "over_budget": bool(daily_budget_pkr and tpr > daily_budget_pkr),
        })
    return plan_days, used


def _budget_note(avg: int, daily_budget_pkr: float) -> str | None:
    if not daily_budget_pkr:
        return None
    if avg > daily_budget_pkr:
        return (f"Avg PKR {avg:,}/day is above your Rs {int(daily_budget_pkr):,} budget "
                f"- pick more daal / egg / vegetable meals.")
    return f"Fits budget: avg PKR {avg:,}/day vs Rs {int(daily_budget_pkr):,}/day."


# =========================================================================== #
# Budget as a HARD constraint: validation -> portion caps -> cost/macro
# optimiser. If the target can't be met inside the budget we ERROR (with the
# minimum viable budget) instead of quietly overspending.
# =========================================================================== #
MIN_DAILY_BUDGET_PKR = 500
MAX_DAILY_BUDGET_PKR = 5000
MIN_MONTHLY_BUDGET_PKR = 15_000
MAX_MONTHLY_BUDGET_PKR = 150_000
_BUDGET_ADVICE = "Try PKR 800-1200/day for balanced nutrition."

# per-unit kcal / protein for meals_prices.csv items  (unit = the CSV "Per X")
_FOOD_NUTRITION = {
    "wheat flour (atta)": (3400, 100), "roti": (130, 3.5), "naan": (260, 7),
    "paratha": (300, 6), "rice (basmati)": (3600, 70), "rice (sella)": (3600, 68),
    "bread (large)": (1080, 36),
    "chicken (whole cut)": (1400, 130), "chicken boneless": (1650, 200),
    "beef (with bone)": (1500, 130), "beef (boneless)": (2000, 200),
    "mutton": (2500, 180), "fish": (1000, 190), "egg": (72, 6),
    "milk (fresh)": (600, 32), "milk (tetrapack)": (610, 32), "yogurt (dahi)": (600, 34),
    "paneer": (2650, 180),
    "daal chana": (3500, 200), "daal moong": (3450, 240), "daal maash": (3400, 250),
    "daal masoor": (3450, 250), "white chana": (3600, 190), "rajma": (3400, 240),
    "potato (aloo)": (770, 20), "onion (piaz)": (400, 11), "tomato (tamatar)": (180, 9),
    "garlic (lashan)": (1490, 63), "ginger (adrak)": (800, 18), "green chilli": (400, 20),
    "spinach (palak)": (230, 29), "ladyfinger (bhindi)": (330, 19),
    "cooking oil": (8840, 0), "banaspati ghee": (9000, 0), "butter": (1450, 2),
    "sugar": (3870, 0), "tea (patti)": (0, 0), "tea (prepared)": (90, 2),
    "banana": (1260, 15), "apple": (520, 3), "dates (khajoor)": (2800, 20),
    "samosa": (130, 3), "pakora": (2400, 45),
    "chicken biryani": (600, 22), "chicken karahi": (900, 95),
}
_STAPLES = {"roti", "naan", "paratha", "rice (basmati)", "rice (sella)", "bread (large)"}
_PROTEINS = {"chicken (whole cut)", "chicken boneless", "beef (with bone)",
             "beef (boneless)", "mutton", "fish", "egg", "milk (fresh)",
             "milk (tetrapack)", "yogurt (dahi)", "paneer", "daal chana", "daal moong",
             "daal maash", "daal masoor", "white chana", "rajma", "chicken karahi"}
# protein price tier -> unlocked by how much budget a meal has
_PROT_CLASS = {
    "chicken (whole cut)": "premium", "chicken boneless": "premium",
    "beef (with bone)": "premium", "beef (boneless)": "premium", "mutton": "premium",
    "fish": "premium", "paneer": "premium", "chicken karahi": "premium",
    "egg": "mid", "yogurt (dahi)": "mid", "milk (fresh)": "mid", "milk (tetrapack)": "mid",
    "daal chana": "legume", "daal moong": "legume", "daal maash": "legume",
    "daal masoor": "legume", "white chana": "legume", "rajma": "legume",
}


def _protein_tier(meal_budget: float) -> set[str]:
    if meal_budget >= 150:
        return {"premium", "mid", "legume"}
    if meal_budget >= 55:
        return {"mid", "legume"}
    return {"legume"}
_VEG = {"potato (aloo)", "onion (piaz)", "tomato (tamatar)", "spinach (palak)",
        "ladyfinger (bhindi)"}
_SNACK_ITEMS = {"egg", "yogurt (dahi)", "milk (fresh)", "banana", "apple",
                "dates (khajoor)", "white chana"}
_MEAT = {"chicken (whole cut)", "chicken boneless", "beef (with bone)",
         "beef (boneless)", "mutton", "fish", "chicken karahi"}
_PREMIUM = _MEAT | {"paneer", "dates (khajoor)", "apple", "chicken boneless"}
_DAIRY = {"milk (fresh)", "milk (tetrapack)", "yogurt (dahi)", "paneer", "butter"}
_HIGH_CARB_STAPLES = _STAPLES  # rice/roti/naan/paratha/bread - biased down for "low carb"
MEAL_KCAL_SHARE = {"Breakfast": 0.25, "Lunch": 0.32, "Dinner": 0.30, "Snack": 0.13}
MAX_ITEM_KCAL_SHARE = 0.25            # no single item > 25% of the day's calories


def validate_budget(daily_budget_pkr: float) -> str | None:
    """Return an error string if the daily budget is impossibly low, else None.
    A budget <= 0 means 'not set' -> no hard constraint (handled elsewhere)."""
    b = float(daily_budget_pkr or 0)
    if b <= 0:
        return None
    if b < MIN_DAILY_BUDGET_PKR:
        return (f"Your budget (PKR {b:,.0f}/day) is too low. "
                f"Minimum viable: PKR {MIN_DAILY_BUDGET_PKR}/day "
                f"(PKR {MIN_MONTHLY_BUDGET_PKR:,}/month). {_BUDGET_ADVICE}")
    return None


def _unit_kind(unit: str) -> str:
    u = (unit or "").lower()
    if "piece" in u or "cup" in u or "plate" in u:
        return "count"
    if "half kg" in u:
        return "halfkg"
    return "frac"                     # kg / liter / loaf / 200 g / 500 ml / dozen


def _portion_cap(name: str, unit: str) -> float:
    """Max quantity of one item in ONE meal, in the item's own unit."""
    n = name.lower()
    if "roti" in n:
        return 3
    if "naan" in n or "paratha" in n:
        return 2
    if n.startswith("rice"):
        return 0.20                   # 200 g
    if n == "egg":
        return 3
    if n in _MEAT:
        return 0.20                   # 200 g meat / fish
    if "banana" in n:
        return 2 / 12                 # 2 of a dozen
    if "apple" in n:
        return 0.30
    if "milk" in n:
        return 0.30                   # 300 ml
    if "yogurt" in n:
        return 0.25
    if "paneer" in n:
        return 0.10
    if n.startswith("daal") or n in ("rajma", "white chana"):
        return 0.18                   # 180 g dry legume (~1.5 katori cooked)
    if "bread" in n:
        return 0.30
    if n in ("cooking oil", "banaspati ghee"):
        return 0.02
    if "butter" in n:
        return 0.15
    if "sugar" in n:
        return 0.03
    return 1 if _unit_kind(unit) != "frac" else 0.25   # "max 1 serving" default


def _min_portion(kind: str) -> float:
    return {"count": 1, "halfkg": 1}.get(kind, 0.03)


def _round_portion(q: float, kind: str, name: str) -> float:
    if q <= 0:
        return 0.0
    if kind in ("count", "halfkg"):
        return float(max(0, int(round(q))))
    if "banana" in name.lower():
        return round(q * 12) / 12.0          # whole bananas; may round to 0
    return round(q, 2)


def _food_pool(preference: str) -> list[dict]:
    """meals_prices.csv rows + macros + role + per-meal cap + efficiency ratio.

    Diet preference filtering (substring match on the lower-cased preference
    string, so any label containing these words is honoured):
      vegetarian    -> no meat, no egg
      vegan         -> no meat, no egg, no dairy
      eggetarian    -> no meat (egg allowed)
      dairy-free    -> no milk/yogurt/paneer/butter (meat/egg still allowed)
      low budget    -> no premium-priced items
      low carb      -> high-carb staples (rice/roti/naan/paratha/bread) get a
                       much smaller per-meal cap so the optimiser leans on
                       protein/veg instead - not banned outright, so calorie
                       targets stay reachable on a tight budget.
    """
    import price_data

    pref = (preference or "").lower()
    veg_only = "vegetarian" in pref and "egg" not in pref
    vegan = "vegan" in pref
    eggetarian = "eggetarian" in pref
    dairy_free = vegan or "dairy-free" in pref or "dairy free" in pref
    low_budget = "low budget" in pref
    low_carb = "low carb" in pref
    high_protein = "high protein" in pref or "high-protein" in pref
    pool = []
    for r in price_data.load_meal_prices()["rows"]:
        key = r["item"].lower()
        if key not in _FOOD_NUTRITION:
            continue
        kcal_u, prot_u = _FOOD_NUTRITION[key]
        if kcal_u <= 0:
            continue
        if (veg_only or eggetarian or vegan) and key in _MEAT:
            continue
        if (veg_only or vegan) and key == "egg":
            continue
        if dairy_free and key in _DAIRY:
            continue
        if low_budget and key in _PREMIUM:
            continue
        price = float(r["price_pkr"])
        kind = _unit_kind(r["unit"])
        cap = _portion_cap(r["item"], r["unit"])
        role = ("staple" if key in _STAPLES else "protein" if key in _PROTEINS
                else "veg" if key in _VEG else "other")
        if low_carb and key in _HIGH_CARB_STAPLES:
            cap *= 0.4
        prot_weight = 1.6 if high_protein else 1.0
        eff = (prot_u * prot_weight + kcal_u / 100.0) / price if price > 0 else 0.0
        pool.append({"item": r["item"], "key": key, "unit": r["unit"], "price": price,
                     "kind": kind, "kcal_u": kcal_u, "prot_u": prot_u, "cap": cap,
                     "role": role, "eff": eff,
                     "snack": key in _SNACK_ITEMS})
    pool.sort(key=lambda c: -c["eff"])            # most cost-efficient first
    return pool


def _take(c: dict, kcal_gap: float, prot_gap: float, budget_left: float,
          day_kcal_target: float, prefer: str = "kcal") -> tuple[float, dict] | None:
    """Largest affordable, cap-respecting, <=25%-daily-kcal portion of `c`.
    `prefer="protein"` sizes to the protein gap (only sensible for real protein
    foods); otherwise sizes to the calorie gap. Never far past either -> no
    ballooning meals."""
    if c["price"] <= 0 or budget_left < c["price"] * _min_portion(c["kind"]):
        return None
    by_kcal = (kcal_gap * 1.10) / c["kcal_u"] if c["kcal_u"] else c["cap"]
    by_prot = (prot_gap * 1.10) / c["prot_u"] if c["prot_u"] else 0.0
    want = by_prot if (prefer == "protein" and prot_gap > 3 and c["prot_u"] > 0) else by_kcal
    by_budget = budget_left / c["price"]
    by_share = (MAX_ITEM_KCAL_SHARE * day_kcal_target) / c["kcal_u"] if c["kcal_u"] else c["cap"]
    q = min(c["cap"], by_budget, by_share, max(want, 0.0))
    if q <= 0:
        return None
    q = _round_portion(q, c["kind"], c["item"])
    step = 1.0 if c["kind"] != "frac" else 0.05
    for _ in range(40):
        if q < _min_portion(c["kind"]) or q * c["price"] <= budget_left + 1e-6:
            break
        q = _round_portion(q - step, c["kind"], c["item"])
    if q < _min_portion(c["kind"]) or q * c["price"] > budget_left + 1e-6:
        return None
    cost = round(q * c["price"], 1)
    return q, {"key": c["key"], "name": c["item"], "qty": q, "unit": c["unit"],
               "kcal": int(round(q * c["kcal_u"])), "protein": round(q * c["prot_u"], 1),
               "price": cost,
               "label": f"{q:g} x {c['item']} ({q:g} x {int(c['price'])} = Rs {int(round(cost))})"}


def _rotate_top(cands: list[dict], keyfn, rot: int, span: int = 4) -> list[dict]:
    """Sort by keyfn, then rotate within the top `span` so different days pick
    different (still-efficient) foods -> variety without abandoning efficiency."""
    s = sorted(cands, key=keyfn)
    if len(s) <= 1:
        return s
    off = rot % min(span, len(s))
    return s[off:] + s[:off]


def _build_meal(slot: str, pool: list[dict], kcal_goal: float, prot_goal: float,
                meal_budget: float, day_kcal_target: float, rot: int,
                low_budget: bool = False) -> dict:
    chosen, spent, kcal, prot = [], 0.0, 0.0, 0.0
    used: set[str] = set()

    def add(c, prefer="kcal") -> bool:
        nonlocal spent, kcal, prot
        got = _take(c, max(kcal_goal - kcal, 0), max(prot_goal - prot, 0),
                    meal_budget - spent, day_kcal_target, prefer)
        if not got:
            return False
        _, item = got
        chosen.append(item)
        used.add(c["key"])
        spent += item["price"]
        kcal += item["kcal"]
        prot += item["protein"]
        return True

    def pick(cands, keyfn, prefer="kcal"):
        for c in _rotate_top([c for c in cands if c["key"] not in used], keyfn, rot):
            if add(c, prefer):
                return

    if slot == "Snack":
        pick([c for c in pool if c["snack"]], lambda c: -c["eff"])
        if kcal < 0.6 * kcal_goal:
            pick([c for c in pool if c["snack"]], lambda c: -c["eff"])
    else:
        # affordable proteins only (a PKR 40 meal can't buy chicken), then STRICT
        # cost-per-macro order: best protein-per-PKR first. Rotated across days for
        # variety. Efficiency-first keeps the feasibility monotonic in budget.
        tier = _protein_tier(meal_budget)
        prot_cands = [c for c in pool if c["role"] == "protein"
                      and _PROT_CLASS.get(c["key"], "legume") in tier] \
            or [c for c in pool if c["role"] == "protein"]
        pick(prot_cands, lambda c: -(c["prot_u"] / c["price"]), prefer="protein")
        # 2. staple to carry the calories
        pick([c for c in pool if c["role"] == "staple"], lambda c: -c["eff"])
        # 3. a vegetable for balance if there is budget headroom
        if meal_budget - spent > 12 and any(c["role"] == "veg" for c in pool):
            pick([c for c in pool if c["role"] == "veg"],
                 lambda c: c["price"] / max(c["kcal_u"], 1))
        # 4. only if still short on protein AND calories: a 2nd protein
        if prot < prot_goal * 0.85 and kcal < kcal_goal:
            pick(prot_cands, lambda c: -(c["prot_u"] / c["price"]), prefer="protein")
        # 5. only if still short on calories: cheapest energy left
        if kcal < 0.85 * kcal_goal:
            pick([c for c in pool if c["role"] in ("staple", "veg")],
                 lambda c: c["price"] / max(c["kcal_u"], 1))

    return {"slot": slot, "name": " + ".join(i["name"] for i in chosen) or "-",
            "portion": 1.0, "kcal": int(round(kcal)), "protein": int(round(prot)),
            "price": int(round(spent)),
            "items": " + ".join(i["label"] for i in chosen)}


def _optimize_day(cal_target: int, prot_target: int, day_budget: float,
                  pool: list[dict], rot: int, label: str, low_budget: bool = False):
    """Build one day inside `day_budget`. Returns (day_dict, feasible: bool).

    Each meal gets its share of the STILL-UNSPENT budget, so a generous budget
    buys more variety / better protein while a tight one still balances."""
    meals, spent = [], 0.0
    slots = list(MEAL_KCAL_SHARE.items())
    for idx, (slot, share) in enumerate(slots):
        remaining_share = sum(s for _, s in slots[idx:])
        meal_budget = (day_budget - spent) * (share / remaining_share)
        m = _build_meal(slot, pool, cal_target * share, prot_target * share,
                        meal_budget, cal_target, rot + idx, low_budget)
        spent += m["price"]
        meals.append(m)
    tk = sum(m["kcal"] for m in meals)
    tp = sum(m["protein"] for m in meals)
    tpr = sum(m["price"] for m in meals)
    # calories must land close; protein is harder on a budget with sane portions
    # so 60% of target counts as "viable" (top up with a whey scoop if wanted).
    feasible = (tpr <= day_budget + 1 and tk >= 0.82 * cal_target
                and tp >= 0.60 * prot_target)
    return ({"day": label, "meals": meals, "total_kcal": tk, "total_protein": tp,
             "total_price": tpr, "over_budget": tpr > day_budget + 1}, feasible)


def _min_feasible_budget(cal_target: int, prot_target: int, pool: list[dict],
                         low_budget: bool = False) -> int:
    for b in range(MIN_DAILY_BUDGET_PKR, MAX_DAILY_BUDGET_PKR + 1, 100):
        _, ok = _optimize_day(cal_target, prot_target, b, pool, 0, "probe", low_budget)
        if ok:
            return b
    return MAX_DAILY_BUDGET_PKR


def build_meal_plan_optimized(calorie_target: int, daily_budget_pkr: float,
                              protein_target_g: int, preference: str = "No preference",
                              days: int = 7) -> dict:
    """Cost-per-macro greedy optimiser. Budget is a HARD ceiling: if the calorie /
    protein target cannot be met inside it, returns {'error', 'min_budget', ...}
    and NO plan."""
    import price_data

    meta = price_data.load_meal_prices()
    if meta["error"] or not meta["rows"]:
        return {"error": f"meals_prices.csv problem: {meta['error'] or 'no rows'}",
                "days": [], "source": "csv-missing", "weekly_cost": 0, "avg_daily_cost": 0,
                "notes": []}

    budget = min(float(daily_budget_pkr), MAX_DAILY_BUDGET_PKR)
    clamped = float(daily_budget_pkr) > MAX_DAILY_BUDGET_PKR
    prot_target = int(protein_target_g or round(calorie_target * 0.25 / 4))
    low_budget = "low budget" in (preference or "").lower()
    pool = _food_pool(preference)
    if not pool:
        return {"error": "No usable foods in meals_prices.csv for this diet preference.",
                "days": [], "source": "csv-missing", "weekly_cost": 0, "avg_daily_cost": 0,
                "notes": []}

    plan_days, feasible_all = [], True
    for i in range(days):
        d, ok = _optimize_day(calorie_target, prot_target, budget, pool, i,
                              DAYS[i] if i < len(DAYS) else f"Day {i + 1}", low_budget)
        feasible_all &= ok
        plan_days.append(d)

    if not feasible_all:
        need = _min_feasible_budget(calorie_target, prot_target, pool, low_budget)
        if need >= MAX_DAILY_BUDGET_PKR:
            msg = (f"A {calorie_target:,} kcal / {prot_target} g-protein target is too "
                   f"demanding for a home-cooked plan from meals_prices.csv at any "
                   f"budget up to PKR {MAX_DAILY_BUDGET_PKR:,}/day with sensible "
                   f"portions. Lower the goal, or add a protein supplement. {_BUDGET_ADVICE}")
        else:
            msg = (f"Your budget (PKR {daily_budget_pkr:,.0f}/day) cannot meet the "
                   f"{calorie_target:,} kcal / {prot_target} g protein target. "
                   f"Minimum viable for this target: about PKR {need:,}/day "
                   f"(PKR {need * 30:,}/month). {_BUDGET_ADVICE}")
        return {
            "error": msg,
            "min_budget": need, "advice": _BUDGET_ADVICE,
            "days": [], "source": "budget-infeasible",
            "weekly_cost": 0, "avg_daily_cost": 0,
            "validation": {"daily_budget": round(daily_budget_pkr), "daily_cost": None,
                           "within_budget": False, "feasible": False},
            "notes": [],
        }

    weekly = sum(d["total_price"] for d in plan_days)
    avg = round(weekly / max(1, len(plan_days)))
    ingredient_prices = [{"item": r["item"], "unit": r["unit"],
                          "price_pkr": r["price_pkr"], "notes": r["notes"]}
                         for r in meta["rows"]]
    notes = [
        f"Budget is a HARD constraint. Foods picked by cost-per-macro efficiency "
        f"= (protein_g + kcal/100) / PKR, then capped per portion "
        f"(roti <=3, rice <=200 g, egg <=3, meat <=150 g, banana <=2, "
        f"no item > 25% of the day's calories).",
        f"Prices from meals_prices.csv ({len(meta['rows'])} items).",
        f"Daily budget: PKR {round(min(daily_budget_pkr, MAX_DAILY_BUDGET_PKR)):,}  ·  "
        f"Daily cost: PKR {avg:,}  ·  "
        + ("WITHIN budget ✅" if avg <= budget + 1 else "OVER budget ❌"),
    ]
    if clamped:
        notes.append(f"Budget above PKR {MAX_DAILY_BUDGET_PKR:,}/day was capped to "
                     f"PKR {MAX_DAILY_BUDGET_PKR:,} for the optimiser.")
    return {
        "days": plan_days, "weekly_cost": weekly, "avg_daily_cost": avg,
        "notes": notes, "source": "csv-optimized",
        "ingredient_prices": ingredient_prices,
        "validation": {"daily_budget": round(min(daily_budget_pkr, MAX_DAILY_BUDGET_PKR)),
                       "daily_cost": avg, "within_budget": avg <= budget + 1,
                       "feasible": True},
    }


def build_meal_plan_csv(calorie_target: int, daily_budget_pkr: float, goal: str,
                        preference: str = "No preference", days: int = 7,
                        use_gemini: bool = True, protein_target_g: int | None = None) -> dict:
    """7-day meal plan. Prices come ONLY from meals_prices.csv.

    Budget handling:
      * budget < PKR 500/day        -> {'error': ...}, no plan (too low to be safe)
      * budget set and >= 500       -> cost-per-macro OPTIMISER; budget is a hard
                                       ceiling. If the calorie/protein target can't
                                       be met inside it -> {'error', 'min_budget'}.
      * no budget entered (<= 0)    -> Gemini arranges the foods (prices fixed from
                                       the CSV), else a local template assembler.
    meals_prices.csv missing/broken -> the built-in static plan.
    """
    import price_data

    err = validate_budget(daily_budget_pkr)
    if err:
        return {"error": err, "days": [], "source": "budget-error",
                "weekly_cost": 0, "avg_daily_cost": 0, "notes": [err],
                "validation": {"daily_budget": round(float(daily_budget_pkr or 0)),
                               "daily_cost": None, "within_budget": False,
                               "feasible": False}}

    meta = price_data.load_meal_prices()
    if meta["error"] or not meta["rows"]:
        fb = build_meal_plan(calorie_target, daily_budget_pkr, days)
        fb["notes"].insert(
            0, f"(meals_prices.csv problem: {meta['error'] or 'no usable rows'} "
               f"- showing the built-in plan so no wrong prices are displayed.)")
        fb["source"] = "csv-missing"
        return fb

    # A real budget makes it a hard constraint -> deterministic optimiser
    # (Gemini can't guarantee staying under the ceiling).
    if float(daily_budget_pkr or 0) >= MIN_DAILY_BUDGET_PKR:
        return build_meal_plan_optimized(calorie_target, float(daily_budget_pkr),
                                         protein_target_g, preference, days)

    price_lines = price_data.meal_price_lines()
    ingredient_prices = [
        {"item": r["item"], "unit": r["unit"], "price_pkr": r["price_pkr"],
         "notes": r["notes"]} for r in meta["rows"]
    ]

    gem_days = _gemini_meal_plan_days(
        price_lines, calorie_target, daily_budget_pkr, goal, preference, days
    ) if use_gemini else None

    if gem_days:
        plan_days = gem_days
        source = "csv+gemini"
        method = ("Gemini arranged the foods and quantities. Every price is fixed "
                  "from meals_prices.csv - it cannot change a price. Arithmetic validated.")
    else:
        plan_days, _ = _assemble_meal_plan_from_csv(
            calorie_target, daily_budget_pkr, preference, days)
        source = "csv"
        method = ("Plan assembled locally from meals_prices.csv - no AI call. "
                  "Each meal = sum of (quantity x CSV unit price).")

    weekly = sum(d["total_price"] for d in plan_days)
    avg = round(weekly / max(1, len(plan_days)))
    notes = [
        f"Prices from meals_prices.csv ({len(meta['rows'])} items). "
        "Roti Rs 25/piece, Rice Rs 430/kg, Chicken Rs 800/kg, Egg Rs 26 each ...",
        method,
    ]
    if (bn := _budget_note(avg, daily_budget_pkr)):
        notes.append(bn)

    return {
        "days": plan_days, "weekly_cost": weekly, "avg_daily_cost": avg,
        "notes": notes, "source": source,
        "ingredient_prices": ingredient_prices,
    }


# --------------------------------------------------------------------------- #
# Workout routines
# --------------------------------------------------------------------------- #
def _goal_key(goal: str) -> str:
    if goal in ("Build muscle", "Mild muscle gain"):
        return "muscle"
    if goal in ("Lose weight", "Mild weight loss"):
        return "loss"
    return "maintain"


WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

GYM_WORKOUTS = {
    "muscle": {
        "Monday": ("Push (chest / shoulders / triceps)", [
            "Barbell bench press — 4 x 6-8",
            "Incline dumbbell press — 3 x 8-10",
            "Seated shoulder press — 3 x 8-10",
            "Lateral raises — 3 x 12-15",
            "Triceps rope pushdown — 3 x 12",
        ]),
        "Tuesday": ("Pull (back / biceps)", [
            "Deadlift — 4 x 5",
            "Lat pulldown / pull-ups — 3 x 8-10",
            "Barbell row — 3 x 8-10",
            "Face pulls — 3 x 15",
            "Barbell curl — 3 x 10-12",
        ]),
        "Wednesday": ("Legs", [
            "Back squat — 4 x 6-8",
            "Romanian deadlift — 3 x 8-10",
            "Leg press — 3 x 10-12",
            "Leg curl — 3 x 12",
            "Standing calf raise — 4 x 15",
        ]),
        "Thursday": ("Push (volume)", [
            "Incline barbell press — 4 x 8",
            "Dumbbell shoulder press — 3 x 10",
            "Cable fly — 3 x 12-15",
            "Lateral raises — 4 x 15",
            "Overhead triceps extension — 3 x 12",
        ]),
        "Friday": ("Pull (volume)", [
            "Pull-ups (weighted if able) — 4 x 8",
            "Seated cable row — 3 x 10-12",
            "Single-arm dumbbell row — 3 x 10",
            "Rear-delt fly — 3 x 15",
            "Hammer curl — 3 x 12",
        ]),
        "Saturday": ("Legs + core", [
            "Front squat — 4 x 8",
            "Hip thrust — 3 x 10-12",
            "Walking lunges — 3 x 12 / leg",
            "Hanging leg raise — 3 x 12",
            "Plank — 3 x 45-60 s",
        ]),
        "Sunday": ("Rest / light walk 30-40 min", []),
    },
    "loss": {
        "Monday": ("Full body A + 10 min cardio", [
            "Goblet squat — 3 x 12",
            "Dumbbell bench press — 3 x 12",
            "Lat pulldown — 3 x 12",
            "Dumbbell RDL — 3 x 12",
            "Incline treadmill walk — 10 min",
        ]),
        "Tuesday": ("HIIT cardio", [
            "Warm-up — 5 min",
            "20 s sprint / 40 s walk x 10 (bike, rower or treadmill)",
            "Cool-down — 5 min",
        ]),
        "Wednesday": ("Full body B", [
            "Leg press — 3 x 15",
            "Seated shoulder press — 3 x 12",
            "Seated cable row — 3 x 12",
            "Leg curl — 3 x 15",
            "Cable crunch — 3 x 15",
        ]),
        "Thursday": ("Steady-state cardio + steps", [
            "Incline walk or cycle — 35-45 min at conversational pace",
            "Target 8-10k steps for the day",
        ]),
        "Friday": ("Full body C", [
            "Romanian deadlift — 3 x 10",
            "Incline dumbbell press — 3 x 12",
            "Assisted pull-ups — 3 x 10",
            "Walking lunges — 3 x 12 / leg",
            "Plank — 3 x 45 s",
        ]),
        "Saturday": ("Active recovery", [
            "Light walk / swim / cycle — 30-40 min",
            "Full-body mobility — 10 min",
        ]),
        "Sunday": ("Rest", []),
    },
    "maintain": {
        "Monday": ("Upper A", [
            "Barbell bench press — 4 x 8",
            "Barbell row — 4 x 8",
            "Seated shoulder press — 3 x 10",
            "Lat pulldown — 3 x 10",
            "Triceps + biceps superset — 3 x 12",
        ]),
        "Tuesday": ("Lower A", [
            "Back squat — 4 x 8",
            "Romanian deadlift — 3 x 10",
            "Leg press — 3 x 12",
            "Calf raise — 4 x 15",
            "Hanging leg raise — 3 x 12",
        ]),
        "Wednesday": ("Cardio + core", [
            "Zone-2 cardio — 30 min",
            "Plank / side plank / dead bug circuit — 3 rounds",
        ]),
        "Thursday": ("Upper B", [
            "Incline dumbbell press — 4 x 10",
            "Pull-ups — 4 x max",
            "Lateral raises — 3 x 15",
            "Cable row — 3 x 12",
            "Face pulls — 3 x 15",
        ]),
        "Friday": ("Lower B", [
            "Deadlift — 3 x 5",
            "Front squat — 3 x 8",
            "Hip thrust — 3 x 12",
            "Leg curl — 3 x 12",
            "Calf raise — 4 x 15",
        ]),
        "Saturday": ("Optional cardio / sport — 30-45 min", []),
        "Sunday": ("Rest", []),
    },
}

HOME_WORKOUTS = {
    "muscle": {
        "Monday": ("Push (bodyweight + bands)", [
            "Push-ups (feet elevated) — 4 x 12-20",
            "Pike push-ups — 3 x 8-12",
            "Band / backpack overhead press — 3 x 12",
            "Bench dips — 3 x 12-15",
            "Diamond push-ups — 2 x AMRAP",
        ]),
        "Tuesday": ("Pull (need a bar or bands)", [
            "Pull-ups / doorway rows — 4 x 8-12",
            "Backpack bent-over row — 3 x 12",
            "Band pull-apart — 3 x 20",
            "Towel curls / backpack curls — 3 x 15",
            "Superman hold — 3 x 30 s",
        ]),
        "Wednesday": ("Legs", [
            "Bulgarian split squat — 4 x 10 / leg",
            "Backpack squat — 4 x 15",
            "Single-leg RDL — 3 x 10 / leg",
            "Glute bridge — 3 x 20",
            "Calf raises (single leg) — 4 x 20",
        ]),
        "Thursday": ("Push volume", [
            "Push-ups — 5 x 15-20",
            "Pike push-ups — 3 x 12",
            "Wide push-ups — 3 x 15",
            "Bench dips — 3 x 20",
        ]),
        "Friday": ("Pull volume + arms", [
            "Rows (bag / table) — 4 x 15",
            "Band pull-apart — 4 x 20",
            "Curls (backpack) — 4 x 15",
            "Reverse snow angels — 3 x 15",
        ]),
        "Saturday": ("Legs + core", [
            "Jump squats — 4 x 12",
            "Walking lunges — 3 x 20",
            "Wall sit — 3 x 45-60 s",
            "Hanging / lying leg raise — 3 x 15",
            "Plank — 3 x 60 s",
        ]),
        "Sunday": ("Rest / walk 30-40 min", []),
    },
    "loss": {
        "Monday": ("Full-body circuit x3-4", [
            "Bodyweight squat — 15",
            "Push-ups — 12",
            "Reverse lunge — 10 / leg",
            "Mountain climbers — 30 s",
            "Rest 60 s, repeat",
        ]),
        "Tuesday": ("HIIT", [
            "Warm-up — 5 min",
            "30 s hard / 30 s easy x 12 (jumping jacks, high knees, burpees, skater hops)",
            "Cool-down — 5 min",
        ]),
        "Wednesday": ("Full-body circuit (lower focus)", [
            "Glute bridge — 20",
            "Squat to calf raise — 15",
            "Superman — 15",
            "Plank shoulder taps — 30 s",
            "3-4 rounds",
        ]),
        "Thursday": ("Walk / cardio", ["Brisk walk 40-50 min", "Aim 9-11k steps"]),
        "Friday": ("Full-body circuit (upper focus)", [
            "Push-ups — 12",
            "Backpack row — 15",
            "Pike push-ups — 10",
            "Bench dips — 15",
            "3-4 rounds",
        ]),
        "Saturday": ("Active recovery", ["Easy walk / cycle 30 min", "Mobility 10 min"]),
        "Sunday": ("Rest", []),
    },
    "maintain": {
        "Monday": ("Upper body", [
            "Push-ups — 4 x 15",
            "Backpack row — 4 x 15",
            "Pike push-ups — 3 x 10",
            "Band pull-apart — 3 x 20",
        ]),
        "Tuesday": ("Lower body", [
            "Backpack squat — 4 x 15",
            "Bulgarian split squat — 3 x 10 / leg",
            "Single-leg RDL — 3 x 10 / leg",
            "Calf raise — 4 x 20",
        ]),
        "Wednesday": ("Cardio + core", ["Run / cycle / walk — 30 min", "Core circuit x3"]),
        "Thursday": ("Full body", [
            "Squat — 3 x 15",
            "Push-ups — 3 x 15",
            "Rows — 3 x 15",
            "Glute bridge — 3 x 20",
        ]),
        "Friday": ("Conditioning", ["EMOM 20 min: 10 squats + 8 push-ups + 12 mountain climbers"]),
        "Saturday": ("Optional sport / walk", []),
        "Sunday": ("Rest", []),
    },
}


def generate_workout(goal: str, gym_access: bool, days_per_week: int | None = None) -> dict:
    key = _goal_key(goal)
    source = GYM_WORKOUTS if gym_access else HOME_WORKOUTS
    plan = source[key]
    out = {}
    for day in WEEK:
        focus, exercises = plan[day]
        out[day] = {"focus": focus, "exercises": list(exercises), "sets": None, "reps": None}
    if days_per_week and 3 <= days_per_week < 6:
        trained = [d for d in WEEK if out[d]["exercises"]]
        for d in trained[days_per_week:]:                     # drop surplus training days
            out[d] = {"focus": "Rest / light walk 30-40 min", "exercises": [],
                      "sets": None, "reps": None}
    return {
        "location": "Gym" if gym_access else "Home",
        "goal_focus": key,
        "experience_level": "Intermediate",
        "source": "static",
        "week": out,
    }


# --------------------------------------------------------------------------- #
# Dynamic workout generation (Gemini) with a strict schema + static fallback
# --------------------------------------------------------------------------- #
EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced"]
WORKOUT_LOCATIONS = ["Home", "Gym", "Both"]
WORKOUT_FOCI = ["Strength", "Hypertrophy", "Endurance", "Fat loss", "General fitness"]
WORKOUT_DAYS_CHOICES = [3, 4, 5, 6]
EQUIPMENT_OPTIONS = [
    "Dumbbells", "Barbell", "Pull-up bar", "Resistance bands", "Kettlebell",
    "Bench", "Cable machine", "Leg press", "Squat rack", "Treadmill/Cardio machine",
    "Backpack (loadable)", "None (bodyweight only)",
]

# experience -> volume / intensity envelope used for the prompt AND validation
_EXP_VOLUME = {
    "Beginner":     {"sets": 3, "reps": "8-12", "ex_max": 6, "rest_min": 3,
                     "wk_sets_max": 95, "days_cap": 4,
                     "intensity": "leave 3-4 reps in reserve; NO training to failure; full rest between sets"},
    "Intermediate": {"sets": 4, "reps": "6-12", "ex_max": 7, "rest_min": 2,
                     "wk_sets_max": 165, "days_cap": 5,
                     "intensity": "1-2 reps in reserve; one top-set AMRAP per lift is fine"},
    "Advanced":     {"sets": 4, "reps": "5-15", "ex_max": 9, "rest_min": 1,
                     "wk_sets_max": 250, "days_cap": 6,
                     "intensity": "intensity techniques allowed: drop sets, rest-pause, myo-reps, supersets"},
}

_GOAL_TO_FOCUS = {
    "Build muscle": "Hypertrophy", "Mild muscle gain": "Hypertrophy",
    "Lose weight": "Fat loss", "Mild weight loss": "Fat loss",
    "Maintain weight": "General fitness",
}

# equipment name -> substrings in an exercise that IMPLY that equipment
_EQUIP_KEYWORDS = {
    "Dumbbells": ("dumbbell", "db ", " db"),
    "Barbell": ("barbell", "deadlift", "back squat", "front squat", "romanian deadlift",
                "rdl", "bench press", "overhead press", "ohp", "power clean", "hip thrust",
                "pendlay", "clean and", "snatch"),
    "Pull-up bar": ("pull-up", "pull up", "pullup", "chin-up", "chin up", "chinup",
                    "hanging leg", "hanging knee", "toes to bar", "toes-to-bar"),
    "Resistance bands": ("band pull-apart", "banded", "resistance band", "band "),
    "Kettlebell": ("kettlebell", "goblet", "kb swing", "turkish get"),
    "Bench": ("bench press", "incline bench", "decline bench", "dumbbell bench", "cable fly"),
    "Cable machine": ("cable", "lat pulldown", "pulldown", "pushdown", "face pull",
                      "rope ", "pec deck", "cable fly", "cable crunch"),
    "Leg press": ("leg press", "leg curl", "leg extension", "hack squat"),
    "Squat rack": ("squat rack", "power rack", "smith machine"),
    "Treadmill/Cardio machine": ("treadmill", "elliptical", "rowing machine", "rower",
                                 "stationary bike", "stairmaster", "assault bike"),
    "Backpack (loadable)": ("backpack", "loaded bag", "book bag"),
}
_BW_OK = (
    "push-up", "pushup", "push up", "squat", "lunge", "plank", "mountain climber",
    "burpee", "glute bridge", "superman", "wall sit", "jump", "run", "walk", "sprint",
    "sit-up", "situp", "crunch", "bench dip", "chair dip", "pike", "handstand", "hollow",
    "bird dog", "dead bug", "calf raise", "bear crawl", "pistol", "nordic", "good morning",
    "flutter", "bicycle", "russian twist", "inchworm", "skater", "high knee",
    "jumping jack", "step-up", "step up", "broad jump", "tuck", "hip hinge", "wall walk",
    "doorway row", "towel", "table row", "reverse snow angel", "superman hold",
)


def _norm_choice(value, allowed: list[str], default: str) -> str:
    v = str(value or "").strip().lower()
    for a in allowed:
        if a.lower() == v or a.lower().split()[0] == v:
            return a
    return default


def _infer_experience(user: dict) -> str:
    act = str(user.get("activity_level", "")).lower()
    if "extremely active" in act or "very active" in act or "physical job" in act:
        return "Advanced"
    if "moderately active" in act:
        return "Intermediate"
    return "Beginner"


def workout_profile(user: dict) -> dict:
    """Build the profile the dynamic generator needs from a DB user dict + any
    optional workout fields the Fitness page collected."""
    gym = bool(user.get("gym_access"))
    exp = _norm_choice(user.get("experience_level"), EXPERIENCE_LEVELS,
                       _infer_experience(user))
    loc = _norm_choice(user.get("workout_location"), WORKOUT_LOCATIONS,
                       "Gym" if gym else "Home")
    focus = _norm_choice(user.get("workout_focus"), WORKOUT_FOCI,
                         _GOAL_TO_FOCUS.get(user.get("goal", ""), "General fitness"))
    equip = user.get("equipment")
    if not equip:
        equip = (["Barbell", "Dumbbells", "Bench", "Cable machine", "Pull-up bar",
                  "Leg press", "Squat rack"] if loc in ("Gym", "Both")
                 else ["Resistance bands", "Backpack (loadable)", "Pull-up bar"])
    equip = [e for e in equip if e in EQUIPMENT_OPTIONS] or ["None (bodyweight only)"]
    try:
        days = int(user.get("workout_days") or user.get("days_per_week") or 4)
    except (TypeError, ValueError):
        days = 4
    days = max(3, min(days, _EXP_VOLUME[exp]["days_cap"]))
    limits = str(user.get("injuries") or user.get("limitations") or "").strip() or "none"
    return {"experience": exp, "location": loc, "goal_focus": focus,
            "equipment": equip, "days_per_week": days, "limitations": limits,
            "goal": user.get("goal", "Maintain weight")}


_WORKOUT_SCHEMA = (
    '{"location":"home|gym|both","goal_focus":"hypertrophy|strength|endurance|fat loss",'
    '"experience_level":"beginner|intermediate|advanced","week":{'
    '"monday":{"focus":"Push","exercises":["Bench press - 4 x 6-8","..."],"sets":4,"reps":"6-8"},'
    '"tuesday":{"focus":"Rest","exercises":[]}, "...":"through sunday"}}'
)


def _workout_prompt(p: dict) -> str:
    env = _EXP_VOLUME[p["experience"]]
    bodyweight = p["equipment"] == ["None (bodyweight only)"]
    return (
        "You are a certified strength & conditioning coach. Based on this profile, "
        "generate a personalised weekly workout routine.\n\n"
        "User Profile:\n"
        f"- Experience: {p['experience']}\n"
        f"- Goal: {p['goal_focus']} (overall goal: {p['goal']})\n"
        f"- Training location: {p['location']}\n"
        f"- Available equipment: {', '.join(p['equipment'])}\n"
        f"- Time available: {p['days_per_week']} days/week\n"
        f"- Injuries / limitations: {p['limitations']}\n\n"
        f"Generate a 7-day split with EXACTLY {p['days_per_week']} training days and "
        f"{7 - p['days_per_week']} rest days.\n"
        "Rules:\n"
        f"- Use ONLY the available equipment. {'No equipment listed -> bodyweight / band / backpack only.' if bodyweight else 'Do NOT program lifts that need gear not in the list.'}\n"
        f"- Work AROUND the stated limitations ({p['limitations']}) - never program a movement that aggravates them.\n"
        f"- {p['experience']} volume/intensity: ~{env['sets']} sets, {env['reps']} reps, "
        f"<= {env['ex_max']} exercises per training day, "
        f">= {env['rest_min']} full rest days. {env['intensity']}.\n"
        "- Each exercise string ends with ' - <sets> x <reps>' (e.g. 'Goblet squat - 3 x 10').\n"
        "- Rest days: \"exercises\": [] and a short \"focus\" like \"Rest\" or \"Light walk\".\n"
        "- 'week' keys are monday..sunday (lowercase).\n"
        "Return ONLY minified JSON, no markdown:\n" + _WORKOUT_SCHEMA
    )


_DOW = {"monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
        "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
        "sunday": "Sunday"}
import re as _re

_SETS_RE = _re.compile(r"(\d+)\s*[x×]\s*\d", _re.I)


def _fmt_exercise(ex, day_sets, day_reps) -> str:
    if isinstance(ex, dict):
        name = str(ex.get("name") or ex.get("exercise") or "").strip()
        s = ex.get("sets") or day_sets
        r = ex.get("reps") or day_reps
        if name and s and r:
            return f"{name} - {s} x {r}"
        return name or "-"
    return str(ex).strip()


def _normalise_week(raw: dict, p: dict) -> dict:
    week = {}
    src = raw if isinstance(raw, dict) else {}
    for lc, title in _DOW.items():
        d = src.get(lc) or src.get(title) or src.get(title.lower()) or {}
        if not isinstance(d, dict):
            d = {}
        day_sets = d.get("sets")
        day_reps = d.get("reps")
        ex_in = d.get("exercises") or d.get("workout") or []
        exercises = [s for s in (_fmt_exercise(e, day_sets, day_reps) for e in ex_in)
                     if s and s != "-"]
        focus = str(d.get("focus") or d.get("name") or
                    ("Rest" if not exercises else "Training")).strip()
        week[title] = {"focus": focus, "exercises": exercises,
                       "sets": day_sets if isinstance(day_sets, int) else None,
                       "reps": str(day_reps) if day_reps else None}
    return week


def _equipment_violations(week: dict, p: dict) -> int:
    have = set(p["equipment"])
    forbidden = {}
    for name, kws in _EQUIP_KEYWORDS.items():
        if name not in have:
            forbidden[name] = kws
    have_kws = tuple(kw for name in have for kw in _EQUIP_KEYWORDS.get(name, ()))
    hits = 0
    for day in week.values():
        for ex in day["exercises"]:
            t = ex.lower()
            if any(b in t for b in _BW_OK) or any(k in t for k in have_kws):
                continue
            if any(any(k in t for k in kws) for kws in forbidden.values()):
                hits += 1
    return hits


def _validate_workout(data: dict, p: dict) -> tuple[bool, str]:
    if not isinstance(data, dict) or not isinstance(data.get("week"), dict):
        return False, "no 'week' object"
    week = _normalise_week(data["week"], p)
    training = [d for d, v in week.items() if v["exercises"]]
    rest = 7 - len(training)
    env = _EXP_VOLUME[p["experience"]]
    if not (max(2, p["days_per_week"] - 1) <= len(training) <= p["days_per_week"] + 1):
        return False, f"{len(training)} training days, asked for {p['days_per_week']}"
    if rest < env["rest_min"] - 1:
        return False, f"only {rest} rest days ({p['experience']} needs >= {env['rest_min']})"
    for d in training:
        n = len(week[d]["exercises"])
        if not (2 <= n <= env["ex_max"] + 3):
            return False, f"{d}: {n} exercises (out of range for {p['experience']})"
    weekly_sets = 0
    for d in training:
        for ex in week[d]["exercises"]:
            m = _SETS_RE.search(ex)
            weekly_sets += int(m.group(1)) if m else 3
    if weekly_sets > env["wk_sets_max"] * 1.15:
        return False, f"~{weekly_sets} weekly sets exceeds the {p['experience']} ceiling"
    viol = _equipment_violations(week, p)
    bodyweight = p["equipment"] == ["None (bodyweight only)"]
    if viol > (0 if bodyweight else max(1, len(training) // 2)):
        return False, f"{viol} exercises need equipment the user does not have"
    return True, "ok"


def _cache_key_workout(p: dict) -> str:
    import hashlib

    raw = "|".join(str(p[k]) for k in
                   ("experience", "location", "goal_focus", "days_per_week",
                    "limitations", "goal")) + "|" + ",".join(sorted(p["equipment"]))
    return "workout:" + hashlib.sha1(raw.encode()).hexdigest()[:20]


def _static_workout(p: dict) -> dict:
    w = generate_workout(p["goal"], p["location"] in ("Gym", "Both"), p["days_per_week"])
    w["location"] = p["location"]
    w["goal_focus"] = p["goal_focus"]
    w["experience_level"] = p["experience"]
    w["source"] = "static"
    return w


def generate_workout_dynamic(profile: dict, use_gemini: bool = True,
                             timeout: float | None = None) -> dict:
    """Gemini-generated weekly routine for `profile`, validated against a strict
    schema + the user's equipment + an experience-appropriate volume envelope.
    Any failure (unavailable / timeout / bad JSON / invalid) -> static fallback.
    Identical profiles are served from a 7-day SQLite cache (no repeat API call)."""
    p = dict(profile)
    p.setdefault("goal", "Maintain weight")
    ckey = _cache_key_workout(p)

    try:
        import database as _db

        hit = _db.get_cached_price(ckey, max_age_days=7)
        if hit:
            hit["source"] = hit.get("source") or "gemini-cached"
            return hit
    except Exception:
        pass

    if use_gemini:
        import gemini_client

        if gemini_client.is_available():
            to = float(timeout or os.environ.get("GEMINI_TIMEOUT", "25"))
            prompt = _workout_prompt(p)
            # One retry on a transient timeout/deadline only (this environment sees
            # occasional transient network blips - see skin_ai/gemini_vision.py for
            # the same pattern). A validation failure is not retried - a 2nd call
            # with the same prompt is unlikely to fix a structurally bad response.
            for attempt in (1, 2):
                try:
                    raw = gemini_client.generate(prompt, temperature=0.5, timeout=to)
                    data = gemini_client.extract_json(raw)
                    ok, reason = _validate_workout(data if isinstance(data, dict) else {}, p)
                    if not ok:
                        break
                    out = {
                        "location": p["location"], "goal_focus": p["goal_focus"],
                        "experience_level": p["experience"], "source": "gemini",
                        "days_per_week": p["days_per_week"],
                        "week": _normalise_week(data["week"], p),
                        "note": (f"AI-personalised for {p['experience'].lower()} · "
                                 f"{p['goal_focus'].lower()} · {p['days_per_week']} days/week · "
                                 f"equipment: {', '.join(p['equipment'])}"
                                 + (f" · working around: {p['limitations']}"
                                    if p["limitations"] != "none" else "")),
                    }
                    try:
                        import database as _db

                        _db.set_cached_price(ckey, "workout", out)
                    except Exception:
                        pass
                    return out
                except Exception as exc:
                    if attempt == 1 and gemini_client._is_timeout(exc):
                        continue
                    break

    fb = _static_workout(p)
    fb["note"] = ("Static routine (AI unavailable or its plan failed validation) - "
                  f"{p['experience'].lower()} {p['goal_focus'].lower()}, "
                  f"{p['days_per_week']} days/week.")
    return fb


def full_assessment(user: dict, use_ai: bool = False, use_csv: bool = False,
                    meal_mode: str | None = None, dynamic_workout: bool = True) -> dict:
    """Convenience wrapper used by app.py. `user` is a DB user dict.

    meal_mode:
      "csv"      -> prices from meals_prices.csv; Gemini arranges foods (default)
      "builtin"  -> static built-in plan (no CSV, no API)
    dynamic_workout:
      True  -> Gemini-personalised weekly routine (validated), static fallback
      False -> static routine only
    (falls back safely; never raises)
    """
    bmr = calculate_bmr(user["weight_kg"], user["height_cm"], user["age"], user["gender"])
    tdee = calculate_tdee(bmr, user.get("activity_level"))
    target = calculate_calorie_target(tdee, user.get("goal"))
    macros = macro_split(target, user.get("goal"))

    period = user.get("budget_period", "daily")
    budget = float(user.get("budget_pkr") or 0)
    daily_budget = budget / 30.0 if period == "monthly" else budget
    preference = user.get("diet_preference", "No preference")

    mode = meal_mode or ("builtin" if (not use_csv and not use_ai) else "csv")
    goal = user.get("goal")
    budget_err = validate_budget(daily_budget)
    if budget_err:
        meal_plan = {"error": budget_err, "days": [], "source": "budget-error",
                     "weekly_cost": 0, "avg_daily_cost": 0, "notes": [budget_err],
                     "validation": {"daily_budget": round(daily_budget),
                                    "daily_cost": None, "within_budget": False,
                                    "feasible": False}}
    elif mode == "builtin":
        meal_plan = build_meal_plan(target, daily_budget)
    else:
        meal_plan = build_meal_plan_csv(target, daily_budget, goal, preference,
                                        protein_target_g=macros["protein_g"])
    prof = workout_profile(user)
    workout = (generate_workout_dynamic(prof, use_gemini=True) if dynamic_workout
               else _static_workout(prof))
    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "calorie_target": target,
        "macros": macros,
        "daily_budget": round(daily_budget),
        "meal_plan": meal_plan,
        "workout": workout,
    }
