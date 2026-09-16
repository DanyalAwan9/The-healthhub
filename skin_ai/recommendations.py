"""
recommendations.py - supplement de-duplication + overdose safety.

A raw skincare supplement list often stacks several products that share an
active ingredient (Zinc x3, Vitamin C x3, Calcium x2). Taken together that is a
real hazard:

  * Zinc  > ~40 mg/day  -> copper deficiency, immune suppression
  * Vitamin C megadoses -> GI upset, oxalate kidney stones
  * Calcium > ~1000 mg supplemental -> vascular / renal load, constipation

`deduplicate_supplements()` collapses the list to ONE product per ingredient
class, prefers all-in-one combos (Surbex-Z covers Zinc + C + B), caps the
routine at 2-3, drops products irrelevant to the routine (e.g. Alpha-Lipoic Acid
for a basic routine), and flags any dose over a conservative daily ceiling.

    from skin_ai.recommendations import deduplicate_supplements, context_from_conditions
    out = deduplicate_supplements(raw_list, context="basic")
    out["supplements"]   # <= 2-3 dicts, each + label / covers / note
    out["safety_note"]   # user-facing string
    out["flags"]         # [] when the combination is within limits
"""
from __future__ import annotations

import re

# ---- 1. INGREDIENT CLASS MAPPING -------------------------------------------
# exact product-name fragment -> the ingredient classes that product contains
PRODUCT_INGREDIENTS: dict[str, set[str]] = {
    "zincat":       {"Zinc"},
    "zinc plus":    {"Zinc"},
    "surbex-z":     {"Zinc", "Vitamin C", "B-Complex"},
    "surbex z":     {"Zinc", "Vitamin C", "B-Complex"},
    "nutra c plus": {"Vitamin C"},
    "nutra c":      {"Vitamin C"},
    "biotin plus":  {"B-Complex"},
    "cac-1000":     {"Calcium", "Vitamin C", "Vitamin D"},
    "cac 1000":     {"Calcium", "Vitamin C", "Vitamin D"},
    "bonex-d":      {"Calcium", "Vitamin D"},
    "bonex d":      {"Calcium", "Vitamin D"},
    "sensolin":     {"Alpha-Lipoic Acid"},
}

# keyword -> class, for any product not in the table above
INGREDIENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Zinc":              ("zinc",),
    "Vitamin C":         ("vitamin c", "vit c", "ascorb"),
    "B-Complex":         ("b-complex", "b complex", "b-comp", "biotin", "vitamin b", "b12", "b6"),
    "Calcium":           ("calcium", "cal-", "cac-", "cac "),
    "Vitamin D":         ("vitamin d", "vit d", "cholecalciferol", "d3 ", "-d3", "vit-d"),
    "Vitamin E":         ("vitamin e", "vit e", "tocopherol"),
    "Alpha-Lipoic Acid": ("alpha-lipoic", "alpha lipoic", "lipoic", "sensolin"),
    "Collagen":          ("collagen",),
    "Selenium":          ("selenium",),
    "Omega-3":           ("omega", "fish oil", "dha", "epa"),
    "Multivitamin":      ("multivitamin", "multi-vitamin", "once daily", "one daily", "centrum"),
}
# a broad multivitamin is treated as already covering these
MULTIVITAMIN_COVERS = {"Zinc", "Vitamin C", "B-Complex", "Vitamin D"}

# ---- 4. SAFETY --------------------------------------------------------------
# _TARGET  = the conservative daily amount we aim for (shown in the message).
# DAILY_LIMITS = the routine-TOTAL ceiling above which we hard-flag. A single
# well-formulated combo at its label dose (Surbex-Z ~22.5 mg zinc) sits between
# the two: fine on its own, flagged the moment a second source is stacked.
_TARGET: dict[str, float] = {"Zinc": 15.0, "Vitamin C": 500.0, "Calcium": 1000.0,
                             "Vitamin D": 2000.0}
DAILY_LIMITS: dict[str, float | None] = {
    "Zinc":              25.0,     # mg   (UL is 40; RDA ~11)
    "Vitamin C":         750.0,    # mg
    "Calcium":          1200.0,    # mg
    "Vitamin D":        4000.0,    # IU   (UL)
    "Vitamin E":         300.0,    # mg
    "Alpha-Lipoic Acid": 600.0,    # mg
    "Selenium":          200.0,    # mcg
    "B-Complex":         None,     # water-soluble, no single number
    "Collagen":          None,
    "Omega-3":           None,
    "Multivitamin":      None,
}
_UNIT = {"Vitamin D": "IU", "Selenium": "mcg"}          # everything else: mg

# well-known combo doses (combos rarely print each amount on the carton)
COMBO_DOSES: dict[str, dict[str, float]] = {
    "surbex-z":  {"Zinc": 22.5, "Vitamin C": 500.0},
    "surbex z":  {"Zinc": 22.5, "Vitamin C": 500.0},
    "cac-1000":  {"Calcium": 1000.0, "Vitamin C": 500.0, "Vitamin D": 400.0},
    "cac 1000":  {"Calcium": 1000.0, "Vitamin C": 500.0, "Vitamin D": 400.0},
    "bonex-d":   {"Calcium": 600.0, "Vitamin D": 400.0},
    "bonex d":   {"Calcium": 600.0, "Vitamin D": 400.0},
}

# which classes matter for which routine  (irrelevant classes are dropped).
# Vitamin D / Calcium are added only when `detected` says a deficiency was seen.
_RELEVANT: dict[str, set[str]] = {
    "basic":   {"Zinc", "Vitamin C", "B-Complex"},
    "acne":    {"Zinc", "Vitamin C"},                      # 1-2 targeted, zinc OR C
    "pigment": {"Vitamin C", "Vitamin E"},
    "general": {"Zinc", "Vitamin C", "B-Complex", "Vitamin D", "Calcium"},
}
_CAP = {"basic": 2, "acne": 2, "pigment": 2, "general": 3}

_AMT = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|iu|g)\b", re.I)


def _text(s: dict) -> str:
    return " ".join(str(s.get(k, "")) for k in ("name", "product", "brand", "dosage")).lower()


def classify(*texts: str) -> set[str]:
    """Ingredient classes a product name / dosage string implies."""
    s = " ".join(t for t in texts if t).lower()
    classes: set[str] = set()
    for frag, cls in PRODUCT_INGREDIENTS.items():
        if frag in s:
            classes |= cls
    for cls, kws in INGREDIENT_KEYWORDS.items():
        if any(k in s for k in kws):
            classes.add(cls)
    if "Multivitamin" in classes:
        classes |= MULTIVITAMIN_COVERS
    return classes


def _dose_for(text: str, cls: str) -> float | None:
    """Best-effort mg (or IU) of `cls` in a product, for the overdose check."""
    low = text.lower()
    for frag, doses in COMBO_DOSES.items():
        if frag in low and cls in doses:
            return doses[cls]
    m = _AMT.search(text or "")                 # single-ingredient: first amount wins
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    if unit == "g":
        val *= 1000.0
    elif unit in ("mcg", "µg", "ug") and _UNIT.get(cls) != "mcg":
        val /= 1000.0
    return val


def context_from_conditions(conditions) -> str:
    """Map detected skin conditions -> a routine context for _RELEVANT / _CAP."""
    t = " ".join(str(c) for c in (conditions or [])).lower()
    if any(k in t for k in ("acne", "blemish", "breakout", "excess oil", "oili")):
        return "acne"
    if any(k in t for k in ("pigment", "dark spot", "melasma", "uneven tone")):
        return "pigment"
    return "basic"


# ---- 2 + 3 + 5: DEDUPLICATION / SMART SELECTION / OUTPUT -----------------
def deduplicate_supplements(supplements, context: str = "basic",
                            detected: set[str] | None = None) -> dict:
    """Collapse a raw supplement list to one product per ingredient class.

    supplements : list of dicts (need at least 'name'; 'dosage'/'product' help)
    context     : 'basic' | 'acne' | 'pigment' | 'general'
    detected    : optional hints, e.g. {'low_vitamin_d'} -> allow one extra pick
    """
    relevant = set(_RELEVANT.get(context, _RELEVANT["basic"]))
    cap = _CAP.get(context, 2)
    if detected and "low_vitamin_d" in detected:
        relevant |= {"Vitamin D", "Calcium"}
        cap += 1

    enriched = []
    for s in supplements or []:
        cls = classify(_text(s))
        rel = cls & relevant
        enriched.append({"raw": s, "classes": cls, "rel": rel,
                         "combo": len(cls) > 1,
                         "suggested": bool(s.get("suggested") or s.get("relevant"))})

    dropped = [{"name": e["raw"].get("name", "?"),
               "reason": "not relevant to the %s routine" % context}
              for e in enriched if not e["rel"]]

    cands = [e for e in enriched if e["rel"]]
    covered: set[str] = set()
    picks: list[dict] = []
    while cands and len(picks) < cap and not covered >= relevant:
        # re-rank each round: most NEW classes, then least overlap with what we
        # already have (don't stack an ingredient), then combo / suggested / price,
        # then (acne) prefer the zinc option.
        def rank(e):
            new = len(e["rel"] - covered)
            overlap = len(e["classes"] & covered)
            return (-new, overlap, -int(e["combo"]), -int(e["suggested"]),
                    -int(context == "acne" and "Zinc" in e["rel"]),
                    float(e["raw"].get("price_pkr", 1e9) or 1e9))

        cands.sort(key=rank)
        e = cands.pop(0)
        if not (e["rel"] - covered):
            dropped.append({"name": e["raw"].get("name", "?"),
                            "reason": "duplicate " + "/".join(sorted(e["rel"]))})
            continue
        picks.append(e)
        covered |= e["rel"]
        # acne: "Zinc OR Vitamin C, not both" - once one is covered, drop the other
        if context == "acne" and covered & {"Zinc", "Vitamin C"}:
            relevant -= {"Zinc", "Vitamin C"}
            for c in cands:
                c["rel"] &= relevant
            cands = [c for c in cands if c["rel"]]
    for e in cands:                                        # everything left over
        dropped.append({"name": e["raw"].get("name", "?"),
                        "reason": "duplicate " + "/".join(sorted(e["rel"]))
                        if e["rel"] & covered else "capped at %d" % cap})

    # ---- 4. safety: sum each class across the FINAL picks --------------
    totals: dict[str, float] = {}
    for e in picks:
        txt = _text(e["raw"])
        for cls in e["rel"]:
            d = _dose_for(txt, cls)
            if d is not None:
                totals[cls] = totals.get(cls, 0.0) + d

    flags: list[str] = []
    for cls, total in sorted(totals.items()):
        lim = DAILY_LIMITS.get(cls)
        if lim is not None and total > lim + 1e-6:
            u = _UNIT.get(cls, "mg")
            tgt = _TARGET.get(cls, lim)
            flags.append(f"{cls} ~{total:g} {u}/day - over the {lim:g} {u} ceiling "
                         f"(target ~{tgt:g} {u}); drop one {cls} source")

    # ---- 5. output format --------------------------------------------
    out_supps = []
    for i, e in enumerate(picks):
        covers = sorted(e["rel"])
        note = ("covers " + ", ".join(covers) + " in one product" if len(covers) > 1
                else "provides " + covers[0])
        out_supps.append({**e["raw"], "label": "Primary" if i == 0 else "Optional",
                          "covers": covers, "note": note})

    if flags:
        safety = ("⚠️ " + "; ".join(flags)
                  + ". Take the lower-dose option or alternate days, and do NOT add "
                    "any other supplement, multivitamin, zinc or vitamin C on top.")
    elif out_supps:
        safety = ("This combination is within safe daily limits. Don't add other "
                  "supplements (multivitamins, zinc, vitamin C, calcium) alongside it.")
    else:
        safety = "No supplements needed for this routine - focus on diet, sleep and sunscreen."

    return {
        "supplements": out_supps,
        "primary": out_supps[0]["name"] if out_supps else None,
        "optional": [s["name"] for s in out_supps[1:]],
        "covered_classes": sorted(covered),
        "flags": flags,
        "safety_note": safety,
        "dropped": dropped,
    }


if __name__ == "__main__":  # demo with the exact list from the brief
    raw = [{"name": n} for n in
           ["Zincat", "Zinc Plus", "Surbex-Z", "Nutra C Plus", "Biotin Plus",
            "CaC-1000", "Bonex-D", "Sensolin"]]
    for ctx in ("basic", "acne"):
        r = deduplicate_supplements(raw, context=ctx)
        print(f"\n=== context: {ctx} ===")
        for s in r["supplements"]:
            print(f"  {s['label']:8} {s['name']:14} - {s['note']}")
        print("  dropped :", ", ".join(f"{d['name']} ({d['reason']})" for d in r["dropped"]))
        print("  flags   :", r["flags"] or "none")
        print("  safety  :", r["safety_note"])
