"""AI Nutritionist backed by the Google Gemini API.

Uses gemini_client (google-generativeai SDK). If Gemini is unavailable
(no API key / SDK / network) a keyword-based fallback responder is used so
the chat still works offline.
"""
from __future__ import annotations

import gemini_client

SYSTEM_PROMPT = (
    "You are HealthHub's AI Nutritionist, an expert dietitian who specialises in "
    "Pakistani cuisine and everyday nutrition for people living in Pakistan.\n"
    "Guidelines:\n"
    "- Give practical, budget-aware advice using foods commonly available in "
    "Pakistan (daal, roti, chawal, chana, seasonal sabzi/fruit, dahi, eggs, "
    "chicken, beef, mutton, fish).\n"
    "- When you mention foods, supplements or products, include an approximate "
    "real price in PKR (local kiryana shops, dhabas, or Daraz for packaged items).\n"
    "- Prefer home-cooked options; call out cheaper swaps (daal/chana/eggs vs meat).\n"
    "- Keep answers concise and structured with short bullet points.\n"
    "- Use metric units and mention rough calorie/protein figures where useful.\n"
    "- You are not a doctor; advise seeing a professional for medical conditions, "
    "pregnancy, or eating disorders.\n"
)


def build_system_prompt(profile: dict | None = None) -> str:
    prompt = SYSTEM_PROMPT
    if profile:
        parts = []
        for key in ("age", "gender", "goal"):
            if profile.get(key):
                parts.append(f"{key}: {profile[key]}")
        if profile.get("height_cm") and profile.get("weight_kg"):
            parts.append(f"height: {profile['height_cm']} cm, weight: {profile['weight_kg']} kg")
        if profile.get("budget_pkr"):
            parts.append(
                f"food budget: PKR {profile['budget_pkr']:.0f} "
                f"({profile.get('budget_period', 'daily')})"
            )
        if profile.get("calorie_target"):
            parts.append(f"daily calorie target: {profile['calorie_target']} kcal")
        if parts:
            prompt += "\nCurrent user profile -> " + "; ".join(parts) + ".\n"
    return prompt


def is_available() -> bool:
    return gemini_client.is_available()


def status() -> str:
    return gemini_client.status()


def _to_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def chat(messages: list[dict], profile: dict | None = None) -> str:
    """Blocking single response."""
    try:
        out = "".join(gemini_client.chat_stream(_to_messages(messages),
                                                build_system_prompt(profile)))
        return out.strip() or _fallback_response(messages)
    except Exception:
        return _fallback_response(messages)


def chat_stream(messages: list[dict], profile: dict | None = None):
    """Yield response chunks; falls back to a single offline chunk on error."""
    try:
        produced = False
        for chunk in gemini_client.chat_stream(_to_messages(messages),
                                               build_system_prompt(profile)):
            produced = True
            yield chunk
        if not produced:
            yield _fallback_response(messages)
    except Exception:
        yield _fallback_response(messages)


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #
_FALLBACK_RULES = [
    (("protein", "muscle", "gain", "bulk"),
     "**Cheap protein sources in Pakistan**\n"
     "- Eggs - ~PKR 25-30 each (12-13 g protein per 2)\n"
     "- Daal chana / masoor / moong - ~PKR 300-400/kg, ~18 g protein per cooked cup\n"
     "- Chicken (boneless) - ~PKR 700-900/kg, ~30 g per 100 g\n"
     "- Dahi / yogurt - ~PKR 220/kg, ~9 g per cup\n"
     "- Whey protein (local) - ~PKR 6,000-9,000/kg, 24 g per scoop\n"
     "Aim for ~1.6-2.2 g protein per kg bodyweight and split it across 3-4 meals."),
    (("lose", "weight", "fat", "cut", "slim", "belly"),
     "**Fat-loss basics (Pakistani plate)**\n"
     "- Keep a ~300-500 kcal daily deficit; don't crash diet.\n"
     "- Half plate sabzi/salad, quarter protein (chicken/daal/eggs), quarter roti/rice.\n"
     "- Swap 2 rotis -> 1 roti + salad; use 1 tsp oil per dish.\n"
     "- Cut sugary chai (use half sugar or none), soft drinks, bakery items, fried snacks.\n"
     "- Walk 8-10k steps/day. Weigh yourself weekly, same time."),
    (("diabet", "sugar", "blood sugar"),
     "**Blood-sugar friendly swaps**\n"
     "- Prefer chana, daal, brown/basmati rice in small portions, whole-wheat roti.\n"
     "- Pair carbs with protein + salad to blunt spikes.\n"
     "- Avoid white bread, sugary chai, mithai, packaged juice.\n"
     "- Karela, methi and cinnamon may help slightly - not a substitute for medication.\n"
     "Please follow your doctor's advice and monitor readings."),
    (("budget", "cheap", "sasta", "expensive", "afford"),
     "**Eating well on a tight budget (~PKR 300-500/day)**\n"
     "- Daal chawal - ~PKR 150/meal\n"
     "- Chana pulao - ~PKR 180\n"
     "- Egg curry + roti - ~PKR 160\n"
     "- Seasonal sabzi + roti - ~PKR 150\n"
     "- Dahi + banana snack - ~PKR 130\n"
     "Buy pulses and rice in bulk, cook in batches, keep eggs and seasonal veg as staples."),
    (("skin", "acne", "pimple", "glow"),
     "**Diet for clearer skin**\n"
     "- Drink 2.5-3 L water/day.\n"
     "- More: seasonal fruit, leafy greens, tomatoes, carrots, walnuts/almonds, dahi.\n"
     "- Less: deep-fried food, excess dairy, sugary chai and bakery items.\n"
     "- Consider omega-3 (fish 2x/week or fish-oil ~PKR 1,500) and zinc ~PKR 400.\n"
     "Give any change 6-8 weeks before judging results."),
    (("meal plan", "diet plan", "routine", "kya khaon", "what should i eat"),
     "**Simple 1-day template (~2,000 kcal)**\n"
     "- Breakfast: 3-egg omelette + 2 roti + chai (no sugar) - ~PKR 150\n"
     "- Lunch: Chicken/daal + 1-2 roti + salad - ~PKR 250-400\n"
     "- Snack: Dahi + fruit or a handful of nuts - ~PKR 120\n"
     "- Dinner: Sabzi or grilled chicken + 1 roti - ~PKR 200-400\n"
     "Scale portions up or down to hit your calorie target."),
]

_FALLBACK_DEFAULT = (
    "I'm running in offline mode right now (Gemini API key not configured), so here's "
    "general guidance:\n\n"
    "- Base meals on daal/chana, whole-wheat roti, seasonal sabzi, dahi, eggs and some "
    "chicken or fish.\n"
    "- Match total calories to your goal (deficit to lose, small surplus to gain).\n"
    "- Get ~1.6-2 g protein per kg bodyweight, 25-35 g fibre, and 2.5-3 L water daily.\n"
    "- Limit sugary chai, soft drinks, fried snacks and bakery items.\n\n"
    "Set GEMINI_API_KEY (or add it to Streamlit secrets) for personalised answers."
)


def _fallback_response(messages: list[dict]) -> str:
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"].lower()
            break
    for keywords, answer in _FALLBACK_RULES:
        if any(k in last_user for k in keywords):
            return answer
    return _FALLBACK_DEFAULT
