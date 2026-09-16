"""Quick connectivity + feature check for the Gemini integration.

Usage:
    # reads key from env, or from .streamlit/secrets.toml as a fallback
    python check_gemini.py
"""
import os
import warnings

warnings.simplefilter("ignore")

# Fallback: load keys from .streamlit/secrets.toml if not already in the env.
try:
    import tomllib  # Python 3.11+

    with open(os.path.join(".streamlit", "secrets.toml"), "rb") as fh:
        data = tomllib.load(fh)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL",
              "SERPAPI_API_KEY", "SERP_API_KEY"):
        if data.get(k):
            os.environ.setdefault(k, str(data[k]))
except Exception:
    pass

import database as db
import gemini_client
import nutritionist
import fitness_engine as fe
import skin_analyzer

db.init_db()   # ensure price_cache / gemplan tables exist

print("status      :", gemini_client.status())
GEMINI_OK = gemini_client.is_available()
if not GEMINI_OK:
    print("No Gemini key - steps 1-2 will use offline fallbacks; CSV steps 3-4 still run.")

# Direct call first so any API error is shown instead of a silent fallback.
if GEMINI_OK:
    print("\n[0/4] Direct Gemini call")
    try:
        print("  ->", gemini_client.generate("Reply with the single word: OK"))
    except Exception as exc:
        print("  API ERROR:", type(exc).__name__, "-", str(exc).splitlines()[0][:160])
        print("  (steps 1-2 will fall back; CSV steps 3-4 do not need Gemini)")

print("\n[1/4] Nutritionist chat")
reply = nutritionist.chat(
    [{"role": "user", "content": "One cheap high-protein Pakistani breakfast under Rs 200?"}]
)
print(reply[:400])
if gemini_client.last_error():
    print("  (fell back - last error:", gemini_client.last_error(), ")")

print("\n[2/4] Meal plan - meals_prices.csv prices (1 day preview)")
plan = fe.build_meal_plan_csv(2000, 800, "Lose weight")
print("source:", plan["source"], "| avg/day PKR", plan["avg_daily_cost"])
print("note:", plan["notes"][0][:90])
for meal in plan["days"][0]["meals"]:
    print(f'  {meal["slot"]:<10} {meal["name"][:34]:<34} {meal["kcal"]:>4} kcal  PKR {meal["price"]}')
    print(f'             {meal.get("items", "")[:100]}')

print("\n[3/4] Skincare recommendations")
from PIL import Image
import product_search

_serp_mode = "search" if product_search.is_available() else "csv"
res = skin_analyzer.analyze(Image.new("RGB", (256, 256), (205, 150, 140)), mode=_serp_mode)
print(f"skin type: {res['skin_type']} | reco source: {res['reco_source']} (asked for {_serp_mode})")
groups = res["recommendations"].get("product_options", [])
for g in groups:
    print(f'  {g["category"]}: {len(g["options"])} options')
    for i, o in enumerate(g["options"][:4], 1):
        print(f'    Option {i}: {o["brand"]} {o["product"][:34]:<34} Rs {o["price_pkr"]}')
for i, p in enumerate(res["recommendations"].get("products", [])[:6], 1):
    if "why" in p:  # search mode
        tag = "verified" if p.get("verified") else f"{p.get('n_sources', 1)} src"
        print(f'  Product {i}: {p["name"][:44]:<44} Rs {p.get("price_pkr")}  [{tag}]')

print("\n[4/4] Price data status")
import price_data
import product_search

s = price_data.status()
print(f'  meals_prices.csv    : {s["meals_rows"]} rows'
      + (f'  ERROR: {s["meals_error"]}' if s["meals_error"] else ""))
print(f'  skincare_brands.csv : {s["skincare_rows"]} rows, {len(s["brands"])} brands'
      + (f'  ERROR: {s["skincare_error"]}' if s["skincare_error"] else ""))
print("  meal price lines   :", "; ".join(price_data.meal_price_lines()[:4]), "...")
print(f'  SerpAPI (skincare) : {product_search.status()}')
if product_search.is_available():
    r = product_search.search_products("CeraVe cleanser price Pakistan Daraz")
    v = product_search.verify_products(r.get("products", []))
    print(f'    test query -> {len(r.get("products", []))} raw hits, {len(v)} products, '
          f'{sum(1 for x in v if x["verified"])} seen on 2+ sites'
          + (f'  ERROR: {r["error"]}' if r.get("error") else ""))
    for x in v[:4]:
        print(f'      {"✅" if x["verified"] else "· "} {x["name"][:50]:<50} '
              f'Rs {x["price_pkr"]}  ({x["n_sources"]} src)')

print("\nDone.")
