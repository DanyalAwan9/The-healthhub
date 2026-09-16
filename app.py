"""HealthHub - Comprehensive Health & Fitness AI Assistant.

Run locally:  streamlit run app.py
AI features (nutritionist chat, AI meal plan, AI skincare) use the Google Gemini
API. Set GEMINI_API_KEY as an env var or in .streamlit/secrets.toml. Without a
key the app still runs using built-in offline logic.
"""
from __future__ import annotations

import io
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

# Make Streamlit secrets visible to plain os.environ so gemini_client picks them up.
for _key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL",
             "SERPAPI_API_KEY", "SERP_API_KEY"):
    try:
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        pass

import database as db
import fitness_engine as fe
import gemini_client
import nutritionist
import price_data
import product_search
import progress_tracker as pt
import skin_analyzer                       # kept for the gentle product-recommendation engine
from skin_ai.inference import SkinAnalyzer  # preprocessing + CNN/CV concerns + Gemini skin type
from skin_ai.recommendations import context_from_conditions, deduplicate_supplements

st.set_page_config(page_title="HealthHub", page_icon="💪", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Global look & feel  (Inter type, card system, 8px spacing, teal accent,
# light/dark-aware via Streamlit theme vars + color-mix).  One injection, all
# pages. Users still toggle Light/Dark from ⋮ → Settings.
# --------------------------------------------------------------------------- #
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --hh-accent:  #10b981;
  --hh-accent-2:#0d9488;
  --hh-warn:    #f59e0b;
  --hh-radius:  12px;
  --hh-radius-sm: 10px;
  --hh-gap:     8px;
  /* static fallbacks; upgraded to theme-aware values just below */
  --hh-border:  rgba(31,41,55,0.10);
  --hh-soft:    rgba(31,41,55,0.62);
  --hh-shadow:  0 1px 2px rgba(15,23,42,0.05), 0 8px 24px rgba(15,23,42,0.07);
  --hh-shadow-sm: 0 1px 2px rgba(15,23,42,0.06);
}
/* Follow whatever theme (light OR dark) Streamlit is currently showing. */
@supports (color: color-mix(in srgb, white, black)) {
  :root {
    --hh-border: color-mix(in srgb, var(--text-color, #1f2937) 14%, transparent);
    --hh-soft:   color-mix(in srgb, var(--text-color, #1f2937) 66%, transparent);
  }
}

/* ---- Typography ---------------------------------------------------------- */
html { font-size: 16px; }
html, body, .stApp, button, input, textarea, select,
[class*="css"], [data-testid="stMarkdownContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.stApp h1 { font-weight: 800; letter-spacing: -0.022em; font-size: 2rem; line-height: 1.2; margin-bottom: .35rem; }
.stApp h2 { font-weight: 700; letter-spacing: -0.014em; font-size: 1.4rem; margin-top: 2rem; }
.stApp h3 { font-weight: 600; letter-spacing: -0.01em; font-size: 1.12rem; margin-top: 1.25rem; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 1rem; line-height: 1.65; }
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--hh-soft) !important; }
a { color: var(--hh-accent-2); text-decoration: none; font-weight: 500; }
a:hover { text-decoration: underline; }

/* ---- Layout / canvas -------------------------------------------------------- */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
footer { display: none; }
.stApp .block-container {
  max-width: 1200px;
  padding-top: 2.4rem;
  padding-bottom: 4rem;
  animation: hhFade .34s ease both;
}
@keyframes hhFade { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: none; } }

/* ---- Sidebar -------------------------------------------------------------- */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--hh-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.hh-brand {
  display: flex; align-items: center; gap: 10px;
  font-weight: 800; font-size: 1.3rem; letter-spacing: -0.02em;
  margin: 0 0 1.1rem 2px;
}
.hh-brand .hh-logo {
  width: 34px; height: 34px; border-radius: 10px;
  display: grid; place-items: center; font-size: 18px;
  background: linear-gradient(135deg, var(--hh-accent-2), var(--hh-accent));
  box-shadow: 0 4px 12px rgba(13,148,136,0.35);
}
/* nav radio -> pill list */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 3px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
  padding: 8px 12px; margin: 0; border-radius: var(--hh-radius-sm);
  transition: background .18s ease, color .18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: var(--hh-border); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(135deg, var(--hh-accent-2), var(--hh-accent));
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) * { color: #fff !important; font-weight: 600; }

/* ---- Cards: bordered containers, forms, expanders, metrics -------------- */
/* stVerticalBlockBorderWrapper is emitted only for st.container(border=True). */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--hh-radius);
  border: 1px solid var(--hh-border) !important;
  box-shadow: var(--hh-shadow);
  background: var(--background-color, #fff);
}
[data-testid="stForm"] {
  border-radius: 14px;
  border: 1px solid var(--hh-border);
  padding: 1.5rem 1.5rem .5rem;
  box-shadow: var(--hh-shadow);
  background: var(--background-color, #fff);
}
[data-testid="stExpander"] {
  border-radius: var(--hh-radius);
  border: 1px solid var(--hh-border);
  box-shadow: var(--hh-shadow-sm);
  overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: var(--hh-accent-2); }
[data-testid="stMetric"] {
  background: var(--secondary-background-color, #f3f4f6);
  border: 1px solid var(--hh-border);
  border-radius: var(--hh-radius);
  padding: 1rem 1.1rem;
}
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.01em; }
[data-testid="stMetricLabel"] { color: var(--hh-soft) !important; }

/* ---- Buttons ----------------------------------------------------------- */
.stButton > button, .stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
  border-radius: var(--hh-radius-sm);
  font-weight: 600;
  padding: 0.5rem 1.15rem;
  border: 1px solid var(--hh-border);
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease, background .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px);
  box-shadow: var(--hh-shadow);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--hh-accent-2), var(--hh-accent));
  color: #fff; border: none;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  filter: brightness(1.05); box-shadow: 0 8px 22px rgba(13,148,136,0.38);
}

/* ---- Inputs ---------------------------------------------------------------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stDateInput input {
  border-radius: var(--hh-radius-sm) !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  border-color: var(--hh-accent-2) !important;
  box-shadow: 0 0 0 3px rgba(13,148,136,0.16) !important;
}
[data-testid="stFileUploaderDropzone"] {
  border-radius: var(--hh-radius); border-style: dashed;
}

/* ---- Chat ------------------------------------------------------------------ */
[data-testid="stChatMessage"] {
  border-radius: 16px;
  padding: 0.85rem 1.1rem;
  margin-bottom: 0.55rem;
  border: 1px solid var(--hh-border);
  background: var(--background-color, #fff);
  box-shadow: var(--hh-shadow-sm);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, rgba(13,148,136,0.12), rgba(16,185,129,0.06));
  border-color: rgba(13,148,136,0.28);
}
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stChatInput"] textarea:focus { box-shadow: none !important; }

/* ---- Tabs -------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--hh-border); }
.stTabs [data-baseweb="tab"] { border-radius: 9px 9px 0 0; padding: 8px 15px; font-weight: 500; }
.stTabs [data-baseweb="tab"]:hover { background: var(--hh-border); }
.stTabs [aria-selected="true"] { color: var(--hh-accent-2); font-weight: 600; }

/* ---- Tables / dataframes -------------------------------------------------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border-radius: var(--hh-radius); overflow: hidden; border: 1px solid var(--hh-border);
}
[data-testid="stTable"] thead tr th { background: var(--secondary-background-color, #f3f4f6); font-weight: 600; }

/* ---- Alerts / progress / spinner --------------------------------------- */
[data-testid="stAlert"] { border-radius: var(--hh-radius); border: 1px solid var(--hh-border); }
.stProgress > div > div > div { background: var(--hh-border); border-radius: 999px; }
.stProgress > div > div > div > div {
  background-image: linear-gradient(90deg, var(--hh-accent-2), var(--hh-accent));
  border-radius: 999px;
}
[data-testid="stSpinner"] > div { border-top-color: var(--hh-accent-2) !important; }

/* ---- Divider spacing (8px grid) --------------------------------------- */
hr { margin: 1.75rem 0; border-color: var(--hh-border); }

/* ---- Mobile ---------------------------------------------------------------- */
@media (max-width: 640px) {
  .stApp .block-container { padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem; }
  .stApp h1 { font-size: 1.6rem; }
}
</style>
"""


def _inject_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


_inject_theme()
db.init_db()

GENDERS = ["Male", "Female"]
BUDGET_PERIODS = ["daily", "monthly"]


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def current_user() -> dict | None:
    name = st.session_state.get("user_name")
    return db.get_user_by_name(name) if name else None


def require_user() -> dict | None:
    user = current_user()
    if not user:
        st.warning("Create or select a profile on the **Home / Profile** page first.")
        st.stop()
    return user


def _render_sources(sources, label="🔗 Web sources for these prices"):
    if not sources:
        return
    with st.expander(f"{label} ({len(sources)})"):
        for s in sources:
            title = (s.get("title") or s.get("uri") or "").strip()
            uri = s.get("uri", "")
            st.markdown(f"- [{title or uri}]({uri})" if uri else f"- {title}")


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Defends st.dataframe() against 'Duplicate column names found' - two
    source keys (e.g. 'product' and 'name') renamed to the same label.
    Keeps the FIRST occurrence of each column name."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    assert len(df.columns) == len(set(df.columns)), df.columns.tolist()
    return df


def _products_table(products: list[dict]) -> pd.DataFrame:
    """One unambiguous 'Product' column (brand+product, or name) - never both."""
    rows = []
    for p in products:
        display = p.get("name") or f"{p.get('brand', '')} {p.get('product', '')}".strip()
        rows.append({
            "Type": p.get("category", ""), "Brand": p.get("brand", ""),
            "Product": display, "PKR": p.get("price_pkr"),
            "Where to buy": p.get("where_to_buy", ""),
            "Authentic source": p.get("authentic_source", ""),
        })
    return _dedup_columns(pd.DataFrame(rows))


# Short "when to use" line for the routine card - one essential step each, plus
# whichever single active/optional category the recommendation engine included.
_ROUTINE_USE = {
    "Gentle cleanser": "Morning & night",
    "Moisturizer": "After cleanser, morning & night",
    "Sunscreen": "Morning only — reapply every 3-4 h outdoors",
    "Alcohol-free toner": "Optional, right after cleansing",
    "Soothing / barrier": "Anytime skin feels tight, red or irritated",
    "Gentle serum": "Before moisturiser",
    "Optional niacinamide serum": "Optional, morning, 2-3x/week",
    "Gentle treatment (azelaic / niacinamide)": "2-3 nights/week - patch-test first",
}
_CORE_ROUTINE_CATS = ["Gentle cleanser", "Moisturizer", "Sunscreen"]


def _render_routine_card(res, rc):
    """ONE card: the single best pick per essential step - no repeated list."""
    groups = {g["category"]: g for g in (rc.get("product_options") or []) if g.get("options")}
    if not groups:
        for p in rc.get("products", []):
            st.markdown(f"- **{p.get('name', p.get('product', ''))}** — "
                        f"Rs {p.get('price_pkr', 0):,}")
        return

    cats = [c for c in _CORE_ROUTINE_CATS if c in groups]
    extra = [c for c in groups if c not in _CORE_ROUTINE_CATS]
    if extra:
        cats.append(extra[0])          # the one active/optional step this level adds

    st.markdown("### 🧴 Your Daily Routine")
    st.caption(f"For {res['skin_type'].lower()} skin — {res['condition']}")
    with st.container(border=True):
        for i, cat in enumerate(cats):
            top = groups[cat]["options"][0]
            price = f"Rs {top['price_pkr']:,}" if top["price_pkr"] else "price n/a"
            st.markdown(f"**✓ {cat.upper()}**")
            st.markdown(f"{top['brand']} {top['product']} — {price}")
            st.caption(f"Use: {_ROUTINE_USE.get(cat, groups[cat].get('usage', ''))}")
            st.caption(f"Why: {top.get('why', 'Gentle, well-suited option.')}")
            if i < len(cats) - 1:
                st.divider()
        note = rc.get("safety_note") or rc.get("brands_note")
        if note:
            st.warning(f"⚠️ **Safety:** {note}")


def _supplements_table(supps: list[dict]) -> pd.DataFrame:
    rows = []
    for s in supps:
        covers = s.get("covers")
        rows.append({
            "Supplement": s.get("name", ""),
            "Covers": ", ".join(covers) if isinstance(covers, (list, tuple)) else (covers or ""),
            "PKR": s.get("price_pkr"), "Dosage": s.get("dosage", ""),
            "Timing": s.get("timing", ""), "Where": s.get("where_to_buy", ""),
        })
    return _dedup_columns(pd.DataFrame(rows))


def _render_supplements_card(rc):
    """ONE card: the Primary supplement pick - alternatives live in an expander."""
    st.markdown("### 💊 Recommended Supplements")
    st.caption("Optional — only if diet alone isn't enough")
    supps = rc.get("supplements") or []
    if not supps:
        st.caption("No supplement rows in the CSV.")
        return

    primary = supps[0]
    with st.container(border=True):
        price = f"Rs {primary['price_pkr']:,}" if primary.get("price_pkr") else "price n/a"
        st.markdown(f"**Primary:** {primary.get('name', '')} — {price}")
        if primary.get("note"):
            st.caption(f"Why: {primary['note']}")
        if primary.get("dosage"):
            timing = f" · {primary['timing']}" if primary.get("timing") else ""
            st.caption(f"Dosage: {primary['dosage']}{timing}")
        note = rc.get("supplement_safety_note", "")
        if note:
            (st.warning if note.startswith("⚠️") else st.info)(f"🛡️ {note}")

    alt = supps[1:]
    dropped = rc.get("supplement_dropped") or []
    if alt or dropped:
        with st.expander("See supplement alternatives"):
            if alt:
                st.dataframe(_supplements_table(alt), hide_index=True,
                            use_container_width=True)
            if dropped:
                st.caption("Not added (would duplicate an ingredient already covered "
                           "above): " + ", ".join(d["name"] for d in dropped))


def _search_product_line(i, p):
    badge = ("✅ seen on 2+ sites" if p.get("verified")
             else f"⚠️ {p.get('n_sources', 1)} source — verify authenticity")
    price = (f"Rs {p['price_pkr']:,}" if p.get("price_pkr")
             else (p.get("price_range") or "price not listed"))
    line = f"**Product {i}: {p['name']}** — {price}"
    if p.get("shop"):
        line += f" — {p['shop']}"
    st.markdown(line + f"  \n  {badge}"
                + (f" · [link]({p['link']})" if p.get("link") else ""))
    if p.get("why"):
        st.markdown(f"> **Why:** {p['why']}")


def _render_search_products(res, rc):
    st.markdown(f"### For your skin condition: **{rc.get('condition_label', res['condition'])}**")
    st.caption("Searched the Pakistan market for available products…  "
               + " · ".join(f"`{q}`" for q in rc.get("searched_queries", [])))
    for e in rc.get("search_errors", []):
        st.caption(f"⚠️ {e}")

    prods = rc.get("products") or []
    if not prods:
        st.warning(rc.get("found_note", "No products found — try a different search."))
        return

    if rc.get("search_summary"):
        st.info(rc["search_summary"])
    st.markdown("**Recommended for you (chosen from the real search results only):**")
    for i, p in enumerate(prods, 1):
        _search_product_line(i, p)

    other = rc.get("other_products") or []
    if other:
        with st.expander(f"➕ Other real products found in the search ({len(other)})"):
            for i, p in enumerate(other, len(prods) + 1):
                _search_product_line(i, p)

    if rc.get("browse_shops"):
        st.caption("Browse: " + " · ".join(rc["browse_shops"]))
    st.caption(rc.get("found_note", ""))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown(
    '<div class="hh-brand"><span class="hh-logo">💪</span><span>HealthHub</span></div>',
    unsafe_allow_html=True,
)

users = db.list_users()
if users:
    options = ["<new profile>"] + users
    idx = 0
    if st.session_state.get("user_name") in users:
        idx = options.index(st.session_state["user_name"])
    chosen = st.sidebar.selectbox("Active profile", options, index=idx)
    st.session_state["user_name"] = None if chosen == "<new profile>" else chosen
else:
    st.sidebar.info("No profiles yet — create one on Home / Profile.")

PAGES = [
    "🏠 Home / Profile",
    "💪 Fitness Assessment",
    "🥗 AI Nutritionist",
    "🔬 Skin Analysis",
    "📈 Progress Tracking",
    "📋 Reports",
]
page = st.sidebar.radio("Navigate", PAGES)

_u = current_user()
if _u:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{_u['name']}**")
    st.sidebar.caption(
        f"{_u['gender']}, {_u['age']} yrs · {_u['height_cm']:.0f} cm · "
        f"{_u['weight_kg']:.1f} kg\nGoal: {_u['goal']}"
    )
st.sidebar.caption(
    "Gemini: " + ("🟢 " + nutritionist.status() if nutritionist.is_available()
                  else "🔴 " + nutritionist.status() + " (offline fallback)")
)


# --------------------------------------------------------------------------- #
# 1. Home / Profile
# --------------------------------------------------------------------------- #
def page_home():
    st.title("🏠 Home / Profile")
    st.write(
        "HealthHub gives you a calorie & macro target, a 7-day Pakistani meal plan, "
        "a workout routine, an AI nutritionist chat, skin analysis and progress tracking."
    )

    u = current_user()
    st.subheader("Edit profile" if u else "Create your profile")

    with st.form("profile_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name / username", value=u["name"] if u else "")
            email = st.text_input("Email (optional)", value=(u.get("email") or "") if u else "",
                                  placeholder="you@example.com")
            age = st.number_input("Age", 12, 100, int(u["age"]) if u else 25)
            gender = st.selectbox(
                "Gender", GENDERS,
                index=GENDERS.index(u["gender"]) if u and u["gender"] in GENDERS else 0,
            )
        with c2:
            height = st.number_input("Height (cm)", 120.0, 230.0,
                                     float(u["height_cm"]) if u else 170.0, step=0.5)
            weight = st.number_input("Weight (kg)", 30.0, 250.0,
                                     float(u["weight_kg"]) if u else 70.0, step=0.1)
            activity = st.selectbox(
                "Activity level", list(fe.ACTIVITY_MULTIPLIERS.keys()),
                index=list(fe.ACTIVITY_MULTIPLIERS).index(u["activity_level"])
                if u and u["activity_level"] in fe.ACTIVITY_MULTIPLIERS else 1,
            )
        with c3:
            goal = st.selectbox(
                "Goal", fe.GOALS,
                index=fe.GOALS.index(u["goal"]) if u and u["goal"] in fe.GOALS else 0,
            )
            budget = st.number_input("Food budget (PKR)", 0.0, 1_000_000.0,
                                     float(u["budget_pkr"]) if u and u["budget_pkr"] else 800.0,
                                     step=50.0)
            period = st.selectbox(
                "Budget period", BUDGET_PERIODS,
                index=BUDGET_PERIODS.index(u["budget_period"])
                if u and u.get("budget_period") in BUDGET_PERIODS else 0,
            )
            diet_pref = st.selectbox(
                "Dietary preference", fe.DIET_PREFERENCES,
                index=fe.DIET_PREFERENCES.index(u["diet_preference"])
                if u and u.get("diet_preference") in fe.DIET_PREFERENCES else 0,
            )
            gym = st.checkbox("I have gym access", value=bool(u["gym_access"]) if u else False)

        submitted = st.form_submit_button("Save profile", type="primary")

    if submitted:
        if not name.strip():
            st.error("Please enter a name.")
        elif email.strip() and "@" not in email:
            st.error("That email doesn't look valid - leave it blank or fix it.")
        else:
            saved = db.upsert_user(name.strip(), int(age), gender, float(height),
                                   float(weight), goal, float(budget), period, gym, activity,
                                   diet_pref, email=email.strip())
            st.session_state["user_name"] = saved["name"]
            st.success(f"Profile saved for {saved['name']}.")
            st.rerun()

    if u:
        bmi = u["weight_kg"] / (u["height_cm"] / 100) ** 2
        cat = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25
               else "Overweight" if bmi < 30 else "Obese")
        st.metric("BMI", f"{bmi:.1f}", cat)
        with st.expander("Danger zone"):
            if st.button("Delete this profile"):
                db.delete_user(u["id"])
                st.session_state["user_name"] = None
                st.rerun()


# --------------------------------------------------------------------------- #
# 2. Fitness Assessment
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _warm_clients() -> dict:
    return {"gemini": gemini_client.is_available(),
            "serpapi": product_search.is_available()}


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_price_status() -> dict:
    return price_data.status()


# Budget bucketed to Rs 25 so near-identical inputs reuse the cached plan.
@st.cache_data(show_spinner=False, max_entries=64)
def _gen_meal_plan(mode: str, calorie_target: int, budget_bucket: int, goal: str,
                   preference: str, protein_target: int = 0) -> dict:
    b = float(budget_bucket)
    if mode == "csv":
        return fe.build_meal_plan_csv(calorie_target, b, goal, preference,
                                      protein_target_g=protein_target or None)
    return fe.build_meal_plan(calorie_target, b)


_MEAL_SOURCE_LABEL = {
    "csv+gemini": "📄 meals_prices.csv prices — Gemini arranged the foods (prices fixed, validated)",
    "csv": "📄 meals_prices.csv — arranged locally (no API call)",
    "csv-optimized": "📄 meals_prices.csv — cost-per-macro optimiser (budget = hard ceiling)",
    "csv-missing": "🧮 built-in food database (meals_prices.csv missing/broken)",
    "budget-error": "🚫 budget below the safe minimum",
    "budget-infeasible": "🚫 target not achievable within budget",
    "builtin": "🧮 built-in food database",
}


@st.cache_data(show_spinner=False, max_entries=64)
def _gen_workout(experience: str, location: str, focus: str, days: int,
                 equipment: tuple, injuries: str, goal: str) -> dict:
    prof = {"experience": experience, "location": location, "goal_focus": focus,
            "days_per_week": days, "equipment": list(equipment),
            "limitations": injuries or "none", "goal": goal}
    return fe.generate_workout_dynamic(prof, use_gemini=True)


_WK_SRC_BADGE = {
    "gemini": "🤖 AI-personalised (Gemini, schema-validated)",
    "gemini-cached": "🤖 AI-personalised (cached — no new API call)",
    "static": "🧮 Static routine (fast fallback)",
}


def _workout_form(u: dict) -> dict:
    """Collect the workout profile; remembers choices per user in session_state."""
    key = f"wk_{u['id']}"
    saved = st.session_state.get(key, {})
    gym = bool(u.get("gym_access"))
    with st.expander("🎛️ Personalise my workout (experience, equipment, injuries…)",
                     expanded=not saved):
        c1, c2, c3 = st.columns(3)
        exp = c1.selectbox("Experience", fe.EXPERIENCE_LEVELS,
                           index=fe.EXPERIENCE_LEVELS.index(
                               saved.get("experience", fe._infer_experience(u))))
        loc = c2.selectbox("Where you train", fe.WORKOUT_LOCATIONS,
                           index=fe.WORKOUT_LOCATIONS.index(
                               saved.get("location", "Gym" if gym else "Home")))
        focus = c3.selectbox("Training focus", fe.WORKOUT_FOCI,
                             index=fe.WORKOUT_FOCI.index(saved.get(
                                 "focus", fe._GOAL_TO_FOCUS.get(u.get("goal", ""),
                                                               "General fitness"))))
        days = st.select_slider("Days per week", fe.WORKOUT_DAYS_CHOICES,
                                value=saved.get("days", 4))
        default_equip = (["Barbell", "Dumbbells", "Bench", "Cable machine", "Pull-up bar"]
                         if loc in ("Gym", "Both")
                         else ["Resistance bands", "Backpack (loadable)", "Pull-up bar"])
        equip = st.multiselect("Available equipment", fe.EQUIPMENT_OPTIONS,
                               default=saved.get("equipment", default_equip),
                               help="Pick nothing / 'None' for a pure bodyweight plan.")
        injuries = st.text_input("Injuries / limitations (optional)",
                                 value=saved.get("injuries", ""),
                                 placeholder="e.g. left knee pain, avoid overhead pressing")
    prof = {"experience": exp, "location": loc, "focus": focus, "days": int(days),
            "equipment": equip or ["None (bodyweight only)"], "injuries": injuries.strip()}
    st.session_state[key] = prof
    return prof


def _assessment_recommendations(result: dict) -> dict:
    """Compact, JSON-able summary saved alongside the full meal_plan/workout
    blobs - what a history view shows without re-parsing those."""
    plan = result.get("meal_plan", {}) or {}
    workout = result.get("workout", {}) or {}
    return {
        "meal_source": plan.get("source"),
        "meal_notes": plan.get("notes", []),
        "budget_validation": plan.get("validation"),
        "workout_source": workout.get("source"),
        "workout_note": workout.get("note"),
    }


def _save_assessment_button(result: dict, u: dict) -> None:
    if st.button("💾 Save this assessment", type="primary"):
        db.save_assessment(
            u["id"], result["bmr"], result["tdee"], result["calorie_target"],
            result["macros"], result["meal_plan"], result["workout"],
            protein_target=result["macros"].get("protein_g"),
            recommendations=_assessment_recommendations(result),
        )
        st.success("Assessment saved to your profile.")

    history = db.get_assessment_history(u["id"], limit=10)
    if history:
        with st.expander(f"📜 Past assessments ({len(history)})"):
            hist_df = pd.DataFrame([{
                "Date": h.get("date") or h["created_at"][:10],
                "Calories": h.get("calorie_target"),
                "Protein (g)": h.get("protein_target"),
                "Meal source": (h.get("recommendations") or {}).get("meal_source", ""),
                "Workout source": (h.get("recommendations") or {}).get("workout_source", ""),
            } for h in history])
            st.dataframe(_dedup_columns(hist_df), hide_index=True, use_container_width=True)


def _render_workout(result: dict, u: dict) -> None:
    wk = result["workout"]
    st.subheader(f"🏋️ Workout Routine — {wk.get('location', 'Home')}"
                 f" · {wk.get('goal_focus', '')} · {wk.get('experience_level', '')}")
    st.caption(_WK_SRC_BADGE.get(wk.get("source"), wk.get("source", "")))
    if wk.get("note"):
        st.caption(wk["note"])
    for day, block in wk["week"].items():
        head = f"{day} — {block['focus']}"
        if block.get("sets") and block.get("reps"):
            head += f"  ({block['sets']} × {block['reps']})"
        with st.expander(head, expanded=day in ("Monday", "Tuesday")):
            if block["exercises"]:
                for ex in block["exercises"]:
                    st.markdown(f"- {ex}")
            else:
                st.markdown("_Rest day._")


def page_fitness():
    st.title("💪 Fitness Assessment")
    u = require_user()

    result = fe.full_assessment(u, dynamic_workout=False)  # fast: static workout placeholder
    gemini_on = nutritionist.is_available()
    _warm_clients()

    budget_bucket = int(round(result["daily_budget"] / 25) * 25)
    pref = u.get("diet_preference", "No preference")
    with st.status("Building your 7-day meal plan…", expanded=True) as _stat:
        _stat.write("📄 Reading real prices from meals_prices.csv…")
        if gemini_on:
            _stat.write("🤖 Gemini is arranging the foods (every price stays fixed "
                        "from the CSV)…")
        else:
            _stat.write("🧮 Gemini not configured — arranging the meals locally…")
        mp = _gen_meal_plan("csv", result["calorie_target"], budget_bucket,
                            u["goal"], pref, result["macros"]["protein_g"])
        _stat.update(label="Meal plan ready", state="complete", expanded=False)

    if mp.get("source") == "csv-missing":
        _gen_meal_plan.clear()
        st.error(mp["notes"][0])
    result["meal_plan"] = mp

    # ---- workout profile + dynamic (Gemini) generation with static fallback ----
    wp = _workout_form(u)
    with st.spinner("🤖 Personalising your workout…" if gemini_on
                    else "Building your workout…"):
        result["workout"] = _gen_workout(
            wp["experience"], wp["location"], wp["focus"], wp["days"],
            tuple(wp["equipment"]), wp["injuries"], u["goal"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BMR", f"{result['bmr']:,} kcal")
    c2.metric("TDEE", f"{result['tdee']:,} kcal")
    c3.metric("Calorie target", f"{result['calorie_target']:,} kcal")
    c4.metric("Daily food budget", f"PKR {result['daily_budget']:,}")

    m = result["macros"]
    st.caption(
        f"Macros: **{m['protein_g']} g protein** · **{m['carbs_g']} g carbs** · "
        f"**{m['fat_g']} g fat**  |  Goal: {u['goal']}  |  Activity: {u['activity_level']}"
    )

    # ---- meal plan ----
    st.subheader("🍽️ 7-Day Pakistani Meal Plan")
    plan = result["meal_plan"]

    # budget is a HARD constraint - an error means we do NOT show a plan
    if plan.get("error"):
        _val = plan.get("validation") or {}
        st.error("🚫 " + plan["error"])
        st.markdown(
            f"- **Daily budget:** PKR {_val.get('daily_budget', result['daily_budget']):,}\n"
            f"- **Daily cost:** {'—' if _val.get('daily_cost') is None else 'PKR %s' % format(_val['daily_cost'], ',')}\n"
            f"- **Status:** ❌ {'budget below the PKR 500/day minimum' if plan.get('source') == 'budget-error' else 'target not achievable within this budget'}"
        )
        if plan.get("min_budget"):
            st.info(f"💡 Raise the food budget to about **PKR {plan['min_budget']:,}/day** "
                    f"(PKR {plan['min_budget'] * 30:,}/month) for this calorie/protein "
                    f"target, or lower the goal. {plan.get('advice', '')}")
        st.caption("Update your budget on the Home / Profile page, then reopen this tab.")
        _render_workout(result, u)
        _save_assessment_button(result, u)
        return

    st.caption("Source: " + _MEAL_SOURCE_LABEL.get(plan.get("source"), plan.get("source", "")))
    for note in plan["notes"]:
        st.info(note)

    _val = plan.get("validation")
    if _val:
        ok = _val.get("within_budget")
        (st.success if ok else st.error)(
            f"Budget check — Daily budget: PKR {_val['daily_budget']:,}  ·  "
            f"Daily cost: PKR {_val['daily_cost']:,}  ·  "
            + ("✅ within budget" if ok else "❌ OVER budget"))

    st.write(
        f"**Weekly food cost ≈ PKR {plan['weekly_cost']:,}**  "
        f"(avg PKR {plan['avg_daily_cost']:,}/day)"
    )

    priced = plan.get("ingredient_prices")
    if priced:
        with st.expander(f"🧾 Real prices from meals_prices.csv ({len(priced)} items)"):
            st.dataframe(
                _dedup_columns(pd.DataFrame(priced).rename(columns={
                    "item": "Item", "unit": "Unit", "price_pkr": "Price (PKR)",
                    "notes": "Notes"})),
                hide_index=True, use_container_width=True)

    has_breakdown = any(m.get("items") for d in plan["days"] for m in d["meals"])
    tabs = st.tabs([d["day"] for d in plan["days"]])
    for tab, day in zip(tabs, plan["days"]):
        with tab:
            rows = []
            for meal in day["meals"]:
                row = {"Meal": meal["slot"], "Food": meal["name"]}
                if has_breakdown:
                    row["Breakdown (real prices)"] = meal.get("items", "")
                row.update({"kcal": meal["kcal"], "Protein (g)": meal["protein"],
                            "PKR": meal["price"]})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            flag = "  ⚠️ over daily budget" if day["over_budget"] else ""
            st.caption(
                f"Day total: **{day['total_kcal']:,} kcal** · "
                f"{day['total_protein']} g protein · **PKR {day['total_price']:,}**{flag}"
            )

    _render_workout(result, u)
    _save_assessment_button(result, u)


# --------------------------------------------------------------------------- #
# 3. AI Nutritionist
# --------------------------------------------------------------------------- #
def page_nutritionist():
    st.title("🥗 AI Nutritionist")
    u = require_user()

    assessment = db.get_latest_assessment(u["id"])
    profile = dict(u)
    if assessment:
        profile["calorie_target"] = assessment["calorie_target"]

    if not nutritionist.is_available():
        st.warning(
            f"Gemini unavailable ({nutritionist.status()}) — running in offline "
            "fallback mode. Add `GEMINI_API_KEY` to your environment or to "
            "`.streamlit/secrets.toml` for full conversational answers."
        )

    history = db.get_chat(u["id"])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("🗑️ Clear chat"):
            db.clear_chat(u["id"])
            st.rerun()

    prompt = st.chat_input("Ask about diet, calories, Pakistani foods, prices...")
    if prompt:
        db.add_chat(u["id"], "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        convo = db.get_chat(u["id"])
        with st.chat_message("assistant"):
            placeholder = st.empty()
            acc = ""
            for chunk in nutritionist.chat_stream(convo, profile):
                acc += chunk
                placeholder.markdown(acc + "▌")
            placeholder.markdown(acc)
        db.add_chat(u["id"], "assistant", acc)
        st.rerun()

    with st.expander("Example questions"):
        st.markdown(
            "- What's a cheap high-protein breakfast under PKR 200?\n"
            "- Build me a 1,800 kcal day using daal, roti and chicken.\n"
            "- I'm always hungry on my diet — what should I change?\n"
            "- Is brown bread better than roti for weight loss?"
        )


# --------------------------------------------------------------------------- #
# 4. Skin Analysis  (analysis via skin_ai, recommendations via skin_analyzer)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _skin_ai():
    """One SkinAnalyzer (MediaPipe + model load once)."""
    return SkinAnalyzer(model_path="skin_ai/skin_model.pth", use_mediapipe=True)


_SKIN_SEV_N = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3, "unknown": 0, "none": 0}
_SKIN_NAMES = {"redness": "Redness / inflammation",
               "hyperpigmentation": "Hyperpigmentation / dark spots",
               "texture": "Uneven texture", "blemishes": "Blemishes / breakouts"}


def _safe_supplements(recs: dict, conditions: list[str]) -> None:
    """De-duplicate the supplement list by active-ingredient class and add an
    overdose safety note (mutates recs in place)."""
    supps = recs.get("supplements") or []
    if not supps:
        return
    dd = deduplicate_supplements(supps, context=context_from_conditions(conditions))
    recs["supplements"] = dd["supplements"]
    recs["supplement_safety_note"] = dd["safety_note"]
    recs["supplement_primary"] = dd["primary"]
    recs["supplement_dropped"] = dd["dropped"]


def _adapt_skin_ai(r: dict, mode: str) -> dict:
    """Map skin_ai SkinAnalyzer.analyze() output onto the schema page_skin() uses;
    recommendations still come from skin_analyzer._recommend()."""
    model = r.get("model", {}) or {}
    if not r.get("assessed", False):
        recs, src = skin_analyzer._recommend("Normal", [], mode=mode, level="basic")
        return {
            "assessed": False, "skin_type": "Unknown", "condition": "Not assessed",
            "conditions": [], "findings": [], "observations": [], "is_clear": None,
            "severity": "none", "advice_band": "Not assessed", "care_level": "basic",
            "summary": r.get("message") or r.get("reason") or "Could not assess this photo.",
            "confidence": 0.0,
            "metrics": {**(r.get("quality") or {}), "skin_fraction": r.get("skin_fraction", 0.0)},
            "cnn": model, "resnet": None, "model": {"trained": bool(model.get("trained"))},
            "backbone": "skin_ai preprocessing (gray-world colour constancy + "
                        "specular/shadow inpaint + occlusion mask)",
            "recommendations": recs, "reco_source": src,
            "disclaimer": r.get("disclaimer", ""),
            "skin_type_confidence": None, "skin_type_source": "not assessed",
            "skin_type_reasoning": "", "skin_type_signals": [], "skin_type_note": "",
        }

    sev = {"redness": r.get("redness_severity", "clear"),
           "hyperpigmentation": r.get("hyperpigmentation_severity", "clear"),
           "texture": r.get("texture_severity", "clear")}
    bc = int(r.get("blemish_count", 0) or 0)
    levels = {k: _SKIN_SEV_N.get(v, 0) for k, v in sev.items()}
    levels["blemishes"] = 0 if bc == 0 else 1 if bc <= 4 else 2 if bc <= 12 else 3
    worst = max(levels.values()) if levels else 0
    care = "treat" if worst >= 3 else "light" if worst >= 2 else "basic"
    conf_pct = round(float(r.get("confidence", 0.0) or 0.0) * 100.0, 1)

    findings, observations, conditions = [], [], []
    for key, lvl in sorted(levels.items(), key=lambda kv: -kv[1]):
        if lvl <= 0:
            continue
        item = {"name": _SKIN_NAMES[key], "metric": key, "score": round(lvl / 3, 3),
                "confidence": conf_pct,
                "severity": ["clear", "mild", "moderate", "severe"][lvl]}
        (findings if lvl >= 2 else observations).append(item)
        if lvl >= 2:
            conditions.append(_SKIN_NAMES[key])

    band = ("Skin looks clear - maintain your routine" if care == "basic"
            else "Minor concerns - basic care" if care == "light"
            else "Visible concerns - consider treatment")
    sev_word = ("high" if worst >= 3 else "medium" if worst >= 2
                else "low" if worst >= 1 else "none")
    skin_type = r.get("skin_type", "Normal")
    recs, src = skin_analyzer._recommend(skin_type, conditions, mode=mode, level=care)
    _safe_supplements(recs, conditions)          # dedupe by ingredient, cap 2-3, overdose check
    summary = (band + ": " + "; ".join(f"{f['name']} ({f['severity']}, {conf_pct:.0f}%)"
                                       for f in findings) + "."
               if conditions else band + ". No significant concerns detected.")
    if care == "basic":
        recs["is_clear_message"] = summary

    st_conf = r.get("skin_type_confidence")
    return {
        "assessed": True, "skin_type": skin_type,
        "condition": "; ".join(conditions) or "Clear", "conditions": conditions,
        "findings": findings, "observations": observations, "is_clear": worst < 2,
        "severity": sev_word, "advice_band": band, "care_level": care,
        "summary": summary, "confidence": conf_pct,
        "metrics": {**(r.get("quality") or {}), "skin_fraction": r.get("skin_fraction", 0.0),
                    "blemish_count": bc, "zones": r.get("zones", {}),
                    "redness_severity": sev["redness"],
                    "hyperpigmentation_severity": sev["hyperpigmentation"],
                    "texture_severity": sev["texture"]},
        "cnn": model, "resnet": None,
        "model": {"trained": bool(model.get("trained")), "arch": model.get("arch", ""),
                  "top_class": skin_type,
                  "top_conf": round(float(st_conf), 3) if st_conf is not None else 0.0},
        "backbone": ("skin_ai: gray-world colour constancy + occlusion mask + "
                     "Google Gemini Vision (skin type); "
                     + ("EfficientNet-B0" if model.get("trained") else "CV screen")
                     + " (blemishes, texture, redness)"),
        "recommendations": recs, "reco_source": src,
        "disclaimer": r.get("disclaimer", ""),
        "primary_concerns": r.get("primary_concerns", []),
        "skin_type_confidence": st_conf,
        "skin_type_source": r.get("skin_type_source", ""),
        "skin_type_reasoning": r.get("skin_type_reasoning", ""),
        "skin_type_signals": r.get("skin_type_signals", []),
        "skin_type_note": r.get("skin_type_note", ""),
    }


def analyze_skin_image(image, mode: str) -> dict:
    try:
        raw = _skin_ai().analyze(image)
    except Exception as e:                       # never crash the page on an analysis error
        return _adapt_skin_ai({"assessed": False, "reason": f"analysis error: {e}"}, mode)
    return _adapt_skin_ai(raw, mode)


def page_skin():
    st.title("🔬 Skin Analysis")
    u = require_user()
    st.caption(
        "**skin_ai** pipeline: MediaPipe face mesh → gray-world colour constancy → "
        "occlusion mask → **Google Gemini Vision** (skin type) + EfficientNet-B0 / CV "
        "(blemishes, texture, redness). **Not a medical diagnosis.**"
    )
    if not gemini_client.is_available():
        st.info("No Gemini API key — skin type falls back to a conservative "
                "default (Combination). Add `GEMINI_API_KEY` to "
                "`.streamlit/secrets.toml` for an accurate Oily / Dry / Normal / "
                "Combination call.")
    if not skin_analyzer.torch_available():
        st.info("PyTorch not installed — blemish/texture analysis will use image "
                "metrics only (install `torch` + `torchvision` for the CNN heads).")

    serp_ok = product_search.is_available()
    src_choice = st.radio(
        "Product source",
        ["📄 CSV brand database", "🔎 Live internet search (SerpAPI)"],
        horizontal=True, index=0,          # CSV first: instant, free, no quota risk
        help="CSV is the curated brand database — instant, free, always available. "
             "Live search queries Google (via SerpAPI) with short, single-condition "
             "searches for newer products, keeps ones seen on 2+ sites, and Gemini "
             "picks only from those real results. Needs SERPAPI_API_KEY (free tier = "
             "100/month) and silently falls back to the CSV if the quota runs out or "
             "nothing real comes back — you'll never see a raw search error.",
    )
    mode = "search" if src_choice.startswith("🔎") else "csv"
    if mode == "search" and not serp_ok:
        st.info(f"Live search unavailable ({product_search.status()}). Add "
                "`SERPAPI_API_KEY` to `.streamlit/secrets.toml` (free tier = 100/month). "
                "Using the CSV database instead.")
        mode = "csv"

    sk_status = _cached_price_status()
    if mode == "csv":
        if sk_status["skincare_error"]:
            st.error(f"skincare_brands.csv problem: {sk_status['skincare_error']} "
                     "— falling back to the built-in list.")
        else:
            st.caption(f"📄 skincare_brands.csv — {sk_status['skincare_rows']} products "
                       f"across {len(sk_status['brands'])} brands.")

    upload = st.file_uploader("Upload a clear, well-lit face photo",
                              type=["jpg", "jpeg", "png", "webp"])
    if upload:
        img_bytes = upload.getvalue()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img_bytes, caption="Uploaded photo", use_container_width=True)
        with c2:
            spin = ("Analysing photo, then searching real products online..."
                    if mode == "search" else "Analysing photo, then matching CSV products...")
            with st.spinner(spin):
                from PIL import Image

                res = analyze_skin_image(Image.open(io.BytesIO(img_bytes)), mode)

            _band = res.get("advice_band", "")
            _conf = float(res.get("confidence", 0) or 0)
            _sev = res.get("severity", "none")
            _sev_txt = f" · severity: {_sev}" if _sev not in ("none", "", None) else ""
            if res.get("assessed") is False:
                st.warning("⚠️ " + res.get("summary", "Could not assess this photo."))
            elif res.get("care_level") == "basic":
                st.success(f"✅ **{_band}** — {_conf:.0f}% concern level{_sev_txt}")
            elif res.get("care_level") == "light":
                st.info(f"🟡 **{_band}** — {_conf:.0f}% concern level{_sev_txt}")
            else:
                st.warning(f"🟠 **{_band}** — {_conf:.0f}% concern level{_sev_txt}")

            _stype = res.get("skin_type", "Normal")
            _stconf = res.get("skin_type_confidence")
            st.metric("Skin type", _stype
                      + (f"  ·  {float(_stconf) * 100:.0f}%" if _stconf else ""))
            _streason = res.get("skin_type_reasoning") or ""
            _src = res.get("skin_type_source", "")
            if res.get("assessed") and _src and _src != "not assessed":
                st.caption("🤖 " + (_streason or _src))       # Gemini Vision explanation
                _sigs = res.get("skin_type_signals") or []
                if _sigs:
                    st.caption("Also weighed: " + " · ".join(_sigs))
                st.caption(res.get("skin_type_note", ""))
            for f in res.get("findings", []):
                st.markdown(f"- **{f.get('name', 'Concern')}** · "
                            f"_severity {f.get('severity', '?')}, "
                            f"{float(f.get('confidence', 0)):.0f}% confidence_")
            for o in res.get("observations", []):
                st.caption(f"· minor: {o.get('name', '').split(' (')[0]} "
                           f"(~{float(o.get('confidence', 0)):.0f}%, not actionable)")
            st.progress(min(1.0, _conf / 100), text=f"Concern level {_conf:.0f}%")
            _rr = res.get("resnet") or {}
            _metrics = res.get("metrics", {}) or {}
            _mdl = res.get("model", {}) or {}
            st.caption(
                f"Read-out: {res.get('backbone', 'CV metrics')} · skin region "
                f"{int(_metrics.get('skin_fraction', 0) * 100)}% of crop"
                + ("" if _mdl.get("trained")
                   else "  ·  no trained CNN — blemish/texture from CV metrics only "
                        "(`python skin_ai/training.py` to train the concern heads)")
            )

        rc = res.get("recommendations", {}) or {}
        _reco_label = {"search": "🔎 live internet search (SerpAPI) + Gemini selection",
                       "csv": "📄 skincare_brands.csv (gentle products)",
                       "builtin": "🧮 built-in list"}
        st.caption("Recommendations from: "
                   + _reco_label.get(res.get("reco_source"), res.get("reco_source", "")))

        if rc.get("is_clear_message"):
            st.markdown("**" + rc["is_clear_message"] + "**")
        if rc.get("level_note"):
            st.info("ℹ️ " + rc["level_note"])

        is_search = res.get("reco_source") == "search"
        if is_search:
            if rc.get("safety_note"):
                st.info("🛡️ **Cautious mode** — " + rc["safety_note"])
            _render_search_products(res, rc)
        else:
            _render_routine_card(res, rc)          # safety note lives inside the card

        _render_supplements_card(rc)

        st.markdown("### 🥗 Diet Tips")
        for tip in rc["diet_changes"]:
            st.markdown(f"- {tip}")

        if not is_search and rc.get("products"):
            with st.expander("👉 See All Brands & Products"):
                st.dataframe(_products_table(rc["products"]), hide_index=True,
                            use_container_width=True)

        if rc.get("authenticity_tips"):
            with st.expander("🔎 How to verify the product is genuine (avoid counterfeits)"):
                for tip in rc["authenticity_tips"]:
                    st.markdown(f"- {tip}")
        with st.expander("Raw metrics"):
            st.json({"metrics": res["metrics"], "cnn": res["cnn"]})
        st.warning(res["disclaimer"])

        if st.button("💾 Save this analysis", type="primary"):
            path = db.save_uploaded_photo(img_bytes, upload.name, prefix=f"skin_u{u['id']}")
            db.add_skin_analysis(u["id"], date.today().isoformat(), res["skin_type"],
                                 res["condition"], res, path)
            st.success("Analysis saved.")

    st.markdown("---")
    st.subheader("Past analyses")
    past = db.get_skin_analyses(u["id"])
    if not past:
        st.caption("Nothing saved yet.")
    for a in past:
        with st.expander(f"{a['date']} — {a['skin_type']} — {a['condition'][:60]}"):
            cols = st.columns([1, 3])
            if a["photo_path"]:
                try:
                    cols[0].image(a["photo_path"], use_container_width=True)
                except Exception:
                    cols[0].caption("(photo missing)")
            cols[1].write(f"**Skin type:** {a['skin_type']}")
            cols[1].write(f"**Condition:** {a['condition']}")


# --------------------------------------------------------------------------- #
# 5. Progress Tracking
# --------------------------------------------------------------------------- #
def page_progress():
    st.title("📈 Progress Tracking")
    u = require_user()

    with st.expander("➕ Log a new entry", expanded=True):
        with st.form("progress_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                d = st.date_input("Date", value=date.today())
                weight = st.number_input("Weight (kg)", 30.0, 300.0,
                                         float(u["weight_kg"]), step=0.1)
                chest = st.number_input("Chest (cm)", 0.0, 200.0, 0.0, step=0.5)
            with c2:
                waist = st.number_input("Waist (cm)", 0.0, 200.0, 0.0, step=0.5)
                hips = st.number_input("Hips (cm)", 0.0, 200.0, 0.0, step=0.5)
                arms = st.number_input("Arms (cm)", 0.0, 100.0, 0.0, step=0.5)
            with c3:
                thighs = st.number_input("Thighs (cm)", 0.0, 120.0, 0.0, step=0.5)
                photo = st.file_uploader("Progress photo", type=["jpg", "jpeg", "png", "webp"])
                notes = st.text_input("Notes")
            save = st.form_submit_button("Save entry", type="primary")

        if save:
            path = ""
            if photo is not None:
                path = db.save_uploaded_photo(photo.getvalue(), photo.name,
                                              prefix=f"prog_u{u['id']}")
            db.add_progress(
                u["id"], d.isoformat(), float(weight),
                chest or None, waist or None, hips or None, arms or None, thighs or None,
                path, notes,
            )
            # keep profile weight in sync with the latest log
            db.upsert_user(u["name"], u["age"], u["gender"], u["height_cm"], float(weight),
                           u["goal"], u["budget_pkr"], u.get("budget_period", "daily"),
                           u["gym_access"], u["activity_level"],
                           u.get("diet_preference", "No preference"), email=u.get("email"))
            st.success("Entry saved.")
            st.rerun()

    df = db.get_progress_df(u["id"])
    if df.empty:
        st.info("No entries yet. Log one above to start seeing charts.")
        return

    c1, c2 = st.columns(2)
    c1.pyplot(pt.weight_figure(df))
    c2.pyplot(pt.measurements_figure(df))

    st.subheader("Entries")
    st.dataframe(pt.measurements_table(df), hide_index=True, use_container_width=True)

    st.subheader("🖼️ Photo timeline")
    photos = pt.photo_entries(df)
    if photos:
        cols = st.columns(min(4, len(photos)))
        for i, rec in enumerate(photos):
            with cols[i % len(cols)]:
                try:
                    st.image(rec["photo_path"], use_container_width=True)
                except Exception:
                    st.caption("(photo missing)")
                w = f" · {rec['weight_kg']:.1f} kg" if pd.notna(rec["weight_kg"]) else ""
                st.caption(f"{rec['date']}{w}")
    else:
        st.caption("No progress photos uploaded yet.")

    st.subheader("🤖 AI progress analysis")
    if st.button("Analyse my progress", type="primary"):
        with st.spinner("Thinking..."):
            st.markdown(pt.ai_feedback(df, u))
    else:
        st.markdown(pt.analyze_progress(df, u))


# --------------------------------------------------------------------------- #
# 6. Reports
# --------------------------------------------------------------------------- #
def build_report(u: dict) -> str:
    lines = [
        f"# HealthHub Report — {u['name']}",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "## Profile",
        f"- Age / gender: {u['age']} / {u['gender']}",
        f"- Height / weight: {u['height_cm']:.0f} cm / {u['weight_kg']:.1f} kg",
        f"- Goal: {u['goal']}",
        f"- Activity: {u['activity_level']}",
        f"- Dietary preference: {u.get('diet_preference', 'No preference')}",
        f"- Food budget: PKR {u['budget_pkr']:.0f} ({u.get('budget_period', 'daily')})",
        f"- Gym access: {'yes' if u['gym_access'] else 'no'}",
    ]
    bmi = u["weight_kg"] / (u["height_cm"] / 100) ** 2
    lines.append(f"- BMI: {bmi:.1f}")

    a = db.get_latest_assessment(u["id"])
    if a:
        lines += [
            "", "## Latest Assessment",
            f"- BMR: {a['bmr']:.0f} kcal",
            f"- TDEE: {a['tdee']:.0f} kcal",
            f"- Calorie target: {a['calorie_target']:.0f} kcal",
            f"- Macros: {a['macros'].get('protein_g')} g P / "
            f"{a['macros'].get('carbs_g')} g C / {a['macros'].get('fat_g')} g F",
            f"- Weekly meal-plan cost: PKR {a['meal_plan'].get('weekly_cost', 0):,}"
            f" (pricing: {a['meal_plan'].get('source', 'builtin')})",
        ]
        if a["meal_plan"].get("error"):
            lines.append(f"- ⚠️ Meal plan not generated: {a['meal_plan']['error']}")

    df = db.get_progress_df(u["id"])
    if not df.empty:
        lines += ["", "## Progress", "", pt.analyze_progress(df, u)]

    skins = db.get_skin_analyses(u["id"])
    if skins:
        s = skins[0]
        lines += [
            "", "## Latest Skin Analysis",
            f"- Date: {s['date']}",
            f"- Skin type: {s['skin_type']}",
            f"- Condition: {s['condition']}",
        ]

    lines += ["", "---", "_Not medical advice. Consult a professional for medical concerns._"]
    return "\n".join(lines)


def page_reports():
    st.title("📋 Personalised Report")
    u = require_user()
    report = build_report(u)
    st.markdown(report)
    st.download_button(
        "⬇️ Download report (Markdown)", report,
        file_name=f"healthhub_{u['name']}_{date.today().isoformat()}.md",
        mime="text/markdown",
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
ROUTES = {
    "🏠 Home / Profile": page_home,
    "💪 Fitness Assessment": page_fitness,
    "🥗 AI Nutritionist": page_nutritionist,
    "🔬 Skin Analysis": page_skin,
    "📈 Progress Tracking": page_progress,
    "📋 Reports": page_reports,
}
ROUTES[page]()
