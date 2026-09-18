"""
HealthHub REST API  —  a thin FastAPI layer over the existing HealthHub Python
modules, so the static front-end in ../frontend can talk to real logic.

    pip install -r backend/requirements-api.txt
    uvicorn backend.api:app --reload --port 8000      # run from the repo root

Then open frontend/index.html (or serve ../frontend on any static host). The
front-end calls http://localhost:8000/api/* and falls back to bundled demo data
whenever this service is not reachable, so it also works fully offline.

Every endpoint is defensive: on an internal error it returns a JSON body with an
"error" key rather than a 500, so the UI degrades instead of breaking.
"""
from __future__ import annotations

import io
import os
import sys
import traceback

# make the repo root importable when run as `uvicorn backend.api:app`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# mirror app.py: surface .streamlit/secrets.toml into os.environ so the shared
# modules (nutritionist, skin_ai/gemini_vision, product_search) see the keys.
def _load_secrets() -> None:
    path = os.path.join(_ROOT, ".streamlit", "secrets.toml")
    try:
        import tomllib
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return
    except Exception:
        return
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL",
              "SERPAPI_API_KEY", "SERP_API_KEY"):
        if k not in os.environ and k in data:
            os.environ[k] = str(data[k])


_load_secrets()

import base64

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database as db
import fitness_engine as fe
import nutritionist

app = FastAPI(title="HealthHub API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    # TODO: replace REPLACE_WITH_NETLIFY_URL below with the actual Netlify URL
    # once Netlify assigns it after deployment.
    allow_origins=[
        "http://localhost:5173",
        "https://REPLACE_WITH_NETLIFY_URL.netlify.app",
    ],
    allow_methods=["*"], allow_headers=["*"],
)

db.init_db()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _user(name: str | None = None) -> dict | None:
    """Resolve a profile strictly by name. No fallback to "the first user" — an
    unknown or missing name means "not signed in", so callers return an empty /
    create-profile state instead of silently adopting someone else's data."""
    name = (name or "").strip()
    return db.get_user_by_name(name) if name else None


def _profile_out(u: dict) -> dict:
    return {
        "exists": True,
        "name": u.get("name", ""), "email": u.get("email") or "",
        "gender": u.get("gender", "Female"), "age": u.get("age"),
        "height_cm": u.get("height_cm"), "weight_kg": u.get("weight_kg"),
        "goal": u.get("goal", "Maintain"),
        "activity": u.get("activity_level", "Moderately active"),
        "budget_pkr": u.get("budget_pkr", 1200),
        "budget_period": u.get("budget_period", "daily"),
        "diet_preference": u.get("diet_preference", "No preference"),
        "gym_access": bool(u.get("gym_access", 0)),
    }


def _guard(fn):
    """Wrap a route body so exceptions become a clean JSON error."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=200)


# --------------------------------------------------------------------------- #
# profile
# --------------------------------------------------------------------------- #
class ProfileIn(BaseModel):
    name: str
    email: str | None = None
    gender: str = "Female"
    age: int = 27
    height_cm: float = 165
    weight_kg: float = 65
    goal: str = "Maintain"
    activity: str = "Moderately active"
    budget_pkr: float = 1200
    budget_period: str = "daily"
    diet_preference: str = "No preference"
    gym_access: bool = False


@app.get("/api/profile")
def get_profile(user: str | None = None):
    """Returns the named profile, or {"exists": false} when `user` is missing or
    unknown. Never invents a default identity."""
    def run():
        u = _user(user)
        return _profile_out(u) if u else {"exists": False}
    return _guard(run)


@app.post("/api/profile")
def save_profile(p: ProfileIn):
    def run():
        u = db.upsert_user(
            name=p.name, age=p.age, gender=p.gender, height_cm=p.height_cm,
            weight_kg=p.weight_kg, goal=p.goal, budget_pkr=p.budget_pkr,
            budget_period=p.budget_period, gym_access=p.gym_access,
            activity_level=p.activity, diet_preference=p.diet_preference,
            email=p.email,
        )
        return {"ok": True, "profile": _profile_out(u)}
    return _guard(run)


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
@app.get("/api/dashboard")
def dashboard(user: str | None = None):
    def run():
        u = _user(user)
        if not u:
            return {"error": "no profile yet"}
        bmr = fe.calculate_bmr(u["weight_kg"], u["height_cm"], u["age"], u["gender"])
        tdee = fe.calculate_tdee(bmr, u.get("activity_level"))
        target = fe.calculate_calorie_target(tdee, u.get("goal"))
        macros = fe.macro_split(target, u.get("goal"))
        h = (u["height_cm"] or 0) / 100.0
        bmi = round((u["weight_kg"] / (h * h)), 1) if h else None

        df = db.get_progress_df(u["id"])
        wchange = 0.0
        if len(df) >= 2:
            wchange = round(float(df["weight_kg"].iloc[-1] - df["weight_kg"].iloc[0]), 1)

        return {
            "calorie_target": target, "protein_target": macros["protein_g"],
            "tdee": round(tdee), "bmi": bmi,
            "streak_days": int(len(df)), "weight_change_kg": wchange,
            "activity": [
                {"icon": "activity", "text": "Profile & targets ready", "when": "now"},
                {"icon": "scale", "text": f"{len(df)} progress entries logged", "when": "history"},
            ],
        }
    return _guard(run)


# --------------------------------------------------------------------------- #
# fitness
# --------------------------------------------------------------------------- #
class AssessIn(BaseModel):
    age: int = 27
    gender: str = "Female"
    height_cm: float = 164
    weight_kg: float = 61
    goal: str = "Lose fat"
    activity: str = "Moderately active (exercise 4-5 days/week)"
    diet_preference: str = "No preference"
    budget_pkr: float = 1200
    budget_period: str = "daily"
    gym_access: str = "No gym (bodyweight/home only)"


# gym_access (from the Fitness Assessment form) -> the workout_location +
# equipment overrides fitness_engine.workout_profile() consumes when building
# the Gemini workout prompt. "No gym" is the safe default for an unrecognised
# value (never assume equipment the user didn't say they have).
def _gym_access_fields(gym_access: str) -> dict:
    g = (gym_access or "").lower()
    if "full gym" in g:
        return {
            "gym_access": True, "workout_location": "Gym",
            "equipment": ["Barbell", "Dumbbells", "Bench", "Cable machine",
                         "Pull-up bar", "Leg press", "Squat rack"],
        }
    if "home gym" in g:
        return {
            "gym_access": False, "workout_location": "Home",
            "equipment": ["Dumbbells", "Resistance bands", "Pull-up bar",
                         "Backpack (loadable)"],
        }
    return {  # "No gym" / anything else -> bodyweight only, no assumptions
        "gym_access": False, "workout_location": "Home",
        "equipment": ["None (bodyweight only)"],
    }


def _adapt_meal_plan(mp: dict) -> dict:
    """Normalise fitness_engine's meal plan to what the UI renders:
    days:[{day, meals:[[slot, text]]}], daily_cost_pkr, within_budget."""
    days_out = []
    for d in (mp.get("days") or [])[:7]:
        meals = []
        for m in (d.get("meals") or []):
            slot = str(m.get("slot") or m.get("name") or "Meal").title()
            text = str(m.get("name") or m.get("text") or "")
            tags = []
            if m.get("kcal"):    tags.append(f"{m['kcal']} kcal")
            if m.get("protein"): tags.append(f"{m['protein']} g protein")
            if tags:
                text += f"  ·  {' / '.join(tags)}"
            meals.append([slot, text])
        days_out.append({"day": str(d.get("day") or f"Day {len(days_out) + 1}"), "meals": meals})
    val = mp.get("validation") or {}
    return {
        "days": days_out,
        "daily_cost_pkr": round(val.get("daily_cost") or mp.get("avg_daily_cost") or 0),
        "within_budget": bool(val.get("within_budget", True)),
        "note": (mp.get("notes") or [None])[0],
        "error": mp.get("error"),
        "min_budget": mp.get("min_budget"),
    }


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _adapt_workout(w: dict) -> dict:
    """fitness_engine returns week:{ 'Monday': {focus, exercises:[str]} , ... }."""
    days = []
    week = w.get("week")
    if isinstance(week, dict):
        ordered = [k for k in _WEEKDAYS if k in week] + [k for k in week if k not in _WEEKDAYS]
        for name in ordered:
            d = week[name] or {}
            items = [str(x) for x in (d.get("exercises") or [])]
            if not items and d.get("focus"):
                items = [str(d["focus"])]            # rest / light days
            days.append({"day": name, "items": items})
    elif isinstance(w.get("days"), list):
        for d in w["days"]:
            raw = d.get("exercises") or d.get("items") or []
            items = [(f"{x.get('name','Exercise')} {x.get('sets','')}×{x.get('reps','')}".strip()
                      if isinstance(x, dict) else str(x)) for x in raw]
            days.append({"day": str(d.get("day") or d.get("name") or f"Day {len(days) + 1}"), "items": items})
    split = " · ".join(str(x) for x in (w.get("location"), w.get("goal_focus"),
                                        w.get("experience_level")) if x) or \
            str(w.get("split") or w.get("summary") or "Weekly routine")
    return {"split": split, "days": days}


@app.post("/api/fitness/assess")
def fitness_assess(a: AssessIn, dynamic: int = 1):
    def run():
        # Backend budget validation - never trust the frontend's min/clamp alone.
        # Same floor fitness_engine enforces (PKR 500/day == PKR 15,000/month);
        # checked again here, up front, so a bypassed/forged request still gets
        # a clear error instead of silently producing an infeasible plan.
        daily_budget = (a.budget_pkr / 30.0) if a.budget_period == "monthly" else a.budget_pkr
        budget_err = fe.validate_budget(daily_budget)
        if budget_err:
            return {"error": budget_err}

        user = {
            "age": a.age, "gender": a.gender, "height_cm": a.height_cm,
            "weight_kg": a.weight_kg, "goal": a.goal, "activity_level": a.activity,
            "diet_preference": a.diet_preference, "budget_pkr": a.budget_pkr,
            "budget_period": a.budget_period, **_gym_access_fields(a.gym_access),
        }
        # dynamic=1 (default) -> Gemini-personalised workout that actually uses
        # gym_access/equipment (validated, 7-day cached, static fallback on any
        # failure). dynamic=0 -> instant static routine (only 2-tier: Home/Gym).
        res = fe.full_assessment(user, use_csv=True, dynamic_workout=bool(dynamic))
        return {
            "bmr": res["bmr"], "tdee": res["tdee"],
            "calorie_target": res["calorie_target"], "macros": res["macros"],
            "meal_plan": _adapt_meal_plan(res.get("meal_plan") or {}),
            "workout": _adapt_workout(res.get("workout") or {}),
        }
    return _guard(run)


# --------------------------------------------------------------------------- #
# nutritionist chat
# --------------------------------------------------------------------------- #
class ChatIn(BaseModel):
    message: str
    user: str | None = None


@app.get("/api/nutrition/history")
def chat_history(user: str | None = None):
    def run():
        u = _user(user)
        if not u:
            return {"messages": [{"role": "bot", "content": "Create a profile to start chatting."}]}
        msgs = db.get_chat(u["id"]) or []
        return {"messages": [{"role": "bot" if m["role"] != "user" else "user",
                              "content": m["content"]} for m in msgs]}
    return _guard(run)


@app.post("/api/nutrition/chat")
def chat_send(c: ChatIn):
    def run():
        u = _user(c.user)
        if not u:
            return {"role": "bot", "content": "Create a profile first so I can tailor answers."}
        db.add_chat(u["id"], "user", c.message)
        convo = [{"role": m["role"], "content": m["content"]} for m in db.get_chat(u["id"])]
        print(f"[nutrition/chat] user={c.user!r} message={c.message!r} "
              f"gemini_available={nutritionist.is_available()}")
        reply = nutritionist.chat(convo, dict(u))
        db.add_chat(u["id"], "assistant", reply)
        return {"role": "bot", "content": reply}
    return _guard(run)


@app.post("/api/nutrition/clear")
def chat_clear(user: str | None = None):
    def run():
        u = _user(user)
        if u:
            db.clear_chat(u["id"])
        return {"ok": True, "messages": []}
    return _guard(run)


# --------------------------------------------------------------------------- #
# skin analysis
# --------------------------------------------------------------------------- #
_SKIN_SEV_N = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3, "unknown": 0, "none": 0}
_SKIN_NAMES = {"redness": "Redness / inflammation",
               "hyperpigmentation": "Hyperpigmentation / dark spots",
               "texture": "Uneven texture", "blemishes": "Blemishes / breakouts"}


def _skin_recommendations(r: dict) -> dict:
    """Reproduce app.py._adapt_skin_ai's recommendation wiring: derive the care
    level + conditions from the analyzer output, then pull real products,
    supplements and diet tips from skincare_brands.csv via skin_analyzer."""
    import skin_analyzer
    from skin_ai.recommendations import context_from_conditions, deduplicate_supplements

    sev = {"redness": r.get("redness_severity", "clear"),
           "hyperpigmentation": r.get("hyperpigmentation_severity", "clear"),
           "texture": r.get("texture_severity", "clear")}
    bc = int(r.get("blemish_count", 0) or 0)
    levels = {k: _SKIN_SEV_N.get(v, 0) for k, v in sev.items()}
    levels["blemishes"] = 0 if bc == 0 else 1 if bc <= 4 else 2 if bc <= 12 else 3
    worst = max(levels.values()) if levels else 0
    care = "treat" if worst >= 3 else "light" if worst >= 2 else "basic"
    conditions = [_SKIN_NAMES[k] for k, lv in levels.items() if lv >= 2]
    skin_type = r.get("skin_type", "Normal")

    recs, src = skin_analyzer._recommend(skin_type, conditions, mode="csv", level=care)

    supps = recs.get("supplements") or []
    supp_note = recs.get("safety_note", "")
    if supps:
        dd = deduplicate_supplements(supps, context=context_from_conditions(conditions))
        recs["supplements"] = dd.get("supplements", supps)
        supp_note = dd.get("safety_note", supp_note)

    # one recommended pick per routine step (real brand + product + price)
    routine = []
    for g in (recs.get("product_options") or []):
        opts = g.get("options") or []
        if not opts:
            continue
        top = opts[0]
        alts = (opts[1:] or []) + (g.get("more_options") or [])
        routine.append({
            "step": g.get("category", ""),
            "product": f"{top.get('brand', '')} {top.get('product', '')}".strip() or "See options",
            "price_pkr": top.get("price_pkr"),
            "why": top.get("why") or g.get("usage", ""),
            "where_to_buy": top.get("where_to_buy", ""),
            "alternatives": [
                {"name": f"{o.get('brand', '')} {o.get('product', '')}".strip(),
                 "price_pkr": o.get("price_pkr")}
                for o in alts[:4] if (o.get("brand") or o.get("product"))
            ],
        })

    supplements = [
        {"name": s.get("name") or f"{s.get('brand', '')} {s.get('product', '')}".strip(),
         "price_pkr": s.get("price_pkr"),
         "dosage": s.get("dosage", ""), "timing": s.get("timing", ""),
         "why": s.get("why") or s.get("note", "")}
        for s in (recs.get("supplements") or [])
    ]

    return {
        "care_level": care,
        "conditions": conditions,
        "routine": routine,
        "supplements": supplements,
        "supplement_note": supp_note,
        "diet": list(recs.get("diet_changes") or []),
        "safety_note": recs.get("safety_note", ""),
        "brands_note": recs.get("brands_note", ""),
        "source": src,
    }


@app.post("/api/skin/analyze")
async def skin_analyze(request: Request):
    """Accepts either multipart/form-data (field `image`) or JSON
    {"image": "<base64>"} (a `data:` URL prefix is stripped)."""
    ct = request.headers.get("content-type", "")
    raw = b""
    if ct.startswith("multipart/form-data"):
        form = await request.form()
        up = form.get("image")
        if up is not None and hasattr(up, "read"):
            raw = await up.read()
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        b64 = (body or {}).get("image", "") or ""
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64, validate=False) if b64 else b""
        except Exception:
            raw = b""
    if not raw:
        return JSONResponse({"error": "no image provided"}, status_code=200)

    def run():
        from PIL import Image
        from skin_ai.inference import SkinAnalyzer

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        r = SkinAnalyzer(model_path="skin_ai/skin_model.pth").analyze(img)
        if not r.get("assessed", False):
            return {"error": r.get("message") or r.get("reason") or "Could not assess this photo."}
        conf = r.get("skin_type_confidence")
        _sev = lambda v: v if v in ("clear", "mild", "moderate", "severe") else "clear"
        concerns = []
        for key, label in (("redness_severity", "Redness"),
                           ("hyperpigmentation_severity", "Pigmentation"),
                           ("texture_severity", "Uneven texture")):
            concerns.append({"name": label, "severity": _sev(r.get(key))})
        bc = int(r.get("blemish_count", 0) or 0)
        concerns.insert(0, {"name": "Blemishes",
                            "severity": "clear" if bc == 0 else "mild" if bc <= 4
                            else "moderate" if bc <= 12 else "severe"})

        reco = {}
        try:
            reco = _skin_recommendations(r)
        except Exception as exc:  # never let reco failure sink the analysis
            traceback.print_exc()
            reco = {"routine": [], "supplements": [], "diet": [],
                    "reco_error": f"{type(exc).__name__}: {exc}"}

        return {
            "skin_type": r.get("skin_type", "Combination"),
            "confidence": round((conf or 0) * 100) if conf is not None else 60,
            "reasoning": r.get("skin_type_reasoning") or r.get("skin_type_source") or "",
            "blemish_count": bc,
            "concerns": concerns,
            "care_level": reco.get("care_level", "basic"),
            "conditions": reco.get("conditions", []),
            "routine": reco.get("routine", []),
            "supplements": reco.get("supplements", []),
            "supplement_note": reco.get("supplement_note", ""),
            "diet": reco.get("diet", []),
            "safety_note": reco.get("safety_note", ""),
            "brands_note": reco.get("brands_note", ""),
            "source": reco.get("source", "csv"),
            "disclaimer": r.get("disclaimer", ""),
        }
    return _guard(run)


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
class ProgressIn(BaseModel):
    weight_kg: float
    waist_cm: float | None = None
    chest_cm: float | None = None
    user: str | None = None


def _entries(u: dict) -> list[dict]:
    df = db.get_progress_df(u["id"])
    out = []
    for _, row in df.iterrows():
        out.append({
            "date": str(row.get("date", ""))[:10],
            "weight_kg": _num(row.get("weight_kg")),
            "waist_cm": _num(row.get("waist_cm")),
            "chest_cm": _num(row.get("chest_cm")),
        })
    return out


def _num(v):
    try:
        f = float(v)
        return None if f != f else round(f, 1)   # NaN -> None
    except (TypeError, ValueError):
        return None


@app.get("/api/progress/entries")
def progress_list(user: str | None = None):
    def run():
        u = _user(user)
        return {"entries": _entries(u) if u else []}
    return _guard(run)


@app.post("/api/progress/entries")
def progress_add(p: ProgressIn):
    def run():
        u = _user(p.user)
        if not u:
            return {"error": "no profile"}
        from datetime import date as _date
        db.add_progress(u["id"], _date.today().isoformat(), p.weight_kg,
                        p.chest_cm, p.waist_cm, None, None, None, None, None)
        return {"ok": True, "entries": _entries(u)}
    return _guard(run)


@app.get("/api/health")
def health():
    return {"ok": True, "gemini": nutritionist.is_available()}
