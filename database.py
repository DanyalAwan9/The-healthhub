"""SQLite persistence layer for HealthHub.

Tables: users, assessments, progress, chat_history, skin_analyses.
All photo uploads are written to data/photos/ and only the path is stored.
"""
import os
import json
import sqlite3
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("HEALTHHUB_DB", os.path.join(BASE_DIR, "healthhub.db"))
PHOTO_DIR = os.path.join(BASE_DIR, "data", "photos")
os.makedirs(PHOTO_DIR, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            email TEXT,
            age INTEGER,
            gender TEXT,
            height_cm REAL,
            weight_kg REAL,
            goal TEXT,
            budget_pkr REAL,
            budget_period TEXT DEFAULT 'daily',
            gym_access INTEGER DEFAULT 0,
            activity_level TEXT,
            diet_preference TEXT DEFAULT 'No preference',
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT,
            bmr REAL,
            tdee REAL,
            calorie_target REAL,
            protein_target REAL,
            macros_json TEXT,
            meal_plan_json TEXT,
            workout_json TEXT,
            recommendations_json TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT,
            weight_kg REAL,
            chest_cm REAL,
            waist_cm REAL,
            hips_cm REAL,
            arms_cm REAL,
            thighs_cm REAL,
            photo_path TEXT,
            notes TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS skin_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT,
            skin_type TEXT,
            condition TEXT,
            details_json TEXT,
            photo_path TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS price_cache (
            key TEXT PRIMARY KEY,
            category TEXT,
            value_json TEXT,
            created_at TEXT
        );
        """
    )
    # lightweight migrations for DBs created before a column existed
    ucols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    if "diet_preference" not in ucols:
        conn.execute("ALTER TABLE users ADD COLUMN diet_preference TEXT DEFAULT 'No preference'")
    if "email" not in ucols:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")

    acols = {r["name"] for r in conn.execute("PRAGMA table_info(assessments)")}
    if "date" not in acols:
        conn.execute("ALTER TABLE assessments ADD COLUMN date TEXT")
    if "protein_target" not in acols:
        conn.execute("ALTER TABLE assessments ADD COLUMN protein_target REAL")
    if "recommendations_json" not in acols:
        conn.execute("ALTER TABLE assessments ADD COLUMN recommendations_json TEXT")
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Users / profiles
# --------------------------------------------------------------------------- #
def upsert_user(name, age, gender, height_cm, weight_kg, goal, budget_pkr,
                budget_period, gym_access, activity_level,
                diet_preference="No preference", email: str | None = None) -> dict:
    """Registration/login is by unique `name`; `email` is an optional profile
    field (also looked up via get_user_by_email). Passing email=None on an
    update leaves the previously-saved email untouched."""
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO users (name, email, age, gender, height_cm, weight_kg, goal, budget_pkr,
                           budget_period, gym_access, activity_level, diet_preference,
                           created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            email=COALESCE(excluded.email, users.email),
            age=excluded.age,
            gender=excluded.gender,
            height_cm=excluded.height_cm,
            weight_kg=excluded.weight_kg,
            goal=excluded.goal,
            budget_pkr=excluded.budget_pkr,
            budget_period=excluded.budget_period,
            gym_access=excluded.gym_access,
            activity_level=excluded.activity_level,
            diet_preference=excluded.diet_preference
        """,
        (name, (email or "").strip() or None, age, gender, height_cm, weight_kg, goal,
         budget_pkr, budget_period, int(bool(gym_access)), activity_level,
         diet_preference or "No preference", _now()),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    conn.close()
    return dict(row)


def get_user_by_name(name):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users():
    conn = get_conn()
    rows = conn.execute("SELECT name FROM users ORDER BY name").fetchall()
    conn.close()
    return [r["name"] for r in rows]


def delete_user(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Assessments
# --------------------------------------------------------------------------- #
def save_assessment(user_id, bmr, tdee, calorie_target, macros, meal_plan, workout,
                    protein_target=None, recommendations=None, date=None):
    """`protein_target` defaults to macros['protein_g'] when not given.
    `recommendations` is any small JSON-able summary (diet/workout notes,
    budget validation, ...) - kept separate from the full meal_plan/workout
    blobs so a history view can render it without re-parsing those."""
    if protein_target is None and isinstance(macros, dict):
        protein_target = macros.get("protein_g")
    conn = get_conn()
    conn.execute(
        """INSERT INTO assessments
           (user_id, date, bmr, tdee, calorie_target, protein_target, macros_json,
            meal_plan_json, workout_json, recommendations_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, date or _now()[:10], bmr, tdee, calorie_target, protein_target,
         json.dumps(macros), json.dumps(meal_plan), json.dumps(workout),
         json.dumps(recommendations) if recommendations is not None else None, _now()),
    )
    conn.commit()
    conn.close()


def _decode_assessment(row) -> dict:
    d = dict(row)
    d["macros"] = json.loads(d.pop("macros_json") or "{}")
    d["meal_plan"] = json.loads(d.pop("meal_plan_json") or "[]")
    d["workout"] = json.loads(d.pop("workout_json") or "{}")
    rec = d.pop("recommendations_json", None)
    d["recommendations"] = json.loads(rec) if rec else {}
    return d


def get_latest_assessment(user_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
    ).fetchone()
    conn.close()
    return _decode_assessment(row) if row else None


def get_assessment_history(user_id, limit: int = 20):
    """Past assessments, newest first - id/date/targets only decoded eagerly;
    full macros/meal_plan/workout/recommendations are still included per row."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [_decode_assessment(r) for r in rows]


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
def add_progress(user_id, date, weight_kg, chest_cm, waist_cm, hips_cm,
                 arms_cm, thighs_cm, photo_path, notes):
    conn = get_conn()
    conn.execute(
        """INSERT INTO progress
           (user_id, date, weight_kg, chest_cm, waist_cm, hips_cm, arms_cm, thighs_cm,
            photo_path, notes, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, date, weight_kg, chest_cm, waist_cm, hips_cm, arms_cm, thighs_cm,
         photo_path, notes, _now()),
    )
    conn.commit()
    conn.close()


def get_progress_df(user_id) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM progress WHERE user_id=? ORDER BY date, id",
            conn, params=(user_id,),
        )
    finally:
        conn.close()
    return df


def delete_progress(entry_id):
    conn = get_conn()
    conn.execute("DELETE FROM progress WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #
def add_chat(user_id, role, content):
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_history (user_id, role, content, created_at) VALUES (?,?,?,?)",
        (user_id, role, content, _now()),
    )
    conn.commit()
    conn.close()


def get_chat(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY id", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_chat(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Skin analyses
# --------------------------------------------------------------------------- #
def add_skin_analysis(user_id, date, skin_type, condition, details, photo_path):
    conn = get_conn()
    conn.execute(
        """INSERT INTO skin_analyses
           (user_id, date, skin_type, condition, details_json, photo_path, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (user_id, date, skin_type, condition, json.dumps(details), photo_path, _now()),
    )
    conn.commit()
    conn.close()


def get_skin_analyses(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM skin_analyses WHERE user_id=? ORDER BY id DESC", (user_id,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d.pop("details_json") or "{}")
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Web price cache
# --------------------------------------------------------------------------- #
def get_cached_price(key: str, max_age_days: int = 7):
    conn = get_conn()
    row = conn.execute(
        "SELECT value_json, created_at FROM price_cache WHERE key=?", (key,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    try:
        age = datetime.now() - datetime.fromisoformat(row["created_at"])
        if age.days > max_age_days:
            return None
        return json.loads(row["value_json"])
    except Exception:
        return None


def set_cached_price(key: str, category: str, value) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO price_cache (key, category, value_json, created_at)
           VALUES (?,?,?,?)
           ON CONFLICT(key) DO UPDATE SET
             category=excluded.category,
             value_json=excluded.value_json,
             created_at=excluded.created_at""",
        (key, category, json.dumps(value), _now()),
    )
    conn.commit()
    conn.close()


def clear_price_cache(category: str | None = None) -> None:
    conn = get_conn()
    if category:
        conn.execute("DELETE FROM price_cache WHERE category=?", (category,))
    else:
        conn.execute("DELETE FROM price_cache")
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Photo helper
# --------------------------------------------------------------------------- #
def save_uploaded_photo(file_bytes: bytes, filename: str, prefix: str = "img") -> str:
    ext = os.path.splitext(filename or "")[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(PHOTO_DIR, f"{prefix}_{stamp}{ext}")
    with open(path, "wb") as fh:
        fh.write(file_bytes)
    return path
