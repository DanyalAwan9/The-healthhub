"""Progress-tracking helpers: charts, tables and progress analysis."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MEASUREMENT_COLS = ["chest_cm", "waist_cm", "hips_cm", "arms_cm", "thighs_cm"]


def weight_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if df is not None and not df.empty and df["weight_kg"].notna().any():
        d = df.dropna(subset=["weight_kg"]).copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.sort_values("date")
        ax.plot(d["date"], d["weight_kg"], marker="o", linewidth=2, color="#2563eb")
        if len(d) >= 2:
            ax.fill_between(d["date"], d["weight_kg"].min(), d["weight_kg"],
                            alpha=0.08, color="#2563eb")
    else:
        ax.text(0.5, 0.5, "No weight data yet", ha="center", va="center",
                transform=ax.transAxes, color="#888")
    ax.set_title("Weight over time")
    ax.set_ylabel("Weight (kg)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def measurements_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    plotted = False
    if df is not None and not df.empty:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.sort_values("date")
        for col in MEASUREMENT_COLS:
            if col in d and d[col].notna().any():
                ax.plot(d["date"], d[col], marker="o", label=col.replace("_cm", "").title())
                plotted = True
    if plotted:
        ax.legend(fontsize=8, ncol=3)
    else:
        ax.text(0.5, 0.5, "No measurement data yet", ha="center", va="center",
                transform=ax.transAxes, color="#888")
    ax.set_title("Body measurements over time")
    ax.set_ylabel("cm")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def measurements_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["date", "weight_kg"] + MEASUREMENT_COLS + ["notes"]
    view = df[[c for c in cols if c in df.columns]].copy()
    return view.sort_values("date", ascending=False).reset_index(drop=True)


def photo_entries(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty or "photo_path" not in df.columns:
        return []
    d = df.dropna(subset=["photo_path"])
    d = d[d["photo_path"].astype(str).str.len() > 0]
    return d.sort_values("date")[["date", "photo_path", "weight_kg"]].to_dict("records")


def analyze_progress(df: pd.DataFrame, user: dict | None = None) -> str:
    if df is None or len(df) < 2:
        return "Log at least two entries (a week or more apart) to get progress analysis."

    d = df.dropna(subset=["weight_kg"]).copy()
    if len(d) < 2:
        return "Need at least two weigh-ins to analyse weight trend."
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date")

    first, last = d.iloc[0], d.iloc[-1]
    days = max(1, (last["date"] - first["date"]).days)
    delta = last["weight_kg"] - first["weight_kg"]
    per_week = delta / days * 7
    goal = (user or {}).get("goal", "") or ""

    lines = [
        f"**Span:** {days} days, {len(d)} weigh-ins.",
        f"**Total change:** {delta:+.1f} kg  ({per_week:+.2f} kg/week).",
    ]

    losing = "loss" in goal.lower() or goal.lower().startswith("lose")
    gaining = "muscle" in goal.lower() or "gain" in goal.lower()

    if losing:
        if -1.0 <= per_week <= -0.25:
            lines.append("On track — a 0.25-1.0 kg/week loss is a healthy, sustainable rate.")
        elif per_week < -1.0:
            lines.append("Losing quite fast (>1 kg/week). Add ~150-250 kcal/day and keep "
                         "protein high to protect muscle.")
        elif per_week > 0:
            lines.append("Weight is trending up despite a fat-loss goal. Tighten portion "
                         "sizes, cut sugary chai / soft drinks, and log intake for a week.")
        else:
            lines.append("Loss has stalled. Recheck portions, step count, and weekend meals; "
                         "a small further calorie cut may be needed.")
    elif gaining:
        if 0.1 <= per_week <= 0.5:
            lines.append("Good lean-gain pace (~0.1-0.5 kg/week).")
        elif per_week > 0.5:
            lines.append("Gaining fast — some of this is likely fat. Trim the surplus by "
                         "~150-200 kcal/day.")
        else:
            lines.append("Not gaining. Add ~250-300 kcal/day (extra roti, dahi, nuts, or a "
                         "shake) and make sure training is progressing.")
    else:
        if abs(per_week) <= 0.25:
            lines.append("Weight is stable, which matches a maintenance goal. Nice consistency.")
        else:
            lines.append(f"Weight is drifting {per_week:+.2f} kg/week. Nudge calories the "
                         "opposite way if you want to hold steady.")

    # measurement note
    for col in ("waist_cm", "chest_cm", "arms_cm"):
        s = d[col].dropna() if col in d else pd.Series(dtype=float)
        if len(s) >= 2:
            md = s.iloc[-1] - s.iloc[0]
            if abs(md) >= 0.5:
                lines.append(f"{col.replace('_cm', '').title()}: {md:+.1f} cm over the period.")

    lines.append("Keep weighing at the same time (morning, after toilet, before breakfast) "
                 "for cleaner data.")
    return "\n\n".join(lines)


def ai_feedback(df: pd.DataFrame, user: dict | None = None) -> str:
    """Rule-based analysis, optionally enriched by the Gemini nutritionist."""
    base = analyze_progress(df, user)
    try:
        import nutritionist

        if nutritionist.is_available() and df is not None and len(df) >= 2:
            summary = base.replace("**", "")
            prompt = (
                "Here is a user's progress summary. Give 3-4 short, encouraging, "
                "actionable bullet points (Pakistani food context, PKR prices where "
                "relevant). Do not repeat the numbers verbatim.\n\n" + summary
            )
            extra = nutritionist.chat([{"role": "user", "content": prompt}], user)
            return base + "\n\n---\n\n**AI coach:**\n\n" + extra
    except Exception:
        pass
    return base
