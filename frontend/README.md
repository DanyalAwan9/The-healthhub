# HealthHub — front-end

A standalone HTML / CSS / vanilla-JS front-end for HealthHub. No build step, no
framework. It talks to a REST API and **falls back to bundled demo data** when
the API isn't running, so every page works offline.

```
frontend/
├── index.html            landing page + top nav
├── pages/
│   ├── dashboard.html     stats + recent activity
│   ├── fitness.html       assessment form → calorie/macro/meal/workout plan
│   ├── nutrition.html     nutritionist chat (bubbles, typing, suggestions)
│   ├── skin.html          image upload → skin-type card + routine
│   ├── progress.html      trend sparkline, add-entry form, table, timeline
│   └── profile.html       editable profile form
├── styles.css            design system: tokens, layout shell, components, animations
├── app.js                theme · app shell · REST client + mocks · page controllers
└── assets/
    ├── logo.svg  favicon.svg
    └── icons.svg         feather-style icon sprite (referenced via <use>)
```

## Run it

Any static server works (the icon sprite needs `http://`, not `file://`):

```bash
cd frontend
python -m http.server 5173
# open http://localhost:5173
```

## Design

| | |
|---|---|
| **Primary** | `#1e3a8a` deep blue (→ `#3b82f6` in dark mode for contrast) |
| **Accent** | `#10b981` green |
| **Surfaces** | `#ffffff` / `#f1f5f9` light · `#0f172a` / `#1e293b` dark |
| **Text** | `#1f2937` light · `#f3f4f6` dark |
| **Type** | Inter (Google Fonts), 16px base, responsive `clamp()` headings |
| **Radius** | 10–22px · **Motion** 220ms `cubic-bezier(.4,0,.2,1)` |

- **Dark / light**: toggle in the top bar. Choice is saved to `localStorage`;
  with no choice it follows the OS (`prefers-color-scheme`). All colours are CSS
  custom properties on `:root` / `[data-theme]`.
- **Responsive**: sidebar collapses to an off-canvas drawer below 1024px; grids
  reflow at 860 / 620px; `prefers-reduced-motion` is respected.
- **Components**: cards, stat tiles, badges, buttons (solid / gradient / outline /
  ghost), forms, range, switch, stepper, progress bar, confidence ring, tables,
  chat bubbles, dropzone, timeline, skeleton loaders, toasts.

## Talking to the backend

`app.js` calls `${HEALTHHUB_API}` (default `http://localhost:8000/api`). Override
before `app.js` loads:

```html
<script>window.HEALTHHUB_API = "https://your-host/api";</script>
```

If a request fails or times out (12s), the matching entry from the `MOCK` map in
`app.js` is used and a small "Demo data" badge appears on the page.

### REST contract

| Method & path | Body | Returns |
|---|---|---|
| `GET  /api/profile` | — | `{name,email,gender,age,height_cm,weight_kg,goal,activity,budget_pkr,budget_period,diet_preference}` |
| `POST /api/profile` | profile object | `{ok,profile}` |
| `GET  /api/dashboard` | — | `{calorie_target,protein_target,tdee,bmi,streak_days,weight_change_kg,activity[]}` |
| `POST /api/fitness/assess` | `{age,gender,height_cm,weight_kg,goal,activity,diet_preference,budget_pkr,budget_period}` | `{bmr,tdee,calorie_target,macros,meal_plan:{days[{day,meals:[[slot,text]]}],daily_cost_pkr,within_budget},workout:{split,days[{day,items[]}]}}` |
| `GET  /api/nutrition/history` | — | `{messages:[{role,content}]}` |
| `POST /api/nutrition/chat` | `{message}` | `{role:"bot",content}` |
| `POST /api/nutrition/clear` | — | `{ok,messages:[]}` |
| `POST /api/skin/analyze` | `multipart/form-data` field `image` **or** JSON `{image:"<base64>"}` | `{skin_type,confidence,reasoning,concerns:[{name,severity}], care_level, routine:[{step,product,price_pkr,why,where_to_buy,alternatives}], supplements:[{name,price_pkr,dosage,timing,why}], supplement_note, diet:[…], brands_note}` |
| `GET  /api/progress/entries` | — | `{entries:[{date,weight_kg,waist_cm,chest_cm}]}` |
| `POST /api/progress/entries` | `{weight_kg,waist_cm,chest_cm}` | `{ok,entries[]}` |

A reference implementation of this contract over the existing HealthHub Python
modules lives in [`../backend/api.py`](../backend/api.py) (FastAPI):

```bash
pip install -r backend/requirements-api.txt
uvicorn backend.api:app --reload --port 8000     # from the repo root
```

> The original Streamlit app (`app.py`) is unchanged and still runs on its own —
> this front-end + `backend/api.py` are an additive, decoupled alternative.
