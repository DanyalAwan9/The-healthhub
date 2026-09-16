import { useState } from 'react';
import Icon from '../components/Icon';
import { useToast } from '../context/ToastContext';
import { fitnessAssess } from '../api/client';
import { MOCK_FITNESS } from '../api/mockData';

const fmt = (n) => (n == null || Number.isNaN(n) ? '—' : Number(n).toLocaleString());

// Harris-Benedict-style activity multipliers (kept in sync with
// fitness_engine.ACTIVITY_MULTIPLIERS — the label text IS the lookup key).
const ACTIVITY_OPTIONS = [
  ['Sedentary (little to no exercise, desk job)', 1.2],
  ['Lightly active (exercise 1-3 days/week)', 1.375],
  ['Moderately active (exercise 4-5 days/week)', 1.55],
  ['Very active (exercise 6-7 days/week)', 1.725],
  ['Extremely active (intense daily training + physical job)', 1.9],
];

const DIET_OPTIONS = [
  'No preference', 'Vegetarian', 'Vegan', 'Halal only',
  'Eggetarian (vegetarian + eggs)', 'High protein focus', 'Low carb', 'Dairy-free',
];

const GYM_OPTIONS = [
  'No gym (bodyweight/home only)',
  'Home gym (dumbbells/resistance bands only)',
  'Full gym access (barbells, machines, cables)',
];

// Same hard floor the backend enforces (fitness_engine.MIN_DAILY_BUDGET_PKR /
// MIN_MONTHLY_BUDGET_PKR) — PKR 500/day == PKR 15,000/month. Frontend clamping
// is a UX nicety only; the backend re-checks this on every request regardless.
const MIN_BUDGET = { daily: 500, monthly: 15000 };

const DEFAULTS = {
  age: 27, gender: 'Female', height_cm: 164, weight_kg: 61, goal: 'Lose fat',
  activity: ACTIVITY_OPTIONS[2][0], diet_preference: 'No preference',
  budget_pkr: 1200, budget_period: 'daily',
  gym_access: GYM_OPTIONS[0],
};

export default function FitnessAssessment() {
  const { show } = useToast();
  const [form, setForm] = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [live, setLive] = useState(true);
  const [dayIndex, setDayIndex] = useState(0);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const clampBudget = (value, period) => {
    const min = MIN_BUDGET[period] ?? MIN_BUDGET.daily;
    const n = Number(value);
    return !value || Number.isNaN(n) || n < min ? min : n;
  };

  const onBudgetBlur = () => setForm((f) => ({ ...f, budget_pkr: clampBudget(f.budget_pkr, f.budget_period) }));

  const onPeriodChange = (e) => {
    const period = e.target.value;
    setForm((f) => {
      // carry the intent across day<->month rather than just re-clamping the raw number
      const converted = period === 'monthly' ? Number(f.budget_pkr) * 30 : Number(f.budget_pkr) / 30;
      return { ...f, budget_period: period, budget_pkr: clampBudget(Math.round(converted), period) };
    });
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const payload = {
      ...form,
      age: Number(form.age), height_cm: Number(form.height_cm),
      weight_kg: Number(form.weight_kg),
      budget_pkr: clampBudget(form.budget_pkr, form.budget_period),   // never trust unblurred input
    };
    try {
      const { data } = await fitnessAssess(payload);
      setResult(data);
      setLive(true);
    } catch {
      setResult(MOCK_FITNESS);
      setLive(false);
      show('Could not reach the assessment service — showing demo data.', 'warn');
    } finally {
      setDayIndex(0);
      setLoading(false);
    }
  };

  const mp = result?.meal_plan || { days: [] };
  const wk = result?.workout || { days: [] };
  const activeDay = mp.days?.[dayIndex];
  const minBudget = MIN_BUDGET[form.budget_period] ?? MIN_BUDGET.daily;

  return (
    <section className="page">
      <div className="page__head">
        <h1>Fitness Assessment</h1>
        <p className="muted">We turn your stats and budget into a calorie target, meal plan and workout.</p>
      </div>

      <div className="stepper">
        <span className={`stepper__item ${result ? 'is-done' : 'is-current'}`}><span className="stepper__num">1</span> Your details</span>
        <span className="stepper__bar" />
        <span className={`stepper__item ${result ? 'is-current' : ''}`}><span className="stepper__num">2</span> Your plan</span>
      </div>

      <div className="card card--pad-lg">
        <form className="form" onSubmit={submit}>
          <fieldset className="fieldset">
            <legend>Body</legend>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="f-age">Age</label>
                <input className="input" id="f-age" type="number" min="14" max="90" value={form.age} onChange={set('age')} required />
              </div>
              <div className="field">
                <label htmlFor="f-gender">Gender</label>
                <select className="select" id="f-gender" value={form.gender} onChange={set('gender')}>
                  <option>Female</option><option>Male</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="f-h">Height (cm)</label>
                <input className="input" id="f-h" type="number" min="120" max="220" value={form.height_cm} onChange={set('height_cm')} required />
              </div>
              <div className="field">
                <label htmlFor="f-w">Weight (kg)</label>
                <input className="input" id="f-w" type="number" min="35" max="200" step="0.1" value={form.weight_kg} onChange={set('weight_kg')} required />
              </div>
            </div>
          </fieldset>

          <fieldset className="fieldset">
            <legend>Goal &amp; lifestyle</legend>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="f-goal">Goal</label>
                <select className="select" id="f-goal" value={form.goal} onChange={set('goal')}>
                  <option>Lose fat</option><option>Maintain</option><option>Build muscle</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="f-act">Activity level</label>
                <select className="select" id="f-act" value={form.activity} onChange={set('activity')}>
                  {ACTIVITY_OPTIONS.map(([label]) => <option key={label}>{label}</option>)}
                </select>
                <span className="hint">Sets your TDEE activity multiplier (1.2–1.9).</span>
              </div>
              <div className="field">
                <label htmlFor="f-diet">Diet preference</label>
                <select className="select" id="f-diet" value={form.diet_preference} onChange={set('diet_preference')}>
                  {DIET_OPTIONS.map((label) => <option key={label}>{label}</option>)}
                </select>
              </div>
              <div className="field">
                <label htmlFor="f-gym">Gym access</label>
                <select className="select" id="f-gym" value={form.gym_access} onChange={set('gym_access')} required>
                  {GYM_OPTIONS.map((label) => <option key={label}>{label}</option>)}
                </select>
                <span className="hint">Determines whether your workout uses bodyweight, dumbbells, or full equipment.</span>
              </div>
            </div>
          </fieldset>

          <fieldset className="fieldset">
            <legend>Food budget</legend>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="f-bud">Budget (PKR)</label>
                <input
                  className="input" id="f-bud" type="number" min={minBudget} step="50"
                  value={form.budget_pkr} onChange={set('budget_pkr')} onBlur={onBudgetBlur} required
                />
              </div>
              <div className="field">
                <label htmlFor="f-per">Per</label>
                <select className="select" id="f-per" value={form.budget_period} onChange={onPeriodChange}>
                  <option value="daily">Day</option><option value="monthly">Month</option>
                </select>
              </div>
            </div>
            <p className="faint mt-1" style={{ fontSize: '.78rem' }}>
              Minimum Rs {fmt(MIN_BUDGET.monthly)}/month (Rs {fmt(MIN_BUDGET.daily)}/day) for balanced nutrition —
              meals are priced from a real grocery list and this floor is enforced on the server too.
            </p>
          </fieldset>

          <div className="row">
            <button className="btn btn-primary" type="submit" disabled={loading}>
              <Icon name="activity" className="ic--sm" /> {loading ? 'Building…' : 'Generate plan'}
            </button>
          </div>
        </form>
      </div>

      {result && result.error && (
        <div className="card mt-4" style={{ boxShadow: 'inset 4px 0 0 var(--danger)' }}>
          <div className="row" style={{ gap: 10 }}>
            <Icon name="alert" style={{ color: 'var(--danger)' }} />
            <div>
              <div style={{ fontWeight: 700 }}>Budget too low</div>
              <p className="muted mt-1" style={{ marginTop: 4 }}>{result.error}</p>
            </div>
          </div>
        </div>
      )}

      {result && !result.error && (
        <div className="mt-4">
          <h2 className="mb-2">Your plan</h2>
          {!live && (
            <div className="mb-3">
              <span className="badge badge--info"><Icon name="alert" className="ic--sm" /> Demo data — backend API not connected</span>
            </div>
          )}
          <div className="grid cols-4">
            <div className="stat"><span className="stat__label">Calories</span><span className="stat__value">{fmt(result.calorie_target)}</span><span className="muted">kcal</span></div>
            <div className="stat"><span className="stat__label">Protein</span><span className="stat__value">{fmt(result.macros?.protein_g)}</span><span className="muted">g</span></div>
            <div className="stat"><span className="stat__label">Carbs</span><span className="stat__value">{fmt(result.macros?.carbs_g)}</span><span className="muted">g</span></div>
            <div className="stat"><span className="stat__label">Fat</span><span className="stat__value">{fmt(result.macros?.fat_g)}</span><span className="muted">g</span></div>
          </div>

          <div className="grid cols-2 mt-3">
            <div className="card">
              <div className="card__head">
                <span className="card__title">7-day meal plan</span>
                {!mp.error && (
                  <span className={`badge ${mp.within_budget ? 'badge--ok' : 'badge--warn'}`}>
                    <Icon name="check" className="ic--sm" /> PKR {fmt(mp.daily_cost_pkr)} / day · {mp.within_budget ? 'within budget' : 'over budget'}
                  </span>
                )}
              </div>
              {mp.error ? (
                <div>
                  <p className="muted">{mp.error}</p>
                  {mp.min_budget != null && (
                    <p className="mt-2">
                      Try at least <b>PKR {fmt(mp.min_budget)}/day</b> (PKR {fmt(mp.min_budget * 30)}/month) for this target.
                    </p>
                  )}
                </div>
              ) : (
                <>
                  <div className="row" style={{ marginBottom: 12 }}>
                    {mp.days.map((day, i) => (
                      <button key={day.day + i} type="button" className={`chip${i === dayIndex ? ' is-active' : ''}`} onClick={() => setDayIndex(i)}>
                        {day.day}
                      </button>
                    ))}
                  </div>
                  {activeDay && activeDay.meals.map(([slot, txt]) => (
                    <div className="kv" key={slot}>
                      <dt>{slot}</dt>
                      <dd style={{ fontWeight: 500, textAlign: 'right', maxWidth: '60%' }}>{txt}</dd>
                    </div>
                  ))}
                </>
              )}
            </div>
            <div className="card">
              <div className="card__head">
                <span className="card__title">Workout</span>
                <span className="badge badge--primary">{wk.split}</span>
              </div>
              <div className="stack" id="fit-workout">
                {wk.days.map((day) => (
                  <div className="card" style={{ boxShadow: 'none' }} key={day.day}>
                    <div className="card__title" style={{ fontSize: '.95rem', marginBottom: 10, color: 'var(--accent)' }}>{day.day}</div>
                    <ul className="list-plain">
                      {day.items.map((it, i) => (
                        <li key={i}><Icon name="check" className="ic--sm" /><span>{it}</span></li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
