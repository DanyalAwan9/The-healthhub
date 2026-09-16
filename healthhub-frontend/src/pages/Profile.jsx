import { useEffect, useState } from 'react';
import { useUser } from '../context/UserContext';
import { useToast } from '../context/ToastContext';
import { getProfile, saveProfile } from '../api/client';

// Kept in sync with fitness_engine.ACTIVITY_MULTIPLIERS (backend/api.py) and
// the same list used on the Fitness Assessment page — the label text IS the
// lookup key the backend uses for the TDEE multiplier.
const ACTIVITY_OPTIONS = [
  'Sedentary (little to no exercise, desk job)',
  'Lightly active (exercise 1-3 days/week)',
  'Moderately active (exercise 4-5 days/week)',
  'Very active (exercise 6-7 days/week)',
  'Extremely active (intense daily training + physical job)',
];

const BLANK = {
  name: '', email: '', gender: 'Female', age: '', height_cm: '', weight_kg: '',
  goal: 'Lose fat', activity: ACTIVITY_OPTIONS[2], budget_pkr: '', budget_period: 'daily',
};

export default function Profile() {
  const { user, login } = useUser();
  const { show } = useToast();
  const [form, setForm] = useState(BLANK);
  const [mode, setMode] = useState('create');
  const [live, setLive] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!user) { setForm(BLANK); setMode('create'); return; }
      try {
        const { data } = await getProfile(user.name);
        if (cancelled) return;
        setLive(true);
        if (data && data.exists !== false && data.name && data.height_cm) {
          setMode('edit');
          setForm((f) => ({ ...f, ...data }));
        } else {
          // signed up but body stats not saved yet — stay in create mode, prefilled
          setMode('create');
          setForm((f) => ({ ...f, name: user.name, email: data?.email || user.email || '' }));
        }
      } catch {
        setLive(false);
        setMode(user.name ? 'create' : 'create');
        setForm((f) => ({ ...f, name: user.name, email: user.email || '' }));
      }
    })();
    return () => { cancelled = true; };
  }, [user]);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const submit = async (e) => {
    e.preventDefault();
    if (!form.name.trim()) { show('Enter a name for your profile.', 'err'); return; }
    setSaving(true);
    const payload = {
      ...form,
      age: form.age ? Number(form.age) : undefined,
      height_cm: form.height_cm ? Number(form.height_cm) : undefined,
      weight_kg: form.weight_kg ? Number(form.weight_kg) : undefined,
      budget_pkr: form.budget_pkr ? Number(form.budget_pkr) : undefined,
    };
    try {
      const { data } = await saveProfile(payload);
      const savedName = data?.profile?.name || form.name.trim();
      login({ name: savedName, email: form.email });
      setMode('edit');
      show(mode === 'edit' ? 'Profile updated.' : 'Profile created.', 'ok');
    } catch {
      show('Could not save profile.', 'err');
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="page">
      <div className="page__head">
        <h1>{mode === 'edit' ? 'Your profile' : 'Create your profile'}</h1>
        <p className="muted">
          {mode === 'edit'
            ? 'Update your details — every target recalculates from these.'
            : 'Fill this in to unlock your dashboard, plan and progress.'}
        </p>
      </div>
      {!live && (
        <div className="mb-3">
          <span className="badge badge--info">Demo data — backend API not connected</span>
        </div>
      )}

      <div className="grid cols-2">
        <div className="card card--pad-lg">
          <form className="form" onSubmit={submit}>
            <fieldset className="fieldset">
              <legend>Identity</legend>
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="pr-name">Name</label>
                  <input className="input" id="pr-name" value={form.name} onChange={set('name')} required autoFocus={mode === 'create'} />
                </div>
                <div className="field">
                  <label htmlFor="pr-email">Email</label>
                  <input className="input" id="pr-email" type="email" value={form.email} onChange={set('email')} />
                </div>
              </div>
            </fieldset>
            <fieldset className="fieldset">
              <legend>Body</legend>
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="pr-gender">Gender</label>
                  <select className="select" id="pr-gender" value={form.gender} onChange={set('gender')}>
                    <option>Female</option><option>Male</option>
                  </select>
                </div>
                <div className="field"><label htmlFor="pr-age">Age</label><input className="input" id="pr-age" type="number" min="14" max="90" value={form.age} onChange={set('age')} required /></div>
                <div className="field"><label htmlFor="pr-h">Height (cm)</label><input className="input" id="pr-h" type="number" value={form.height_cm} onChange={set('height_cm')} required /></div>
                <div className="field"><label htmlFor="pr-w">Weight (kg)</label><input className="input" id="pr-w" type="number" step="0.1" value={form.weight_kg} onChange={set('weight_kg')} required /></div>
              </div>
            </fieldset>
            <fieldset className="fieldset">
              <legend>Goal &amp; budget</legend>
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="pr-goal">Goal</label>
                  <select className="select" id="pr-goal" value={form.goal} onChange={set('goal')}>
                    <option>Lose fat</option><option>Maintain</option><option>Build muscle</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="pr-act">Activity</label>
                  <select className="select" id="pr-act" value={form.activity} onChange={set('activity')}>
                    {ACTIVITY_OPTIONS.map((label) => <option key={label}>{label}</option>)}
                  </select>
                </div>
                <div className="field"><label htmlFor="pr-bud">Food budget (PKR)</label><input className="input" id="pr-bud" type="number" value={form.budget_pkr} onChange={set('budget_pkr')} /></div>
                <div className="field">
                  <label htmlFor="pr-per">Budget period</label>
                  <select className="select" id="pr-per" value={form.budget_period} onChange={set('budget_period')}>
                    <option value="daily">Daily</option><option value="monthly">Monthly</option>
                  </select>
                </div>
              </div>
            </fieldset>
            <button className="btn btn-primary" type="submit" disabled={saving}>
              {saving ? 'Saving…' : mode === 'edit' ? 'Save changes' : 'Create profile'}
            </button>
          </form>
        </div>

        <div className="stack">
          <div className="card">
            <div className="row">
              <span className="avatar" style={{ width: 52, height: 52, fontSize: '1.1rem' }}>
                {form.name ? form.name.trim().charAt(0).toUpperCase() : '?'}
              </span>
              <div>
                <div style={{ fontWeight: 700, fontSize: '1.05rem' }}>Your identity</div>
                <div className="muted" style={{ fontSize: '.9rem' }}>Used only to label your data locally.</div>
              </div>
            </div>
          </div>
          <div className="card">
            <div className="card__title mb-2">Data &amp; privacy</div>
            <p className="muted" style={{ fontSize: '.9rem' }}>HealthHub is not a medical service. Recommendations are general and cautious by design.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
