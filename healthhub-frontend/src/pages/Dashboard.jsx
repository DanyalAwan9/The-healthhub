import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import Icon from '../components/Icon';
import LoadingSpinner from '../components/LoadingSpinner';
import { useUser } from '../context/UserContext';
import { getDashboard } from '../api/client';
import { MOCK_DASHBOARD } from '../api/mockData';

const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? '—' : Number(n).toLocaleString(undefined, { maximumFractionDigits: d }));

function NoProfile({ started }) {
  return (
    <div className="empty" style={{ padding: '72px 20px' }}>
      <Icon name="user" />
      <h2 style={{ margin: '14px 0 6px' }}>{started ? 'Finish your profile' : 'No profile yet'}</h2>
      <p className="muted">
        {started
          ? 'Add your body stats on the profile page to unlock your dashboard, plan and progress.'
          : 'Create a profile to unlock your dashboard, plan and progress.'}
      </p>
      <Link className="btn btn-primary mt-3" to="/profile">{started ? 'Go to profile' : 'Create profile'}</Link>
    </div>
  );
}

export default function Dashboard() {
  const { user } = useUser();
  const [state, setState] = useState({ loading: true, data: null, live: true, incomplete: false });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { data } = await getDashboard(user?.name);
        if (cancelled) return;
        if (data && data.error) setState({ loading: false, data: null, live: true, incomplete: true });
        else setState({ loading: false, data, live: true, incomplete: false });
      } catch {
        if (!cancelled) setState({ loading: false, data: MOCK_DASHBOARD, live: false, incomplete: false });
      }
    })();
    return () => { cancelled = true; };
  }, [user?.name]);

  if (state.loading) {
    return (
      <section className="page">
        <div className="page__head"><h1>Welcome back 👋</h1><p className="muted">Here's where you stand today.</p></div>
        <div className="grid cols-4">
          {Array.from({ length: 4 }).map((_, i) => <LoadingSpinner key={i} lines={2} />)}
        </div>
      </section>
    );
  }
  if (state.incomplete) return <section className="page"><NoProfile started /></section>;

  const d = state.data;
  const stats = [
    ['Calorie target', fmt(d.calorie_target), 'kcal / day', 'flame'],
    ['Protein target', fmt(d.protein_target), 'g / day', 'target'],
    ['BMI', fmt(d.bmi, 1), 'normal range', 'activity'],
    ['Streak', fmt(d.streak_days), 'days active', 'check-circle'],
  ];
  const wc = d.weight_change_kg ?? 0;

  return (
    <section className="page">
      <div className="page__head">
        <h1>Welcome back 👋</h1>
        <p className="muted">Here's where you stand today.</p>
      </div>
      {!state.live && (
        <div className="mb-3">
          <span className="badge badge--info"><Icon name="alert" className="ic--sm" /> Demo data — backend API not connected</span>
        </div>
      )}

      <div className="grid cols-4">
        {stats.map(([label, value, sub, icon]) => (
          <div className="stat" key={label}>
            <span className="stat__label">{label}</span>
            <span className="stat__value">{value}</span>
            <span className="muted" style={{ fontSize: '.82rem', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <Icon name={icon} className="ic--sm" /> {sub}
            </span>
          </div>
        ))}
      </div>

      <div className="grid cols-2 mt-3">
        <div className="card">
          <div className="card__head"><span className="card__title">Recent activity</span></div>
          <ul className="list-plain">
            {(d.activity || []).map((a, i) => (
              <li key={i} className="row" style={{ padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
                <span className="avatar sm" style={{ background: 'var(--surface-2)', color: 'var(--primary)' }}>
                  <Icon name={a.icon} className="ic--sm" />
                </span>
                <span>{a.text}</span>
                <span className="spacer" />
                <span className="faint" style={{ fontSize: '.8rem' }}>{a.when}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="stack">
          <div className="card">
            <div className="row-between">
              <div>
                <div className="stat__label">Weight change (30 days)</div>
                <div className="stat__value">{wc > 0 ? '+' : ''}{fmt(wc, 1)} kg</div>
              </div>
              <span className={`badge ${wc <= 0 ? 'badge--ok' : 'badge--warn'}`}>{wc <= 0 ? 'On track' : 'Review plan'}</span>
            </div>
          </div>
          <div className="card">
            <div className="card__title mb-2">Jump back in</div>
            <div className="stack-sm">
              <Link className="btn btn-outline btn-block" to="/fitness">Update fitness plan</Link>
              <Link className="btn btn-outline btn-block" to="/skin">Run a skin analysis</Link>
              <Link className="btn btn-outline btn-block" to="/progress">Log this week's weight</Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
