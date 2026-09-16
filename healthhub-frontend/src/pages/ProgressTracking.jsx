import { useEffect, useMemo, useState } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { useUser } from '../context/UserContext';
import { useToast } from '../context/ToastContext';
import { getProgress, addProgress } from '../api/client';
import { MOCK_PROGRESS_ENTRIES } from '../api/mockData';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? '—' : Number(n).toLocaleString(undefined, { maximumFractionDigits: d }));

// Chart.js dark theme config — reads the same CSS variables as the rest of
// the app so it always matches the design system.
function chartOptions() {
  const css = getComputedStyle(document.documentElement);
  const tok = (n, f) => css.getPropertyValue(n).trim() || f;
  const grid = tok('--border', '#2a2a2a');
  const label = tok('--text-muted', '#b3b3b3');
  return {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: { grid: { color: grid }, ticks: { color: label, maxRotation: 0, autoSkipPadding: 16 } },
      y: { grid: { color: grid }, ticks: { color: label } },
    },
    plugins: {
      legend: { labels: { color: label, usePointStyle: true, boxWidth: 8, padding: 16 } },
      tooltip: { padding: 10, cornerRadius: 8, usePointStyle: true },
    },
  };
}

export default function ProgressTracking() {
  const { user } = useUser();
  const { show } = useToast();
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);
  const [noProfile, setNoProfile] = useState(false);
  const [form, setForm] = useState({ weight_kg: '', waist_cm: '', chest_cm: '' });

  useEffect(() => {
    if (!user) { setNoProfile(true); setLoading(false); return; }
    let cancelled = false;
    (async () => {
      try {
        const { data } = await getProgress(user.name);
        if (!cancelled) { setEntries(data.entries || []); setLive(true); setLoading(false); }
      } catch {
        if (!cancelled) { setEntries(MOCK_PROGRESS_ENTRIES); setLive(false); setLoading(false); }
      }
    })();
    return () => { cancelled = true; };
  }, [user]);

  const submit = async (e) => {
    e.preventDefault();
    const payload = {
      weight_kg: Number(form.weight_kg),
      waist_cm: form.waist_cm ? Number(form.waist_cm) : null,
      chest_cm: form.chest_cm ? Number(form.chest_cm) : null,
    };
    try {
      const { data } = await addProgress(payload, user.name);
      setEntries(data.entries || []);
      setForm({ weight_kg: '', waist_cm: '', chest_cm: '' });
      show('Entry saved.', 'ok');
    } catch {
      show('Could not save entry.', 'err');
    }
  };

  const chartData = useMemo(() => {
    const primary = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#509bf5';
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#1ED760';
    const amber = '#f59e0b';
    const ds = (label, key, color, hidden = false) => ({
      label, data: entries.map((e) => e[key] ?? null),
      borderColor: color, backgroundColor: `${color}22`,
      borderWidth: 2.5, tension: 0.35, pointRadius: 3, pointHoverRadius: 5,
      pointBackgroundColor: color, spanGaps: true, hidden,
    });
    return {
      labels: entries.map((e) => e.date),
      datasets: [
        ds('Weight (kg)', 'weight_kg', primary),
        ds('Waist (cm)', 'waist_cm', accent),
        ds('Chest (cm)', 'chest_cm', amber, true),
      ],
    };
  }, [entries]);

  if (noProfile) {
    return (
      <section className="page">
        <div className="empty" style={{ padding: '72px 20px' }}>
          <p className="muted">Create a profile to start tracking progress.</p>
        </div>
      </section>
    );
  }

  const first = entries[0], last = entries[entries.length - 1] || {};
  const dW = first && last ? (last.weight_kg - first.weight_kg) : 0;
  const dWa = first && last ? (last.waist_cm - first.waist_cm) : 0;

  return (
    <section className="page">
      <div className="page__head">
        <h1>Progress Tracking</h1>
        <p className="muted">Log weekly — the trend matters more than any single day.</p>
      </div>
      {!live && (
        <div className="mb-3">
          <span className="badge badge--info">Demo data — backend API not connected</span>
        </div>
      )}

      {!loading && (
        <div className="grid cols-4">
          <div className="stat"><span className="stat__label">Current weight</span><span className="stat__value">{fmt(last.weight_kg, 1)} kg</span></div>
          <div className="stat">
            <span className="stat__label">Weight change</span>
            <span className="stat__value">{dW > 0 ? '+' : ''}{fmt(dW, 1)} kg</span>
            <span className={`stat__delta ${dW <= 0 ? 'up' : 'down'}`}>{dW <= 0 ? 'improving' : 'watch'}</span>
          </div>
          <div className="stat"><span className="stat__label">Waist</span><span className="stat__value">{fmt(last.waist_cm, 1)} cm</span></div>
          <div className="stat">
            <span className="stat__label">Waist change</span>
            <span className="stat__value">{dWa > 0 ? '+' : ''}{fmt(dWa, 1)} cm</span>
            <span className={`stat__delta ${dWa <= 0 ? 'up' : 'down'}`}>{dWa <= 0 ? 'improving' : 'watch'}</span>
          </div>
        </div>
      )}

      <div className="grid cols-2 mt-3">
        <div className="card">
          <div className="card__title mb-2">Weight &amp; measurements</div>
          <div style={{ height: 240, position: 'relative' }}>
            {entries.length >= 2 && <Line data={chartData} options={chartOptions()} />}
          </div>
        </div>
        <div className="card">
          <div className="card__title mb-2">Add an entry</div>
          <form className="form" onSubmit={submit}>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="p-w">Weight (kg)</label>
                <input className="input" id="p-w" type="number" step="0.1" required
                  value={form.weight_kg} onChange={(e) => setForm((f) => ({ ...f, weight_kg: e.target.value }))} />
              </div>
              <div className="field">
                <label htmlFor="p-wa">Waist (cm)</label>
                <input className="input" id="p-wa" type="number" step="0.1"
                  value={form.waist_cm} onChange={(e) => setForm((f) => ({ ...f, waist_cm: e.target.value }))} />
              </div>
              <div className="field">
                <label htmlFor="p-c">Chest (cm)</label>
                <input className="input" id="p-c" type="number" step="0.1"
                  value={form.chest_cm} onChange={(e) => setForm((f) => ({ ...f, chest_cm: e.target.value }))} />
              </div>
            </div>
            <button className="btn btn-accent" type="submit">Save entry</button>
          </form>
        </div>
      </div>

      <div className="grid cols-2 mt-3">
        <div className="card card--flush">
          <div className="card__title" style={{ padding: '22px 22px 0' }}>History</div>
          <div className="table-wrap" style={{ border: 0 }}>
            <table className="table">
              <thead><tr><th>Date</th><th>Weight</th><th>Waist</th><th>Chest</th></tr></thead>
              <tbody>
                {[...entries].reverse().map((e) => (
                  <tr key={e.date}>
                    <td>{e.date}</td><td>{fmt(e.weight_kg, 1)} kg</td><td>{fmt(e.waist_cm, 1)} cm</td><td>{fmt(e.chest_cm, 1)} cm</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <div className="card__title mb-3">Timeline</div>
          <div className="timeline">
            {[...entries].reverse().map((e) => (
              <div className="timeline__item" key={e.date}>
                <div className="timeline__date">{e.date}</div>
                <div>Weight <b>{fmt(e.weight_kg, 1)} kg</b> · waist <b>{fmt(e.waist_cm, 1)} cm</b></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
