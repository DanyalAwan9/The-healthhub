import { useState } from 'react';
import { useUser } from '../context/UserContext';

// Shown instead of the routed app when there is no profile in context yet.
// Collects name + email only — the profile record itself (with real body
// stats) is created on the Profile page, so nothing is ever saved with
// placeholder height/weight defaults.
export default function AuthGate() {
  const { login } = useUser();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const submit = (e) => {
    e.preventDefault();
    if (!name.trim()) { setError('Enter your name.'); return; }
    if (!/^\S+@\S+\.\S+$/.test(email)) { setError('Enter a valid email.'); return; }
    setError('');
    login({ name: name.trim(), email: email.trim() });
    // App will now render the routed shell; Profile page opens in create
    // mode (name/email prefilled) so the user can add body stats.
  };

  return (
    <div className="authgate">
      <div className="authgate__card">
        <h2>Welcome to HealthHub</h2>
        <p className="muted" style={{ marginTop: -6 }}>
          Create a profile to get your targets, meal plan and progress tracking.
        </p>
        <form className="form" onSubmit={submit} noValidate>
          <div className="field">
            <label htmlFor="auth-name">Your name</label>
            <input
              id="auth-name" className="input" type="text" autoComplete="name"
              placeholder="e.g. Ali Raza" value={name}
              onChange={(e) => setName(e.target.value)} required
            />
          </div>
          <div className="field">
            <label htmlFor="auth-email">Your email</label>
            <input
              id="auth-email" className="input" type="email" autoComplete="email"
              placeholder="you@example.com" value={email}
              onChange={(e) => setEmail(e.target.value)} required
            />
          </div>
          {error && <p style={{ color: 'var(--danger)', fontSize: '.85rem' }}>{error}</p>}
          <button className="btn btn-primary btn-block" type="submit">Create profile</button>
        </form>
        <p className="faint" style={{ fontSize: '.8rem' }}>
          Stored on this device — no password. A profile already saved here loads automatically.
        </p>
      </div>
    </div>
  );
}
