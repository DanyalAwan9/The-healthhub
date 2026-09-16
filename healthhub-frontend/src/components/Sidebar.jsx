import { NavLink, useNavigate } from 'react-router-dom';
import Icon from './Icon';
import { useUser } from '../context/UserContext';
import { useToast } from '../context/ToastContext';

const NAV = [
  { to: '/', label: 'Dashboard', icon: 'grid', end: true },
  { to: '/fitness', label: 'Fitness', icon: 'dumbbell' },
  { to: '/nutrition', label: 'Nutritionist', icon: 'chat' },
  { to: '/skin', label: 'Skin Analysis', icon: 'sparkles' },
  { to: '/progress', label: 'Progress', icon: 'trending' },
  { to: '/profile', label: 'Profile', icon: 'user' },
];

export default function Sidebar({ open, onNavigate }) {
  const { user, logout } = useUser();
  const { show } = useToast();
  const navigate = useNavigate();

  const handleSignOut = () => {
    logout();
    show('Signed out.', 'ok');
    navigate('/profile');
  };

  return (
    <div className={`sidebar${open ? ' is-open' : ''}`}>
      <NavLink className="brand" to="/" onClick={onNavigate}>
        <img className="brand__logo" src="/logo.svg" alt="" />
        <span className="brand__name">Health<span>Hub</span></span>
      </NavLink>

      <nav className="nav">
        <div className="nav__label">Menu</div>
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            onClick={onNavigate}
            className={({ isActive }) => `nav__link${isActive ? ' is-active' : ''}`}
          >
            <Icon name={n.icon} className="ic--sm" />
            <span>{n.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__foot">
        <span className="avatar sm">{user?.name ? user.name.trim().charAt(0).toUpperCase() : '?'}</span>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontWeight: 600, fontSize: '.9rem' }}>{user?.name || 'Not signed in'}</div>
          <button
            type="button"
            className="faint"
            style={{ fontSize: '.78rem', padding: 0, border: 0, background: 'none', cursor: 'pointer' }}
            onClick={handleSignOut}
          >
            {user ? 'Sign out' : 'Create profile'}
          </button>
        </div>
      </div>
    </div>
  );
}
