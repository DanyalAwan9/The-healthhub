import { useLocation } from 'react-router-dom';
import Icon from './Icon';
import { useUser } from '../context/UserContext';

const TITLES = {
  '/': ['Dashboard', 'Your health at a glance'],
  '/fitness': ['Fitness Assessment', 'Calorie target, meal plan & workout'],
  '/nutrition': ['AI Nutritionist', 'Chat about diet, calories & prices'],
  '/skin': ['Skin Analysis', 'Upload a selfie for a skin-type read'],
  '/progress': ['Progress Tracking', 'Weight & measurements over time'],
  '/profile': ['Profile', 'Your details drive every recommendation'],
};

export default function Navbar({ onHamburger }) {
  const { pathname } = useLocation();
  const { user } = useUser();
  const [title, subtitle] = TITLES[pathname] || ['HealthHub', ''];

  return (
    <header className="topbar">
      <button className="icon-btn hamburger" aria-label="Menu" onClick={onHamburger}>
        <Icon name="menu" />
      </button>
      <div className="topbar__title">
        {title}
        <small>{subtitle}</small>
      </div>
      <div className="spacer" />
      <span className="avatar">{user?.name ? user.name.trim().charAt(0).toUpperCase() : '?'}</span>
    </header>
  );
}
