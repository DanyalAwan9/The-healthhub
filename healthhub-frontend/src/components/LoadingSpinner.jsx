// The pulsing skeleton loader used across every page (dashboard stats,
// chat "thinking" bubble, skin results, etc.) — mirrors .skeleton in theme.css.
export default function LoadingSpinner({ variant = 'card', lines = 2 }) {
  if (variant === 'line') return <div className="skeleton line" />;
  if (variant === 'block') return <div className="skeleton block" />;
  if (variant === 'inline') {
    return (
      <span className="typing" aria-label="Loading">
        <i />
        <i />
        <i />
      </span>
    );
  }
  // 'card' (default): a card-shaped skeleton, e.g. a stat tile while loading
  return (
    <div className="card">
      <div className="skeleton line" />
      {Array.from({ length: Math.max(0, lines - 1) }).map((_, i) => (
        <div key={i} className="skeleton line sm" />
      ))}
    </div>
  );
}
