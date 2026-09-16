// Confidence ring with dynamic colour by value: green >80%, yellow 50-79%,
// red <50% (the vanilla version was a single accent colour — this adds the
// tri-colour logic requested for the React rebuild).
export default function ConfidenceRing({ value = 0, label = 'confidence' }) {
  const v = Number(value) || 0;
  const color = v >= 80 ? 'var(--success)' : v >= 50 ? 'var(--warning)' : 'var(--danger)';
  return (
    <div className="ring" style={{ '--val': v, '--ring-color': color }}>
      <div className="ring__label">
        <b>{Math.round(v)}%</b>
        <span>{label}</span>
      </div>
    </div>
  );
}
