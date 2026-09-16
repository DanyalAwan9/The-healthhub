// Renders one symbol from /public/icons.svg (the same feather-style sprite
// used by the vanilla frontend).
export default function Icon({ name, className = '', ...rest }) {
  return (
    <svg className={`ic ${className}`.trim()} aria-hidden="true" {...rest}>
      <use href={`/icons.svg#i-${name}`} />
    </svg>
  );
}
