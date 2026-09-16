// One skincare / supplement recommendation: brand+name, price, why, where to
// buy. `supplement` adds the accent highlight strip used for supplement picks.
export default function ProductCard({ step, brand, name, price, why, whereToBuy, supplement = false }) {
  const rs = (v) => (v == null || Number.isNaN(v) ? '—' : `Rs ${Number(v).toLocaleString()}`);
  return (
    <div className={`product-card${supplement ? ' product-card--supplement' : ''}`}>
      <div className="product-card__top">
        <div>
          {step && <div className="faint" style={{ fontSize: '.76rem', textTransform: 'uppercase', letterSpacing: '.04em' }}>{step}</div>}
          <div className="product-card__name">{name}</div>
          {brand && <div className="product-card__brand">{brand}</div>}
        </div>
        <div className="product-card__price">{rs(price)}</div>
      </div>
      {why && <div className="product-card__why">{why}</div>}
      {whereToBuy && <div className="product-card__buy">Buy: {whereToBuy}</div>}
    </div>
  );
}
