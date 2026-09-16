import { useRef, useState } from 'react';
import Icon from '../components/Icon';
import LoadingSpinner from '../components/LoadingSpinner';
import ConfidenceRing from '../components/ConfidenceRing';
import ProductCard from '../components/ProductCard';
import { useToast } from '../context/ToastContext';
import { skinAnalyze } from '../api/client';
import { MOCK_SKIN_RESULT } from '../api/mockData';

const SEV_CLASS = { clear: 'badge--ok', mild: 'badge--warn', moderate: 'badge--warn', severe: 'badge--danger' };

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function SkinAnalysis() {
  const { show } = useToast();
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [live, setLive] = useState(true);

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) { show('Please choose an image file.', 'err'); return; }
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError('');
    setLoading(true);
    try {
      const base64 = await fileToBase64(file);
      const { data } = await skinAnalyze(base64);
      if (data && data.error) { setError(data.error); }
      else { setResult(data); setLive(true); }
    } catch {
      setResult(MOCK_SKIN_RESULT);
      setLive(false);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  };

  return (
    <section className="page">
      <div className="page__head">
        <h1>Skin Analysis</h1>
        <p className="muted">Upload a clear, front-facing selfie in even light. Not a medical diagnosis.</p>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card__title mb-2">Photo</div>
          <div
            className={`dropzone${dragging ? ' is-drag' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <Icon name="upload" />
            <p><b>Click to upload</b> or drag an image here</p>
            <p className="faint" style={{ fontSize: '.82rem' }}>JPG or PNG · face clearly visible</p>
          </div>
          <input
            ref={inputRef} type="file" accept="image/*" className="hidden"
            onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
          />
          {preview && (
            <div className="mt-3">
              <img className="preview-img" src={preview} alt="Uploaded selfie preview" />
            </div>
          )}
        </div>

        <div className="card">
          <div className="card__title mb-2">What you'll get</div>
          <ul className="list-plain">
            <li><Icon name="check" className="ic--sm" /><span>Skin type — Oily, Dry, Normal or Combination</span></li>
            <li><Icon name="check" className="ic--sm" /><span>A confidence score and a short explanation</span></li>
            <li><Icon name="check" className="ic--sm" /><span>Colour-coded concerns (blemishes, texture, redness)</span></li>
            <li><Icon name="check" className="ic--sm" /><span>Real products, supplements and diet tips</span></li>
          </ul>
        </div>
      </div>

      <div className="mt-4">
        {loading && <LoadingSpinner lines={3} />}

        {!loading && error && (
          <div className="empty"><Icon name="alert" /><p>{error}</p></div>
        )}

        {!loading && !error && !result && (
          <div className="empty"><Icon name="image" /><p>Your analysis will appear here.</p></div>
        )}

        {!loading && result && (
          <>
            {!live && (
              <div className="mb-3">
                <span className="badge badge--info"><Icon name="alert" className="ic--sm" /> Demo data — backend API not connected</span>
              </div>
            )}
            <div className="grid cols-2">
              <div className="card">
                <div className="card__title">Skin type</div>
                <div className="row" style={{ gap: 20, marginTop: 10 }}>
                  <ConfidenceRing value={result.confidence} />
                  <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{result.skin_type}</div>
                    <div className="muted" style={{ fontSize: '.9rem' }}>Gemini Vision read</div>
                  </div>
                </div>
              </div>
              <div className="card">
                <div className="card__title">Why</div>
                <p className="muted mt-2">{result.reasoning}</p>
                <div className="row mt-3">
                  {(result.concerns || []).map((c) => (
                    <span key={c.name} className={`badge ${SEV_CLASS[c.severity] || ''}`}>{c.name} · {c.severity}</span>
                  ))}
                </div>
              </div>
            </div>

            {result.routine?.length > 0 && (
              <div className="card mt-3">
                <div className="card__head">
                  <span className="card__title">Recommended routine</span>
                  <span className="badge badge--primary"><Icon name="sparkles" className="ic--sm" /> {(result.care_level || 'basic').toUpperCase()} care</span>
                </div>
                <div className="stack mt-2">
                  {result.routine.map((r, i) => (
                    <ProductCard key={i} step={r.step} name={r.product} price={r.price_pkr} why={r.why} whereToBuy={r.where_to_buy} />
                  ))}
                </div>
                {result.brands_note && <p className="faint mt-3" style={{ fontSize: '.8rem' }}>{result.brands_note}</p>}
              </div>
            )}

            <div className="grid cols-2 mt-3">
              <div className="card">
                <div className="card__title mb-2"><Icon name="pill" className="ic--sm" /> Supplements</div>
                {result.supplements?.length ? (
                  <>
                    <div className="stack">
                      {result.supplements.map((s, i) => (
                        <ProductCard key={i} name={s.name} price={s.price_pkr}
                          why={[s.dosage, s.timing].filter(Boolean).join(' · ') + (s.why ? ` — ${s.why}` : '')}
                          supplement />
                      ))}
                    </div>
                    {result.supplement_note && <p className="faint mt-3" style={{ fontSize: '.8rem' }}>⚠️ {result.supplement_note}</p>}
                  </>
                ) : (
                  <p className="muted">None needed — skin looks clear enough that food covers it.</p>
                )}
              </div>
              <div className="card">
                <div className="card__title mb-2"><Icon name="leaf" className="ic--sm" /> Diet changes</div>
                <ul className="list-plain">
                  {(result.diet?.length ? result.diet : ['Balanced plate, seasonal fruit & veg, 2.5–3 L water/day']).map((t, i) => (
                    <li key={i}><Icon name="check" className="ic--sm" /><span>{t}</span></li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
