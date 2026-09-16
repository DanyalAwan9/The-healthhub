"""
gemini_vision.py - skin-type classification via the Google Gemini Vision API.

This replaces the EfficientNet ``skin_type`` head (which was ~61% accurate and,
because it never saw specular shine, could not reliably tell Oily from Dry).
Everything else in ``skin_ai`` is unchanged - face detection, gray-world colour
constancy, and the CV + CNN blemish / texture / redness detectors all still run.
Only the Oily / Dry / Normal / Combination call now comes from Gemini.

    from skin_ai.gemini_vision import classify_skin_type_gemini
    result = classify_skin_type_gemini(pil_image)        # SkinAnalysisResult | None

Environment:
    GEMINI_API_KEY / GOOGLE_API_KEY   API key (required; never hard-coded)
    GEMINI_MODEL                      preferred model (default: gemini-flash-latest)
    GEMINI_SKIN_TIMEOUT              per-request timeout, seconds (default 20)

SDK: the new ``google-genai`` package (``from google import genai``).
Install with ``pip install google-genai`` (``google-generativeai`` is the old one).

``classify_skin_type_gemini`` returns ``None`` - not an exception - when the key
or SDK is missing or every model in the fallback chain failed, so the caller can
fall back to a conservative default.
"""
from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

log = logging.getLogger("healthhub.skin_ai")

_CANON = ("Oily", "Dry", "Normal", "Combination")

# gemini-2.0-flash / gemini-2.5-flash are retired for newer API keys, so the
# request's hard-coded model is not enough on its own. Try GEMINI_MODEL, then the
# rolling aliases, then concrete versions - unknown ones just 404 and are skipped.
# Mirrors gemini_client._FALLBACK_MODELS.
_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

_TIMEOUT = float(os.environ.get("GEMINI_SKIN_TIMEOUT",
                                os.environ.get("GEMINI_TIMEOUT", "20")))

_PROMPT = (
    "Analyze this facial image and classify skin type.\n"
    "Consider: lighting, glare vs real sebum, surface texture, ambient conditions.\n"
    "Be specific and accurate."
)

_last_model: str | None = None
_last_error: str | None = None


class SkinAnalysisResult(BaseModel):
    skin_type: str = Field(description="'Oily' | 'Dry' | 'Normal' | 'Combination'")
    confidence: int = Field(description="0-100")
    reasoning: str = Field(description="Brief explanation")


def last_model() -> str | None:
    """Model that served the most recent successful classification."""
    return _last_model


def last_error() -> str | None:
    """Human-readable string for the most recent failure, if any."""
    return _last_error


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available() -> bool:
    """True if an API key and the google-genai SDK are both present."""
    if not _api_key():
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


def _model_chain() -> list[str]:
    primary = os.environ.get("GEMINI_MODEL", "").strip()
    chain: list[str] = []
    for name in [primary, *_FALLBACK_MODELS]:
        if name and name not in chain:
            chain.append(name)
    return chain


def _canon_type(s: str) -> str:
    """Map a free-text label onto one of the four canonical skin types."""
    t = (s or "").strip().lower()
    for c in _CANON:
        if c.lower() in t:
            return c
    return "Combination"          # conservative default for an unrecognised label


def _is_transient(exc: Exception) -> bool:
    n = type(exc).__name__
    s = str(exc).lower()
    return ("connect" in n.lower() or "ssl" in s or "eof" in s or "timed out" in s
            or "timeout" in s or "503" in s or "502" in s or "unavailable" in s)


def classify_skin_type_gemini(pil_image, blemish_severity: float | None = None,
                              timeout: float | None = None) -> "SkinAnalysisResult | None":
    """Classify skin type (Oily / Dry / Normal / Combination) with Gemini Vision.

    Parameters
    ----------
    pil_image        a preprocessed PIL RGB image - face-cropped and colour-constant
                     (gray-world), shine NOT inpainted so Gemini can weigh glare vs
                     real sebum itself.
    blemish_severity optional 0..1 hint from the local blemish detector; added to
                     the prompt as corroborating (not deciding) evidence.
    timeout          per-request seconds (default GEMINI_SKIN_TIMEOUT, 20).

    Returns a ``SkinAnalysisResult`` (skin_type canonicalised, confidence clamped
    0-100), or ``None`` if Gemini is unavailable / every fallback model failed.
    """
    global _last_model, _last_error

    key = _api_key()
    if not key:
        _last_error = "no GEMINI_API_KEY / GOOGLE_API_KEY in the environment"
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as e:                        # SDK not installed
        _last_error = f"google-genai SDK not importable: {e}"
        log.warning("skin-type: %s - install `google-genai`", _last_error)
        return None

    prompt = _PROMPT
    if blemish_severity is not None:
        prompt += (f"\nAn independent local detector puts blemish / breakout severity "
                   f"at {float(blemish_severity):.2f} on a 0-1 scale - treat that as "
                   f"corroborating evidence, not the deciding factor.")

    to = float(timeout or _TIMEOUT)
    try:
        client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(to * 1000)),  # ms
        )
    except Exception as e:
        try:
            client = genai.Client(api_key=key)                       # older SDK signature
        except Exception as e2:
            _last_error = f"genai.Client() failed: {e2 or e}"
            log.warning("skin-type: %s", _last_error)
            return None

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SkinAnalysisResult,
        temperature=0.2,
    )

    last_exc: Exception | None = None
    for mdl in _model_chain():
        for attempt in (1, 2):                    # one retry for a transient TLS / 5xx drop
            try:
                resp = client.models.generate_content(
                    model=mdl, contents=[pil_image, prompt], config=cfg,
                )
                out = SkinAnalysisResult.model_validate_json(resp.text)
                out.skin_type = _canon_type(out.skin_type)
                out.confidence = int(max(0, min(100, int(out.confidence))))
                _last_model, _last_error = mdl, None
                log.info("skin-type: Gemini(%s) -> %s (%d%%) - %s",
                         mdl, out.skin_type, out.confidence,
                         (out.reasoning or "")[:100])
                return out
            except Exception as e:                # noqa: BLE001
                last_exc = e
                if _is_transient(e) and attempt == 1:
                    log.info("skin-type: %s transient (%s), retrying once",
                             mdl, type(e).__name__)
                    continue
                _last_error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                break                            # move to the next model

    log.warning("skin-type: Gemini classification failed on every model (%s)",
                _last_error or last_exc)
    return None


if __name__ == "__main__":                       # quick manual check
    import sys

    from PIL import Image

    if len(sys.argv) < 2:
        print("usage: python skin_ai/gemini_vision.py <image>")
        raise SystemExit(2)
    r = classify_skin_type_gemini(Image.open(sys.argv[1]).convert("RGB"))
    print(r if r is not None else f"unavailable: {last_error()}")
