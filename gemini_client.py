"""Thin wrapper around the Google Gemini API (google-generativeai SDK).

Centralises configuration so nutritionist.py, fitness_engine.py and
skin_analyzer.py can all share one code path. Every public helper raises
on failure so callers can fall back to their offline behaviour.

Resilience: each Gemini model has its own free-tier quota bucket, so on a
quota (429 / ResourceExhausted) or model-not-found error the call is retried
against the next model in _model_chain() automatically.

Environment:
    GEMINI_API_KEY / GOOGLE_API_KEY   API key (required)
    GEMINI_MODEL                      primary model, default "gemini-flash-latest"
    GEMINI_FALLBACK_MODELS            comma-separated override for the fallback chain
"""
from __future__ import annotations

import json
import os
import re
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from google.generativeai import GenerativeModel
        import google.generativeai as genai

    _SDK_OK = True
except Exception:  # pragma: no cover
    _SDK_OK = False

DEFAULT_MODEL = "gemini-flash-latest"

# Tried in order after the primary model. `-latest` aliases first (most likely
# to exist on any key); concrete versions after. Unknown ones raise NotFound and
# are simply skipped.
_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

_configured = False
_last_error: str | None = None
_active_model: str | None = None

GEN_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "15"))  # seconds per call, so a
# stalled/blocked connection fails fast instead of hanging - callers with a heavier
# prompt (fitness_engine's meal/workout generation) pass their own longer `timeout=`.

# Error class names that mean "this model won't work now, try another".
_SWITCHABLE = {"ResourceExhausted", "TooManyRequests", "NotFound", "FailedPrecondition"}


def _is_timeout(exc: Exception) -> bool:
    n = type(exc).__name__
    s = str(exc).lower()
    return (n in ("DeadlineExceeded", "RetryError", "TimeoutError", "GatewayTimeout")
            or "timeout" in s or "deadline" in s or "504" in s)


def last_error() -> str | None:
    """Human-readable string for the most recent Gemini call failure, if any."""
    return _last_error


def active_model() -> str | None:
    """Model name that served the most recent successful call."""
    return _active_model


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _model_chain() -> list[str]:
    primary = model_name()
    env_fb = os.environ.get("GEMINI_FALLBACK_MODELS", "")
    fallbacks = [m.strip() for m in env_fb.split(",") if m.strip()] or _FALLBACK_MODELS
    chain: list[str] = []
    for name in [primary, *fallbacks]:
        if name and name not in chain:
            chain.append(name)
    return chain


def _ensure() -> bool:
    global _configured
    if not _SDK_OK:
        return False
    key = _api_key()
    if not key:
        return False
    if not _configured:
        # transport="rest" avoids the default gRPC transport, which was
        # consistently hanging until DeadlineExceeded (504) on this network
        # (gRPC/HTTP2 blocked or throttled) while plain HTTPS reached Google fine.
        genai.configure(api_key=key, transport="rest")
        _configured = True
    return True


def is_available() -> bool:
    return _ensure()


def status() -> str:
    if not _SDK_OK:
        return "SDK not installed"
    if not _api_key():
        return "no API key"
    if _active_model and _active_model != model_name():
        return f"ready ({model_name()} -> using {_active_model})"
    return f"ready ({model_name()})"


def _make_model(name: str, system_instruction: str | None):
    if not _ensure():
        raise RuntimeError("Gemini unavailable: " + status())
    try:
        return GenerativeModel(name, system_instruction=system_instruction)
    except TypeError:  # very old SDK without system_instruction kwarg
        return GenerativeModel(name)


def _record(exc: Exception) -> None:
    global _last_error
    _last_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"


def _switchable(exc: Exception) -> bool:
    return type(exc).__name__ in _SWITCHABLE


def generate(prompt: str, system_instruction: str | None = None,
             temperature: float = 0.7, timeout: float | None = None) -> str:
    """Single-shot completion, with automatic model fallback on quota/not-found.

    `timeout` (default GEMINI_TIMEOUT, 25s) caps each request. A timeout is NOT
    retried down the model chain - it raises so the caller can fall back fast.
    """
    global _last_error, _active_model
    to = float(timeout or GEN_TIMEOUT)
    last_exc: Exception | None = None
    for name in _model_chain():
        try:
            model = _make_model(name, system_instruction)
            resp = model.generate_content(
                prompt, generation_config={"temperature": temperature},
                request_options={"timeout": to},
            )
            _last_error = None
            _active_model = name
            return (getattr(resp, "text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _record(exc)
            if _is_timeout(exc):
                raise
            if _switchable(exc):
                continue
            raise
    raise last_exc if last_exc else RuntimeError("Gemini: no model available")


def stream(prompt: str, system_instruction: str | None = None,
           temperature: float = 0.7):
    """Yield text for a single prompt, with model fallback before first token.

    Fetches the full response in one non-streaming call and yields it as a
    single chunk. `transport="rest"` (see `_ensure`) is used to route around
    gRPC being blocked/throttled on some networks, and this SDK's REST
    transport doesn't reliably support server-streamed (`stream=True`)
    responses, so real token streaming isn't available while on REST.
    """
    global _last_error, _active_model
    last_exc: Exception | None = None
    for name in _model_chain():
        try:
            model = _make_model(name, system_instruction)
            resp = model.generate_content(
                prompt, generation_config={"temperature": temperature},
                request_options={"timeout": GEN_TIMEOUT},
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                _active_model = name
                _last_error = None
                yield text
                return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _record(exc)
            if not _switchable(exc):
                raise
            continue
    if last_exc:
        raise last_exc


_ROLE_MAP = {"user": "user", "assistant": "model", "model": "model", "system": "user"}


def chat_stream(messages: list[dict], system_instruction: str | None = None,
                temperature: float = 0.7):
    """Multi-turn chat. `messages` is [{'role': 'user'|'assistant', 'content': str}, ...].

    The last message must be from the user. Falls back across models before
    yielding. Fetches the full reply in one non-streaming call and yields it as
    a single chunk - see `stream()` above for why (REST transport + this SDK).
    """
    global _last_error, _active_model
    if not messages:
        return
    history = [
        {"role": _ROLE_MAP.get(m["role"], "user"), "parts": [m["content"]]}
        for m in messages[:-1]
    ]
    last_exc: Exception | None = None
    for name in _model_chain():
        try:
            model = _make_model(name, system_instruction)
            chat = model.start_chat(history=history)
            resp = chat.send_message(
                messages[-1]["content"],
                generation_config={"temperature": temperature},
                request_options={"timeout": GEN_TIMEOUT},
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                _active_model = name
                _last_error = None
                yield text
                return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _record(exc)
            if not _switchable(exc):
                raise
            continue
    if last_exc:
        raise last_exc


_search_tool_cache: object | None = None
_search_tool_tried = False


def _search_tool():
    """Return a Google-Search grounding tool in whatever form this SDK accepts."""
    global _search_tool_cache, _search_tool_tried
    if _search_tool_tried:
        return _search_tool_cache
    _search_tool_tried = True
    candidates = []
    try:
        candidates.append(genai.protos.Tool(
            google_search=genai.protos.Tool.GoogleSearch()))
    except Exception:
        pass
    try:
        candidates.append(genai.protos.Tool(
            google_search_retrieval=genai.protos.GoogleSearchRetrieval()))
    except Exception:
        pass
    candidates.append("google_search_retrieval")  # string form, older SDKs
    _search_tool_cache = candidates[0] if candidates else None
    return _search_tool_cache


def _grounding_sources(response) -> list[dict]:
    out, seen = [], set()

    def _add(uri, title=""):
        uri = (uri or "").strip()
        if uri and uri not in seen:
            seen.add(uri)
            out.append({"title": (title or uri).strip(), "uri": uri})

    try:
        for cand in getattr(response, "candidates", []) or []:
            gm = getattr(cand, "grounding_metadata", None)
            for chunk in (getattr(gm, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None) or getattr(chunk, "retrieved_context", None)
                if web:
                    _add(getattr(web, "uri", ""), getattr(web, "title", ""))
            cm = getattr(cand, "citation_metadata", None)
            for c in (getattr(cm, "citation_sources", None)
                      or getattr(cm, "citations", None) or []):
                _add(getattr(c, "uri", "") or getattr(c, "url", ""),
                     getattr(c, "title", ""))
    except Exception:
        pass
    return out


def generate_grounded(prompt: str, system_instruction: str | None = None,
                      temperature: float = 0.2) -> dict:
    """Web-grounded completion. Returns {'text', 'sources': [{title,uri}], 'model'}.

    Raises on failure (quota, no grounding support, ...) so callers can fall back.
    """
    global _last_error, _active_model
    tool = _search_tool()
    if tool is None:
        raise RuntimeError("no grounding tool available in SDK")
    tools = tool if isinstance(tool, str) else [tool]
    last_exc: Exception | None = None
    for name in _model_chain():
        try:
            if not _ensure():
                raise RuntimeError("Gemini unavailable: " + status())
            try:
                model = GenerativeModel(name, tools=tools,
                                        system_instruction=system_instruction)
            except TypeError:
                model = GenerativeModel(name, tools=tools)
            resp = model.generate_content(
                prompt, generation_config={"temperature": temperature},
                request_options={"timeout": GEN_TIMEOUT},
            )
            _last_error = None
            _active_model = name
            return {
                "text": (getattr(resp, "text", "") or "").strip(),
                "sources": _grounding_sources(resp),
                "model": name,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _record(exc)
            if _switchable(exc):
                continue
            raise
    raise last_exc if last_exc else RuntimeError("Gemini: no model available")


def extract_json(text: str):
    """Best-effort parse of a JSON object/array out of an LLM response."""
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = candidate.find(opener), candidate.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(candidate[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in response")
