"""Real product search for Pakistan via SerpAPI (Google Search / Shopping).

    pip install google-search-results        # provides `serpapi`
    SERPAPI_API_KEY=...                       # free tier = 100 searches/month

Everything degrades safely: no key / no package / API error -> empty result
with an `error` string (callers fall back to the CSV database). Query results
are cached in SQLite (price_cache table) for 7 days to protect the quota.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re

import database as db

try:
    from serpapi import GoogleSearch

    _PKG_OK = True
except Exception:  # pragma: no cover
    _PKG_OK = False

log = logging.getLogger("healthhub.product_search")

CACHE_DAYS = 7
SERP_TIMEOUT = float(os.environ.get("SERPAPI_TIMEOUT", "5"))   # seconds per query

# phrases SerpAPI / the HTTP layer use for "you're out of searches" or "blocked"
_QUOTA_MARKERS = ("run out of searches", "out of searches", "quota", "exceeded",
                  "rate limit", "too many requests", "429", "403", "insufficient credit",
                  "account has been suspended", "invalid api key", "unauthorized")


def _is_quota_error(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in _QUOTA_MARKERS)


def _log_quota_exhausted(query: str, msg: str) -> None:
    log.warning("SerpAPI quota/auth issue on query %r: %s - falling back to CSV", query, msg)
_PKR_RE = re.compile(r"(?:Rs\.?|PKR|RS|₨)\s?([\d][\d,]{1,8})", re.IGNORECASE)
_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ml|g|gm|gram|kg|oz|pcs?|pack|tablets?|caps?|softgels?)\b",
                      re.IGNORECASE)
_NOISE_RE = re.compile(
    r"\b(price|prices|in|pakistan|karachi|lahore|islamabad|daraz|buy|online|best|"
    r"original|authentic|genuine|shop|store|com|pk|sale|offer|deal|new|imported)\b",
    re.IGNORECASE)


def api_key() -> str | None:
    return os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERP_API_KEY")


def is_available() -> bool:
    return _PKG_OK and bool(api_key())


def status() -> str:
    if not _PKG_OK:
        return "google-search-results not installed"
    if not api_key():
        return "SERPAPI_API_KEY not set"
    return "ready"


def extract_pkr(*texts) -> int | None:
    for t in texts:
        if not t:
            continue
        m = _PKR_RE.search(str(t))
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None


_STOP = {"the", "a", "an", "for", "with", "and", "of", "in", "by", "new", "some",
         "original", "pack", "set", "kit", "combo", "ml", "gm", "g"}


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"\d+(?:\.\d+)?\s?%", " ", t)          # 10% , 2 %
    t = re.sub(r"(rs\.?|pkr|₨)\s?[\d,]+", " ", t)     # prices
    t = _SIZE_RE.sub(" ", t)                          # 236ml , 60 caps
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)                    # leftover bare numbers
    t = _NOISE_RE.sub(" ", t)
    return " ".join(t.split())


def _sig_tokens(title: str) -> list[str]:
    return [w for w in normalize_title(title).split()
            if w not in _STOP and len(w) > 1][:6]


def _source_of(row: dict) -> str:
    return (row.get("source") or row.get("displayed_link") or "").split("/")[0].strip().lower()


# price token with the text that precedes it (the product name fragment)
_PRICE_WITH_LEAD = re.compile(
    r"([^.;|·•\n]{4,85}?)\s*(?:[-–—:]|\bfor\b)?\s*(?:Rs\.?|PKR|₨|RS)\s?\.?\s?"
    r"([1-9][\d,]{1,7}(?:\.\d{1,2})?)",
    re.IGNORECASE)
_LEAD_SPLIT = re.compile(
    r"[·•;|…]|\s[-–—]\s|\bincluding\b|\bsuch as\b|\bfrom\b(?=\s)|\s+and\s+(?=[a-z])",
    re.IGNORECASE)
_LEAD_TRIM = re.compile(r"^\s*(?:\d+\s+)?(?:buy|shop|get|best|top|our|price of|starting|"
                        r"only|new pack of|for)\s+", re.IGNORECASE)
_TRAIL_JUNK = re.compile(r"\s+(?:regular price|sale price|price|for|at|under|now|only|"
                         r"reviews?|rating|revi\w*|in pakistan|pakistan)\b.*$", re.IGNORECASE)
_FILLER = {"dishes under", "products", "price list", "online", "best price", "for sale",
           "buy online", "and more", "shop now", "click here", "read more", "learn more",
           "dishes", "skin tone", "uneven skin tone"}
_PROSE_WORDS = {"and", "in", "for", "of", "with", "to", "your", "this", "that",
                "it", "on", "as", "are", "is", "you", "we", "our", "helps", "which"}


_PRICE_ONLY = re.compile(r"(?:Rs\.?|PKR|₨|RS)\s?\.?\s?([1-9][\d,]{1,7}(?:\.\d{1,2})?)",
                         re.IGNORECASE)
_BAD_LAST_WORD = {"under", "for", "at", "from", "with", "of", "and", "the", "to", "a"}


def _clean_frag(frag: str) -> str | None:
    frag = _LEAD_SPLIT.split(frag)[-1]
    frag = _LEAD_TRIM.sub("", frag)
    frag = _TRAIL_JUNK.sub("", frag).strip(" -–—:·•,.\t")
    frag = re.sub(r"\s+", " ", frag)
    low = frag.lower()
    words = low.split()
    letters = sum(c.isalpha() for c in frag)
    if (len(frag) < 5 or letters < 4 or not (2 <= len(words) <= 8)
            or low in _FILLER or words[-1] in _BAD_LAST_WORD
            or words[0] in _PROSE_WORDS
            or sum(w in _PROSE_WORDS for w in words) >= 3):   # reads like a sentence
        return None
    return frag


def parse_snippet_products(text: str, source: str, link: str, max_items: int = 12) -> list[dict]:
    """Pull every '<product name> ... Rs <price>' pair out of a search snippet.

    Broad queries ('oily skin acne treatment Pakistan price') mostly return
    listing pages whose snippets pack several real products + prices - this
    turns each snippet into multiple candidate products. Two passes: adjacent
    (name immediately before price) and clause-level (split on . ; · then look
    for a price at the end of the clause).
    """
    if not text:
        return []
    out, seen = [], set()

    def _add(frag, raw_price):
        low = (frag or "").lower()
        if not frag or low in seen:
            return
        try:
            price = int(float(str(raw_price).replace(",", "")))
        except ValueError:
            return
        if price < 50 or price > 200000:
            return
        seen.add(low)
        out.append({"title": frag, "price_pkr": price, "source": source,
                    "link": link, "kind": "snippet"})

    for m in _PRICE_WITH_LEAD.finditer(text):
        _add(_clean_frag(m.group(1)), m.group(2))
        if len(out) >= max_items:
            return out

    for clause in re.split(r"[.;\n]| – | - | · |•", text):
        pm = _PRICE_ONLY.search(clause)
        if pm:
            _add(_clean_frag(clause[:pm.start()]), pm.group(1))
            if len(out) >= max_items:
                break
    return out


def search_products(query: str, num: int = 20, use_cache: bool = True,
                    timeout: float | None = None) -> dict:
    """Return {'query', 'products': [ {title, price_pkr, source, link, kind} ], 'error'}.

    `timeout` (default SERPAPI_TIMEOUT, 5s) caps the HTTP request; a cache hit
    returns instantly and never touches the network.
    """
    key = f"serp:{query.lower().strip()}"
    if use_cache:
        try:
            cached = db.get_cached_price(key, CACHE_DAYS)
        except Exception:               # DB not initialised yet - skip the cache
            cached = None
        if cached is not None:
            cached["cached"] = True
            return cached

    if not is_available():
        return {"query": query, "products": [], "listings": [], "error": status(),
                "quota_exhausted": False, "cached": False}

    params = {
        "engine": "google", "q": query, "api_key": api_key(),
        "google_domain": "google.com.pk", "gl": "pk", "hl": "en",
        "location": "Pakistan", "num": num,
        "timeout": float(timeout or SERP_TIMEOUT),   # google-search-results reads this
    }
    try:
        data = GoogleSearch(params).get_dict()
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        quota = _is_quota_error(msg)
        if quota:
            _log_quota_exhausted(query, msg)
        return {"query": query, "products": [], "listings": [], "error": msg,
                "quota_exhausted": quota, "cached": False}
    if data.get("error"):
        msg = str(data["error"])
        quota = _is_quota_error(msg)
        if quota:
            _log_quota_exhausted(query, msg)
        return {"query": query, "products": [], "listings": [], "error": msg,
                "quota_exhausted": quota, "cached": False}

    products, listings = [], []
    for r in data.get("shopping_results", []) or []:
        price = extract_pkr(r.get("price")) or (
            int(r["extracted_price"]) if isinstance(r.get("extracted_price"), (int, float))
            and r["extracted_price"] > 50 else None)
        products.append({
            "title": (r.get("title") or "").strip(),
            "price_pkr": price,
            "source": (r.get("source") or "").strip(),
            "link": r.get("product_link") or r.get("link") or "",
            "kind": "shopping",
        })
    for r in data.get("immersive_products", []) or []:
        products.append({
            "title": (r.get("title") or "").strip(),
            "price_pkr": extract_pkr(r.get("price")),
            "source": (r.get("source") or "").strip(),
            "link": r.get("link") or "", "kind": "shopping",
        })
    for r in data.get("organic_results", []) or []:
        src = (r.get("source") or r.get("displayed_link") or "").strip()
        link = r.get("link") or ""
        snippet = r.get("snippet") or ""
        # the listing page itself (where to browse), plus every priced item in its snippet
        listings.append({
            "title": (r.get("title") or "").strip(),
            "price_pkr": extract_pkr(r.get("title")),
            "source": src, "link": link, "kind": "listing",
        })
        products.extend(parse_snippet_products(snippet, src, link))

    products = [p for p in products if p["title"]]
    out = {
        "query": query,
        "products": products,
        "listings": [x for x in listings if x["title"]],
        "error": None if (products or listings) else "no results",
        "quota_exhausted": False,
        "cached": False,
    }
    if use_cache and (products or listings):
        try:
            db.set_cached_price(key, "serp", out)
        except Exception:
            pass
    return out


def search_many(queries: list[str], timeout: float | None = None,
                max_workers: int = 4) -> dict:
    """Run several queries CONCURRENTLY. Returns {query: search_products-result}.

    Cache hits resolve instantly. Any query still running after the timeout
    window yields an empty {'error': 'timeout'} result rather than blocking.
    """
    queries = list(dict.fromkeys(q for q in queries if q))
    if not queries:
        return {}
    per = float(timeout or SERP_TIMEOUT)
    out: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(queries))) as ex:
        futs = {ex.submit(search_products, q, 20, True, per): q for q in queries}
        try:
            for fut in concurrent.futures.as_completed(futs, timeout=per + 3):
                q = futs[fut]
                try:
                    out[q] = fut.result(timeout=0)
                except Exception as exc:  # noqa: BLE001
                    out[q] = {"query": q, "products": [], "listings": [],
                              "error": type(exc).__name__, "cached": False}
        except concurrent.futures.TimeoutError:
            pass
    for q in queries:
        out.setdefault(q, {"query": q, "products": [], "listings": [],
                           "error": "timeout", "cached": False})
    return out


def search_shopping(query: str, num: int = 20, use_cache: bool = True) -> dict:
    """Google Shopping engine - structured products with clean names + prices.

    NOTE: SerpAPI's google_shopping engine does not support Pakistan
    (`gl=pk` -> "Unsupported country"), so HealthPub does not use this for the
    Pakistan market - it relies on parse_snippet_products() over normal results
    instead. Kept for reuse in supported regions. Same shape as search_products().
    """
    key = f"serpshop:{query.lower().strip()}"
    if use_cache:
        cached = db.get_cached_price(key, CACHE_DAYS)
        if cached is not None:
            cached["cached"] = True
            return cached
    if not is_available():
        return {"query": query, "products": [], "listings": [], "error": status(),
                "cached": False}
    params = {
        "engine": "google_shopping", "q": query, "api_key": api_key(),
        "google_domain": "google.com.pk", "gl": "pk", "hl": "en",
        "location": "Pakistan", "num": num,
    }
    try:
        data = GoogleSearch(params).get_dict()
    except Exception as exc:  # noqa: BLE001
        return {"query": query, "products": [], "listings": [],
                "error": f"{type(exc).__name__}: {exc}", "cached": False}
    if data.get("error"):
        return {"query": query, "products": [], "listings": [],
                "error": str(data["error"]), "cached": False}

    products = []
    for r in data.get("shopping_results", []) or []:
        ep = r.get("extracted_price")
        price = extract_pkr(r.get("price")) or (
            int(ep) if isinstance(ep, (int, float)) and ep > 50 else None)
        products.append({
            "title": (r.get("title") or "").strip(),
            "price_pkr": price,
            "source": (r.get("source") or r.get("store") or "").strip(),
            "link": r.get("product_link") or r.get("link") or "",
            "kind": "shopping",
        })
    products = [p for p in products if p["title"]]
    out = {"query": query, "products": products, "listings": [],
           "error": None if products else "no results", "cached": False}
    if use_cache and products:
        db.set_cached_price(key, "serp", out)
    return out


def verify_products(raw: list[dict]) -> list[dict]:
    """Group near-identical titles (token-overlap, not exact match); a product is
    'verified' when it shows up on 2+ distinct sources (e.g. Daraz + a pharmacy)."""
    groups: list[dict] = []
    for p in raw:
        toks = set(_sig_tokens(p["title"]))
        if len(toks) < 2:
            continue
        best, best_score = None, 0.0
        for g in groups:
            inter = len(toks & g["tokens"])
            union = len(toks | g["tokens"]) or 1
            score = inter / union
            if inter >= 2 and score > 0.5 and score > best_score:
                best, best_score = g, score
        if best is None:
            best = {"name": p["title"], "tokens": set(toks), "prices": [],
                    "sources": set(), "links": [], "kinds": set(), "mentions": 0}
            groups.append(best)
        else:
            best["tokens"] |= toks
        if len(p["title"]) < len(best["name"]):
            best["name"] = p["title"]        # shortest = cleanest display name
        if p["price_pkr"]:
            best["prices"].append(p["price_pkr"])
        src = _source_of(p)
        if src:
            best["sources"].add(src)
        if p["link"]:
            best["links"].append(p["link"])
        best["kinds"].add(p["kind"])
        best["mentions"] += 1

    merged = []
    for g in groups:
        prices = sorted(g["prices"])
        n_src = len(g["sources"])
        merged.append({
            "name": g["name"],
            "price_pkr": prices[len(prices) // 2] if prices else None,
            "price_low": prices[0] if prices else None,
            "price_high": prices[-1] if prices else None,
            "sources": sorted(g["sources"]),
            "n_sources": n_src,
            "n_mentions": g["mentions"],
            "verified": n_src >= 2,
            "link": g["links"][0] if g["links"] else "",
        })
    merged.sort(key=lambda m: (not m["verified"], -m["n_mentions"],
                               m["price_pkr"] is None, -m["n_sources"]))
    return merged
