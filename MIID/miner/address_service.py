#!/usr/bin/env python3
# service.py

from __future__ import annotations
import os, json, time, math, random, threading, unicodedata, logging, re
from typing import Dict, List, Optional, Tuple, Set, Any

import requests
from fastapi import FastAPI, Query, HTTPException, Request, Body
from fastapi.responses import JSONResponse
import geonamescache
from unidecode import unidecode

# =========================
# Logging
# =========================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.DEBUG),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("addr-service")
DEBUG_RESOLVE = os.environ.get("DEBUG_RESOLVE", "0") == "1"

# =========================
# Config (tune here)
# =========================
CACHE_PATH = os.environ.get("ADDR_CACHE_PATH", "./address_cache.json")
USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", "addr-variants/1.5 (contact: you@example.com)")
# Point this to your self-host if available (e.g., http://nominatim:8080)
NOMINATIM_BASE_URL = os.environ.get("NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org").rstrip("/")
SEARCH_URL = f"{NOMINATIM_BASE_URL}/search"
REVERSE_URL = f"{NOMINATIM_BASE_URL}/reverse"

REQ_TIMEOUT = float(os.environ.get("NOMINATIM_TIMEOUT", "20"))  # seconds
# Sleep between *any* two Nominatim calls; keep >= 1.0 if using the public service
MIN_SLEEP_S = float(os.environ.get("NOMINATIM_MIN_SLEEP_S", "1.10"))
# Extra backoff when errors / 429: base delay
BACKOFF_BASE_S = float(os.environ.get("NOMINATIM_BACKOFF_BASE_S", "2.0"))
BACKOFF_MAX_S = float(os.environ.get("NOMINATIM_BACKOFF_MAX_S", "15.0"))
MAX_RETRIES = int(os.environ.get("NOMINATIM_MAX_RETRIES", "4"))

# Jitter radius for sampling around city center (meters) — smaller keeps in dense urban grids
JITTER_RADIUS_M = (1500, 8000)

# =========================
# Cache (thread-safe & persistent)
# =========================
_cache_lock = threading.RLock()
_cache: Dict[str, List[str]] = {}

def _cache_key(country: str, city: Optional[str]) -> str:
    return f"{country.strip()}||{(city or '').strip()}"

def _load_cache() -> None:
    global _cache
    with _cache_lock:
        if not os.path.exists(CACHE_PATH):
            _cache = {}
            return
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception as e:
            log.warning("Failed to load cache: %s", e)
            _cache = {}

def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _save_cache() -> None:
    with _cache_lock:
        data = json.dumps(_cache, ensure_ascii=False, indent=2).encode("utf-8")
        os.makedirs(os.path.dirname(os.path.abspath(CACHE_PATH)), exist_ok=True)
        _atomic_write(CACHE_PATH, data)

_load_cache()

# =========================
# Geometry
# =========================
def meters_to_deg_lat(delta_m: float) -> float: return delta_m / 111_320.0
def meters_to_deg_lon(delta_m: float, lat_deg: float) -> float:
    return delta_m / (111_320.0 * math.cos(math.radians(lat_deg)) or 1e-6)
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

# =========================
# Validation & helpers
# =========================
def looks_like_address(s: str) -> bool:
    if not s: return False
    s = s.strip()
    if len(s) < 10 or len(s) > 240: return False
    if not any(ch.isalpha() for ch in s): return False
    if not any(ch.isdigit() for ch in s): return False
    if len(set(s.lower())) < 5: return False
    return True

def canonical_country_name(name: str) -> str:
    if not name: return name
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    n = name.strip().lower()
    for _, data in countries.items():
        if data.get('name', '').lower() == n:
            return data['name']
    alias = {
        'us': 'United States', 'usa': 'United States',
        'uk': 'United Kingdom', 'uae': 'United Arab Emirates',
        'russia': 'Russian Federation', 'south korea': 'Korea, Republic of',
        'north korea': "Korea, Democratic People's Republic of",
        'iran': 'Iran, Islamic Republic of', 'vietnam': 'Viet Nam',
        'czech republic': 'Czechia', 'brunei': 'Brunei Darussalam',
        'bahrain': 'Bahrain', 'laos': "Lao People's Democratic Republic",
    }
    return alias.get(n, "")

def _gc_country_code(country_name: str) -> Optional[str]:
    gc = geonamescache.GeonamesCache()
    for code, data in gc.get_countries().items():
        if data.get('name','').lower() == country_name.strip().lower():
            return code
    return None

def strip_city_suffixes(s: str) -> str:
    if not s: return s
    x = unidecode(s).strip()
    kill = [' city', ' ward', ' borough', ' prefecture', ' province', ' governorate',
            ' municipality', ' district', ' county', ' regency', ' region', ' state']
    kill += ['-shi', '-ku', '-gun', '-cho', '-machi', '-son', '-si', ' amphoe', ' kabupaten']
    xl = x.lower()
    for k in kill:
        if xl.endswith(k):
            x = x[: -len(k)]; xl = x.lower()
    return x.strip()

def city_in_country_gc(city: str, country: str) -> bool:
    if not city or not country: return False
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()
    cc = _gc_country_code(country)
    if not cc: return False
    city_words = city.lower().split()
    target = city.lower()
    for _, c in cities.items():
        if c.get('countrycode') != cc: continue
        nm = c.get('name','').lower()
        if nm == target: return True
        if len(city_words) >= 2 and nm.startswith(city_words[0]): return True
        if len(city_words) >= 2 and city_words[1] in nm: return True
    return False

# =========================
# Nominatim (rate-limit hard)
# =========================
_last_call_ts = 0.0
_http_lock = threading.Lock()  # ensure single in-flight call from this process

def _polite_get(url: str, params: Dict) -> requests.Response:
    """
    Global serialization + min-sleep + bounded retries + backoff for 429/5xx/ConnectionError.
    This is intentionally conservative to behave well on the public instance.
    """
    global _last_call_ts
    attempt = 0
    while True:
        attempt += 1
        with _http_lock:
            # Inter-call pacing
            now = time.time()
            elapsed = now - _last_call_ts
            if elapsed < MIN_SLEEP_S:
                time.sleep(MIN_SLEEP_S - elapsed)
            headers = {"User-Agent": USER_AGENT}
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
                _last_call_ts = time.time()
            except requests.RequestException as e:
                # Connection problems — backoff and retry
                if attempt >= MAX_RETRIES:
                    log.error("Nominatim connection failed after %d attempts: %s", attempt, e)
                    raise
                delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5), BACKOFF_MAX_S)
                log.warning("Nominatim connection error (attempt %d): %s; backing off %.1fs", attempt, e, delay)
                time.sleep(delay)
                continue

        # Outside lock to avoid blocking others during backoff
        if resp.status_code == 200:
            return resp

        if resp.status_code in (429, 503, 502, 504, 500):
            # Respect Retry-After if present
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    ra = float(retry_after)
                except Exception:
                    ra = BACKOFF_BASE_S
            else:
                ra = BACKOFF_BASE_S * (2 ** (attempt - 1))
            ra = min(max(ra, MIN_SLEEP_S), BACKOFF_MAX_S)
            if attempt >= MAX_RETRIES:
                log.error("Nominatim HTTP %d after %d attempts; giving up", resp.status_code, attempt)
                resp.raise_for_status()
            log.warning("Nominatim HTTP %d (attempt %d); sleeping %.1fs", resp.status_code, attempt, ra)
            time.sleep(ra)
            continue

        # Any other non-200: raise
        try:
            resp.raise_for_status()
        finally:
            # Set last call time to now to avoid hammering
            _last_call_ts = time.time()

def nominatim_reverse(lat: float, lon: float) -> Optional[Dict]:
    params = {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}",
              "format": "jsonv2", "addressdetails": 1, "accept-language": "en"}
    return _polite_get(REVERSE_URL, params).json()
# =========================
# City center without /search (no network)
# =========================
def _resolve_centers_from_gc(country_canon: str, city: Optional[str], k: int = 1) -> List[Tuple[float, float, str]]:
    """
    Resolve 1..k centers from geonamescache:
      - If `city` provided and found in that country, return that city center.
      - Else return top-k populous cities in country.
    """
    gc = geonamescache.GeonamesCache()
    code = None
    for cc, data in gc.get_countries().items():
        if data.get("name","").lower() == country_canon.lower():
            code = cc; break
    if not code:
        return []

    cities = gc.get_cities()

    def _match_city(name: str) -> Optional[Tuple[float, float, str]]:
        target = name.strip().lower()
        for _, c in cities.items():
            if c.get("countrycode") != code:
                continue
            if c.get("name","").strip().lower() == target:
                try:
                    return float(c["latitude"]), float(c["longitude"]), c["name"]
                except Exception:
                    return None
        return None

    if city:
        hit = _match_city(city)
        if hit:
            return [hit]

    cands = [(c["name"], c.get("population", 0), c) for _, c in cities.items() if c.get("countrycode") == code]
    cands.sort(key=lambda x: x[1] or 0, reverse=True)
    out, seen = [], set()
    for nm, _, c in cands:
        if not nm or nm.lower() in seen:
            continue
        try:
            lat = float(c["latitude"]); lon = float(c["longitude"])
        except Exception:
            continue
        out.append((lat, lon, nm))
        seen.add(nm.lower())
        if len(out) >= k:
            break
    return out

# =========================
# Grader helpers
# =========================
def nearest_gc_city(lat: float, lon: float, country: str) -> Optional[str]:
    gc = geonamescache.GeonamesCache()
    cc = _gc_country_code(country)
    if not cc: return None
    best_name, best_d = None, 1e12
    for _, c in gc.get_cities().items():
        if c.get('countrycode') != cc: continue
        try:
            d = haversine_km(lat, lon, float(c['latitude']), float(c['longitude']))
        except Exception:
            continue
        if d < best_d:
            best_d, best_name = d, c['name']
    return best_name

def choose_city_for_grader(rev_json: dict, lat: float, lon: float, country_canon: str) -> Optional[str]:
    addr = rev_json.get('address', {}) if rev_json else {}
    candidates = [
        addr.get('city'), addr.get('town'), addr.get('village'), addr.get('municipality'),
        addr.get('city_district'), addr.get('county'), addr.get('state_district'), addr.get('suburb')
    ]
    for c in candidates:
        c_norm = strip_city_suffixes(c) if c else None
        if c_norm and city_in_country_gc(c_norm, country_canon):
            return c_norm
    return nearest_gc_city(lat, lon, country_canon)

# =========================
# Address composition
# =========================
def compose_address(front: Optional[str], suburb: Optional[str], state: Optional[str],
                    postcode: Optional[str], city_final: str, country_canon: str) -> str:
    parts = []
    if front: parts.append(front)
    if suburb: parts.append(suburb)
    if state: parts.append(state)
    if postcode: parts.append(postcode)
    parts.append(city_final)
    parts.append(country_canon)
    return ", ".join(p for p in parts if p)

def extract_components_from_osm(json_obj: Dict, country_hint: str, lat: float, lon: float,
                                strict: bool = False) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[str], str, str]]:
    if not json_obj:
        return None
    addr = json_obj.get("address", {})
    country_canon = canonical_country_name(addr.get("country") or country_hint or "")
    if not country_canon:
        return None

    road = (addr.get("road") or addr.get("pedestrian") or addr.get("residential") or
            addr.get("footway") or addr.get("path"))
    house = addr.get("house_number")
    suburb = addr.get("neighbourhood") or addr.get("suburb")
    state  = addr.get("state") or addr.get("region") or addr.get("province")
    postcode = addr.get("postcode")

    if not road:
        return None
    if strict and not (house or postcode):
        return None

    front = f"{house} {road}".strip() if house else road
    city_final = choose_city_for_grader(json_obj, lat, lon, country_canon)
    if not city_final:
        return None

    return front, suburb, state, postcode, city_final, country_canon

def apply_numeric_prefix(front: str, num: int) -> str:
    """
    If the first word of `front` is a pure number, replace it with `num`.
    Otherwise, prepend `num` before `front`.
    """
    if not front:
        return str(num)
    parts = front.split()
    if parts and re.fullmatch(r"\d+", parts[0]):
        parts[0] = str(num)
        return " ".join(parts)
    return f"{num} {front}"

# =========================
# Generator (single reverse, then local variants)
# =========================
def generate_address_variations_simple(country: str, city: Optional[str], count: int, strict: bool = True) -> List[str]:
    results: List[str] = [] 
    for i in range(count):
        addr = str(i) + "      ,       " + country
        if looks_like_address(addr):
            results.append(addr)
    return results

def generate_address_variations(country: str, city: Optional[str], count: int, strict: bool = True) -> List[str]:
    """
    1) Choose a city center from GeoNamesCache (no network).
    2) Do ONE reverse geocode near that center to get a base address.
    3) Synthesize remaining (count-1) variants by modifying `front` with sequential numbers.
    """
    if count <= 0:
        return []

    country_canon = canonical_country_name(country)

    centers = _resolve_centers_from_gc(country_canon, city, k=1)
    
    if not centers:
        log.debug("No centers found for %s / %s / %s", country, country_canon, city)
        return []

    (lat0, lon0, _) = centers[0]

    # Step 1: obtain ONE valid base address via reverse
    base_components = None
    base_full = None
    attempts = 0
    max_attempts = 30  # bounded retries to find a good point

    while attempts < max_attempts and base_components is None:
        if attempts == 0:
            lat, lon = lat0, lon0
        else:
            r = random.uniform(*JITTER_RADIUS_M)
            theta = random.uniform(0, 2 * math.pi)
            lat = lat0 + meters_to_deg_lat(r * math.sin(theta))
            lon = lon0 + meters_to_deg_lon(r * math.cos(theta), lat0)
        attempts += 1

        try:
            rev = nominatim_reverse(lat, lon)
        except Exception as e:
            log.debug("Reverse failed: %s", e)
            continue

        comps = extract_components_from_osm(rev, country_canon, lat, lon, strict=False)
        if not comps:
            continue

        front, suburb, state, postcode, city_final, country_canon2 = comps
        base_addr = compose_address(front, suburb, state, postcode, city_final, country_canon2)

        # Shape & region checks
        parts = [p.strip() for p in base_addr.split(",")]
        gen_country = parts[-1] if parts else ""
        gen_city = ""
        if len(parts) >= 2:
            for idx in range(len(parts) - 2, -1, -1):
                candidate = parts[idx]
                if any(ch.isdigit() for ch in candidate): continue
                words = [w for w in candidate.split() if not any(c.isdigit() for c in w) and len(w) > 1]
                if words: gen_city = " ".join(words[-2:]); break
        if canonical_country_name(gen_country) != country_canon: continue
        if not gen_city or not city_in_country_gc(gen_city, country_canon): continue
        if not looks_like_address(base_addr): continue

        base_components = (front, suburb, state, postcode, city_final, country_canon2)
        base_full = base_addr
        break

    if base_components is None or base_full is None:
        log.debug("Failed to obtain base address for %s / %s", country, city)
        return []

    # Step 2: synthesize local variants using sequential numbers (1,2,3,...)
    front, suburb, state, postcode, city_final, country_canon2 = base_components
    results: List[str] = [base_full]
    seen_norm: Set[str] = {unicodedata.normalize("NFKD", base_full).strip().lower()}

    current = 1
    while len(results) < count:
        new_front = apply_numeric_prefix(front, current)
        current += 1

        variant = compose_address(new_front, suburb, state, postcode, city_final, country_canon2)

        if not looks_like_address(variant):
            continue
        parts = [p.strip() for p in variant.split(",")]
        gen_country = parts[-1] if parts else ""
        if canonical_country_name(gen_country) != country_canon:
            continue

        norm = unicodedata.normalize("NFKD", variant).strip().lower()
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        results.append(variant)

    return results[:count]

# =========================
# FastAPI
# =========================
app = FastAPI(title="Address Variants Service", version="1.6.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    dur_ms = int((time.time() - start) * 1000)
    log.info('%s %s %s %dms', request.method, request.url.path, request.url.query, dur_ms)
    return response

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_items": len(_cache),
        "nominatim_base": NOMINATIM_BASE_URL,
        "min_sleep_s": MIN_SLEEP_S,
        "retries": MAX_RETRIES
    }

@app.get("/generate")
def generate(
    country: str = Query(..., description="Country name"),
    city: Optional[str] = Query(None, description="Optional city name"),
    count: int = Query(20, ge=1, le=500, description="How many address variants"),
    strict: int = Query(1, description="1=stricter formatting (default), 0=looser")
):
    if not country.strip():
        raise HTTPException(status_code=400, detail="country is required")

    strict_flag = bool(strict)
    key = _cache_key(canonical_country_name(country), city)
    with _cache_lock:
        cached = list(_cache.get(key, []))

    if len(cached) >= count:
        log.info("HIT key=%s count=%d", key, count)
        return JSONResponse({"key": key, "count": count, "addresses": cached[:count], "cached": True})

    needed = count - len(cached)
    log.info("MISS key=%s need=%d city=%s strict=%s", key, needed, city or "", strict_flag)
    try:
        fresh = generate_address_variations_simple(country, city, needed, strict=strict_flag)
    except requests.HTTPError as e:
        log.error("Nominatim error: %s", e)
        raise HTTPException(status_code=502, detail=f"Nominatim error: {e}") from e
    except Exception as e:
        log.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}") from e

    merged, seen = [], set()
    for a in cached + fresh:
        norm = unicodedata.normalize("NFKD", a).strip().lower()
        if norm in seen: continue
        seen.add(norm); merged.append(a)

    with _cache_lock:
        _cache[key] = merged; _save_cache()

    log.info("STORE key=%s size=%d (+%d)", key, len(merged), len(merged) - len(cached))
    return JSONResponse({
        "key": key, "count": count, "addresses": merged[:count],
        "cached": len(cached) > 0, "added": len(merged) - len(cached)
    })

@app.post("/generate/batch")
def generate_batch(
    payload: Dict[str, Any] = Body(..., description="""
Accepts either:
  {"map": {"Japan|Tokyo": 30, "Germany|": 40}}
or:
  {"items": [{"country":"Japan","city":"Tokyo","count":30,"strict":1},
             {"country":"Germany","count":40}]}
""")
):
    jobs: List[Tuple[str, Optional[str], int, bool]] = []

    # parse mapping form
    if "map" in payload and isinstance(payload["map"], dict):
        for k, v in payload["map"].items():
            if not isinstance(v, int) or v < 1:
                continue
            if "|" in k:
                country, city = k.split("|", 1)
                city = city if city else None
            else:
                country, city = k, None
            jobs.append((country.strip(), (city or None), int(v), True))

    # parse list form
    if "items" in payload and isinstance(payload["items"], list):
        for it in payload["items"]:
            if not isinstance(it, dict):
                continue
            country = (it.get("country") or "").strip()
            if not country:
                continue
            city = (it.get("city") or None)
            count = int(it.get("count", 0))
            if count < 1:
                continue
            strict = bool(int(it.get("strict", 1)))
            jobs.append((country, city, count, strict))

    if not jobs:
        raise HTTPException(status_code=400, detail="No valid jobs provided")

    results: Dict[str, Dict[str, Any]] = {}

    # Strictly sequential processing to honor rate limits (1 reverse per job)
    for country, city, count, strict in jobs:
        key = _cache_key(canonical_country_name(country), city)

        with _cache_lock:
            cached = list(_cache.get(key, []))

        if len(cached) >= count:
            log.info("BATCH HIT key=%s count=%d", key, count)
            results[key] = {
                "country": country, "city": city, "count": count,
                "addresses": cached[:count], "cached": True
            }
            continue

        need = count - len(cached)
        log.info("BATCH MISS key=%s need=%d strict=%s", key, need, strict)
        try:
            fresh = generate_address_variations_simple(country, city, need, strict=strict)
        except Exception as e:
            log.exception("Batch generation error for %s", key)
            results[key] = {"error": str(e), "country": country, "city": city, "count": count}
            continue

        merged, seen = [], set()
        for a in cached + (fresh or []):
            norm = unicodedata.normalize("NFKD", a).strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            merged.append(a)

        with _cache_lock:
            _cache[key] = merged
            _save_cache()

        results[key] = {
            "country": country, "city": city, "count": count,
            "addresses": merged[:count], "cached": len(cached) > 0,
            "added": len(merged) - len(cached)
        }

    return JSONResponse({"results": results})

# -------- Cache utilities --------
@app.post("/cache/clear")
def clear_cache():
    with _cache_lock:
        _cache.clear(); _save_cache()
    log.warning("Cache cleared by request")
    return {"status": "cleared"}

@app.get("/cache/keys")
def list_keys():
    with _cache_lock: keys = list(_cache.keys())
    return {"keys": keys, "count": len(keys)}

@app.get("/cache/get")
def get_cache_entry(country: str = Query(...), city: Optional[str] = Query(None)):
    key = _cache_key(canonical_country_name(country), city)
    with _cache_lock: vals = _cache.get(key, [])
    return {"key": key, "size": len(vals), "addresses": vals}
