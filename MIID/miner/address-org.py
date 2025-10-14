#!/usr/bin/env python3
# service.py

from __future__ import annotations
import os, json, time, math, random, threading, unicodedata, logging
from typing import Dict, List, Optional, Tuple, Set, Any

import requests
from fastapi import FastAPI, Query, HTTPException, Request, Body
from fastapi.responses import JSONResponse
import geonamescache
from unidecode import unidecode

import asyncio

BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "10"))


# =========================
# Logging
# =========================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.DEBUG),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("addr-service")
DEBUG_RESOLVE = os.environ.get("DEBUG_RESOLVE", "0") == "1"

# =========================
# Config
# =========================
CACHE_PATH = os.environ.get("ADDR_CACHE_PATH", "./address_cache.json")
USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", "addr-variants/1.4 (contact: you@example.com)")
SEARCH_URL = "https://nominatim.openstreetmap.org/search"
REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
REQ_TIMEOUT = 15
MIN_SLEEP_S = 1.05
JITTER_RADIUS_M = (2500, 12000)
OVERSAMPLE_FACTOR = 3

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
    return alias.get(n, name)

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
# Nominatim
# =========================
_last_call_ts = 0.0
def _polite_get(url: str, params: Dict) -> requests.Response:
    global _last_call_ts
    now = time.time(); elapsed = now - _last_call_ts
    if elapsed < MIN_SLEEP_S: time.sleep(MIN_SLEEP_S - elapsed)
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
    _last_call_ts = time.time()
    resp.raise_for_status(); return resp

def nominatim_search_place(q: str, country_code_hint: Optional[str] = None) -> Optional[Tuple[float, float, Dict]]:
    params = {"q": q, "format": "jsonv2", "limit": 1,
              "addressdetails": 1, "extratags": 1, "accept-language": "en"}
    if country_code_hint: params["countrycodes"] = country_code_hint.lower()
    js = _polite_get(SEARCH_URL, params).json()
    if not js: return None
    hit = js[0]; return float(hit["lat"]), float(hit["lon"]), hit

def nominatim_reverse(lat: float, lon: float) -> Optional[Dict]:
    params = {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}",
              "format": "jsonv2", "addressdetails": 1, "accept-language": "en"}
    return _polite_get(REVERSE_URL, params).json()

def nominatim_search_text(text: str) -> bool:
    try:
        params = {"q": text, "format": "jsonv2", "limit": 1, "accept-language": "en"}
        return bool(_polite_get(SEARCH_URL, params).json())
    except Exception as e:
        log.debug("DEBUG_RESOLVE failed for '%s': %s", text, e); return False

# =========================
# City selection
# =========================
def pick_seed_cities(country: str, city: Optional[str], k: int = 1) -> List[str]:
    if city: return [city]
    gc = geonamescache.GeonamesCache()
    code = None
    for cc, data in gc.get_countries().items():
        if data.get("name", "").lower() == country.strip().lower():
            code = cc; break
    if not code: return []
    cands = [(c.get("name"), c.get("population", 0))
             for _, c in gc.get_cities().items() if c.get("countrycode") == code]
    seen, result = set(), []
    for nm, _ in cands:
        if nm and nm.lower() not in seen:
            result.append(nm); seen.add(nm.lower())
        if len(result) >= k: break
    return result

def nearest_gc_city(lat: float, lon: float, country: str) -> Optional[str]:
    """Prefer larger cities to avoid neighborhood/ward names."""
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
# Address formatter (harder/stricter)
# =========================
def compose_address(front: Optional[str], suburb: Optional[str], state: Optional[str],
                    postcode: Optional[str], city_final: str, country_canon: str) -> str:
    parts = []
    if front: parts.append(front)     # e.g., "12 Main St"
    if suburb: parts.append(suburb)
    if state: parts.append(state)
    if postcode: parts.append(postcode)  # digits before city
    parts.append(city_final)             # penultimate
    parts.append(country_canon)          # last
    return ", ".join(p for p in parts if p)

def format_address_from_osm(json_obj: Dict, country_hint: str, lat: float, lon: float,
                            strict: bool = True) -> Optional[str]:
    if not json_obj: return None
    addr = json_obj.get("address", {})
    country_canon = canonical_country_name(addr.get("country") or country_hint or "")
    if not country_canon: return None

    # Only use street/house info. Never amenity/business names.
    road = (addr.get("road") or addr.get("pedestrian") or addr.get("residential") or
            addr.get("footway") or addr.get("path"))
    house = addr.get("house_number")
    suburb = addr.get("neighbourhood") or addr.get("suburb")
    state  = addr.get("state") or addr.get("region") or addr.get("province")
    postcode = addr.get("postcode")

    # HARDER: require a road; if strict, also require either house_number or postcode.
    if not road:
        return None
    if strict and not (house or postcode):
        return None

    front = f"{house} {road}".strip() if house else road  # road is guaranteed here

    city_final = choose_city_for_grader(json_obj, lat, lon, country_canon)
    
    if not city_final:
        print(f"city_final is None!!")
        return None

    return compose_address(front, suburb, state, postcode, city_final, country_canon)

# =========================
# Generator
# =========================
def generate_address_variations(country: str, city: Optional[str], count: int, strict: bool = False) -> List[str]:
    country_canon = canonical_country_name(country)
    country_code = _gc_country_code(country_canon)

    seeds = pick_seed_cities(country_canon, city, k=1) or [None]
    centers: List[Tuple[float, float, str]] = []
    for seed_city in seeds:
        try:
            q = f"{seed_city}, {country_canon}" if seed_city else country_canon
            res = nominatim_search_place(q, country_code)
            if res:
                lat, lon, _ = res; centers.append((lat, lon, seed_city or ""))
        except Exception as e:
            log.debug("Seed search failed for %s: %s", seed_city or country_canon, e)
    if not centers: return []

    # oversample more when strict to meet count
    oversample = OVERSAMPLE_FACTOR + (1 if strict else 0)
    target_samples = max(count * oversample, count + 12)

    results: List[str] = []
    seen_norm: Set[str] = set()
    attempts, center_idx = 0, 0

    while len(results) < count and attempts < target_samples:
        lat0, lon0, _ = centers[center_idx]; center_idx = (center_idx + 1) % len(centers)
        r = random.uniform(*JITTER_RADIUS_M); theta = random.uniform(0, 2 * math.pi)
        lat = lat0 + meters_to_deg_lat(r * math.sin(theta))
        lon = lon0 + meters_to_deg_lon(r * math.cos(theta), lat0)

        try:
            rev = nominatim_reverse(lat, lon)
        except Exception as e:
            attempts += 1; log.debug("Reverse failed: %s", e); continue
        attempts += 1

        addr_str = format_address_from_osm(rev, country_canon, lat, lon, strict=strict)
        if not addr_str or not looks_like_address(addr_str): 
            print(f"addr_str is None or not looks_like_address!!")
            continue

        parts = [p.strip() for p in addr_str.split(",")]
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

        if DEBUG_RESOLVE:
            ok = nominatim_search_text(addr_str)
            log.info("RESOLVE_TEST address='%s' nominatim_ok=%s", addr_str, ok)

        norm = unicodedata.normalize("NFKD", addr_str).strip().lower()
        if norm in seen_norm: continue
        seen_norm.add(norm); results.append(addr_str)

    # Top-up loop (still strict by default)
    # if len(results) < count:
    #     extra_needed = count - len(results)
    #     extra_samples = extra_needed * (6 if strict else 4)
    #     for _ in range(extra_samples):
    #         lat0, lon0, _ = random.choice(centers)
    #         r = random.uniform(*JITTER_RADIUS_M); theta = random.uniform(0, 2 * math.pi)
    #         lat = lat0 + meters_to_deg_lat(r * math.sin(theta))
    #         lon = lon0 + meters_to_deg_lon(r * math.cos(theta), lat0)
    #         try:
    #             rev = nominatim_reverse(lat, lon)
    #         except Exception:
    #             continue
    #         addr_str = format_address_from_osm(rev, country_canon, lat, lon, strict=strict)
    #         if not addr_str or not looks_like_address(addr_str): continue

    #         parts = [p.strip() for p in addr_str.split(",")]
    #         gen_country = parts[-1] if parts else ""
    #         gen_city = ""
    #         if len(parts) >= 2:
    #             for idx in range(len(parts) - 2, -1, -1):
    #                 candidate = parts[idx]
    #                 if any(ch.isdigit() for ch in candidate): continue
    #                 words = [w for w in candidate.split() if not any(c.isdigit() for c in w) and len(w) > 1]
    #                 if words: gen_city = " ".join(words[-2:]); break
    #         if canonical_country_name(gen_country) != country_canon: continue
    #         if not gen_city or not city_in_country_gc(gen_city, country_canon): continue

    #         if DEBUG_RESOLVE:
    #             ok = nominatim_search_text(addr_str)
    #             log.info("RESOLVE_TEST address='%s' nominatim_ok=%s", addr_str, ok)

    #         norm = unicodedata.normalize("NFKD", addr_str).strip().lower()
    #         if norm in seen_norm: continue
    #         seen_norm.add(norm); results.append(addr_str)
    #         if len(results) >= count: break

    return results[:count]

# =========================
# FastAPI
# =========================
app = FastAPI(title="Address Variants Service", version="1.4.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    dur_ms = int((time.time() - start) * 1000)
    log.info('%s %s %s %dms', request.method, request.url.path, request.url.query, dur_ms)
    return response

@app.get("/health")
def health():
    return {"status": "ok", "cache_items": len(_cache)}

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
        fresh = generate_address_variations(country, city, needed, strict=strict_flag)
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
async def generate_batch(
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
            # "Country|City" or "Country|" or "Country"
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

    sem = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def _process_job(country: str, city: Optional[str], count: int, strict: bool):
        key = _cache_key(canonical_country_name(country), city)

        # fast path: check cache snapshot
        with _cache_lock:
            cached = list(_cache.get(key, []))
        if len(cached) >= count:
            log.info("BATCH HIT key=%s count=%d", key, count)
            return key, {
                "country": country, "city": city, "count": count,
                "addresses": cached[:count], "cached": True
            }

        need = count - len(cached)
        log.info("BATCH MISS key=%s need=%d strict=%s", key, need, strict)

        # run the blocking generator off the event loop
        async with sem:
            try:
                fresh = await asyncio.to_thread(
                    generate_address_variations, country, city, need, strict
                )
            except Exception as e:
                log.exception("Batch generation error for %s", key)
                return key, {"error": str(e), "country": country, "city": city, "count": count}

        # merge + dedup + persist
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

        return key, {
            "country": country, "city": city, "count": count,
            "addresses": merged[:count], "cached": len(cached) > 0,
            "added": len(merged) - len(cached)
        }

    # fire all jobs concurrently (unordered completion)
    tasks = [asyncio.create_task(_process_job(cn, ct, n, st)) for (cn, ct, n, st) in jobs]
    pairs = await asyncio.gather(*tasks)

    # build results dict from completed tasks
    results: Dict[str, Dict[str, Any]] = {k: v for k, v in pairs}
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
