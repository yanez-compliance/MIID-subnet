# prefetch_daemon.py  (single-thread; city>=1, country>=100; resume+merge; rich logs; city sharding)
# python3 -m venv venv
# pip install requests geonamescache
# 
# Usage examples:
# python collect_city_info.py daemon --city-shard-index 0 --city-shard-count 10
# python collect_city_info.py search-city "Tokyo" --country "Japan" --num-addresses 20
# python collect_city_info.py search-city "Paris" --num-addresses 50
# python collect_city_info.py search-city "Tokyo" "Paris" "Berlin" --num-addresses 30
# python collect_city_info.py search-city "London" "Manchester" "Liverpool" --country "United Kingdom" --num-addresses 15
import geonamescache
from typing import List, Optional, Dict, Tuple
import os
import json
import requests
import time
import random
import math
import signal
import argparse
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from math import ceil
import re
import string

# ======================================================================================
# Static data
# ======================================================================================
gc = geonamescache.GeonamesCache()
CITIES = gc.get_cities()          # dict keyed by geonameid
COUNTRIES = gc.get_countries()    # dict keyed by ISO2 (upper)

SPECIAL_REGIONS = ["luhansk", "crimea", "donetsk", "west sahara"]
# ======================================================================================
# Config (env defaults)
# ======================================================================================
CONTACT_EMAIL = os.getenv("NOMINATIM_CONTACT", "AddressListService/1.0 (contact@example.com)")
DEFAULT_ADDRESSES_PER_CITY = int(os.getenv("ADDRESSES_PER_CITY", "1"))   # baseline >=1 per city
DEFAULT_MIN_ADDRESSES_PER_COUNTRY = int(os.getenv("MIN_ADDRS_PER_COUNTRY", "100"))
DEFAULT_MAX_ATTEMPTS_FACTOR = float(os.getenv("MAX_ATTEMPTS_FACTOR", "1.0"))  # attempts ≈ factor * target
DEFAULT_JITTER_RADIUS_M = int(os.getenv("JITTER_RADIUS_M", "5000"))           # 5km around center
DEFAULT_REVERSE_MIN_INTERVAL_S = float(os.getenv("REVERSE_MIN_INTERVAL_S", "0.5"))  # be nice to Nominatim
DEFAULT_CHECK_MIN_INTERVAL_S = float(os.getenv("CHECK_MIN_INTERVAL_S", "1.0"))      # validation rate limit

# Sharding defaults
DEFAULT_CITY_SHARD_INDEX = int(os.getenv("CITY_SHARD_INDEX", "0"))
DEFAULT_CITY_SHARD_COUNT = int(os.getenv("CITY_SHARD_COUNT", "1"))

# ======================================================================================
# Output layout
# ======================================================================================
ADDGEN_DIR = Path("addgen")
GEO_OUT_DIR = ADDGEN_DIR / "geo"
ADDRS_OUT_DIR = ADDGEN_DIR / "addrs"
DIAG_OUT_DIR = ADDGEN_DIR / "diag"   # optional diagnostics
for p in [GEO_OUT_DIR, ADDRS_OUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# In-memory caches (per run)
ADDR_CHECKED_CACHE: Dict[str, bool] = {}
GEO_CACHE: Dict[Tuple[float, float], dict] = {}

# graceful stop flag
SHOULD_STOP = False

# logger
logger = logging.getLogger("addgen")

# ======================================================================================
# Utils
# ======================================================================================

def looks_like_address(address: str) -> bool:
    address = address.strip().lower()

    # Keep all letters (Latin and non-Latin) and numbers
    # Using a more compatible approach for Unicode characters
    address_len = re.sub(r'[^\w]', '', address.strip(), flags=re.UNICODE)

    if len(address_len) < 30:
        logger.warning(f"Address has less than 30 characters: {address}")
        return False
    if len(address_len) > 300:  # maximum length check

        return False

    # Count letters (both Latin and non-Latin) - using \w which includes Unicode letters
    letter_count = len(re.findall(r'[^\W\d]', address, flags=re.UNICODE))
    if letter_count < 20:
        logger.warning(f"Address has less than 20 letters: {address}")
        return False

    if re.match(r"^[^a-zA-Z]*$", address):  # no letters at all
        logger.warning(f"Address has no letters: {address}")
        return False
    if len(set(address)) < 5:  # all chars basically the same
        logger.warning(f"Address has all same chars: {address}")
        return False
        
    # Has at least two digit (street number)
    number_groups = re.findall(r"\d+", address)
    if len(number_groups) < 2:
        logger.warning(f"Address has less than 2 numbers: {address}")
        return False

    if address.count(",") < 2:
        logger.warning(f"Address has less than 2 commas: {address}")
        return False
    
    # Check for special characters that should not be in addresses
    special_chars = ['`', ':', '%', '$', '@', '*', '^', '[', ']', '{', '}']
    if any(char in address for char in special_chars):
        logger.warning(f"Address has special characters: {address}")
        return False
    return True


# Global country name mapping to handle variations between miner submissions and geonames data
# All keys and values are lowercase for case-insensitive matching
COUNTRY_MAPPING = {
    # Korea variations
    "korea, south": "south korea",
    "korea, north": "north korea",
    
    # Cote d'Ivoire variations
    "cote d ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "ivory coast": "ivory coast",
    
    # Gambia variations
    "the gambia": "gambia",
    "gambia": "gambia",
    
    # Netherlands variations
    "netherlands": "the netherlands",
    "holland": "the netherlands",
    
    # Congo variations
    "congo, democratic republic of the": "democratic republic of the congo",
    "drc": "democratic republic of the congo",
    "congo, republic of the": "republic of the congo",
    
    # Burma/Myanmar variations
    "burma": "myanmar",

    # Bonaire variations
    'bonaire': 'bonaire, saint eustatius and saba',
    
    # Additional common variations
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "britain": "united kingdom",
    "uae": "united arab emirates",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "u.k.": "united kingdom",
}


def extract_city_country(address: str) -> tuple:
    if not address:
        return "", ""

    address = address.lower()
    
    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 2:
        return "", ""
    
    country = parts[-1]
    country = COUNTRY_MAPPING.get(country.lower(), country.lower())
    
    # If no country found, return empty
    if not country:
        return "", ""

    # Check each section from right to left (excluding the country)
    for i in range(2, len(parts) + 1):
        candidate_index = -i
        if abs(candidate_index) > len(parts):
            break
        
        candidate_part = parts[candidate_index]
        if not candidate_part:
            continue
            
        words = candidate_part.split()
        
        # Try different combinations of words (1-2 words max)
        # Start with 2 words, then 1 word for better city matching
        for num_words in range(len(words)):
            current_word = words[num_words]

            # Try current word
            candidates = [current_word]

            # Also try current + previous (if exists)
            if num_words > 0:
                prev_plus_current = words[num_words - 1] + " " + words[num_words]
                candidates.append(prev_plus_current)

            for city_candidate in candidates:
                # Skip if contains numbers or is too short
                if any(char.isdigit() for char in city_candidate):
                    continue

                # Validate the city exists in the country
                if city_in_country(city_candidate, country):
                    return city_candidate, country

    return "", country

def city_in_country(city_name: str, country_name: str) -> bool:
    """
    Check if a city is actually in the specified country using geonamescache.
    
    Args:
        city_name: Name of the city
        country_name: Name of the country
        
    Returns:
        True if city is in country, False otherwise
    """
    if not city_name or not country_name:
        return False
    
    try:
        cities, countries = CITIES, COUNTRIES
        
        city_name_lower = city_name.lower()
        country_name_lower = country_name.lower()
        
        # Find country code
        country_code = None
        for code, data in countries.items():
            if data.get('name', '').lower() == country_name_lower:
                country_code = code
                break
        
        if not country_code:
            return False
        
        # Only check cities that are actually in the specified country
        city_words = city_name_lower.split()
        
        for city_id, city_data in cities.items():
            # Skip cities not in the target country
            if city_data.get("countrycode", "") != country_code:
                continue
                
            city_data_name = city_data.get("name", "").lower()
            
            # Check exact match first
            if city_data_name == city_name_lower:
                return True
            # Check first word match
            elif len(city_words) >= 2 and city_data_name.startswith(city_words[0]):
                return True
            # Check second word match
            elif len(city_words) >= 2 and city_words[1] in city_data_name:
                return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Error checking city '{city_name}' in country '{country_name}': {str(e)}")
        return False

def validate_address_region(generated_address: str, seed_address: str) -> bool:
    """
    Validate that generated address has correct region from seed address.
    
    Special handling for disputed regions not in geonames:
    - Luhansk, Crimea, Donetsk, West Sahara
    
    Args:
        generated_address: The generated address to validate
        seed_address: The seed address to match against
        
    Returns:
        True if region is valid, False otherwise
    """
    if not generated_address or not seed_address:
        return False
    
    # Special handling for disputed regions not in geonames
    SPECIAL_REGIONS = ["luhansk", "crimea", "donetsk", "west sahara"]
    
    # Check if seed address is one of the special regions
    seed_lower = seed_address.lower()
    if seed_lower in SPECIAL_REGIONS:
        # If seed is a special region, check if that region appears in generated address
        gen_lower = generated_address.lower()
        return seed_lower in gen_lower
    
    # Extract city and country from both addresses
    gen_city, gen_country = extract_city_country(generated_address)
    seed_city, seed_country = seed_address.lower(), seed_address.lower()
    
    # If no city was extracted from generated address, it's an error
    if not gen_city:
        return False
    
    # If no country was extracted from generated address, it's an error
    if not gen_country:
        return False
    
    # Check if either city or country matches
    city_match = gen_city and seed_city and gen_city == seed_city
    country_match = gen_country and seed_country and gen_country == seed_country
    
    if not (city_match or country_match):
        logger.warning(f"No city or country match for generated address: {generated_address} and seed address: {seed_address}")
        return False
    
    # If we have both city and country, validate city is in country
    if gen_city and gen_country:
        return city_in_country(gen_city, gen_country)
    
    return True

def check_with_nominatim(address: str) -> bool:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json"}
        response = requests.get(url, params=params, headers={"User-Agent": "address-checker"}, timeout=10)
        return len(response.json()) > 0
    except requests.exceptions.Timeout:
        logging.warning(f"API timeout for address: {address}")
        return "TIMEOUT"
    except:
        return False

def init_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(lvl)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_country_name_from_code(cc: str) -> Optional[str]:
    cdata = COUNTRIES.get(cc)
    return cdata.get("name") if cdata else None

def slugify(s: str) -> str:
    out = []
    for ch in s.strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "unnamed"

def city_slug(city: dict) -> str:
    name = city.get("name", "UnknownCity")
    gid = str(city.get("geonameid", ""))
    return f"{slugify(name)}_{gid}" if gid else slugify(name)

def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# single-thread rate limiting
_last_reverse_ts = 0.0
_last_check_ts = 0.0
def rl_sleep(kind: str):
    global _last_reverse_ts, _last_check_ts
    now = time.monotonic()
    if kind == "reverse":
        wait = max(0.0, DEFAULT_REVERSE_MIN_INTERVAL_S - (now - _last_reverse_ts))
        if wait > 0:
            time.sleep(wait)
        _last_reverse_ts = time.monotonic()
    else:
        wait = max(0.0, DEFAULT_CHECK_MIN_INTERVAL_S - (now - _last_check_ts))
        if wait > 0:
            time.sleep(wait)
        _last_check_ts = time.monotonic()

# ======================================================================================
# Sharding helpers (deterministic city assignment)
# ======================================================================================
def city_shard_id(city: dict, shard_count: int) -> int:
    """
    Deterministically assign a city to a shard in [0, shard_count-1].
    Prefer geonameid; fallback to stable hash of name+cc+lat+lon.
    """
    if shard_count <= 1:
        return 0
    gid = city.get("geonameid")
    try:
        return int(gid) % shard_count
    except Exception:
        name = city.get("name", "")
        cc = city.get("countrycode", "")
        lat = str(city.get("latitude", ""))
        lon = str(city.get("longitude", ""))
        h = hashlib.md5(f"{name}|{cc}|{lat}|{lon}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) % shard_count

def shard_filter(cities: List[dict], shard_index: int, shard_count: int) -> List[dict]:
    if shard_count <= 1:
        return cities
    return [c for c in cities if city_shard_id(c, shard_count) == shard_index]

# ======================================================================================
# Nominatim reverse (in-memory only)
# ======================================================================================
def nominatim_search(city_name: str, country_name: str = None) -> Optional[dict]:
    """
    Search for a city using Nominatim search API to get coordinates.
    
    Args:
        city_name: Name of the city to search for
        country_name: Optional country name to narrow search
        
    Returns:
        Dictionary with city info including lat, lon, or None if not found
    """
    rl_sleep("reverse")  # Use same rate limiting
    url = "https://nominatim.openstreetmap.org/search"
    
    query = city_name
    if country_name:
        query = f"{city_name}, {country_name}"
    
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {"User-Agent": CONTACT_EMAIL}
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        results = r.json()
        
        if not results:
            logger.warning("nominatim_search: no results for '%s'", query)
            return None
        
        result = results[0]
        addr = result.get("address", {}) or {}
        
        city_info = {
            "name": city_name,
            "latitude": float(result.get("lat")),
            "longitude": float(result.get("lon")),
            "countrycode": addr.get("country_code", "").upper(),
            "geonameid": f"nominatim_{result.get('place_id', '')}",
            "display_name": result.get("display_name"),
        }
        
        logger.info("nominatim_search: found '%s' at (%.6f, %.6f)", 
                   city_name, city_info["latitude"], city_info["longitude"])
        return city_info
        
    except Exception as e:
        logger.warning("nominatim_search error for '%s': %s", query, e)
        return None

def nominatim_reverse(lat, lon, lang="en"):
    key = (round(float(lat), 5), round(float(lon), 5))
    if key in GEO_CACHE:
        return GEO_CACHE[key]

    rl_sleep("reverse")
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat, "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
        "accept-language": lang
    }
    headers = {"User-Agent": CONTACT_EMAIL}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        logger.warning("reverse_error lat=%.6f lon=%.6f err=%s", lat, lon, e)
        raise

    addr = j.get("address", {}) or {}
    data = {
        "display_name": j.get("display_name"),
        "road": addr.get("road"),
        "house_number": addr.get("house_number"),
        "neighbourhood": addr.get("neighbourhood"),
        "suburb": addr.get("suburb"),
        "city_district": addr.get("city_district"),
        "city": addr.get("city") or addr.get("town") or addr.get("village"),
        "state": addr.get("state"),
        "postcode": addr.get("postcode"),
        "country": addr.get("country"),
        "country_code": addr.get("country_code")
    }
    GEO_CACHE[key] = data
    return data

def get_geoinfo_from_lat_lon(lat, lon):
    return nominatim_reverse(lat, lon)

def find_city_by_name(city_name: str, country_name: str = None) -> Optional[dict]:
    """
    Find a city by name, first searching geonamescache, then falling back to Nominatim.
    
    Args:
        city_name: Name of the city to search for
        country_name: Optional country name to narrow search
        
    Returns:
        Dictionary with city info including lat, lon, or None if not found
    """
    city_name_lower = city_name.lower()
    country_code = None
    
    # If country name provided, find its code
    if country_name:
        country_name_lower = country_name.lower()
        # Apply country mapping
        country_name_lower = COUNTRY_MAPPING.get(country_name_lower, country_name_lower)
        
        for code, data in COUNTRIES.items():
            if data.get('name', '').lower() == country_name_lower:
                country_code = code
                break
    
    # Search in geonamescache first
    logger.info("Searching for city '%s' in geonamescache...", city_name)
    for city_id, city_data in CITIES.items():
        # If country specified, only check cities in that country
        if country_code and city_data.get("countrycode", "") != country_code:
            continue
        
        city_data_name = city_data.get("name", "").lower()
        
        # Check for exact match or close match
        if city_data_name == city_name_lower or city_name_lower in city_data_name:
            logger.info("Found city '%s' in geonamescache: %s (%s)", 
                       city_name, city_data.get("name"), city_data.get("countrycode"))
            return city_data
    
    # If not found in geonamescache, try Nominatim search
    logger.info("City '%s' not found in geonamescache, searching via Nominatim...", city_name)
    return nominatim_search(city_name, country_name)

# ======================================================================================
# Diagnostics helpers
# ======================================================================================
def diag_path(cc: str, city: dict) -> Path:
    return DIAG_OUT_DIR / cc.upper() / f"{city_slug(city)}.json"

def save_diagnostics(cc: str, city: dict, diag: dict):
    p = diag_path(cc, city)
    atomic_write_json(p, diag)

# ======================================================================================
# Address generation (multi per city) with fallbacks + diagnostics
# ======================================================================================
def _meters_to_deg_latlon(d_m: float, lat0_deg: float) -> Tuple[float, float]:
    dlat = d_m / 111_320.0
    dlon = d_m / (111_320.0 * max(0.000001, math.cos(math.radians(lat0_deg))))
    return dlat, dlon

def _random_point_near(lat0: float, lon0: float, radius_m: float) -> Tuple[float, float]:
    r = radius_m * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    dlat, dlon = _meters_to_deg_latlon(r, lat0)
    return lat0 + dlat * math.sin(theta), lon0 + dlon * math.cos(theta)

def _compose_address(geoinfo: dict, fallback_city: str, fallback_country: str) -> Optional[List[str]]:
    addresses = []
    parts = []
    display_name = geoinfo.get("display_name")
    road = geoinfo.get("road")
    neighbourhood = geoinfo.get("neighbourhood") or geoinfo.get("suburb") or geoinfo.get("city_district")

    city = fallback_city
    state = geoinfo.get("state")
    country = fallback_country
    postcode = geoinfo.get("postcode")
    if not postcode:
        postcode = make_postcode()
    house_number = geoinfo.get("house_number")
    if not house_number:
        house_number = make_house_number()
    
    # compose address
    if not city or not country:
        return None

    if road:
        parts.append(road)

    elif neighbourhood:
        parts.append(neighbourhood)

    parts.append(city)
    
    if state:
        parts.append(state)
    
    parts.append(country)
    
    generated_address = ", ".join(p for p in parts if p)
    
    addresses.append(display_name)
    addresses.append(generated_address)
    
    for idx, address in enumerate(addresses):
        if len(re.findall(r"\d+", address)) == 0:
            address_parts = address.split(", ")
            address_parts.insert(0, house_number)
            address_parts[len(address_parts) - 2] = postcode + " " + address_parts[len(address_parts) - 2]
            address = ", ".join(address_parts)
            addresses[idx] = ", ".join(address_parts)
        elif len(re.findall(r"\d+", address)) == 1:
            address_parts = address.split(", ")
            if len(re.findall(r"\d+", address_parts[0])) == 1:
                address_parts[len(address_parts) - 2] = postcode + " " + address_parts[len(address_parts) - 2]
            else:
                address_parts.insert(0, house_number)
            addresses[idx] = ", ".join(address_parts)
    
    return addresses

def make_postcode():
    return "".join(random.choices(string.digits, k=5))

def make_house_number():
    return "".join(random.choices(string.digits, k=3))

def validate_address(addr_str: str, seed_address: str) -> bool:
    # logger.info(f"validating address: {addr_str} with seed: {seed_address}")

    val = ADDR_CHECKED_CACHE.get(addr_str)
    if val is True:
        return True
    if val in (False, "TIMEOUT"):
        return False

    # Step 1: quick shape check
    looks_like = looks_like_address(addr_str)

    # Step 2: region constraint
    region_match = False
    if looks_like:
        region_match = validate_address_region(addr_str, seed_address)

    if not looks_like:
        logger.warning(f"address does not look like an address: {addr_str}")
        ADDR_CHECKED_CACHE[addr_str] = False
        return False

    if not region_match:
        logger.warning(f"NO SEED MATCH: {addr_str} : {seed_address}")
        ADDR_CHECKED_CACHE[addr_str] = False
        return False
    
    
    try:
        result = check_with_nominatim(addr_str)
        time.sleep(1)  # gentle pace for the search call
    except Exception as e:
        logger.warning("validate_error addr=%r err=%s", addr_str, e)
        result = False

    ADDR_CHECKED_CACHE[addr_str] = bool(result)
    return bool(result)

def generate_addresses_for_city(
    city: dict,
    target_total_for_city: int,
    jitter_radius_m: int,
    max_attempts: Optional[int],
    lang: str,
    min_required: int = 1,
    collect_diag: bool = False,
    existing_seen: Optional[set] = None
) -> Tuple[List[str], List[dict], dict]:
    """
    Returns (valid_addresses, samples, diag) for THIS RUN (not including preexisting).
    Ensures at least `min_required` are produced if possible via escalation.
    """
    name = city.get("name")
    cc = city.get("countrycode")
    lat0 = float(city.get("latitude"))
    lon0 = float(city.get("longitude"))
    country_name = get_country_name_from_code(cc) or ""
    state_fallback = None

    target = max(min_required, int(target_total_for_city))
    attempts_limit = int(max_attempts if max_attempts is not None else max(20, target * DEFAULT_MAX_ATTEMPTS_FACTOR))

    valid_addresses: List[str] = []
    samples: List[dict] = []
    seen_addresses: set = set(existing_seen or ())

    counters = {
        "attempts": 0,
        "reverse_ok": 0,
        "reverse_err": 0,
        "compose_none": 0,
        "duplicate": 0,
        "validated_true": 0,
        "validated_false": 0,
        "attempts_limit_hit": False,
        "fallback_city_country": False,
        "escalations": 0,
    }
    failures_examples = []

    # Escalating search radii to improve chances
    radii = [jitter_radius_m, int(jitter_radius_m*2), int(jitter_radius_m*4), int(jitter_radius_m*8)]
    for ridx, radius in enumerate(radii):
        while len(valid_addresses) < target and counters["attempts"] < attempts_limit and not SHOULD_STOP:
            counters["attempts"] += 1
            lat, lon = _random_point_near(lat0, lon0, radius)
            try:
                geoinfo = get_geoinfo_from_lat_lon(lat, lon)
                time.sleep(0.5)  # gentle pace for reverse calls
                counters["reverse_ok"] += 1
            except Exception as e:
                counters["reverse_err"] += 1
                if collect_diag and len(failures_examples) < 10:
                    failures_examples.append({"reason":"reverse_error","lat":lat,"lon":lon,"error":str(e)})
                continue

            addr_candidates = _compose_address(
                geoinfo,
                fallback_city=name,
                fallback_country=country_name,
            )
            
            logger.info(f"addr_candidates: {addr_candidates}")

            if not addr_candidates:
                counters["compose_none"] += 1
                if collect_diag and len(failures_examples) < 10:
                    failures_examples.append({"reason": "compose_none", "lat": lat, "lon": lon, "geoinfo": geoinfo})
                continue

            for addr_str in addr_candidates:
                if addr_str in seen_addresses:
                    counters["duplicate"] += 1
                    continue

                ok = validate_address(addr_str, seed_address=country_name)
                seen_addresses.add(addr_str)

                if ok:
                    counters["validated_true"] += 1
                    logger.info(f"OK address: {addr_str}")
                    valid_addresses.append(addr_str)
                    samples.append({"lat": lat, "lon": lon, "geoinfo": geoinfo, "address": addr_str})
                    break
                else:
                    counters["validated_false"] += 1
                    logger.info(f"NO address: {addr_str}")
                    if collect_diag and len(failures_examples) < 10:
                        failures_examples.append({"reason": "validate_false", "lat": lat, "lon": lon, "geoinfo": geoinfo, "address": addr_str})


        if len(valid_addresses) >= max(min_required, target) or counters["attempts"] >= attempts_limit or SHOULD_STOP:
            break
        counters["escalations"] += 1  # move to next, larger radius

    if counters["attempts"] >= attempts_limit and len(valid_addresses) < target:
        counters["attempts_limit_hit"] = True

    diag = {
        "city": name,
        "country_code": cc,
        "target_for_city": target,
        "produced_now": len(valid_addresses),
        "attempts_limit": attempts_limit,
        "counters": counters,
        "examples": failures_examples if collect_diag else [],
        "updated_at": _now_iso(),
    }
    return valid_addresses, samples, diag

def generate_addresses_for_city_from_samples(
    city: dict,
    target_total_for_city: int,
    samples: List[dict],
    min_required: int = 1,
    existing_seen: Optional[set] = None
) -> List[str]:
    """
    Returns (valid_addresses, samples, diag) for THIS RUN (not including preexisting).
    Ensures at least `min_required` are produced if possible via escalation.
    """
    name = city.get("name")
    cc = city.get("countrycode")
    country_name = get_country_name_from_code(cc) or ""

    valid_addresses: List[str] = []
    seen_addresses: set = set(existing_seen or ())
  

    # Escalating search radii to improve chances
    for sample in samples:
        addr_candidates = _compose_address(
            sample.get("geoinfo"),
            fallback_city=name,
            fallback_country=country_name,
        )
        
        for addr_str in addr_candidates:
            if addr_str in seen_addresses:
                continue
            ok = validate_address(addr_str, seed_address=country_name)
            seen_addresses.add(addr_str)

            if ok:
                logger.info(f"OK address: {addr_str}")
                valid_addresses.append(addr_str)
                break
    return valid_addresses

# ======================================================================================
# File IO (merge-friendly)
# ======================================================================================
def _addrs_path(cc: str, city: dict) -> Path:
    return ADDRS_OUT_DIR / cc.upper() / f"{city_slug(city)}.json"

def _geo_path(cc: str, city: dict) -> Path:
    return GEO_OUT_DIR / cc.upper() / f"{city_slug(city)}.json"

def load_existing_city(cc: str, city: dict) -> Tuple[List[str], List[dict]]:
    addrs_p = _addrs_path(cc, city)
    geo_p = _geo_path(cc, city)
    addrs = []
    samples = []
    try:
        if addrs_p.exists():
            with open(addrs_p, "r", encoding="utf-8") as f:
                d = json.load(f)
                addrs = list(d.get("addresses") or [])
        if geo_p.exists():
            with open(geo_p, "r", encoding="utf-8") as f:
                d = json.load(f)
                samples = list(d.get("samples") or [])
    except Exception as e:
        logger.warning("load_existing_error [%s] %s: %s", cc, city.get("name"), e)
    return addrs, samples

def save_city_outputs(cc: str, city: dict, addresses_all: List[str], samples_all: List[dict], target_for_city: int):
    addrs_path = _addrs_path(cc, city)
    geo_path = _geo_path(cc, city)

    addrs_data = {
        "country_code": cc,
        "country": get_country_name_from_code(cc),
        "city": city.get("name"),
        "geonameid": city.get("geonameid"),
        "target_addresses": target_for_city,
        "count": len(addresses_all),
        "addresses": addresses_all,
        "updated_at": _now_iso(),
        "_kind": "checked_addrs"
    }
    geo_data = {
        "country_code": cc,
        "country": get_country_name_from_code(cc),
        "city": city.get("name"),
        "geonameid": city.get("geonameid"),
        "target_addresses": target_for_city,
        "samples": samples_all,
        "updated_at": _now_iso(),
        "_kind": "geo_samples"
    }

    atomic_write_json(addrs_path, addrs_data)
    atomic_write_json(geo_path, geo_data)

# ======================================================================================
# Planning helpers (city>=1, country>=min_total)
# ======================================================================================
def planned_cities_for_country(cc: str) -> List[dict]:
    # stable order: by geonameid if present
    cities = [c for _, c in CITIES.items() if c.get("countrycode") == cc]
    try:
        cities.sort(key=lambda c: int(c.get("geonameid", 0)))
    except Exception:
        cities.sort(key=lambda c: (c.get("name",""), c.get("geonameid","")))
    return cities

def country_existing_total(cc: str, cities: List[dict]) -> Tuple[int, List[int]]:
    totals = []
    s = 0
    for city in cities:
        addrs, _ = load_existing_city(cc, city)
        n = len(addrs)
        totals.append(n)
        s += n
    return s, totals

# ======================================================================================
# Daemon
# ======================================================================================
def run_daemon(
    addresses_per_city: int = DEFAULT_ADDRESSES_PER_CITY,
    jitter_radius_m: int = DEFAULT_JITTER_RADIUS_M,
    countries_in: Optional[List[str]] = None,
    countries_exclude: Optional[List[str]] = None,
    resume: bool = True,
    max_attempts: Optional[int] = None,
    min_addresses_per_country: int = DEFAULT_MIN_ADDRESSES_PER_COUNTRY,
    collect_diag: bool = False,
    city_shard_index: int = DEFAULT_CITY_SHARD_INDEX,
    city_shard_count: int = DEFAULT_CITY_SHARD_COUNT,
):
    # Validate shard params
    if city_shard_count < 1:
        raise ValueError("--city-shard-count must be >= 1")
    if not (0 <= city_shard_index < city_shard_count):
        raise ValueError("--city-shard-index must satisfy 0 <= index < count")

    only_cc = [cc.upper() for cc in (countries_in or [])] or list(COUNTRIES.keys())
    exclude_cc = set([cc.upper() for cc in (countries_exclude or [])])

    planned = []
    per_country_plan = {}  # cc -> list of (city, target_total_for_city)

    logger.info("Shard configuration: index=%d, count=%d", city_shard_index, city_shard_count)

    # Build plan per country (targets computed across ALL cities of the country)
    for cc in only_cc:
        if cc in exclude_cc:
            continue
        all_cities = planned_cities_for_country(cc)
        if not all_cities:
            logger.warning("[%s] No cities found", cc)
            continue

        # existing totals (from any prior runs possibly done by other shards/machines)
        existing_total, existing_per_city = country_existing_total(cc, all_cities)

        # Step A: ensure >=1 per city baseline (or provided baseline)
        base_per_city = max(1, addresses_per_city)
        desired_per_city = [max(base_per_city, existing_per_city[i]) for i in range(len(all_cities))]

        # Step B: ensure >= min per country (computed across ALL cities for this country)
        current_sum = sum(desired_per_city)
        if current_sum < min_addresses_per_country:
            shortfall = min_addresses_per_country - current_sum
            extra_each = ceil(shortfall / len(all_cities))
            desired_per_city = [desired_per_city[i] + extra_each for i in range(len(all_cities))]

        # Now apply SHARD FILTER so each machine gets a disjoint subset
        shard_cities = shard_filter(all_cities, city_shard_index, city_shard_count)

        # Prepare city list for this shard
        city_targets = []
        for city in shard_cities:
            i = all_cities.index(city)  # use index in all_cities to pick the right target
            target_total = desired_per_city[i]
            if resume:
                # If already >= target, skip
                if existing_per_city[i] >= target_total:
                    continue
            city_targets.append((city, target_total))

        if not city_targets:
            logger.info("[%s] Shard %d/%d — already satisfies targets for assigned cities",
                        cc, city_shard_index, city_shard_count)
            continue

        per_country_plan[cc] = city_targets
        planned.extend([(cc, city, target) for city, target in city_targets])

        logger.info("[%s] plan (shard %d/%d): assigned_cities=%d/%d, existing_total=%d, "
                    "country_min=%d, per-city baseline=%d",
                    cc, city_shard_index, city_shard_count, len(shard_cities), len(all_cities),
                    existing_total, min_addresses_per_country, base_per_city)

    total_cities = len(planned)
    processed = 0
    logger.info("Starting (single-thread). Planned city tasks for shard %d/%d: %d",
                city_shard_index, city_shard_count, total_cities)

    for cc, city, target_total_for_city in planned:
        if SHOULD_STOP:
            break

        city_name = city.get("name")
        # Load existing and compute how many more we need for this city
        existing_addrs, existing_samples = load_existing_city(cc, city)
        need = max(0, target_total_for_city - len(existing_addrs))
        if need == 0 and resume:
            processed += 1
            logger.info("[%d/%d] [%s] %s already has %d (>= target %d) — skipped",
                        processed, total_cities, cc, city_name, len(existing_addrs), target_total_for_city)
            continue

        try:
            re_checked_addrs = []
            if not resume:
                re_checked_addrs = generate_addresses_for_city_from_samples(
                    city,
                    target_total_for_city=need if resume else target_total_for_city,
                    samples=existing_samples,
                    existing_seen=set(existing_addrs),
                )

            new_addrs, new_samples, diag = generate_addresses_for_city(
                city,
                target_total_for_city=need if resume else target_total_for_city,
                jitter_radius_m=jitter_radius_m,
                max_attempts=max_attempts,
                lang="en",
                min_required=1,  # enforce at least one NEW when we need any
                collect_diag=collect_diag or logger.level <= logging.DEBUG,
                existing_seen=set(existing_addrs),
            )

            # Merge with existing (dedupe, preserve order: existing first)
            merged_addrs = list(dict.fromkeys(existing_addrs + new_addrs + re_checked_addrs))
            merged_samples = existing_samples + new_samples

            save_city_outputs(cc, city, merged_addrs, merged_samples, target_total_for_city)
            processed += 1

            if new_addrs:
                logger.info("[%d/%d] [%s] %s -> +%d (now %d / target %d)",
                            processed, total_cities, cc, city_name,
                            len(new_addrs), len(merged_addrs), target_total_for_city)
            else:
                c = diag["counters"]
                logger.warning(
                    "[%d/%d] [%s] %s -> +0 (need %d) attempts=%d ok=%d rev_err=%d compose_none=%d dup=%d val_true=%d val_false=%d escalations=%d fallback=%s limit=%s",
                    processed, total_cities, cc, city_name, need,
                    c["attempts"], c["reverse_ok"], c["reverse_err"], c["compose_none"], c["duplicate"],
                    c["validated_true"], c["validated_false"], c["escalations"],
                    c["fallback_city_country"], c["attempts_limit_hit"]
                )

            if collect_diag:
                DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)
                save_diagnostics(cc, city, diag)

        except Exception as e:
            processed += 1
            logger.error("[%d/%d] [%s] %s ERROR: %s", processed, total_cities, cc, city_name, e)

    logger.info("Done (shard %d/%d). Processed %d/%d city tasks.",
                city_shard_index, city_shard_count, processed, total_cities)

# ======================================================================================
# Search by city name
# ======================================================================================
def run_search_city_single(
    city_name: str,
    country_name: Optional[str] = None,
    num_addresses: int = 10,
    jitter_radius_m: int = DEFAULT_JITTER_RADIUS_M,
    max_attempts: Optional[int] = None,
    collect_diag: bool = False,
) -> bool:
    """
    Search for a single city by name and generate addresses for it.
    
    Args:
        city_name: Name of the city to search for
        country_name: Optional country name to narrow search
        num_addresses: Number of addresses to generate
        jitter_radius_m: Radius in meters for address jittering
        max_attempts: Maximum attempts to generate addresses
        collect_diag: Whether to collect diagnostics
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Searching for city: %s%s", city_name, 
                f" in {country_name}" if country_name else "")
    
    # Find the city
    city = find_city_by_name(city_name, country_name)
    
    if not city:
        logger.error("City '%s' not found!", city_name)
        return False
    
    # Extract city info
    city_name_resolved = city.get("name")
    cc = city.get("countrycode")
    lat = city.get("latitude")
    lon = city.get("longitude")
    
    if not lat or not lon:
        logger.error("Could not get coordinates for city '%s'", city_name)
        return False
    
    logger.info("Found city: %s (%s) at coordinates (%.6f, %.6f)", 
                city_name_resolved, cc, lat, lon)
    
    # Load existing addresses if any
    existing_addrs, existing_samples = load_existing_city(cc, city)
    logger.info("Existing addresses: %d", len(existing_addrs))
    
    # Generate addresses
    logger.info("Generating %d addresses for %s...", num_addresses, city_name_resolved)
    
    try:
        new_addrs, new_samples, diag = generate_addresses_for_city(
            city,
            target_total_for_city=num_addresses,
            jitter_radius_m=jitter_radius_m,
            max_attempts=max_attempts,
            lang="en",
            min_required=1,
            collect_diag=collect_diag or logger.level <= logging.DEBUG,
            existing_seen=set(existing_addrs),
        )
        
        # Merge with existing
        merged_addrs = list(dict.fromkeys(existing_addrs + new_addrs))
        merged_samples = existing_samples + new_samples
        
        # Save outputs
        save_city_outputs(cc, city, merged_addrs, merged_samples, num_addresses)
        
        logger.info("Generated %d new addresses (total: %d)", len(new_addrs), len(merged_addrs))
        
        if new_addrs:
            logger.info("Sample addresses:")
            for i, addr in enumerate(new_addrs[:5], 1):
                logger.info("  %d. %s", i, addr)
        else:
            c = diag["counters"]
            logger.warning(
                "Could not generate addresses. Stats: attempts=%d ok=%d rev_err=%d compose_none=%d dup=%d val_true=%d val_false=%d",
                c["attempts"], c["reverse_ok"], c["reverse_err"], c["compose_none"], c["duplicate"],
                c["validated_true"], c["validated_false"]
            )
        
        if collect_diag:
            DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)
            save_diagnostics(cc, city, diag)
            logger.info("Diagnostics saved to: %s", diag_path(cc, city))
        
        return True
        
    except Exception as e:
        logger.error("Error processing city '%s': %s", city_name_resolved, e)
        return False

def run_search_cities(
    city_names: List[str],
    country_name: Optional[str] = None,
    num_addresses: int = 10,
    jitter_radius_m: int = DEFAULT_JITTER_RADIUS_M,
    max_attempts: Optional[int] = None,
    collect_diag: bool = False,
):
    """
    Search for multiple cities by name and generate addresses for each.
    
    Args:
        city_names: List of city names to search for
        country_name: Optional country name to narrow search (applies to all cities)
        num_addresses: Number of addresses to generate per city
        jitter_radius_m: Radius in meters for address jittering
        max_attempts: Maximum attempts to generate addresses
        collect_diag: Whether to collect diagnostics
    """
    total_cities = len(city_names)
    logger.info("Processing %d cities...", total_cities)
    
    results = {"successful": 0, "failed": 0, "cities": []}
    
    for idx, city_name in enumerate(city_names, 1):
        if SHOULD_STOP:
            logger.info("Stopping early due to signal...")
            break
        
        logger.info("\n[%d/%d] Processing city: %s", idx, total_cities, city_name)
        logger.info("=" * 80)
        
        success = run_search_city_single(
            city_name=city_name,
            country_name=country_name,
            num_addresses=num_addresses,
            jitter_radius_m=jitter_radius_m,
            max_attempts=max_attempts,
            collect_diag=collect_diag,
        )
        
        if success:
            results["successful"] += 1
            results["cities"].append({"name": city_name, "status": "success"})
        else:
            results["failed"] += 1
            results["cities"].append({"name": city_name, "status": "failed"})
        
        logger.info("=" * 80 + "\n")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Total cities: %d", total_cities)
    logger.info("Successful: %d", results["successful"])
    logger.info("Failed: %d", results["failed"])
    logger.info("\nDetailed results:")
    for city_result in results["cities"]:
        status_symbol = "✓" if city_result["status"] == "success" else "✗"
        logger.info("  %s %s", status_symbol, city_result["name"])
    logger.info("=" * 80)

# ======================================================================================
# CLI
# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="AddGen daemon (single-thread) — city≥1, country≥N; outputs in addgen/geo and addgen/addrs")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daemon", help="Run address generation over countries/cities")
    d.add_argument("--addresses-per-city", type=int, default=DEFAULT_ADDRESSES_PER_CITY,
                   help="Baseline target per city (at least 1). Country min may raise this.")
    d.add_argument("--min-addresses-per-country", type=int, default=DEFAULT_MIN_ADDRESSES_PER_COUNTRY,
                   help="Minimum total addresses to generate per country (default 100).")
    d.add_argument("--jitter-radius-m", type=int, default=DEFAULT_JITTER_RADIUS_M)
    d.add_argument("--country-in", nargs="*", default=None, help="Only these ISO2 country codes (e.g., JP US DE). Default = all.")
    d.add_argument("--country-exclude", nargs="*", default=None, help="Exclude these ISO2 country codes.")
    d.add_argument("--no-resume", action="store_true", help="Ignore existing addgen files; recompute targets regardless of current counts.")
    d.add_argument("--max-attempts", type=int, default=None, help="Override max attempts per city (default scales with target).")
    d.add_argument("--diag", action="store_true", help="Write per-city diagnostics JSON under addgen/diag.")

    # New: city sharding controls for multi-machine runs
    d.add_argument("--city-shard-index", type=int, default=DEFAULT_CITY_SHARD_INDEX,
                   help="This machine's shard index (0-based).")
    d.add_argument("--city-shard-count", type=int, default=DEFAULT_CITY_SHARD_COUNT,
                   help="Total number of shards/machines (>=1).")

    d.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")

    # New: search by city name (supports multiple cities)
    s = sub.add_parser("search-city", help="Search for one or more cities by name and generate addresses")
    s.add_argument("city_names", type=str, nargs="+", help="Name(s) of the city/cities to search for (space-separated for multiple)")
    s.add_argument("--country", type=str, default=None, help="Optional country name to narrow search (applies to all cities)")
    s.add_argument("--num-addresses", type=int, default=10, help="Number of addresses to generate per city (default: 10)")
    s.add_argument("--jitter-radius-m", type=int, default=DEFAULT_JITTER_RADIUS_M, help="Radius in meters for address jittering")
    s.add_argument("--max-attempts", type=int, default=None, help="Maximum attempts to generate addresses per city")
    s.add_argument("--diag", action="store_true", help="Write diagnostics JSON")
    s.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")

    return p.parse_args()

def _signal_handler(sig, frame):
    global SHOULD_STOP
    logger.info("Received signal %s. Stopping after current city…", sig)
    SHOULD_STOP = True

if __name__ == "__main__":
    args = parse_args()
    init_logging(args.log_level)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.cmd == "daemon":
        run_daemon(
            addresses_per_city=args.addresses_per_city,
            jitter_radius_m=args.jitter_radius_m,


            countries_in=args.country_in,
            countries_exclude=args.country_exclude,
            resume=not args.no_resume,
            max_attempts=args.max_attempts,
            min_addresses_per_country=args.min_addresses_per_country,
            collect_diag=args.diag,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
        )
    elif args.cmd == "search-city":
        run_search_cities(
            city_names=args.city_names,
            country_name=args.country,
            num_addresses=args.num_addresses,
            jitter_radius_m=args.jitter_radius_m,
            max_attempts=args.max_attempts,
            collect_diag=args.diag,
        )

