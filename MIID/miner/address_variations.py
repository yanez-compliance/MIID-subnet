#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import time
import math
import requests
from typing import Any, Dict, List, Optional, Tuple


def _ensure_project_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_on_path()
from MIID.validator.reward import (
    looks_like_address,
    validate_address_region,
    _grade_address_variations,
    check_with_nominatim,
)  # type: ignore
import geonamescache  # type: ignore


def load_seed_addresses(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seed_list = data.get("seed_names_with_labels", [])
    return [item.get("address", "").strip() for item in seed_list if isinstance(item, dict)]



def normalize_seed(seed: str) -> str:
    return seed.strip()


def resolve_seed_region(seed: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve seed string into (city, country) using geonamescache.

    - If seed matches a known country name → (None, country)
    - Else if seed matches a known city name anywhere → (city, country_of_city)
    - Else return (None, seed) to treat as country fallback
    """
    if not seed:
        return None, None
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    cities = gc.get_cities()

    seed_lower = seed.lower()

    # Check country exact match
    for code, data in countries.items():
        if data.get("name", "").lower() == seed_lower:
            return None, data["name"]

    # Check city exact match; fall back to contains
    best_city = None
    for city_id, cdata in cities.items():
        name = cdata.get("name", "")
        if name.lower() == seed_lower or seed_lower in name.lower():
            best_city = cdata
            break
    if best_city:
        country_code = best_city.get("countrycode")
        country_name = countries.get(country_code, {}).get("name")
        return best_city.get("name"), country_name

    # Fallback: treat as country string
    return None, seed


def build_address(street_num: int, street_name: str, city: str, postal: str, country: str) -> str:
    parts = [f"{street_num} {street_name}", f"{city} {postal}", country]
    return ", ".join([p for p in parts if p and p.strip()])


def _accept_language_for_seed(seed: str) -> str:
    s = seed.lower()
    if any(x in s for x in ["côte d'ivoire", "polynésie française", "albanie"]):
        return "fr-FR,fr;q=0.9"
    # Use French if non-ascii present (heuristic for localized names)
    if any(ord(ch) > 127 for ch in seed):
        return "fr-FR,fr;q=0.9"
    return "en-US,en;q=0.9"


def _get_place_bbox(seed: str, accept_language: str) -> Optional[Tuple[float, float, float, float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": seed,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "address-finder", "Accept-Language": accept_language}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        bb = arr[0].get("boundingbox")
        if not bb or len(bb) != 4:
            return None
        south, north, west, east = map(float, bb)
        return (south, north, west, east)
    except Exception:
        return None


def _get_place_info(seed: str, accept_language: str) -> Optional[Tuple[Tuple[float, float, float, float], Tuple[float, float]]]:
    """Return (bbox, center) for the seed using Nominatim search.

    bbox: (south, north, west, east)
    center: (lat, lon)
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": seed,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "address-finder", "Accept-Language": accept_language}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        item = arr[0]
        bb = item.get("boundingbox")
        if not bb or len(bb) != 4:
            return None
        south, north, west, east = map(float, bb)
        center_lat = float(item.get("lat"))
        center_lon = float(item.get("lon"))
        return (south, north, west, east), (center_lat, center_lon)
    except Exception:
        return None


def _reverse_geocode(lat: float, lon: float, accept_language: str) -> Optional[str]:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 18,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "address-finder", "Accept-Language": accept_language}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        obj = r.json()
        print(f"obj: {obj}")
        disp = obj.get("display_name")
        if disp and isinstance(disp, str):
            return disp
        return None
    except Exception:
        return None


def generate_variations_for_seed(seed_address: str, per_seed: int = 5) -> List[str]:
    variations: List[str] = []
    seed_norm = normalize_seed(seed_address)
    seed_norm = "France"
    if not seed_norm:
        return variations

    # Use Nominatim to find a bounding box for the seed, then reverse-geocode random points
    accept_lang = _accept_language_for_seed(seed_norm)
    place = _get_place_info(seed_norm, accept_lang)
    if not place:
        return variations

    bbox, center = place
    south, north, west, east = bbox
    if center and isinstance(center, tuple) and len(center) == 2:
        center_lat, center_lon = center
    else:
        center_lat = (south + north) / 2.0
        center_lon = (west + east) / 2.0

    results: List[str] = []
    last_api_call = 0.0
    attempts = 0
    max_attempts = max(per_seed * 24 * 2, 96)  # two full angle sweeps if needed

    # Precompute shuffled angles (0..345 step 15)
    base_angles = list(range(0, 360, 15))
    random.shuffle(base_angles)
    angle_idx = 0

    while len(results) < per_seed and attempts < max_attempts:
        attempts += 1
        # Select bearing and random distance up to 30km
        if angle_idx >= len(base_angles):
            random.shuffle(base_angles)
            angle_idx = 0
        bearing_deg = base_angles[angle_idx]
        angle_idx += 1
        distance_km = random.uniform(1.0, 30.0)

        # Compute offset from center
        bearing_rad = math.radians(bearing_deg)
        # Convert km offsets to degrees at center latitude
        delta_lat_deg = (distance_km * math.cos(bearing_rad)) / 111.0
        lon_scale = max(0.000001, math.cos(math.radians(center_lat)))
        delta_lon_deg = (distance_km * math.sin(bearing_rad)) / (111.32 * lon_scale)
        lat = center_lat + delta_lat_deg
        lon = center_lon + delta_lon_deg

        # Ensure candidate remains inside the bbox
        if not (south <= lat <= north and west <= lon <= east):
            continue

        # rate limit ~1 req/sec
        now = time.time()
        delta = now - last_api_call
        # if delta < 1.0:
        #     time.sleep(1.0 - delta)
        addr = _reverse_geocode(lat, lon, accept_lang)
        print(f"addr: {addr}")
        last_api_call = time.time()

        if not addr:
            continue
        if not looks_like_address(addr):
            continue
        if not validate_address_region(addr, seed_norm):
            continue

        # Optional: confirm via boolean check
        now = time.time()
        delta = now - last_api_call
        if delta < 1.0:
            time.sleep(1.0 - delta)
        ok = check_with_nominatim(addr)
        last_api_call = time.time()
        if ok is True:
            results.append(addr)

    return results


def build_variations_payload(seed_addresses: List[str], per_seed: int) -> Dict[str, List[List[str]]]:
    payload: Dict[str, List[List[str]]] = {}
    for idx, seed in enumerate(seed_addresses):
        key = f"seed_{idx}"
        addrs = generate_variations_for_seed(seed, per_seed=per_seed)
        # Each entry is [name_variation, dob_variation, address_variation]; we only need index 2
        payload[key] = [["", "", a] for a in addrs]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate address variations validated by _grade_address_variations")
    default_input = os.path.join(os.path.dirname(__file__), "..", "query", "0.json")
    parser.add_argument("--input", default=os.path.abspath(default_input), help="Path to input JSON with seed_names_with_labels")
    parser.add_argument("--per-seed", type=int, default=5, help="Number of address variations to generate per seed (API-validated when possible)")
    parser.add_argument("--output", default="address_variations_output.json", help="Path to write output JSON")
    args = parser.parse_args()

    seed_addresses = load_seed_addresses(args.input)
    if not seed_addresses:
        print("No seed addresses found in input JSON.")
        sys.exit(1)

    variations = build_variations_payload(seed_addresses, args.per_seed)

    # Grade using project validator
    score = _grade_address_variations(variations, seed_addresses, miner_metrics={})

    result: Dict[str, Any] = {
        "input": args.input,
        "per_seed": args.per_seed,
        "seed_addresses": seed_addresses,
        "variations": variations,
        "grading": score,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()


