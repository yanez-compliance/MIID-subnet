
import geonamescache
from typing import List, Optional

import os
import json
import hashlib
import requests


gc = geonamescache.GeonamesCache()
CITIES = gc.get_cities()

COUNTRIES = gc.get_countries()

def _gc_country_code(country_name: str) -> Optional[str]:
    for code, data in COUNTRIES.items():
        if data.get('name','').lower() == country_name.strip().lower():
            return code
    return None

def get_cities_by_country_code(cc: str) -> List[str]:
    return [c for _, c in CITIES.items() if c.get("countrycode") == cc]

def nominatim_reverse(lat, lon, lang="en"):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat, "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
        "accept-language": lang
    }
    headers = {"User-Agent": "YourAppName/1.0 (contact@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    addr = j.get("address", {})
    return {
        "display_name": j.get("display_name"),
        "road": addr.get("road"),
        "neighbourhood": addr.get("neighbourhood"),
        "city": addr.get("city") or addr.get("town") or addr.get("village"),
        "state": addr.get("state"),
        "postcode": addr.get("postcode"),
        "country": addr.get("country"),
        "country_code": addr.get("country_code")
    }

print(nominatim_reverse(35.681236, 139.767125))


def get_country_name_from_code(cc: str) -> Optional[str]:
    """
    Given a country code (cc), return the country name, or None if not found.
    """
    cdata = COUNTRIES.get(cc)
    if cdata:
        return cdata.get("name")
    return None

def get_geoinfo_from_lat_lon(lat, lon):
    # Cache geoinfo result to file by lat, lon

    def _latlon_cache_path(lat, lon):
        # Fuzzy round to 5 decimals for stability
        lat_ = round(float(lat), 5)
        lon_ = round(float(lon), 5)
        # Use a hash for extra safety as file name
        key = f"{lat_}_{lon_}"
        hash_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_dir = "latlon_cache"
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{hash_key}.json")

    path = _latlon_cache_path(lat, lon)
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data
        except Exception:
            # Read fail: fall through to fresh call
            pass

    data = nominatim_reverse(lat, lon)
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return data


def get_available_addr_list(seed_country, limit=10):
    cc = _gc_country_code(seed_country)
    seed_words = seed_country.split()
    city_list = []
    addr_list = []
    
    if cc is None:
        for city in CITIES:
            city_words = city.split()
            if city == seed_country or city_words[0] in seed_words or len(city_words) > 1 and city_words[1] in seed_words:
                print("matching special")
                country_name = get_country_name_from_code(city.get("countrycode"))
                city_list.append((city.get("name"), country_name, city.get("latitude"), city.get("longitude")))
                if len(city_list) >= limit:
                    break
        if len(city_list) == 0:
            print(f"No cities found for {seed_country}")
            return []
    else:
        city_infos = get_cities_by_country_code(cc)
        for city in city_infos:
            city_list.append((city.get("name"), seed_country, city.get("latitude"), city.get("longitude")))
            if len(city_list) >= limit:
                break
    
    for (city, country, lat, lon) in city_list:
        addr = country
        
        geoinfo = get_geoinfo_from_lat_lon(lat, lon)
        
        state = geoinfo.get("state")
        if state:
            addr = f"{state}, " + addr
        
        addr = f"{city}, " + addr

        road = geoinfo.get("road")
        if road:
            addr = f"{road}, " + addr
        
        addr_list.append(addr)
    
    return addr_list

addrs = ['Luhansk', 'Fiji', 'Montserrat', 'Burkina Faso', 'Iran', 'Kuwait', 'Serbia', 'Montserrat', 'Cameroon', 'Georgia', 'Bolivia', 'Brazil', 'Haiti', 'Belarus', 'Saint Pierre and Miquelon']

for addr in addrs:
    city_list = get_available_addr_list(addr)
    print(city_list)
    if len(city_list) == 0:
        print(f"No city list for {addr}")




