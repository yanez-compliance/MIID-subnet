# addr_service.py
# Serve addresses from pre-fetched DB (addgen/addrs/<CC>/*.json)
# - country_code -> [addresses] pool (loaded from disk, reloadable)
# - POST /locations: allocate unique addresses per (round_id, seed) with persistence
# - GET /stats: pool sizes + rounds tracked
# - POST /reload: reload pools from disk
# - GET /rounds: list known rounds and seeds
# - GET /rounds/{round_id}: detail for a round
# - POST /rounds/reset: reset one round or all
#
# Environment (optional):
#   ADDGEN_ADDRS_DIR="addgen/addrs"
#   STATE_DIR="state"
#   STATE_FILE="allocations.json"
#   AUTOSAVE_INTERVAL_SEC="5"            # background autosave period
#   LOG_LEVEL="INFO"
#   IMMEDIATE_SAVE="0"                   # "1" to write after each change

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
import threading
import random
import geonamescache
import os
import time
from contextlib import contextmanager
import random

# ------------------------------------------------------------------------------
# Config & Logging
# ------------------------------------------------------------------------------
ADDRS_BASE_DIR = Path(os.getenv("ADDGEN_ADDRS_DIR", "addgen/addrs")).resolve()
STATE_DIR = Path(os.getenv("STATE_DIR", "state")).resolve()
STATE_FILE = os.getenv("STATE_FILE", "allocations.json")
STATE_PATH = STATE_DIR / STATE_FILE
AUTOSAVE_INTERVAL_SEC = float(os.getenv("AUTOSAVE_INTERVAL_SEC", "5"))
IMMEDIATE_SAVE = os.getenv("IMMEDIATE_SAVE", "0") == "1"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("addrsvc")

# ------------------------------------------------------------------------------
# Country metadata
# ------------------------------------------------------------------------------
_gc = geonamescache.GeonamesCache()
COUNTRIES: Dict[str, Dict] = _gc.get_countries()  # ISO2 -> data

# Build helpers for name lookup (case-insensitive)
_COUNTRY_NAME_TO_CODE: Dict[str, str] = {v["name"].strip().lower(): k for k, v in COUNTRIES.items()}
_ALIASES = {
    "united states of america": "US",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "uk": "GB",
    "united kingdom": "GB",
    "russia": "RU",
    "south korea": "KR",
    "korea, south": "KR",
    "north korea": "KP",
    "korea, north": "KP",
    "czech republic": "CZ",
    "viet nam": "VN",
    "hong kong": "HK",
    "macao": "MO",
    "uae": "AE",
    "united arab emirates": "AE",
}
_COUNTRY_NAME_TO_CODE.update(_ALIASES)

def normalize_country_code(cc: str) -> Optional[str]:
    if not cc:
        return None
    cc = cc.strip().upper()
    return cc if cc in COUNTRIES else None

def infer_country_code_from_seed(seed: str) -> Optional[str]:
    """
    Try to map a seed string (which might be 'JP', 'Japan', or 'Tokyo, Japan')
    to an ISO2 country code.
    """
    if not seed:
        return None
    s = seed.strip()

    # ISO2 direct
    cc = normalize_country_code(s)
    if cc:
        return cc

    # Last token (e.g., "Tokyo, Japan" -> "Japan"), then whole string.
    parts = [p.strip() for p in s.split(",") if p.strip()]
    candidates = []
    if parts:
        candidates.append(parts[-1])
    candidates.append(s)

    for cand in candidates:
        low = cand.lower()
        if low in _COUNTRY_NAME_TO_CODE:
            return _COUNTRY_NAME_TO_CODE[low]
        for name, code in _COUNTRY_NAME_TO_CODE.items():
            if name in low or low in name:
                return code
    return None


def normalize_address(addr_str):
    """Normalize address string by removing extra spaces and standardizing format"""
    if not addr_str:
        return ""
    # Remove extra spaces, convert to lowercase, and standardize common separators
    normalized = " ".join(addr_str.split()).lower()
    # Replace common separators with spaces
    normalized = normalized.replace(",", " ").replace(";", " ").replace("-", " ")
    # Remove multiple spaces
    normalized = " ".join(normalized.split())
    return normalized
            
def _seed_key(seed: str) -> str:
    return (seed or "").strip().lower()

# ------------------------------------------------------------------------------
# In-memory pools and persistent usage state
# ------------------------------------------------------------------------------
# Pool: country_code -> (list of addresses (unique))
_POOL: Dict[str, List[str]] = {}

# Persistent "used" map:
#   { round_id: { seed_key: [addresses_already_served] } }
# In-memory representation is sets for quick lookup;
# saved on disk as lists.
_USED: Dict[str, Dict[str, set]] = {}
_USED_V2: Dict[str, Dict[str, set]] = {}

# Locks
_pool_lock = threading.RLock()
_used_lock = threading.RLock()

# Persistence flags
_STATE_DIRTY = False
_STOP_AUTOSAVE = threading.Event()

# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------
def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _save_state():
    """Write _USED to disk atomically."""
    global _STATE_DIRTY
    with _used_lock:
        # convert to lists
        serializable = {
            "version": 1,
            "rounds": {rid: {seed: sorted(list(addrs)) for seed, addrs in seeds.items()}
                       for rid, seeds in _USED.items()}
        }
        _atomic_write_json(STATE_PATH, serializable)
        _STATE_DIRTY = False
        logger.debug("State saved: %s", STATE_PATH)

def _load_state():
    """Load _USED from disk."""
    if not STATE_PATH.exists():
        logger.info("No existing state, starting fresh: %s", STATE_PATH)
        return
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        rounds = data.get("rounds", {})
        restored: Dict[str, Dict[str, set]] = {}
        for rid, seeds in rounds.items():
            restored[rid] = {seed: set(addrs or []) for seed, addrs in (seeds or {}).items()}
        with _used_lock:
            _USED.clear()
            _USED.update(restored)
        logger.info("Loaded state: rounds=%d (%s)", len(_USED), STATE_PATH)
    except Exception as e:
        logger.warning("Failed to load state from %s: %s", STATE_PATH, e)

def _mark_dirty():
    global _STATE_DIRTY
    _STATE_DIRTY = True
    if IMMEDIATE_SAVE:
        try:
            _save_state()
        except Exception as e:
            logger.error("Immediate save failed: %s", e)

def _autosave_loop():
    logger.info("Autosave thread started (interval=%ss)", AUTOSAVE_INTERVAL_SEC)
    while not _STOP_AUTOSAVE.is_set():
        _STOP_AUTOSAVE.wait(AUTOSAVE_INTERVAL_SEC)
        if _STOP_AUTOSAVE.is_set():
            break
        try:
            if _STATE_DIRTY:
                _save_state()
        except Exception as e:
            logger.error("Autosave failed: %s", e)
    logger.info("Autosave thread stopped")

# ------------------------------------------------------------------------------
# Pool loading
# ------------------------------------------------------------------------------
def load_pool(base_dir: Path) -> Dict[str, List[str]]:
    """
    Scan addgen/addrs/<CC>/*.json and build country_code -> [addresses] pool.
    Duplicates across files are de-duplicated. List order is randomized.
    """
    if not base_dir.exists():
        logger.warning("Addresses base dir does not exist: %s", base_dir)
        return {}

    new_pool: Dict[str, set] = {}
    count_files = 0
    for cc_dir in sorted(base_dir.iterdir()):
        if not cc_dir.is_dir():
            continue
        cc = cc_dir.name.upper()
        if cc not in COUNTRIES:
            logger.debug("Skipping unknown country code directory: %s", cc)
            continue
        for jf in cc_dir.glob("*.json"):
            count_files += 1
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                addrs = data.get("addresses") or []
                if not addrs:
                    continue
                s = new_pool.setdefault(cc, set())
                s.update(a.strip() for a in addrs if isinstance(a, str) and a.strip())
            except Exception as e:
                logger.warning("Failed reading %s: %s", jf, e)

    # convert sets to randomized lists
    pool_list: Dict[str, List[str]] = {}
    total_addrs = 0
    for cc, s in new_pool.items():
        lst = list(s)
        random.shuffle(lst)
        pool_list[cc] = lst
        total_addrs += len(lst)

    logger.info("Loaded pool: %d country buckets, %d files, %d total addresses",
                len(pool_list), count_files, total_addrs)
    return pool_list

def refresh_pool():
    global _POOL
    with _pool_lock:
        _POOL = load_pool(ADDRS_BASE_DIR)

# ------------------------------------------------------------------------------
# Allocation logic
# ------------------------------------------------------------------------------
def allocate_for_seed(round_id: str, seed: str, per_seed: int) -> Tuple[str, List[str], bool, Optional[str]]:
    """
    Returns: (country_code, allocated_addresses, exhausted, warning_msg)
    - exhausted=True if we couldn't allocate `per_seed` unique addresses
    - warning_msg set if exhausted or country not found
    """
    SPECIAL_REGIONS = ["luhansk", "crimea", "donetsk", "west sahara"]
    is_special = False
    
    if any(region in seed.lower() for region in ["luhansk", "crimea", "donetsk"]):
        cc = "UA"
        is_special = True
    elif any(region in seed.lower() for region in ["west sahara"]):
        cc = "EH"
        is_special = True
    else:
        cc = infer_country_code_from_seed(seed)


    with _pool_lock:
        if isinstance(cc, list):
            pool_list = [a for c in cc for a in _POOL.get(c, [])]
        else:   
            pool_list = _POOL.get(cc, [])

    if not pool_list:
        msg = f"No pool for country {cc} (seed={seed!r}); please prefetch more."
        logger.warning(msg)
        return cc, [], True, msg

    skey = _seed_key(seed)
    with _used_lock:
        round_map = _USED.setdefault(round_id, {})
        used = round_map.setdefault(skey, set())

        # available = pool minus already used for this round+seed
        if is_special:
            available = [a for a in pool_list if a not in used and seed.lower() in a.lower()]
        else:
            available = [a for a in pool_list if a not in used]
                                                                                                                                    
        if not available:
            msg = f"Pool exhausted for (round={round_id}, seed={seed!r}); prefetch/expand pool for {cc}."
            logger.warning(msg)
            return cc, [], True, msg

        # take up to per_seed

        exhausted = len(available) < per_seed + per_seed/3
        
        allocated = random.sample(available, per_seed) if not exhausted else []    

        used.update(allocated)
        _mark_dirty()

        if exhausted:
            msg = (f"Insufficient pool for (round={round_id}, seed={seed!r}) in {cc}: "
                   f"requested {per_seed}, served {0}, remaining {len(available)}.")
            logger.warning(msg)
            return cc, [], True, msg

        return cc, allocated, False, None

def allocate_for_seed_random(round_id: str, seed: str, per_seed: int) -> Tuple[str, List[str], bool, Optional[str]]:
    """
    Random allocation version - selects addresses randomly instead of sequentially.
    Returns: (country_code, allocated_addresses, exhausted, warning_msg)
    - exhausted=True if we couldn't allocate `per_seed` unique addresses
    - warning_msg set if exhausted or country not found
    """
    is_special = False
    
    if any(region in seed.lower() for region in ["luhansk", "crimea", "donetsk"]):
        cc = "UA"
        is_special = True
    elif any(region in seed.lower() for region in ["west sahara"]):
        cc = "EH"
        is_special = True
    else:
        cc = infer_country_code_from_seed(seed)


    with _pool_lock:
        if isinstance(cc, list):
            pool_list = [a for c in cc for a in _POOL.get(c, [])]
        else:   
            pool_list = _POOL.get(cc, [])

    if not pool_list:
        import os
        seed_file_path = "missing_seeds.txt"
        try:
            # Check if seed already in file
            if os.path.exists(seed_file_path):
                with open(seed_file_path, "r", encoding="utf-8") as f:
                    existing_seeds = set(line.strip() for line in f)
            else:
                existing_seeds = set()
            if seed not in existing_seeds:
                with open(seed_file_path, "a", encoding="utf-8") as f:
                    f.write(seed + "\n")
        except Exception as e:
            logger.warning(f"Could not write missing seed {seed} to file: {e}")
    
        msg = f"No pool for country {cc} (seed={seed!r}); please prefetch more."
        
        logger.warning(msg)
        return cc, [], True, msg

    skey = _seed_key(seed)
    with _used_lock:
        round_map = _USED_V2.setdefault(round_id, {})
        used = round_map.setdefault(skey, set())

        # available = pool minus already used for this round+seed
        if is_special:
            available = [a for a in pool_list if seed.lower() in a.lower()]
        else:
            available = pool_list
            
        if not available:
            msg = f"Pool exhausted for (round={round_id}, seed={seed!r}); prefetch/expand pool for {cc}."
            logger.warning(msg)
            # INSERT_YOUR_CODE
            # Save the seed to a file (append mode) if it's not already there.
            # File path and existence checks. We avoid duplicate writes.
            import os
            seed_file_path = "missing_seeds.txt"
            try:
                # Check if seed already in file
                if os.path.exists(seed_file_path):
                    with open(seed_file_path, "r", encoding="utf-8") as f:
                        existing_seeds = set(line.strip() for line in f)
                else:
                    existing_seeds = set()
                if seed not in existing_seeds:
                    with open(seed_file_path, "a", encoding="utf-8") as f:
                        f.write(seed + "\n")
            except Exception as e:
                logger.warning(f"Could not write missing seed {seed} to file: {e}")
            return cc, [], True, msg

        # take up to per_seed, but RANDOMLY instead of sequentially
        n = min(per_seed, len(available))
        
        # Randomly sample from available addresses
        allocated = random.sample(available, n)
        
        # # if 
        # if n <= per_seed:
        #     used.update(allocated)

        _mark_dirty()

        exhausted = n < per_seed
        if exhausted:
            import os
            seed_file_path = "missing_seeds.txt"
            try:
                # Check if seed already in file
                if os.path.exists(seed_file_path):
                    with open(seed_file_path, "r", encoding="utf-8") as f:
                        existing_seeds = set(line.strip() for line in f)
                else:
                    existing_seeds = set()
                if seed not in existing_seeds:
                    with open(seed_file_path, "a", encoding="utf-8") as f:
                        f.write(seed + "\n")
            except Exception as e:
                logger.warning(f"Could not write missing seed {seed} to file: {e}")

            msg = (f"Insufficient pool for (round={round_id}, seed={seed!r}) in {cc}: "
                   f"requested {per_seed}, served {n}, remaining 0.")
            logger.warning(msg)
            return cc, allocated, True, msg

        return cc, allocated, False, None

# ------------------------------------------------------------------------------
# FastAPI models
# ------------------------------------------------------------------------------
class LocationRequest(BaseModel):
    round_id: str = Field(..., description="Unique round identifier; uniqueness enforced per (round_id, seed).")
    seed_addresses: List[str] = Field(..., description="List of seed strings (country names/codes or 'City, Country').")
    per_seed: int = Field(1, ge=1, le=100, description="How many addresses to allocate per seed (default=1).")

class SeedAllocation(BaseModel):
    seed: str
    country_code: Optional[str]
    allocated: List[str]
    exhausted: bool
    warning: Optional[str] = None

class LocationResponse(BaseModel):
    round_id: str
    results: List[SeedAllocation]
    warnings: List[str] = []

class StatsResponse(BaseModel):
    countries: Dict[str, int]
    total_addresses: int
    rounds: int

class ReloadResponse(BaseModel):
    countries: int
    total_addresses: int

class RoundsListResponse(BaseModel):
    rounds: Dict[str, int]  # round_id -> number of seeds tracked
    total_rounds: int

class RoundDetail(BaseModel):
    round_id: str
    seeds: Dict[str, int]  # seed_key -> num_used_addresses
    total_seeds: int
    total_used: int

class RoundResetRequest(BaseModel):
    round_id: Optional[str] = Field(None, description="If omitted, resets ALL rounds. If provided, resets only that round.")

class RoundResetResponse(BaseModel):
    reset_all: bool
    round_id: Optional[str]
    remaining_rounds: int

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Address Allocation Service",
    description="Serves pre-fetched addresses with per-round per-seed uniqueness (persistent).",
    version="1.1.0",
)

@app.on_event("startup")
def _startup():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    refresh_pool()
    _load_state()
    # start autosave thread
    t = threading.Thread(target=_autosave_loop, name="autosave", daemon=True)
    t.start()

@app.on_event("shutdown")
def _shutdown():
    _STOP_AUTOSAVE.set()
    # best-effort final save
    try:
        if _STATE_DIRTY:
            _save_state()
    except Exception as e:
        logger.error("Final save failed: %s", e)

@app.get("/health")
def health():
    return {"status": "ok", "addr_base": str(ADDRS_BASE_DIR), "state": str(STATE_PATH)}

@app.get("/stats", response_model=StatsResponse)
def stats():
    with _pool_lock:
        sizes = {cc: len(lst) for cc, lst in _POOL.items()}
        total = sum(sizes.values())
    with _used_lock:
        rounds = len(_USED)
    return StatsResponse(countries=sizes, total_addresses=total, rounds=rounds)

@app.post("/reload", response_model=ReloadResponse)
def reload_pool():
    refresh_pool()
    with _pool_lock:
        sizes = {cc: len(lst) for cc, lst in _POOL.items()}
        total = sum(sizes.values())
    return ReloadResponse(countries=len(sizes), total_addresses=total)

@app.get("/rounds", response_model=RoundsListResponse)
def list_rounds():
    with _used_lock:
        info = {rid: len(seeds) for rid, seeds in _USED.items()}
    return RoundsListResponse(rounds=info, total_rounds=len(info))

@app.get("/rounds/{round_id}", response_model=RoundDetail)
def round_detail(round_id: str):
    with _used_lock:
        seeds = _USED.get(round_id)
        if seeds is None:
            raise HTTPException(status_code=404, detail="round_id not found")
        detail = {seed: len(addrs) for seed, addrs in seeds.items()}
        total_used = sum(detail.values())
    return RoundDetail(round_id=round_id, seeds=detail, total_seeds=len(detail), total_used=total_used)

@app.post("/rounds/reset", response_model=RoundResetResponse)
def round_reset(req: RoundResetRequest):
    with _used_lock:
        if req.round_id:
            if req.round_id in _USED:
                del _USED[req.round_id]
                _mark_dirty()
            else:
                raise HTTPException(status_code=404, detail="round_id not found")
            remaining = len(_USED)
            return RoundResetResponse(reset_all=False, round_id=req.round_id, remaining_rounds=remaining)
        else:
            _USED.clear()
            _mark_dirty()
            return RoundResetResponse(reset_all=True, round_id=None, remaining_rounds=0)

@app.post("/locations", response_model=LocationResponse)
def get_locations(req: LocationRequest):
    if not req.seed_addresses:
        raise HTTPException(status_code=400, detail="seed_addresses cannot be empty")
    results: List[SeedAllocation] = []
    warnings: List[str] = []

    for seed in req.seed_addresses:
        cc_inferred = infer_country_code_from_seed(seed)
        alloc_cc, addrs, exhausted, warn = allocate_for_seed(req.round_id, seed, req.per_seed)
        if warn:
            warnings.append(warn)
        results.append(SeedAllocation(
            seed=seed,
            country_code=alloc_cc or cc_inferred,
            allocated=addrs,
            exhausted=exhausted,
            warning=warn
        ))

    return LocationResponse(round_id=req.round_id, results=results, warnings=warnings)

@app.post("/locations/v2", response_model=LocationResponse)
def get_locations(req: LocationRequest):
    if not req.seed_addresses:
        raise HTTPException(status_code=400, detail="seed_addresses cannot be empty")
    results: List[SeedAllocation] = []
    warnings: List[str] = []

    for seed in req.seed_addresses:
        cc_inferred = infer_country_code_from_seed(seed)
        alloc_cc, addrs, exhausted, warn = allocate_for_seed_random(req.round_id, seed, req.per_seed)
        if warn:
            warnings.append(warn)
        results.append(SeedAllocation(
            seed=seed,
            country_code=alloc_cc or cc_inferred,
            allocated=addrs,
            exhausted=exhausted,
            warning=warn
        ))

    return LocationResponse(round_id=req.round_id, results=results, warnings=warnings)
