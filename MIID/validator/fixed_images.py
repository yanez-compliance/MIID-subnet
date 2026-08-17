# MIID/validator/fixed_images.py
#
# Daily + tomorrow image-of-the-day (IOTD) for the real screen-replay path.
# Fetched from the MIID image API and cached under fixed_image_cache/.
# Refreshed when the cache is empty (cold start) or at the UTC day boundary.

import base64
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import bittensor as bt

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Static sandbox pool (git-tracked). Used only when VALIDATOR_SENDS_SEED_IMAGE
# is False, so miners can practice without a live API seed.
FIXED_IMAGE_DIR = Path(__file__).parent / "fixed_image"

# Live IOTD cache (today + tomorrow). Never write API bytes into FIXED_IMAGE_DIR
# — that would wipe the git-tracked sandbox pool.
FIXED_IMAGE_CACHE_DIR = Path(__file__).parent / "fixed_image_cache"
SEED_META_PATH = FIXED_IMAGE_CACHE_DIR / "seed_meta.json"
SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")

# When True, the validator fetches today's and tomorrow's IOTD from the API
# and sends both to miners alongside the per-round face-variation image.
VALIDATOR_SENDS_SEED_IMAGE = True

SeedTriple = Tuple[str, str, str]  # (filename, base64, seed_date)


def _utc_today() -> str:
    """Return today's date in UTC as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_tomorrow() -> str:
    """Return tomorrow's date in UTC as YYYY-MM-DD."""
    return (datetime.now(timezone.utc).date() + timedelta(days=1)).strftime("%Y-%m-%d")


def _list_image_files(directory: Path) -> List[Path]:
    """List image files currently stored in a directory."""
    if not directory.exists():
        return []
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(ext))
    return sorted(files)


def is_fixed_image_dir_empty() -> bool:
    """True when no IOTD has been cached yet."""
    return len(_list_image_files(FIXED_IMAGE_CACHE_DIR)) == 0


def _load_meta() -> dict:
    if not SEED_META_PATH.exists():
        return {}
    try:
        with open(SEED_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_meta(today: dict, tomorrow: dict) -> None:
    FIXED_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "today": today,
        "tomorrow": tomorrow,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SEED_META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _slot_meta(meta: dict, slot: str) -> dict:
    """Return today/tomorrow slot, including legacy {filename, seed_date} meta."""
    slot_data = meta.get(slot)
    if isinstance(slot_data, dict) and slot_data:
        return slot_data
    if slot == "today" and meta.get("seed_date"):
        return {
            "filename": meta.get("filename"),
            "seed_date": meta.get("seed_date"),
            "source_filename": meta.get("filename"),
        }
    return {}


def needs_fixed_image_refresh() -> bool:
    """Refresh when cache is empty or either IOTD is from a prior UTC day."""
    if is_fixed_image_dir_empty():
        return True
    meta = _load_meta()
    today_meta = _slot_meta(meta, "today")
    tomorrow_meta = _slot_meta(meta, "tomorrow")
    if not today_meta.get("filename") or not today_meta.get("seed_date"):
        return True
    if not tomorrow_meta.get("filename") or not tomorrow_meta.get("seed_date"):
        return True
    if today_meta.get("seed_date") != _utc_today():
        return True
    if tomorrow_meta.get("seed_date") != _utc_tomorrow():
        return True

    today_path = FIXED_IMAGE_CACHE_DIR / today_meta["filename"]
    tomorrow_path = FIXED_IMAGE_CACHE_DIR / tomorrow_meta["filename"]
    return not today_path.exists() or not tomorrow_path.exists()


def _clear_cache_image_files() -> None:
    """Remove previous IOTD bytes from the cache; keep meta/.gitkeep."""
    for path in _list_image_files(FIXED_IMAGE_CACHE_DIR):
        try:
            path.unlink()
        except Exception as e:
            bt.logging.warning(f"Failed to remove old cached IOTD {path.name}: {e}")


def _parse_seed_slot(data: dict, slot: str) -> Optional[Tuple[str, bytes, str]]:
    """Parse one today/tomorrow slot (or the legacy top-level image) from the API."""
    slot_data = data.get(slot) or {}
    item = slot_data.get("image") or {}
    if slot == "today" and not item:
        item = data.get("image") or {}
    filename = item.get("filename")
    b64 = item.get("data_base64")
    if not filename or not b64:
        return None
    seed_date = slot_data.get("seed_date") or (data.get("seed_date") if slot == "today" else None)
    if not seed_date:
        seed_date = _utc_today() if slot == "today" else _utc_tomorrow()
    return filename, base64.standard_b64decode(b64), seed_date


def _fetch_seed_pair_from_api(wallet) -> Optional[Tuple[Tuple[str, bytes, str], Optional[Tuple[str, bytes, str]]]]:
    """Fetch today's and tomorrow's IOTD from the MIID API.

    Calls POST /fixed_image/<hotkey>, which resolves deterministically by UTC
    calendar date server-side. Newer API versions return both `today` and
    `tomorrow`; older versions only return today's image (legacy `image` /
    `seed_date` fields).

    Returns:
        ((today_filename, raw, date), (tomorrow_filename, raw, date) or None)
        or None on failure to fetch even today.
    """
    if not REQUESTS_AVAILABLE:
        bt.logging.warning("requests library not available. Cannot fetch fixed image.")
        return None

    try:
        from MIID.utils.sign_message import sign_message

        hotkey = wallet.hotkey
        hotkey_address = hotkey.ss58_address
        message_to_sign = (
            f"Hotkey: {hotkey} \n timestamp: {time.time()} \n request: fixed_image"
        )
        signed_contents = sign_message(wallet, message_to_sign, output_file=None)

        server_url = os.environ.get("MIID_IMAGES_SERVER", "http://52.44.186.20:5000")
        url = f"{server_url.rstrip('/')}/fixed_image/{hotkey_address}"
        response = requests.post(url, json={"signature": signed_contents}, timeout=30)

        if response.status_code != 200:
            bt.logging.warning(
                f"Fixed image API returned {response.status_code}: {response.text[:200]}"
            )
            return None

        data = response.json()
        today = _parse_seed_slot(data, "today")
        if today is None:
            bt.logging.warning("Fixed image API response missing today's filename or data_base64")
            return None
        tomorrow = _parse_seed_slot(data, "tomorrow")
        if tomorrow is None:
            bt.logging.warning(
                "Fixed image API response missing tomorrow's IOTD; "
                "sending today only until the API is updated."
            )
        return today, tomorrow

    except Exception as e:
        bt.logging.error(f"Error fetching fixed image from API: {e}")
        return None


def _write_seed_file(source_filename: str, raw: bytes, seed_date: str, slot: str) -> str:
    """Write one IOTD into the cache and return the on-disk filename."""
    ext = Path(source_filename).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = ".png"
    saved_name = f"{slot}_seed_{seed_date}{ext}"
    image_path = FIXED_IMAGE_CACHE_DIR / saved_name
    image_path.write_bytes(raw)
    return saved_name


def fetch_and_save_fixed_image(wallet) -> Optional[Tuple[str, Path]]:
    """Download today's (and tomorrow's) IOTD and save under fixed_image_cache/.

    Returns:
        (today_filename, path) on success, None on failure.
    """
    result = _fetch_seed_pair_from_api(wallet)
    if result is None:
        return None

    today, tomorrow = result
    today_source, today_raw, today_date = today
    FIXED_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _clear_cache_image_files()

    today_saved = _write_seed_file(today_source, today_raw, today_date, "today")
    today_meta = {
        "filename": today_saved,
        "seed_date": today_date,
        "source_filename": today_source,
    }
    tomorrow_meta = {}
    if tomorrow is not None:
        tomorrow_source, tomorrow_raw, tomorrow_date = tomorrow
        tomorrow_saved = _write_seed_file(tomorrow_source, tomorrow_raw, tomorrow_date, "tomorrow")
        tomorrow_meta = {
            "filename": tomorrow_saved,
            "seed_date": tomorrow_date,
            "source_filename": tomorrow_source,
        }

    _save_meta(today_meta, tomorrow_meta)

    bt.logging.info(
        f"Saved IOTD today={today_saved} ({len(today_raw)} bytes) "
        f"source={today_source} seed_date={today_date} UTC"
        + (
            f" | tomorrow={tomorrow_meta.get('filename')} "
            f"source={tomorrow_meta.get('source_filename')} "
            f"seed_date={tomorrow_meta.get('seed_date')} UTC"
            if tomorrow_meta
            else " | tomorrow=unavailable"
        )
    )
    return today_saved, FIXED_IMAGE_CACHE_DIR / today_saved


def ensure_daily_fixed_image(wallet) -> Optional[Tuple[str, Path]]:
    """Ensure the cache holds today's and tomorrow's IOTD.

    Fetches when:
      - the cache has no image (cold start / empty), or
      - the UTC calendar day has rolled over past 00:00:00, or
      - tomorrow's slot is missing

    Returns:
        (filename, path) for today's seed, or None if unavailable.
    """
    FIXED_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not needs_fixed_image_refresh():
        today_meta = _slot_meta(_load_meta(), "today")
        filename = today_meta.get("filename")
        path = FIXED_IMAGE_CACHE_DIR / filename if filename else None
        if path and path.exists():
            bt.logging.debug(
                f"Using cached IOTD: today={filename} "
                f"(seed_date={today_meta.get('seed_date')})"
            )
            return filename, path
        return None

    reason = "empty cache" if is_fixed_image_dir_empty() else "new UTC day or missing tomorrow slot"
    bt.logging.info(f"Refreshing IOTD pair ({reason})")
    return fetch_and_save_fixed_image(wallet)


def list_fixed_image_pool() -> List[str]:
    """List filenames of the static fixed-image pool miners choose from.

    Used only in sandbox mode (VALIDATOR_SENDS_SEED_IMAGE=False) to build the
    miner-facing instructions dynamically, so the text always matches
    whatever images actually sit in fixed_image/ without a hardcoded count.
    """
    return [p.name for p in _list_image_files(FIXED_IMAGE_DIR)]


def _load_slot_base64(slot: str) -> Optional[SeedTriple]:
    """Load one cached IOTD slot as (source_filename, base64, seed_date)."""
    meta = _slot_meta(_load_meta(), slot)
    filename = meta.get("filename")
    if not filename:
        return None
    path = FIXED_IMAGE_CACHE_DIR / filename
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        # Prefer the original API filename so miners report the same name
        # the rest of the network uses.
        reported_name = meta.get("source_filename") or path.name
        seed_date = meta.get("seed_date") or (_utc_today() if slot == "today" else _utc_tomorrow())
        return reported_name, base64.b64encode(raw).decode("utf-8"), seed_date
    except Exception as e:
        bt.logging.error(f"Failed to load cached IOTD {path}: {e}")
        return None


def load_fixed_image_base64() -> Optional[Tuple[str, str]]:
    """Load today's cached IOTD as (filename, base64). Backward-compat helper."""
    today = _load_slot_base64("today")
    if today is None:
        return None
    return today[0], today[1]


def load_seed_pair_base64() -> Tuple[Optional[SeedTriple], Optional[SeedTriple]]:
    """Load cached today + tomorrow IOTD as (filename, base64, seed_date) triples."""
    return _load_slot_base64("today"), _load_slot_base64("tomorrow")
