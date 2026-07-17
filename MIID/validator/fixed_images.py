# MIID/validator/fixed_images.py
#
# Daily fixed seed image for the real screen-replay path.
# Fetched from the MIID image API and cached under fixed_image/.
# Refreshed when the folder is empty (cold start) or at the UTC day boundary.

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import bittensor as bt

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


FIXED_IMAGE_DIR = Path(__file__).parent / "fixed_image"
SEED_META_PATH = FIXED_IMAGE_DIR / "seed_meta.json"
SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _utc_today() -> str:
    """Return today's date in UTC as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _list_image_files() -> List[Path]:
    """List image files currently stored in fixed_image/."""
    if not FIXED_IMAGE_DIR.exists():
        return []
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(FIXED_IMAGE_DIR.glob(ext))
    return sorted(files)


def is_fixed_image_dir_empty() -> bool:
    """True when no seed image has been saved yet."""
    return len(_list_image_files()) == 0


def _load_meta() -> dict:
    if not SEED_META_PATH.exists():
        return {}
    try:
        with open(SEED_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_meta(filename: str, seed_date: str) -> None:
    FIXED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "filename": filename,
        "seed_date": seed_date,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SEED_META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def needs_fixed_image_refresh() -> bool:
    """Refresh when folder is empty or the stored seed is from a prior UTC day."""
    if is_fixed_image_dir_empty():
        return True
    meta = _load_meta()
    stored_date = meta.get("seed_date")
    if not stored_date:
        return True
    return stored_date != _utc_today()


def _clear_image_files() -> None:
    """Remove previous seed image bytes; keep meta/.gitkeep."""
    for path in _list_image_files():
        try:
            path.unlink()
        except Exception as e:
            bt.logging.warning(f"Failed to remove old fixed image {path.name}: {e}")


def _fetch_fixed_image_from_api(wallet) -> Optional[Tuple[str, bytes, str]]:
    """Fetch today's fixed seed image from the MIID API.

    Calls the dedicated POST /fixed_image/<hotkey> endpoint, which resolves
    deterministically by UTC calendar date server-side — every validator that
    calls it on the same day gets identical bytes (unlike the random,
    pool-consuming /image/<hotkey> endpoint used for variation base images).

    Returns:
        Tuple of (filename, raw_bytes, seed_date) where seed_date is the
        server's authoritative UTC date string, or None on failure.
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
        item = data.get("image") or {}
        filename = item.get("filename")
        b64 = item.get("data_base64")
        if not filename or not b64:
            bt.logging.warning("Fixed image API response missing filename or data_base64")
            return None

        seed_date = data.get("seed_date") or _utc_today()
        return filename, base64.standard_b64decode(b64), seed_date

    except Exception as e:
        bt.logging.error(f"Error fetching fixed image from API: {e}")
        return None


def fetch_and_save_fixed_image(wallet) -> Optional[Tuple[str, Path]]:
    """Download today's fixed seed and save it under fixed_image/.

    Clears any previous image files, writes the new bytes, and updates seed_meta.json.
    Uses the server's authoritative seed_date (not the local clock) so all
    validators agree on which day a given image belongs to.

    Returns:
        (filename, path) on success, None on failure.
    """
    result = _fetch_fixed_image_from_api(wallet)
    if result is None:
        return None

    filename, raw, seed_date = result
    FIXED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    _clear_image_files()

    # Keep a stable on-disk name while preserving extension from the API filename.
    ext = Path(filename).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = ".png"
    saved_name = f"daily_seed_{seed_date}{ext}"
    image_path = FIXED_IMAGE_DIR / saved_name
    image_path.write_bytes(raw)
    _save_meta(saved_name, seed_date)

    bt.logging.info(
        f"Saved daily fixed image: {saved_name} ({len(raw)} bytes) "
        f"source={filename} seed_date={seed_date} UTC"
    )
    return saved_name, image_path


def ensure_daily_fixed_image(wallet) -> Optional[Tuple[str, Path]]:
    """Ensure fixed_image/ holds today's seed.

    Fetches when:
      - the directory has no image (cold start / empty), or
      - the UTC calendar day has rolled over past 00:00:00

    Returns:
        (filename, path) for the current seed, or None if unavailable.
    """
    FIXED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not needs_fixed_image_refresh():
        files = _list_image_files()
        if files:
            meta = _load_meta()
            filename = meta.get("filename") or files[0].name
            path = FIXED_IMAGE_DIR / filename
            if not path.exists():
                path = files[0]
                filename = path.name
            bt.logging.debug(
                f"Using cached fixed image: {filename} (seed_date={meta.get('seed_date')})"
            )
            return filename, path
        return None

    reason = "empty directory" if is_fixed_image_dir_empty() else "new UTC day (00:00:00)"
    bt.logging.info(f"Refreshing daily fixed image ({reason})")
    return fetch_and_save_fixed_image(wallet)


def load_fixed_image_base64() -> Optional[Tuple[str, str]]:
    """Load the cached fixed seed as (filename, base64)."""
    files = _list_image_files()
    if not files:
        return None
    meta = _load_meta()
    path = FIXED_IMAGE_DIR / meta["filename"] if meta.get("filename") else files[0]
    if not path.exists():
        path = files[0]
    try:
        raw = path.read_bytes()
        return path.name, base64.b64encode(raw).decode("utf-8")
    except Exception as e:
        bt.logging.error(f"Failed to load fixed image {path}: {e}")
        return None
