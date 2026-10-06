"""
Image endpoints for validators:
  POST /images/<hotkey>       — all base images for that validator folder
  POST /image/<hotkey>        — random image from batch pool (consume + recycle)
  POST /fixed_image/<hotkey>  — deterministic today+tomorrow seed images
"""

import base64
import json
import random
import shutil
import threading
from datetime import datetime, timezone, timedelta

from flask import Blueprint, jsonify

from MIID.datasets.signature_auth import verify_request_signature

# Pool paths shared with operator scripts
import sys

sys.path.insert(0, "/home/ubuntu/YanezMIIDManage")
from media_pools.config import (
    BASE_IMAGES_DIR,
    BATCH_DIR,
    USED_DIR,
    BATCH_LOG,
    ALLOWED_EXT,
    FIXED_IMAGE_POOL_DIR,
    FIXED_IMAGE_LOG,
    HOTKEY_TO_FOLDER,
)

image_bp = Blueprint("images", __name__)

_batch_lock = threading.Lock()
_fixed_image_lock = threading.Lock()


@image_bp.route("/images/<hotkey>", methods=["POST"])
def get_validator_images(hotkey):
    """
    Return all base images for this validator (that hotkey's folder under base_images).
    Only hotkeys in HOTKEY_TO_FOLDER can call this endpoint.
    """
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify(
                {
                    "error": "Unauthorized hotkey or no image folder for this validator"
                }
            ),
            403,
        )

    err_resp, err_status = verify_request_signature(tmp_prefix="tmp_signature_images")
    if err_resp is not None:
        return err_resp, err_status

    folder_name = HOTKEY_TO_FOLDER.get(hotkey)
    if not folder_name:
        return jsonify({"error": "No image folder configured for this validator"}), 404

    images_dir = BASE_IMAGES_DIR / folder_name
    if not images_dir.is_dir():
        return jsonify({"error": f"Image folder not found: {folder_name}"}), 404

    images = []
    for f in sorted(images_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
            try:
                b64 = base64.standard_b64encode(f.read_bytes()).decode("ascii")
                images.append({"filename": f.name, "data_base64": b64})
            except Exception:
                pass

    return (
        jsonify(
            {
                "validator_folder": folder_name,
                "verified_by": hotkey,
                "images": images,
                "count": len(images),
            }
        ),
        200,
    )


def _recycle_if_empty():
    """
    Called inside _batch_lock. If BATCH_DIR has no images remaining, move every
    image from USED_DIR back to BATCH_DIR so the pool restarts, and record the
    event in BATCH_LOG.
    """
    batch_images = [
        f
        for f in BATCH_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT
    ]
    if batch_images:
        return

    used_images = (
        [
            f
            for f in USED_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT
        ]
        if USED_DIR.is_dir()
        else []
    )
    if not used_images:
        return

    recycled_names = []
    errors = []
    for f in used_images:
        dest = BATCH_DIR / f.name
        try:
            shutil.move(str(f), str(dest))
            recycled_names.append(f.name)
        except Exception as e:
            errors.append({"filename": f.name, "error": str(e)})
            print(f"[ERROR] Failed to recycle {f.name}: {e}")

    print(
        f"[INFO] Batch pool recycled: moved {len(recycled_names)} images "
        f"from USED_DIR back to BATCH_DIR"
    )

    try:
        with open(BATCH_LOG, "r", encoding="utf-8") as lf:
            log = json.load(lf)

        log.setdefault("recycle_events", []).append(
            {
                "recycled_at": datetime.now(timezone.utc).isoformat(),
                "images_recycled": len(recycled_names),
                "recycled_filenames": recycled_names,
                "errors": errors,
            }
        )
        log["images_remaining"] = len(recycled_names)
        log["recycle_count"] = log.get("recycle_count", 0) + 1

        with open(BATCH_LOG, "w", encoding="utf-8") as lf:
            json.dump(log, lf, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to log recycle event: {e}")


@image_bp.route("/image/<hotkey>", methods=["POST"])
def get_validator_image(hotkey):
    """
    Pick a random image from BATCH_DIR, move it to USED_DIR, log the move,
    and return the image to the requesting validator.
    """
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify(
                {
                    "error": "Unauthorized hotkey or no image folder for this validator"
                }
            ),
            403,
        )

    err_resp, err_status = verify_request_signature(tmp_prefix="tmp_signature_image")
    if err_resp is not None:
        return err_resp, err_status

    with _batch_lock:
        _recycle_if_empty()

        available = [
            f
            for f in BATCH_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT
        ]
        if not available:
            return jsonify({"error": "No images left in batch pool"}), 404

        chosen = random.choice(available)

        try:
            image_bytes = chosen.read_bytes()
        except Exception as e:
            return jsonify({"error": f"Failed to read image: {str(e)}"}), 500

        USED_DIR.mkdir(parents=True, exist_ok=True)
        dest = USED_DIR / chosen.name
        shutil.move(str(chosen), str(dest))

        try:
            with open(BATCH_LOG, "r", encoding="utf-8") as lf:
                log = json.load(lf)

            log["images_moved_to_used_count"] = (
                log.get("images_moved_to_used_count", 0) + 1
            )
            log["images_remaining"] = len(available) - 1

            log.setdefault("moves_to_used_images", []).append(
                {
                    "filename": chosen.name,
                    "moved_at": datetime.now(timezone.utc).isoformat(),
                    "validator_hotkey": hotkey,
                    "dest_path": str(dest),
                }
            )

            with open(BATCH_LOG, "w", encoding="utf-8") as lf:
                json.dump(log, lf, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to update batch log: {e}")

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return (
        jsonify(
            {
                "verified_by": hotkey,
                "image": {"filename": chosen.name, "data_base64": b64},
            }
        ),
        200,
    )


def _get_daily_fixed_image_path(day_offset=0):
    """
    Deterministically pick the fixed seed image for UTC today + day_offset.

    Selection is keyed off the UTC calendar date (proleptic Gregorian ordinal),
    not request order or hotkey, so every validator that calls this endpoint
    on the same UTC day resolves to the exact same file.

    Returns:
        (path, seed_date) where seed_date is YYYY-MM-DD UTC, or (None, None).
    """
    if not FIXED_IMAGE_POOL_DIR.is_dir():
        return None, None

    pool = sorted(
        f
        for f in FIXED_IMAGE_POOL_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT
    )
    if not pool:
        return None, None

    utc_date = datetime.now(timezone.utc).date() + timedelta(days=day_offset)
    return pool[utc_date.toordinal() % len(pool)], utc_date.strftime("%Y-%m-%d")


def _encode_image_payload(path):
    """Read an image file and return {filename, data_base64}."""
    image_bytes = path.read_bytes()
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return {"filename": path.name, "data_base64": b64}


def _log_fixed_image_serve(hotkey, filename, seed_date):
    """
    Append a serve record to FIXED_IMAGE_LOG.
    Must be called under _fixed_image_lock.
    """
    try:
        if FIXED_IMAGE_LOG.is_file():
            with open(FIXED_IMAGE_LOG, "r", encoding="utf-8") as lf:
                log = json.load(lf)
        else:
            log = {
                "initialized_at": datetime.now(timezone.utc).isoformat(),
                "pool_dir": str(FIXED_IMAGE_POOL_DIR),
                "endpoint": "/fixed_image/<hotkey>",
                "serves_to_validators": [],
                "daily_assignments": {},
                "manifest": [],
            }

        served_at = datetime.now(timezone.utc).isoformat()
        log.setdefault("serves_to_validators", []).append(
            {
                "filename": filename,
                "seed_date": seed_date,
                "served_at": served_at,
                "validator_hotkey": hotkey,
                "note": "same image for all validators on this UTC day",
            }
        )
        log["serves_to_validators_count"] = len(log["serves_to_validators"])

        assignments = log.setdefault("daily_assignments", {})
        day_entry = assignments.setdefault(
            seed_date,
            {
                "filename": filename,
                "validators": [],
                "serve_count": 0,
            },
        )
        day_entry["filename"] = filename
        if hotkey not in day_entry["validators"]:
            day_entry["validators"].append(hotkey)
        day_entry["serve_count"] = len(day_entry["validators"])
        day_entry["last_served_at"] = served_at

        utc_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if seed_date == utc_today:
            log["today_seed_date"] = seed_date
            log["today_selected_filename"] = filename
        else:
            log["tomorrow_seed_date"] = seed_date
            log["tomorrow_selected_filename"] = filename

        with open(FIXED_IMAGE_LOG, "w", encoding="utf-8") as lf:
            json.dump(log, lf, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to update fixed image serve log: {e}")


@image_bp.route("/fixed_image/<hotkey>", methods=["POST"])
def get_fixed_image(hotkey):
    """
    Return today's AND tomorrow's fixed seed images for the screen-replay challenge.

    Unlike /image/<hotkey>, this endpoint is NOT random and does NOT consume
    the pool: the same UTC calendar day always resolves to the same pair of
    files in FIXED_IMAGE_POOL_DIR.
    """
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify(
                {
                    "error": "Unauthorized hotkey or no image folder for this validator"
                }
            ),
            403,
        )

    err_resp, err_status = verify_request_signature(
        tmp_prefix="tmp_signature_fixed_image"
    )
    if err_resp is not None:
        return err_resp, err_status

    with _fixed_image_lock:
        today_path, today_date = _get_daily_fixed_image_path(0)
        tomorrow_path, tomorrow_date = _get_daily_fixed_image_path(1)
        if today_path is None or tomorrow_path is None:
            return (
                jsonify(
                    {"error": "No fixed image pool configured or pool is empty"}
                ),
                404,
            )

        try:
            today_payload = _encode_image_payload(today_path)
            tomorrow_payload = _encode_image_payload(tomorrow_path)
        except Exception as e:
            return jsonify({"error": f"Failed to read fixed image: {str(e)}"}), 500

        _log_fixed_image_serve(hotkey, today_path.name, today_date)
        _log_fixed_image_serve(hotkey, tomorrow_path.name, tomorrow_date)

    return (
        jsonify(
            {
                "verified_by": hotkey,
                "seed_date": today_date,
                "image": today_payload,
                "today": {
                    "seed_date": today_date,
                    "image": today_payload,
                },
                "tomorrow": {
                    "seed_date": tomorrow_date,
                    "image": tomorrow_payload,
                },
            }
        ),
        200,
    )
