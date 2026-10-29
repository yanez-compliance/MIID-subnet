"""
Voice endpoints for validators:
  POST /voice/<hotkey> — random reference WAV

Pool selection (language always random en|es):
  - HOTKEY_TO_FOLDER[hotkey] == "testnet" OR VOICE_USE_TEST_POOL → test_net_en/ or test_net_es/
  - otherwise → synthethic-voices-english/ or synthethic-voices-spanish/

Each successful serve appends to VOICE_BATCH_LOG (clips stay in place).
"""

import base64
import json
import random
import re
import threading
from datetime import datetime, timezone

from flask import Blueprint, jsonify

from MIID.datasets.signature_auth import verify_request_signature

import sys

sys.path.insert(0, "/home/ubuntu/YanezMIIDManage")
from media_pools.config import (
    HOTKEY_TO_FOLDER,
    VOICE_BATCH_DIR,
    VOICE_BATCH_LOG,
    VOICE_ALLOWED_EXT,
    VOICE_TRANSCRIPTS,
    VOICE_GENDER_BY_PREFIX,
    VOICE_POOL_BY_LANGUAGE,
    VOICE_TESTNET_POOL_BY_LANGUAGE,
    VOICE_USE_TEST_POOL,
)

voice_bp = Blueprint("voice", __name__)

_voice_batch_lock = threading.Lock()

_VOICE_NAME_RE = re.compile(r"^voice_(M[1-4])_\d+\.wav$", re.IGNORECASE)


def _gender_from_filename(filename: str):
    """Return male/female from voice_M{1-4}_N.wav, or None if unrecognized."""
    match = _VOICE_NAME_RE.match(filename)
    if not match:
        return None
    prefix = match.group(1).upper()
    return VOICE_GENDER_BY_PREFIX.get(prefix)


def _select_pool(hotkey: str):
    """
    Return (language, pool_name, pool_dir) for this validator.

    Language is chosen uniformly at random (en or es). Testnet validators
    and sandbox (VOICE_USE_TEST_POOL) draw from test_net_en/test_net_es;
    everyone else from the synthetic pools.
    """
    language = random.choice(["en", "es"])
    folder = HOTKEY_TO_FOLDER.get(hotkey)
    pools = (
        VOICE_TESTNET_POOL_BY_LANGUAGE
        if folder == "testnet" or VOICE_USE_TEST_POOL
        else VOICE_POOL_BY_LANGUAGE
    )
    pool_name = pools[language]
    return language, pool_name, VOICE_BATCH_DIR / pool_name


def _empty_serve_log() -> dict:
    return {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "voice_batch_dir": str(VOICE_BATCH_DIR),
        "log_path": str(VOICE_BATCH_LOG),
        "in_place": True,
        "total_voices_at_init": 0,
        "voices_served_count": 0,
        "unique_voices_served": 0,
        "unique_voices_never_served": 0,
        "served": [],
        "never_served": [],
        "serves": [],
        "by_filename": {},
    }


def _log_voice_serve(
    *,
    hotkey: str,
    filename: str,
    pool_name: str,
    language: str,
    gender,
    source_path,
) -> None:
    """
    Append an in-place serve record to VOICE_BATCH_LOG.

    Updates by_filename serve_count plus served / never_served summary lists.
    Failures are logged but do not fail the HTTP response.
    """
    try:
        if VOICE_BATCH_LOG.is_file():
            with open(VOICE_BATCH_LOG, "r", encoding="utf-8") as lf:
                log = json.load(lf)
        else:
            log = _empty_serve_log()

        served_at = datetime.now(timezone.utc).isoformat()
        pool_key = f"{pool_name}/{filename}"

        serve_record = {
            "filename": filename,
            "pool": pool_name,
            "pool_key": pool_key,
            "language": language,
            "gender": gender,
            "served_at": served_at,
            "validator_hotkey": hotkey,
            "in_place": True,
            "source_path": str(source_path),
        }
        log.setdefault("serves", []).append(serve_record)
        log["voices_served_count"] = log.get("voices_served_count", 0) + 1

        by_filename = log.setdefault("by_filename", {})
        entry = by_filename.get(pool_key)
        if entry is None:
            entry = {
                "filename": filename,
                "pool": pool_name,
                "language": language,
                "source_path": str(source_path),
                "serve_count": 0,
                "last_served_at": None,
                "validators": [],
            }
            by_filename[pool_key] = entry

        entry["serve_count"] = int(entry.get("serve_count") or 0) + 1
        entry["last_served_at"] = served_at
        validators = entry.setdefault("validators", [])
        if hotkey not in validators:
            validators.append(hotkey)

        served = sorted(
            k for k, e in by_filename.items() if int(e.get("serve_count") or 0) > 0
        )
        never_served = sorted(
            k for k, e in by_filename.items() if int(e.get("serve_count") or 0) == 0
        )
        log["served"] = served
        log["never_served"] = never_served
        log["unique_voices_served"] = len(served)
        log["unique_voices_never_served"] = len(never_served)

        VOICE_BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(VOICE_BATCH_LOG, "w", encoding="utf-8") as lf:
            json.dump(log, lf, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to update voice serve log: {e}")


@voice_bp.route("/voice/<hotkey>", methods=["POST"])
def get_validator_voice(hotkey):
    """
    Serve a reference voice clip for the voice-clone challenge.

    Picks language uniformly at random (en or es), then a random WAV from
    the matching pool (testnet or sandbox: test_net_*; else synthethic-voices-*).
    Transcript is the fixed challenge text for that language. Gender is
    inferred from M1/M3 (male) vs M2/M4 (female). Logs the serve to
    VOICE_BATCH_LOG without moving the file.
    """
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify({"error": "Unauthorized hotkey or no folder for this validator"}),
            403,
        )

    err_resp, err_status = verify_request_signature(
        tmp_prefix="tmp_signature_voice", expected_hotkey=hotkey
    )
    if err_resp is not None:
        return err_resp, err_status

    language, pool_name, lang_dir = _select_pool(hotkey)

    with _voice_batch_lock:
        available = []
        if lang_dir.is_dir():
            available = [
                f
                for f in lang_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VOICE_ALLOWED_EXT
            ]

        if not available:
            return (
                jsonify(
                    {
                        "error": f"No voices left in batch pool ({lang_dir})",
                    }
                ),
                404,
            )

        chosen = random.choice(available)
        try:
            voice_bytes = chosen.read_bytes()
        except Exception as e:
            return jsonify({"error": f"Failed to read voice: {str(e)}"}), 500

        filename = chosen.name
        gender = _gender_from_filename(filename)
        _log_voice_serve(
            hotkey=hotkey,
            filename=filename,
            pool_name=pool_name,
            language=language,
            gender=gender,
            source_path=chosen,
        )

    transcript = VOICE_TRANSCRIPTS.get(language, "")

    b64 = base64.standard_b64encode(voice_bytes).decode("ascii")
    voice_payload = {
        "filename": filename,
        "data_base64": b64,
        "language": language,
        "transcript": transcript,
    }
    if gender is not None:
        voice_payload["gender"] = gender

    return (
        jsonify(
            {
                "verified_by": hotkey,
                "voice": voice_payload,
            }
        ),
        200,
    )
