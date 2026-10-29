"""
Voice endpoints for validators:
  POST /voice/<hotkey> — random reference WAV (consumed once)

Pool selection (language prefers random en|es among non-empty pools):
  - HOTKEY_TO_FOLDER[hotkey] == "testnet" OR VOICE_USE_TEST_POOL → test_net_en/ or test_net_es/
  - otherwise → synthethic-voices-english/ or synthethic-voices-spanish/

Each successful serve moves the WAV into used_voice_batch/<pool>/ and
appends to VOICE_BATCH_LOG so the same clip is not sent again.
"""

import base64
import json
import random
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify

from MIID.datasets.signature_auth import verify_request_signature

import sys

sys.path.insert(0, "/home/ubuntu/YanezMIIDManage")
from media_pools.config import (
    HOTKEY_TO_FOLDER,
    VOICE_BATCH_DIR,
    VOICE_USED_DIR,
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


def _pools_for_hotkey(hotkey: str) -> dict:
    folder = HOTKEY_TO_FOLDER.get(hotkey)
    if folder == "testnet" or VOICE_USE_TEST_POOL:
        return VOICE_TESTNET_POOL_BY_LANGUAGE
    return VOICE_POOL_BY_LANGUAGE


def _list_wavs(pool_dir: Path):
    if not pool_dir.is_dir():
        return []
    return [
        f
        for f in pool_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VOICE_ALLOWED_EXT
    ]


def _select_pool(hotkey: str):
    """
    Return (language, pool_name, pool_dir) for this validator.

    Prefer a random language among pools that still have WAVs. Fall back to
    any configured language (caller handles empty).
    """
    pools = _pools_for_hotkey(hotkey)
    non_empty = []
    for language, pool_name in pools.items():
        pool_dir = VOICE_BATCH_DIR / pool_name
        if _list_wavs(pool_dir):
            non_empty.append((language, pool_name, pool_dir))

    if non_empty:
        return random.choice(non_empty)

    # Both empty — return a random configured pool so the 404 path is clear.
    language = random.choice(list(pools.keys()))
    pool_name = pools[language]
    return language, pool_name, VOICE_BATCH_DIR / pool_name


def _empty_serve_log() -> dict:
    return {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "voice_batch_dir": str(VOICE_BATCH_DIR),
        "used_voice_batch_dir": str(VOICE_USED_DIR),
        "log_path": str(VOICE_BATCH_LOG),
        "in_place": False,
        "total_voices_at_init": 0,
        "voices_served_count": 0,
        "voices_moved_to_used_count": 0,
        "unique_voices_served": 0,
        "unique_voices_never_served": 0,
        "served": [],
        "never_served": [],
        "serves": [],
        "moves_to_used": [],
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
    dest_path: Path,
    voices_remaining: int,
) -> None:
    """
    Append a consume/move serve record to VOICE_BATCH_LOG.

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
            "moved_at": served_at,
            "validator_hotkey": hotkey,
            "in_place": False,
            "source_path": str(source_path),
            "dest_path": str(dest_path),
        }
        log.setdefault("serves", []).append(serve_record)
        log.setdefault("moves_to_used", []).append(serve_record)
        log["voices_served_count"] = log.get("voices_served_count", 0) + 1
        log["voices_moved_to_used_count"] = log.get("voices_moved_to_used_count", 0) + 1
        log["voices_remaining"] = voices_remaining
        log["in_place"] = False
        log["used_voice_batch_dir"] = str(VOICE_USED_DIR)

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
        entry["dest_path"] = str(dest_path)
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

    Picks a language with remaining WAVs, reads a random unused clip, moves it
    into used_voice_batch/<pool>/, and logs the serve. Transcript is the fixed
    challenge text for that language. Gender is inferred from M1/M3 (male) vs
    M2/M4 (female).
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

    with _voice_batch_lock:
        language, pool_name, lang_dir = _select_pool(hotkey)
        available = _list_wavs(lang_dir)

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

        used_pool_dir = VOICE_USED_DIR / pool_name
        used_pool_dir.mkdir(parents=True, exist_ok=True)
        dest = used_pool_dir / chosen.name
        try:
            shutil.move(str(chosen), str(dest))
        except Exception as e:
            return jsonify({"error": f"Failed to move voice to used: {str(e)}"}), 500

        filename = chosen.name
        gender = _gender_from_filename(filename)
        remaining = len(_list_wavs(lang_dir))
        # Remaining across both language pools for this validator class.
        pools = _pools_for_hotkey(hotkey)
        total_remaining = sum(
            len(_list_wavs(VOICE_BATCH_DIR / name)) for name in pools.values()
        )
        _log_voice_serve(
            hotkey=hotkey,
            filename=filename,
            pool_name=pool_name,
            language=language,
            gender=gender,
            source_path=chosen,
            dest_path=dest,
            voices_remaining=total_remaining,
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
