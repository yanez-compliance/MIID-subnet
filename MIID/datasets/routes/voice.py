"""
Voice endpoints for validators:
  POST /voice/<hotkey> — random reference WAV

Pool selection (language always random en|es):
  - HOTKEY_TO_FOLDER[hotkey] == "testnet" OR VOICE_USE_TEST_POOL → test_net_en/ or test_net_es/
  - otherwise → synthethic-voices-english/ or synthethic-voices-spanish/
"""

import base64
import random
import re
import threading

from flask import Blueprint, jsonify

from MIID.datasets.signature_auth import verify_request_signature

import sys

sys.path.insert(0, "/home/ubuntu/YanezMIIDManage")
from media_pools.config import (
    HOTKEY_TO_FOLDER,
    VOICE_BATCH_DIR,
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
    Return (language, pool_dir) for this validator.

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
    return language, VOICE_BATCH_DIR / pools[language]


@voice_bp.route("/voice/<hotkey>", methods=["POST"])
def get_validator_voice(hotkey):
    """
    Serve a reference voice clip for the voice-clone challenge.

    Picks language uniformly at random (en or es), then a random WAV from
    the matching pool (testnet or sandbox: test_net_*; else synthethic-voices-*).
    Transcript is the fixed challenge text for that language. Gender is
    inferred from M1/M3 (male) vs M2/M4 (female).
    """
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify({"error": "Unauthorized hotkey or no folder for this validator"}),
            403,
        )

    err_resp, err_status = verify_request_signature(tmp_prefix="tmp_signature_voice")
    if err_resp is not None:
        return err_resp, err_status

    language, lang_dir = _select_pool(hotkey)

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

    transcript = VOICE_TRANSCRIPTS.get(language, "")
    gender = _gender_from_filename(filename)

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
