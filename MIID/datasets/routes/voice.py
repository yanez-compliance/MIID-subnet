"""
Voice endpoints for validators:
  POST /voice/<hotkey> — random reference WAV from en/ or es/ pool
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


@voice_bp.route("/voice/<hotkey>", methods=["POST"])
def get_validator_voice(hotkey):
    """
    Serve a reference voice clip for the voice-clone challenge.

    Picks language uniformly at random (en or es), then a random WAV from
    VOICE_BATCH_DIR/<language>/. Transcript is the fixed challenge text for
    that language. Gender is inferred from M1/M3 (male) vs M2/M4 (female).
    """
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify({"error": "Unauthorized hotkey or no folder for this validator"}),
            403,
        )

    err_resp, err_status = verify_request_signature(tmp_prefix="tmp_signature_voice")
    if err_resp is not None:
        return err_resp, err_status

    language = random.choice(["en", "es"])
    lang_dir = VOICE_BATCH_DIR / language

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
