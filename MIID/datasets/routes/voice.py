"""
Voice endpoints for validators:
  POST /voice/<hotkey> — reference WAV for the voice-clone challenge

Validators only call this API. Pool selection and serve bookkeeping live on
the core machine (YanezMIIDManage.media_pools.voice_serve).
"""

import sys

from flask import Blueprint, jsonify

from MIID.datasets.signature_auth import verify_request_signature

sys.path.insert(0, "/home/ubuntu/YanezMIIDManage")
from media_pools.config import HOTKEY_TO_FOLDER
from media_pools.voice_serve import allocate_voice_for_hotkey

voice_bp = Blueprint("voice", __name__)


@voice_bp.route("/voice/<hotkey>", methods=["POST"])
def get_validator_voice(hotkey):
    """Return one reference voice clip chosen by the core machine."""
    if hotkey not in HOTKEY_TO_FOLDER:
        return (
            jsonify({"error": "Unauthorized hotkey or no folder for this validator"}),
            403,
        )

    err_resp, err_status = verify_request_signature(hotkey)
    if err_resp is not None:
        return err_resp, err_status

    voice_payload, err = allocate_voice_for_hotkey(hotkey)
    if err is not None:
        body, status = err
        return jsonify(body), status

    return (
        jsonify(
            {
                "verified_by": hotkey,
                "voice": voice_payload,
            }
        ),
        200,
    )
