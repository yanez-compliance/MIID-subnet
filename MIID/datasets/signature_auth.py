"""Shared signature verification for Flask media / upload routes."""

import os
import time
from typing import Optional, Tuple

from flask import jsonify, request

from MIID.datasets.config import DATA_DIR
from MIID.utils.verify_message import verify_message


def verify_request_signature(tmp_prefix: str = "tmp_signature"):
    """
    Verify the JSON body has a valid ``signature`` field.

    Returns:
        (None, None) on success.
        (response, status_code) on failure — caller should ``return response, status_code``.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.get_json()
    signature_text = data.get("signature") if data else None
    if not signature_text:
        return jsonify({"error": "Missing 'signature' in JSON payload"}), 400

    tmp_signature_filename = os.path.join(
        DATA_DIR, f"{tmp_prefix}_{time.time()}.txt"
    )
    with open(tmp_signature_filename, "w", encoding="utf-8") as tmp_file:
        tmp_file.write(signature_text)

    try:
        verify_message(tmp_signature_filename)
    except ValueError as e:
        os.remove(tmp_signature_filename)
        return jsonify({"error": f"Signature verification failed: {str(e)}"}), 400

    os.remove(tmp_signature_filename)
    return None, None


def get_json_after_auth(tmp_prefix: str = "tmp_signature") -> Tuple[Optional[dict], Optional[tuple]]:
    """
    Verify signature and return the request JSON.

    Returns:
        (data, None) on success.
        (None, (response, status_code)) on failure.
    """
    err_resp, err_status = verify_request_signature(tmp_prefix=tmp_prefix)
    if err_resp is not None:
        return None, (err_resp, err_status)
    return request.get_json(), None
