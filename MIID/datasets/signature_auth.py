"""Verify validator request proofs and reject reuse across API workers."""

import os
import re
import sqlite3
import time
from contextlib import closing

from bittensor_wallet import Keypair
from flask import current_app, jsonify, request

from MIID.datasets.config import DATA_DIR
from MIID.utils.validator_requests import PROOF_HEADER, request_message

MAX_PROOF_AGE = 120
MAX_CLOCK_SKEW = 15
PROOF_PATTERN = re.compile(r"1\.([0-9]{1,12})\.([0-9a-f]{32})\.([0-9a-f]{128})")


def _consume_nonce(hotkey, nonce, now, expires_at):
    path = os.path.join(DATA_DIR, "validator_request_nonces.sqlite3")
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS request_nonces ("
            "hotkey TEXT NOT NULL, nonce TEXT NOT NULL, expires_at INTEGER NOT NULL, "
            "PRIMARY KEY (hotkey, nonce))"
        )
        db.execute("DELETE FROM request_nonces WHERE expires_at < ?", (now,))
        cursor = db.execute(
            "INSERT OR IGNORE INTO request_nonces VALUES (?, ?, ?)",
            (hotkey, nonce, expires_at),
        )
        return cursor.rowcount == 1


def verify_request_signature(expected_hotkey):
    if not request.is_json or request.query_string:
        return jsonify({"error": "Expected a JSON request without query parameters"}), 400

    proof = PROOF_PATTERN.fullmatch(request.headers.get(PROOF_HEADER, ""))
    if proof is None:
        return jsonify({"error": "Missing or invalid validator request proof"}), 401

    timestamp_text, nonce, signature = proof.groups()
    timestamp = int(timestamp_text)
    now = int(time.time())
    if not now - MAX_PROOF_AGE <= timestamp <= now + MAX_CLOCK_SKEW:
        return jsonify({"error": "Validator request proof expired or not yet valid"}), 401

    message = request_message(
        request.method, request.path, request.get_data(), timestamp, nonce
    )
    try:
        key = Keypair(ss58_address=expected_hotkey, ss58_format=42)
        valid = key.verify(data=message, signature=bytes.fromhex(signature))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return jsonify({"error": "Invalid signature for validator request"}), 403

    if not isinstance(request.get_json(silent=True), dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    try:
        if not _consume_nonce(expected_hotkey, nonce, now, timestamp + MAX_PROOF_AGE):
            return jsonify({"error": "Validator request proof already used"}), 409
    except sqlite3.Error:
        current_app.logger.exception("Could not record validator request nonce")
        return jsonify({"error": "Validator request verification unavailable"}), 503

    return None, None
