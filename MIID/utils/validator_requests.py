"""Signed HTTPS requests for the MIID validator data service."""

import hashlib
import json
import os
import secrets
import time
from urllib.parse import urlsplit

import requests

PROOF_HEADER = "X-MIID-Request-Proof"
MAX_ALLOCATION_BATCH = 32


def request_message(method, path, body, timestamp, nonce):
    return json.dumps(
        [
            "miid-validator-request-v1",
            os.environ.get("MIID_VALIDATOR_AUDIENCE", "sn54"),
            method,
            path,
            hashlib.sha256(body).hexdigest(),
            timestamp,
            nonce,
        ],
        separators=(",", ":"),
    ).encode("utf-8")


def post_validator_request(path, hotkey, payload):
    base_url = os.environ.get("MIID_VALIDATOR_API_URL") or os.environ.get(
        "MIID_IMAGES_SERVER", ""
    )
    url = urlsplit(base_url)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError("Set MIID_VALIDATOR_API_URL to the HTTPS validator API base URL")

    request_path = f"{url.path.rstrip('/')}{path}"
    body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    timestamp = int(time.time())
    nonce = secrets.token_hex(16)
    signature = hotkey.sign(
        data=request_message("POST", request_path, body, timestamp, nonce)
    ).hex()
    return requests.post(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={
            "Content-Type": "application/json",
            PROOF_HEADER: f"1.{timestamp}.{nonce}.{signature}",
        },
        timeout=30,
        allow_redirects=False,
    )
