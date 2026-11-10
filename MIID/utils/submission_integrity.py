"""Verify the existing miner commitment without loading any scoring models.

The signature authenticates challenge + plaintext SHA-256, not arbitrary JSON
fields. Path checks are consistency checks, not S3 access control.
"""
import hashlib
import hmac
import re

from bittensor_wallet import Keypair


def validate_challenge_id(challenge_id):
    if not isinstance(challenge_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", challenge_id):
        raise ValueError("Invalid challenge identifier")
    return challenge_id


def verify_commitment(challenge_id, hotkey, image_hash, signature, s3_key, path_signature):
    validate_challenge_id(challenge_id)
    if not isinstance(image_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", image_hash):
        raise ValueError("Invalid submission hash")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-fA-F]{128}", signature):
        raise ValueError("Invalid submission signature encoding")
    if not isinstance(path_signature, str) or not re.fullmatch(r"[0-9a-fA-F]{16}", path_signature):
        raise ValueError("Invalid submission path component")
    parts = s3_key.split("/") if isinstance(s3_key, str) else []
    if (len(parts) != 6 or parts[:4] != ["submissions", challenge_id, hotkey, path_signature]
            or any(p in ("", ".", "..") or "\\" in p for p in parts)
            or not parts[-1].endswith(".tlock")):
        raise ValueError("Submission path does not match miner and challenge")
    try:
        valid = Keypair(ss58_address=hotkey).verify(
            data=f"challenge:{challenge_id}:hash:{image_hash}".encode(),
            signature=bytes.fromhex(signature),
        )
    except Exception as exc:
        raise ValueError("Invalid submission signer") from exc
    if not valid:
        raise ValueError("Submission signature does not match miner and challenge")


def verify_plaintext(data, image_hash):
    if not isinstance(data, bytes) or not data:
        raise ValueError("Submission decryption returned no bytes")
    if not isinstance(image_hash, str) or not hmac.compare_digest(hashlib.sha256(data).hexdigest(), image_hash):
        raise ValueError("Stored submission differs from the signed hash")
