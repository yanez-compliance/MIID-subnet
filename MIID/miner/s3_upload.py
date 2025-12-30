# MIID/miner/s3_upload.py
#
# Phase 4: Local storage module for miners (replaces S3 in sandbox).
# Stores encrypted images locally for testing. YANEZ will provide real S3 later.

import os
import time
import json
import bittensor as bt
from pathlib import Path
from typing import Optional


# Local storage directory (created under /tmp or user-specified)
LOCAL_STORAGE_DIR = Path(os.environ.get("MIID_LOCAL_STORAGE", "/tmp/miid_submissions"))

# Placeholder for future S3 URL (TBD by YANEZ)
S3_UPLOAD_URL = "https://placeholder-s3-url.example.com"  # TBD
S3_BUCKET_NAME = "miid-image-variations"  # TBD


def ensure_local_storage():
    """Ensure local storage directory exists."""
    LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def upload_to_s3(
    encrypted_data: bytes,
    miner_hotkey: str,
    signature: str,
    image_hash: str,
    target_round: int,
    challenge_id: str,
    variation_type: str
) -> Optional[str]:
    """Store encrypted image locally (simulates S3 upload).

    LOCAL STORAGE IMPLEMENTATION:
    Saves files to local filesystem for sandbox testing.
    Structure mirrors what would be in S3.

    Args:
        encrypted_data: Tlock-encrypted image bytes
        miner_hotkey: Miner's hotkey address for path
        signature: Wallet signature proving ownership
        image_hash: SHA256 hash of original image
        target_round: Drand round for reveal
        challenge_id: Challenge identifier
        variation_type: Type of variation (pose, expression, etc.)

    Returns:
        S3-style key (path) if successful, None if failed
    """
    try:
        ensure_local_storage()

        # Generate S3-style key path
        timestamp = int(time.time())
        s3_key = f"submissions/{challenge_id}/{miner_hotkey}/{variation_type}_{timestamp}.png.tlock"

        # Create full local path
        local_path = LOCAL_STORAGE_DIR / s3_key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Write encrypted data
        with open(local_path, "wb") as f:
            f.write(encrypted_data)

        # Write metadata file alongside
        metadata_path = local_path.with_suffix(".meta.json")
        metadata = {
            "hotkey": miner_hotkey,
            "signature": signature,
            "image_hash": image_hash,
            "target_round": target_round,
            "challenge_id": challenge_id,
            "variation_type": variation_type,
            "timestamp": timestamp,
            "size_bytes": len(encrypted_data)
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        bt.logging.info(
            f"[LOCAL STORAGE] Saved: {s3_key} "
            f"({len(encrypted_data)} bytes)"
        )

        return s3_key

    except Exception as e:
        bt.logging.error(f"[LOCAL STORAGE] Failed to save: {e}")
        return None


def download_from_s3(s3_key: str) -> Optional[bytes]:
    """Download encrypted file from local storage.

    Used by post-validator to retrieve encrypted images.

    Args:
        s3_key: Path to the encrypted file (S3-style key)

    Returns:
        Encrypted bytes, or None if not found
    """
    try:
        local_path = LOCAL_STORAGE_DIR / s3_key
        if not local_path.exists():
            bt.logging.warning(f"[LOCAL STORAGE] File not found: {s3_key}")
            return None

        with open(local_path, "rb") as f:
            return f.read()

    except Exception as e:
        bt.logging.error(f"[LOCAL STORAGE] Failed to read: {e}")
        return None


def get_s3_metadata(s3_key: str) -> Optional[dict]:
    """Get metadata for a stored file.

    Args:
        s3_key: Path to the file (S3-style key)

    Returns:
        Metadata dict, or None if not found
    """
    try:
        local_path = LOCAL_STORAGE_DIR / s3_key
        metadata_path = local_path.with_suffix(".meta.json")

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    except Exception as e:
        bt.logging.error(f"[LOCAL STORAGE] Failed to read metadata: {e}")
        return None


def validate_s3_key(s3_key: str) -> bool:
    """Validate S3 key format.

    Args:
        s3_key: S3 key to validate

    Returns:
        True if key format is valid
    """
    if not s3_key:
        return False

    if not s3_key.startswith("submissions/"):
        return False

    parts = s3_key.split("/")
    if len(parts) < 3:
        return False

    if not s3_key.endswith(".tlock"):
        return False

    return True


def generate_s3_key(
    challenge_id: str,
    miner_hotkey: str,
    variation_type: str,
    timestamp: Optional[int] = None
) -> str:
    """Generate a valid S3 key for an image submission.

    Args:
        challenge_id: Challenge identifier
        miner_hotkey: Miner's hotkey address
        variation_type: Type of variation
        timestamp: Optional timestamp (defaults to current time)

    Returns:
        S3 key string
    """
    if timestamp is None:
        timestamp = int(time.time())

    return f"submissions/{challenge_id}/{miner_hotkey}/{variation_type}_{timestamp}.png.tlock"


def list_local_submissions(challenge_id: Optional[str] = None) -> list:
    """List all local submissions.

    Args:
        challenge_id: Optional filter by challenge ID

    Returns:
        List of submission info dicts
    """
    ensure_local_storage()
    submissions = []

    submissions_dir = LOCAL_STORAGE_DIR / "submissions"
    if not submissions_dir.exists():
        return []

    for challenge_dir in submissions_dir.iterdir():
        if not challenge_dir.is_dir():
            continue

        if challenge_id and challenge_dir.name != challenge_id:
            continue

        for miner_dir in challenge_dir.iterdir():
            if not miner_dir.is_dir():
                continue

            for file_path in miner_dir.glob("*.tlock"):
                metadata = get_s3_metadata(str(file_path.relative_to(LOCAL_STORAGE_DIR)))
                submissions.append({
                    "s3_key": str(file_path.relative_to(LOCAL_STORAGE_DIR)),
                    "challenge_id": challenge_dir.name,
                    "miner_hotkey": miner_dir.name,
                    "file_size": file_path.stat().st_size,
                    "metadata": metadata
                })

    return submissions


def get_storage_stats() -> dict:
    """Get local storage statistics.

    Returns:
        Dict with storage stats
    """
    ensure_local_storage()

    total_files = 0
    total_bytes = 0

    for root, dirs, files in os.walk(LOCAL_STORAGE_DIR):
        for f in files:
            if f.endswith(".tlock"):
                total_files += 1
                total_bytes += (Path(root) / f).stat().st_size

    return {
        "storage_path": str(LOCAL_STORAGE_DIR),
        "total_submissions": total_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2)
    }
