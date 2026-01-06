# MIID/miner/s3_upload.py
#
# Phase 4: S3 upload module for miners.
# Uploads encrypted images to YANEZ S3 bucket.
# Falls back to local storage if S3 is not configured.

import os
import time
import json
import bittensor as bt
from pathlib import Path
from typing import Optional

# Try to import boto3 for S3 uploads
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    bt.logging.warning("boto3 not installed. S3 uploads disabled, using local storage.")


# =============================================================================
# S3 Configuration
# =============================================================================
S3_BUCKET_NAME = os.environ.get("MIID_S3_BUCKET", "yanez-miid-sn54")
S3_REGION = os.environ.get("MIID_S3_REGION", "us-east-1")

# Set to True to use S3, False for local storage only
USE_S3 = os.environ.get("MIID_USE_S3", "true").lower() == "true" and BOTO3_AVAILABLE

# Local storage directory (fallback or for sandbox testing)
LOCAL_STORAGE_DIR = Path(os.environ.get("MIID_LOCAL_STORAGE", "/tmp/miid_submissions"))

# S3 client (initialized lazily)
_s3_client = None


def get_s3_client():
    """Get or create S3 client."""
    global _s3_client
    if _s3_client is None and BOTO3_AVAILABLE:
        try:
            _s3_client = boto3.client('s3', region_name=S3_REGION)
            bt.logging.info(f"[S3] Initialized S3 client for bucket: {S3_BUCKET_NAME}")
        except Exception as e:
            bt.logging.error(f"[S3] Failed to create S3 client: {e}")
            return None
    return _s3_client


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
    variation_type: str,
    path_signature: str
) -> Optional[str]:
    """Upload encrypted image to S3 (or local storage as fallback).

    Uploads to YANEZ S3 bucket: s3://yanez-miid-sn54/
    Falls back to local storage if S3 is not configured or unavailable.

    SECURITY: path_signature prevents malicious miners from writing to
    other miners' paths. Only the miner with the private key can generate
    the correct path_signature for their hotkey.

    Args:
        encrypted_data: Tlock-encrypted image bytes
        miner_hotkey: Miner's hotkey address for path
        signature: Wallet signature proving ownership
        image_hash: SHA256 hash of original image
        target_round: Drand round for reveal
        challenge_id: Challenge identifier
        variation_type: Type of variation (pose, expression, etc.)
        path_signature: Unique path component derived from miner's signature
                       Format: sign(challenge_id:miner_hotkey)[:16]

    Returns:
        S3 key (path) if successful, None if failed
    """
    # Generate S3 key path with path_signature for security
    timestamp = int(time.time())
    s3_key = f"submissions/{challenge_id}/{miner_hotkey}/{path_signature}/{variation_type}_{timestamp}.png.tlock"

    # Prepare metadata
    metadata = {
        "hotkey": miner_hotkey,
        "signature": signature,
        "image_hash": image_hash,
        "target_round": str(target_round),
        "challenge_id": challenge_id,
        "variation_type": variation_type,
        "path_signature": path_signature,
        "timestamp": str(timestamp),
        "size_bytes": str(len(encrypted_data))
    }

    # Try S3 upload if configured
    if USE_S3:
        try:
            s3_client = get_s3_client()
            if s3_client:
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=s3_key,
                    Body=encrypted_data,
                    Metadata=metadata,
                    ContentType='application/octet-stream'
                )
                bt.logging.info(
                    f"[S3] Uploaded: s3://{S3_BUCKET_NAME}/{s3_key} "
                    f"({len(encrypted_data)} bytes)"
                )
                return s3_key
        except NoCredentialsError:
            bt.logging.warning("[S3] No AWS credentials found. Falling back to local storage.")
        except ClientError as e:
            bt.logging.warning(f"[S3] Upload failed: {e}. Falling back to local storage.")
        except Exception as e:
            bt.logging.warning(f"[S3] Unexpected error: {e}. Falling back to local storage.")

    # Fallback to local storage
    try:
        ensure_local_storage()

        # Create full local path
        local_path = LOCAL_STORAGE_DIR / s3_key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Write encrypted data
        with open(local_path, "wb") as f:
            f.write(encrypted_data)

        # Write metadata file alongside
        metadata_path = local_path.with_suffix(".meta.json")
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
    """Download encrypted file from S3 (or local storage as fallback).

    Used by post-validator to retrieve encrypted images.

    Args:
        s3_key: Path to the encrypted file (S3 key)

    Returns:
        Encrypted bytes, or None if not found
    """
    # Try S3 download if configured
    if USE_S3:
        try:
            s3_client = get_s3_client()
            if s3_client:
                response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                data = response['Body'].read()
                bt.logging.debug(f"[S3] Downloaded: s3://{S3_BUCKET_NAME}/{s3_key}")
                return data
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                bt.logging.warning(f"[S3] File not found: {s3_key}")
            else:
                bt.logging.warning(f"[S3] Download failed: {e}. Trying local storage.")
        except Exception as e:
            bt.logging.warning(f"[S3] Unexpected error: {e}. Trying local storage.")

    # Fallback to local storage
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
        s3_key: Path to the file (S3 key)

    Returns:
        Metadata dict, or None if not found
    """
    # Try S3 if configured
    if USE_S3:
        try:
            s3_client = get_s3_client()
            if s3_client:
                response = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                metadata = response.get('Metadata', {})
                bt.logging.debug(f"[S3] Got metadata for: {s3_key}")
                return metadata
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                bt.logging.warning(f"[S3] File not found: {s3_key}")
            else:
                bt.logging.warning(f"[S3] Failed to get metadata: {e}. Trying local storage.")
        except Exception as e:
            bt.logging.warning(f"[S3] Unexpected error: {e}. Trying local storage.")

    # Fallback to local storage
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

    Expected format:
    submissions/{challenge_id}/{miner_hotkey}/{path_signature}/{variation_type}_{timestamp}.png.tlock

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
    # Expected: submissions / challenge_id / miner_hotkey / path_signature / filename
    if len(parts) < 5:
        return False

    if not s3_key.endswith(".tlock"):
        return False

    return True


def generate_s3_key(
    challenge_id: str,
    miner_hotkey: str,
    variation_type: str,
    path_signature: str,
    timestamp: Optional[int] = None
) -> str:
    """Generate a valid S3 key for an image submission.

    Args:
        challenge_id: Challenge identifier
        miner_hotkey: Miner's hotkey address
        variation_type: Type of variation
        path_signature: Unique path component from miner's signature
        timestamp: Optional timestamp (defaults to current time)

    Returns:
        S3 key string
    """
    if timestamp is None:
        timestamp = int(time.time())

    return f"submissions/{challenge_id}/{miner_hotkey}/{path_signature}/{variation_type}_{timestamp}.png.tlock"


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
