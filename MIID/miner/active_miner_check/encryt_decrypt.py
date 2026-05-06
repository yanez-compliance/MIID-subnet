"""Active miner check: encrypt/decrypt round-trip with drand timelock.

Reads every image from
    MIID/miner/active_miner_check/image_test/seed_image/
encrypts each one with drand timelock for a target round ~1 minute in the
future, writes the ciphertext to
    MIID/miner/active_miner_check/image_test/encryption/
then waits for the drand beacon to release the signature, decrypts the
ciphertext, and writes the recovered image to
    MIID/miner/active_miner_check/image_test/decryption/

Run from the project root or directly:
    python MIID/miner/active_miner_check/encryt_decrypt.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

# Make the project importable when executed as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MIID.miner.drand_encrypt import (  # noqa: E402
    decrypt_with_drand,
    encrypt_for_drand,
    is_timelock_available,
)


# Drand Quicknet (3-second period). Same chain used by drand_utils.py.
DRAND_QUICKNET_URL = (
    "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
)
FALLBACK_GENESIS = 1692803367
FALLBACK_PERIOD = 3

DELAY_SECONDS = 60  # ~1 minute lock
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

BASE_DIR = Path(__file__).resolve().parent / "image_test"
SEED_DIR = BASE_DIR / "seed_image"
ENC_DIR = BASE_DIR / "encryption"
DEC_DIR = BASE_DIR / "decryption"


def _log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def get_drand_info() -> tuple[int, int]:
    try:
        r = requests.get(f"{DRAND_QUICKNET_URL}/info", timeout=5)
        r.raise_for_status()
        info = r.json()
        return int(info["genesis_time"]), int(info["period"])
    except Exception as e:
        _log("WARN", f"Falling back to default drand info: {e}")
        return FALLBACK_GENESIS, FALLBACK_PERIOD


def calculate_target_round(delay_seconds: int) -> tuple[int, int, int]:
    genesis, period = get_drand_info()
    target_time = int(time.time()) + delay_seconds
    target_round = (target_time - genesis) // period + 1
    reveal_ts = genesis + (target_round - 1) * period
    return target_round, reveal_ts, period


def fetch_signature(round_number: int) -> bytes | None:
    try:
        r = requests.get(f"{DRAND_QUICKNET_URL}/public/{round_number}", timeout=5)
        if r.status_code == 200:
            return bytes.fromhex(r.json()["signature"])
    except Exception as e:
        _log("WARN", f"Signature fetch failed for round {round_number}: {e}")
    return None


def wait_for_signature(
    round_number: int, reveal_ts: int, timeout: int = 180
) -> bytes | None:
    wait = max(0, reveal_ts + 2 - int(time.time()))
    if wait > 0:
        _log("INFO", f"Waiting {wait}s for drand round {round_number}...")
        time.sleep(wait)

    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        sig = fetch_signature(round_number)
        if sig:
            return sig
        time.sleep(2)
    return None


def collect_seed_images() -> list[Path]:
    if not SEED_DIR.exists():
        _log("ERROR", f"Seed image directory not found: {SEED_DIR}")
        return []
    return sorted(
        p for p in SEED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> int:
    if not is_timelock_available():
        _log(
            "ERROR",
            "timelock package is not installed. "
            "Install with: pip install timelock",
        )
        return 1

    images = collect_seed_images()
    if not images:
        _log("ERROR", f"No images found in {SEED_DIR}")
        return 1

    ENC_DIR.mkdir(parents=True, exist_ok=True)
    DEC_DIR.mkdir(parents=True, exist_ok=True)

    target_round, reveal_ts, period = calculate_target_round(DELAY_SECONDS)
    _log(
        "INFO",
        f"Encrypting {len(images)} image(s) for drand round {target_round} "
        f"(reveal in ~{DELAY_SECONDS}s, period={period}s)",
    )

    encrypted: list[tuple[Path, Path]] = []
    for img in images:
        plaintext = img.read_bytes()
        ciphertext = encrypt_for_drand(plaintext, target_round)
        out = ENC_DIR / f"{img.stem}.enc"
        out.write_bytes(ciphertext)
        encrypted.append((out, img))
        _log(
            "OK",
            f"Encrypted {img.name} -> {out.name} "
            f"({len(plaintext)} -> {len(ciphertext)} bytes)",
        )

    signature = wait_for_signature(target_round, reveal_ts)
    if signature is None:
        _log("ERROR", f"Could not retrieve drand signature for round {target_round}")
        return 2

    failures = 0
    for enc_path, original in encrypted:
        try:
            ciphertext = enc_path.read_bytes()
            plaintext = decrypt_with_drand(ciphertext, signature)
            out = DEC_DIR / original.name
            out.write_bytes(plaintext)
            ok = plaintext == original.read_bytes()
            status = "OK" if ok else "MISMATCH"
            if not ok:
                failures += 1
            _log(
                status,
                f"Decrypted {enc_path.name} -> {out.name} "
                f"({len(plaintext)} bytes)",
            )
        except Exception as e:
            failures += 1
            _log("ERROR", f"Decryption failed for {enc_path.name}: {e}")

    if failures:
        _log("ERROR", f"{failures} image(s) failed the round-trip check")
        return 3

    _log("DONE", "Encrypt/decrypt round-trip succeeded for all images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
