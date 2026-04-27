#!/usr/bin/env python3
"""Local drand timelock round-trip tester for miners.

Reads images from image_test/seed_image, encrypts with the same stack as
MIID.miner.drand_encrypt (Quicknet IBE + ephemeral key), writes ciphertext to
image_test/encryption, waits until the target drand round is available (~delay),
then decrypts and writes PNG/JPEG/etc. to image_test/decryption.

Run from repo root (MIID-subnet):

    python MIID/miner/active_miner_check/encrypt_decrypt.py

Requires timelock (see requirements-miner.txt; Linux x86_64 + Python 3.10 for wheels).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from MIID.miner.drand_encrypt import (  # noqa: E402
    decrypt_with_drand,
    encrypt_for_drand,
    is_timelock_available,
)
from MIID.validator.drand_utils import (  # noqa: E402
    calculate_target_round,
    wait_for_round,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SEED_DIR = SCRIPT_DIR / "image_test" / "seed_image"
ENC_DIR = SCRIPT_DIR / "image_test" / "encryption"
DEC_DIR = SCRIPT_DIR / "image_test" / "decryption"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _list_seed_images() -> List[Path]:
    if not SEED_DIR.is_dir():
        return []
    return sorted(
        p
        for p in SEED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _ensure_dirs() -> None:
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    DEC_DIR.mkdir(parents=True, exist_ok=True)


def run(delay_seconds: int, wait_timeout: int) -> int:
    if not is_timelock_available():
        print(
            "timelock is not installed or failed to import.\n"
            "Install pinned deps: pip install -r requirements-miner.txt\n"
            "(Linux x86_64 + Python 3.10 needed for prebuilt WASM wheels.)",
            file=sys.stderr,
        )
        return 1

    seeds = _list_seed_images()
    if not seeds:
        print(f"No images found under {SEED_DIR}", file=sys.stderr)
        return 1

    _ensure_dirs()

    target_round, reveal_ts = calculate_target_round(delay_seconds)
    now = int(time.time())
    print(
        f"Target drand round={target_round} (reveal ~{reveal_ts}, "
        f"~{max(0, reveal_ts - now)}s from now, requested delay={delay_seconds}s)"
    )

    enc_map: Dict[str, Dict[str, object]] = {}
    for path in seeds:
        data = path.read_bytes()
        ciphertext = encrypt_for_drand(data, target_round)
        # Same naming convention as production uploads: <name>.<ext>.tlock
        out_name = f"{path.name}.tlock"
        out_path = ENC_DIR / out_name
        out_path.write_bytes(ciphertext)
        enc_map[out_name] = {
            "seed": path.name,
            "target_round": target_round,
            "ciphertext_bytes": len(ciphertext),
        }
        print(f"Encrypted {path.name} -> {out_path.name} ({len(ciphertext)} bytes)")

    meta_path = ENC_DIR / "batch_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "target_round": target_round,
                "reveal_timestamp": reveal_ts,
                "delay_seconds": delay_seconds,
                "files": enc_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {meta_path.name}")

    print(f"Waiting for drand round {target_round} (timeout {wait_timeout}s)...")
    signature = wait_for_round(target_round, timeout=wait_timeout)
    if not signature:
        print("Failed to obtain drand signature for round.", file=sys.stderr)
        return 1

    for path in seeds:
        enc_name = f"{path.name}.tlock"
        enc_path = ENC_DIR / enc_name
        ciphertext = enc_path.read_bytes()
        plaintext = decrypt_with_drand(ciphertext, signature)
        dec_path = DEC_DIR / path.name
        dec_path.write_bytes(plaintext)
        ok = plaintext == path.read_bytes()
        print(f"Decrypted {enc_name} -> {dec_path.name} (bytes match seed: {ok})")

    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Seconds ahead to target for drand round (default: 30)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Max seconds to wait for drand signature (default: 300)",
    )
    args = parser.parse_args(argv)
    if args.delay < 5:
        print("--delay should be at least a few seconds (drand period is 3s).", file=sys.stderr)
        return 1
    return run(args.delay, args.wait_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
