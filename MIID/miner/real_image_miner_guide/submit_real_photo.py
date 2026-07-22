r"""Real screen-replay photo submitter.

This is the ONE file miners run after dropping a real photo into inbox/.
See README.md in this folder for the full walkthrough.

What it does:
  1. Finds the single image file you placed in inbox/ (next to this script).
  2. Asks a few quick questions about the photo (camera used, device the
     seed image was displayed on, which visual cues are visible) — unless
     you already answered them via CLI flags.
  3. Moves your photo into staged/ (so inbox/ is free for next time) and
     rewrites screen_replay.json with the photo path + your answers, then
     flips "ready" to true.
  4. That's it. The miner process (neurons/miner.py) is always running and
     checks screen_replay.json on every validator query. As soon as it sees
     "ready": true it takes over: encrypts your photo, uploads it to S3 under
     the real submission path (this is the ONLY S3 upload that happens — no
     separate/duplicate raw upload here), sends it to the validator, and
     automatically flips "ready" back to false so it won't resubmit the same
     photo again.

Usage:
    python MIID/miner/real_image_miner_guide/submit_real_photo.py

    # Or answer everything up front (no prompts):
    python MIID/miner/real_image_miner_guide/submit_real_photo.py \
        --camera "iPhone 15 Pro" --device phone \
        --moire --glare --keystone --gamma --edge-crop

Run from the project root or directly (this script fixes up sys.path itself).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
from pathlib import Path

# Make the project importable when executed as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402

# Must match SCREEN_REPLAY_DEVICE_TYPES in MIID/validator/image_variations.py
DEVICE_TYPES = ["phone", "tablet", "laptop", "monitor", "tv"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

HERE = Path(__file__).resolve().parent
INBOX_DIR = HERE / "inbox"
STAGED_DIR = HERE / "staged"
SCREEN_REPLAY_JSON = HERE / "screen_replay.json"


def _log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def find_inbox_photo() -> Path:
    """Return the single image file sitting in inbox/, or exit with an error."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        p for p in INBOX_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not candidates:
        _log("ERROR", f"No image found in {INBOX_DIR}")
        _log("ERROR", "Drop your real photo (jpg/jpeg/png/webp/bmp) in that folder and re-run.")
        sys.exit(1)

    if len(candidates) > 1:
        _log(
            "WARN",
            f"Found {len(candidates)} images in inbox/, using the first one alphabetically: "
            f"{candidates[0].name}. Remove the others if that's not what you want.",
        )

    return candidates[0]


def validate_image(path: Path) -> None:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        _log("ERROR", f"'{path}' does not look like a valid image: {e}")
        sys.exit(1)


def prompt_if_missing(value: str | None, prompt: str, default: str = "") -> str:
    if value:
        return value
    entered = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return entered or default


def prompt_device(value: str | None) -> str:
    if value and value in DEVICE_TYPES:
        return value
    while True:
        entered = input(f"Device the seed image was displayed on ({'/'.join(DEVICE_TYPES)}): ").strip().lower()
        if entered in DEVICE_TYPES:
            return entered
        print(f"  Please enter one of: {', '.join(DEVICE_TYPES)}")


def prompt_bool_if_missing(flag_value: bool, prompt: str) -> bool:
    if flag_value:
        return True
    entered = input(f"{prompt} [y/N]: ").strip().lower()
    return entered in ("y", "yes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", help="Camera/phone used to take the photo, e.g. 'iPhone 15 Pro'")
    parser.add_argument("--device", choices=DEVICE_TYPES, help="Device the seed image was displayed on")
    parser.add_argument("--date", help="Capture date YYYY-MM-DD (UTC). Defaults to today.")
    parser.add_argument("--moire", action="store_true", help="Moiré / pixel grid interference visible")
    parser.add_argument("--glare", action="store_true", help="Screen glare hotspots visible")
    parser.add_argument("--keystone", action="store_true", help="Perspective / keystone distortion visible")
    parser.add_argument("--gamma", action="store_true", help="Gamma / contrast shift visible")
    parser.add_argument("--edge-crop", action="store_true", dest="edge_crop", help="Screen edge/bezel/crop cues visible")
    args = parser.parse_args()

    photo_path = find_inbox_photo()
    validate_image(photo_path)
    photo_bytes = photo_path.read_bytes()
    _log("OK", f"Found photo: {photo_path.name} ({len(photo_bytes)} bytes)")

    camera_used = prompt_if_missing(args.camera, "Camera/phone used to take the photo (e.g. 'iPhone 15 Pro')")
    device_photographed = prompt_device(args.device)
    date = args.date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

    print("\nWhich visual cues are clearly visible in your photo? (report honestly — it's fine if none apply)")
    moire = prompt_bool_if_missing(args.moire, "  Moiré / pixel grid interference")
    glare = prompt_bool_if_missing(args.glare, "  Screen glare hotspots")
    keystone = prompt_bool_if_missing(args.keystone, "  Perspective / keystone distortion")
    gamma = prompt_bool_if_missing(args.gamma, "  Gamma / contrast shift")
    edge_crop = prompt_bool_if_missing(args.edge_crop, "  Screen edge / bezel / crop cues")

    # Move (not copy) the photo out of inbox/ into staged/ so inbox/ is free
    # for the next submission and the miner process has a stable local path
    # to read from (drand encryption needs the raw bytes at query time).
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    staged_name = f"{date}_{photo_path.stem}{photo_path.suffix}"
    staged_path = STAGED_DIR / staged_name
    shutil.move(str(photo_path), str(staged_path))
    _log("OK", f"Moved photo to {staged_path}")

    data = {
        "ready": True,
        "photo_path": str(staged_path.resolve()),
        "date": date,
        "camera_used": camera_used,
        "device_photographed": device_photographed,
        "moire_pixel_grid": moire,
        "screen_glare_hotspots": glare,
        "perspective_keystone_distortion": keystone,
        "gamma_contrast_shift": gamma,
        "edge_crop_cues": edge_crop,
    }
    with open(SCREEN_REPLAY_JSON, "w") as f:
        json.dump(data, f, indent=2)

    _log("DONE", f"Wrote {SCREEN_REPLAY_JSON} with ready=true.")
    _log(
        "DONE",
        "Nothing else to do — your miner process will pick this up and submit it "
        "on its next validator query, then flip ready back to false automatically.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
