r"""Real screen-replay photo/video submitter.

This is the ONE file miners run after dropping a face close-up + environment
pair into inbox/. See README.md in this folder for the full walkthrough.

Photo / video roles (important):
  1. FACE CLOSE-UP — face on screen dominant and centered, minimal angular
     distortion (near head-on). Still photo for the 3 photo variants, or a
     short video for the 3 video variants.
  2. ENVIRONMENT — always a still photo of the whole screen/device in its
     surroundings; angular distortion, keystone, glare, etc. are fine.

Capture variants (pick one — see --variant):
  Photo:
    1. seed_unchanged — seed as-is
    2. seed_smiling — seed edited to smile
    3. seed_eyes_closed — seed edited to eyes closed
  Video:
    4. seed_video_blinking — blink seed-video
    5. seed_video_smiling — smile seed-video
    6. seed_video_smile_and_blink — smile + blink seed-video
  (Every variant also needs an environment still. Device/camera are asked separately.)

There's no limit on how many times you can run this — submit as many
different real captures as you want, whenever you have them ready. The only
rule is that each one must be a genuinely new capture: never re-run this on
the same media twice, and never reuse a file from a previous submission —
duplicates are filtered out and penalised.

Usage:
    python MIID/miner/real_image_miner_guide/submit_real_photo.py

    # Photo variant, non-interactive:
    python MIID/miner/real_image_miner_guide/submit_real_photo.py \
        --variant seed_smiling \
        --face closeup.jpg --env wide.jpg \
        --camera "iPhone 15 Pro" --device phone \
        --moire --glare

    # Video variant:
    python MIID/miner/real_image_miner_guide/submit_real_photo.py \
        --variant seed_video_blinking \
        --face replay.mp4 --env wide.jpg \
        --camera "iPhone 15 Pro" --device laptop

Run from the project root or directly (this script fixes up sys.path itself).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

# Make the project importable when executed as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402

# Must match constants in MIID/validator/image_variations.py
DEVICE_TYPES = ["phone", "tablet", "laptop", "monitor", "tv"]
CAPTURE_VARIANTS = [
    "seed_unchanged",
    "seed_smiling",
    "seed_eyes_closed",
    "seed_video_blinking",
    "seed_video_smiling",
    "seed_video_smile_and_blink",
]
VIDEO_VARIANTS = frozenset({
    "seed_video_blinking",
    "seed_video_smiling",
    "seed_video_smile_and_blink",
})

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v"}
# Phone camera "High Efficiency" formats — converted to JPEG on drop so the
# rest of the pipeline (validate / stage / encrypt) only ever sees common formats.
HEIC_EXTENSIONS = {".heic", ".heif"}

HERE = Path(__file__).resolve().parent
INBOX_DIR = HERE / "inbox"
STAGED_DIR = HERE / "staged"
SCREEN_REPLAY_JSON = HERE / "screen_replay.json"

# Holds captures submitted while a previous one was still waiting for the
# miner process to pick it up, so re-running this script never loses/
# overwrites a pending submission. See queue_existing_pending_capture().
QUEUE_DIR = HERE / "queue"

# Sandbox mode: shared pool of fixed images (checked into git) that miners
# choose from at random, since the validator isn't sending a seed image right
# now (see VALIDATOR_SENDS_SEED_IMAGE in MIID/validator/fixed_images.py).
FIXED_IMAGE_POOL_DIR = PROJECT_ROOT / "MIID" / "validator" / "fixed_image"

VARIANT_HELP = {
    "seed_unchanged": "Seed as-is (no edit)",
    "seed_smiling": "Seed smiling (still)",
    "seed_eyes_closed": "Seed eyes closed (still)",
    "seed_video_blinking": "Seed blink video",
    "seed_video_smiling": "Seed smile video",
    "seed_video_smile_and_blink": "Seed smile + blink video",
}


def _log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def _convert_heic_to_jpeg(src: Path, dest: Path) -> None:
    """Decode one HEIC/HEIF file to JPEG. Tries pillow-heif, then CLI tools."""
    try:
        from pillow_heif import register_heif_opener
        from PIL import Image

        register_heif_opener()
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb.save(dest, format="JPEG", quality=95)
        return
    except ImportError:
        pass
    except Exception as e:
        _log("ERROR", f"Failed to convert '{src.name}' with pillow-heif: {e}")
        sys.exit(1)

    for cmd in (
        ["magick", str(src), str(dest)],
        ["convert", str(src), str(dest)],
    ):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                if dest.exists() and dest.stat().st_size > 0:
                    return
            except Exception as e:
                _log("ERROR", f"Failed to convert '{src.name}' with {cmd[0]}: {e}")
                sys.exit(1)

    if shutil.which("heif-convert"):
        try:
            subprocess.run(
                ["heif-convert", str(src), str(dest)],
                check=True,
                capture_output=True,
            )
            if dest.exists() and dest.stat().st_size > 0:
                return
        except Exception as e:
            _log("ERROR", f"Failed to convert '{src.name}' with heif-convert: {e}")
            sys.exit(1)

    _log(
        "ERROR",
        f"Found phone HEIC/HEIF photo '{src.name}' but no converter is available.",
    )
    _log(
        "ERROR",
        "Install one of:  pip install pillow-heif   OR   apt install libheif-examples / imagemagick",
    )
    sys.exit(1)


def convert_heic_in_inbox() -> None:
    """Convert any .heic/.heif files in inbox/ to JPEG, then remove the originals."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    heic_files = sorted(
        p for p in INBOX_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in HEIC_EXTENSIONS
    )
    if not heic_files:
        return

    for src in heic_files:
        dest = src.with_suffix(".jpg")
        if dest.exists():
            _log(
                "ERROR",
                f"Cannot convert '{src.name}': '{dest.name}' already exists in inbox/. "
                "Remove one of them and re-run.",
            )
            sys.exit(1)
        _convert_heic_to_jpeg(src, dest)
        src.unlink()
        _log("OK", f"Converted phone HEIC → JPEG: {src.name} → {dest.name}")


def _list_inbox_media() -> tuple[list[Path], list[Path]]:
    """Return (image_files, video_files) currently in inbox/."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    convert_heic_in_inbox()
    images = sorted(
        p for p in INBOX_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    videos = sorted(
        p for p in INBOX_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return images, videos


def find_inbox_pair(variant: str) -> tuple[Path, Path]:
    """Find the primary + environment files for this variant, or exit."""
    images, videos = _list_inbox_media()

    if variant in VIDEO_VARIANTS:
        if len(videos) != 1 or len(images) != 1:
            _log(
                "ERROR",
                f"Variant '{variant}' needs exactly 1 video + 1 environment image in "
                f"{INBOX_DIR} (found {len(videos)} video(s), {len(images)} image(s)).",
            )
            _log(
                "ERROR",
                f"Accepted video extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}",
            )
            sys.exit(1)
        # Primary = video, env = image (roles fixed by media type)
        return videos[0], images[0]

    if videos:
        _log(
            "ERROR",
            f"Found video file(s) in inbox/ but variant '{variant}' expects two still "
            "photos. Use a seed_video_* variant for video, or remove "
            "the video from inbox/.",
        )
        sys.exit(1)

    if len(images) < 2:
        _log("ERROR", f"Found {len(images)} image(s) in {INBOX_DIR}, need exactly 2.")
        _log(
            "ERROR",
            "Drop TWO photos of the same on-screen capture "
            "(1 face close-up + 1 environment shot) and re-run.",
        )
        sys.exit(1)

    if len(images) > 2:
        _log(
            "ERROR",
            f"Found {len(images)} images in {INBOX_DIR}, expected exactly 2 "
            "(face close-up + environment of ONE capture). Remove the extras.",
        )
        sys.exit(1)

    return images[0], images[1]


def _resolve_inbox_name(name: str, candidates: list[Path]) -> Path:
    """Match a --face/--env argument to an inbox file (by name or path)."""
    needle = Path(name)
    by_name = {p.name: p for p in candidates}
    if needle.name in by_name:
        return by_name[needle.name]
    try:
        resolved = needle.resolve()
    except Exception:
        resolved = None
    for p in candidates:
        if resolved is not None and p.resolve() == resolved:
            return p
    _log(
        "ERROR",
        f"'{name}' is not one of the inbox files: "
        f"{', '.join(p.name for p in candidates)}",
    )
    sys.exit(1)


def assign_photo_roles(
    photo_a: Path,
    photo_b: Path,
    face_name: str | None = None,
    env_name: str | None = None,
    *,
    primary_is_video: bool = False,
) -> tuple[Path, Path]:
    """Decide which file is the face close-up vs the environment shot.

    Returns (face_closeup_path, environment_path).
    """
    if primary_is_video:
        # find_inbox_pair already returned (video, image)
        return photo_a, photo_b

    candidates = [photo_a, photo_b]

    if face_name and env_name:
        face = _resolve_inbox_name(face_name, candidates)
        env = _resolve_inbox_name(env_name, candidates)
        if face == env:
            _log("ERROR", "--face and --env must refer to two different inbox files.")
            sys.exit(1)
        return face, env

    if face_name or env_name:
        _log("ERROR", "Pass both --face and --env, or neither (you'll be prompted).")
        sys.exit(1)

    print("\nAssign roles for the two inbox photos:")
    print("  FACE CLOSE-UP = face dominant + centered, minimal angular distortion")
    print("  ENVIRONMENT   = whole screen/device in its surroundings (distortion OK)")
    print(f"  1. {photo_a.name}")
    print(f"  2. {photo_b.name}")
    while True:
        entered = input(
            "Which file is the FACE CLOSE-UP? Enter 1 or 2: "
        ).strip()
        if entered == "1":
            return photo_a, photo_b
        if entered == "2":
            return photo_b, photo_a
        print("  Please enter 1 or 2.")


def validate_image(path: Path) -> None:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        _log("ERROR", f"'{path}' does not look like a valid image: {e}")
        sys.exit(1)


def validate_video(path: Path) -> None:
    """Lightweight video check: non-empty + recognizable container magic when possible."""
    try:
        raw = path.read_bytes()
    except Exception as e:
        _log("ERROR", f"Cannot read video '{path}': {e}")
        sys.exit(1)
    if len(raw) < 32:
        _log("ERROR", f"'{path.name}' is too small to be a video.")
        sys.exit(1)
    # MP4/MOV/M4V usually contain 'ftyp' near the start; WebM starts with EBML (0x1A45DFA3)
    head = raw[:64]
    if path.suffix.lower() in {".mp4", ".mov", ".m4v"} and b"ftyp" not in head:
        _log(
            "ERROR",
            f"'{path.name}' does not look like an MP4/MOV container (missing 'ftyp').",
        )
        sys.exit(1)
    if path.suffix.lower() == ".webm" and not head.startswith(b"\x1a\x45\xdf\xa3"):
        _log("ERROR", f"'{path.name}' does not look like a WebM container.")
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
        entered = input(f"Device the seed was displayed on ({'/'.join(DEVICE_TYPES)}): ").strip().lower()
        if entered in DEVICE_TYPES:
            return entered
        print(f"  Please enter one of: {', '.join(DEVICE_TYPES)}")


def prompt_bool_if_missing(flag_value: bool, prompt: str) -> bool:
    if flag_value:
        return True
    entered = input(f"{prompt} [y/N]: ").strip().lower()
    return entered in ("y", "yes")


def prompt_capture_variant(value: str | None) -> str:
    if value and value in CAPTURE_VARIANTS:
        return value
    if value:
        _log("ERROR", f"Unknown --variant '{value}'. Choose one of: {', '.join(CAPTURE_VARIANTS)}")
        sys.exit(1)

    print("\nWhich capture variant is this submission?")
    for i, key in enumerate(CAPTURE_VARIANTS, 1):
        print(f"  {i}. {key}")
        print(f"     {VARIANT_HELP[key]}")
    while True:
        entered = input(f"Enter a number (1-{len(CAPTURE_VARIANTS)}) or the variant name: ").strip()
        if entered.isdigit() and 1 <= int(entered) <= len(CAPTURE_VARIANTS):
            return CAPTURE_VARIANTS[int(entered) - 1]
        if entered in CAPTURE_VARIANTS:
            return entered
        print(f"  Please enter 1-{len(CAPTURE_VARIANTS)}, or one of: {', '.join(CAPTURE_VARIANTS)}")


def list_pool_images() -> list[str]:
    """List filenames in the shared fixed_image/ pool (sandbox seed images)."""
    if not FIXED_IMAGE_POOL_DIR.exists():
        return []
    return sorted(
        p.name for p in FIXED_IMAGE_POOL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def prompt_seed_image(value: str | None) -> str:
    """Get which fixed_image/ pool file the miner used as their screen-replay seed."""
    pool = list_pool_images()

    if value:
        if not pool or value in pool:
            return value
        print(f"  Warning: '{value}' isn't in the known pool ({FIXED_IMAGE_POOL_DIR}); using it anyway.")
        return value

    if not pool:
        entered = input(
            "Filename of the seed image you displayed and photographed "
            f"(couldn't list {FIXED_IMAGE_POOL_DIR}, type it manually): "
        ).strip()
        return entered

    print(f"\nWhich fixed_image/ pool image did you randomly pick (base seed)?")
    for i, name in enumerate(pool, 1):
        print(f"  {i}. {name}")
    while True:
        entered = input(f"Enter a number (1-{len(pool)}) or the filename: ").strip()
        if entered.isdigit() and 1 <= int(entered) <= len(pool):
            return pool[int(entered) - 1]
        if entered in pool:
            return entered
        print(f"  Please enter a number 1-{len(pool)}, or one of: {', '.join(pool)}")


def queue_existing_pending_capture() -> None:
    """Back up a still-pending capture into queue/ instead of overwriting it."""
    if not SCREEN_REPLAY_JSON.exists():
        return
    try:
        with open(SCREEN_REPLAY_JSON, "r") as f:
            existing = json.load(f)
    except Exception:
        return
    if not bool(existing.get("ready", False)):
        return  # nothing pending — safe to overwrite

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    queued_name = f"queued_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    shutil.copy(str(SCREEN_REPLAY_JSON), str(QUEUE_DIR / queued_name))
    _log(
        "OK",
        f"A previous capture was still waiting to be sent — queued it as "
        f"queue/{queued_name}. Your miner will submit it automatically right "
        f"after this new one. This new capture takes the active slot now.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=CAPTURE_VARIANTS,
        help="Capture variety track for this submission",
    )
    parser.add_argument(
        "--seed-image",
        help="Filename of the fixed_image/ pool image you randomly picked and photographed",
    )
    parser.add_argument(
        "--face",
        help="Inbox filename of the FACE CLOSE-UP (photo or video)",
    )
    parser.add_argument(
        "--env",
        help="Inbox filename of the ENVIRONMENT still photo",
    )
    parser.add_argument("--camera", help="Camera/phone used to take the photo, e.g. 'iPhone 15 Pro'")
    parser.add_argument("--device", choices=DEVICE_TYPES, help="Device the seed image was displayed on")
    parser.add_argument("--date", help="Capture date YYYY-MM-DD (UTC). Defaults to today.")
    parser.add_argument("--moire", action="store_true", help="Moiré / pixel grid interference visible")
    parser.add_argument("--glare", action="store_true", help="Screen glare hotspots visible")
    parser.add_argument("--keystone", action="store_true", help="Perspective / keystone distortion visible")
    parser.add_argument("--gamma", action="store_true", help="Gamma / contrast shift visible")
    parser.add_argument("--edge-crop", action="store_true", dest="edge_crop", help="Screen edge/bezel/crop cues visible")
    args = parser.parse_args()

    capture_variant = prompt_capture_variant(args.variant)
    _log("OK", f"Capture variant: {capture_variant} — {VARIANT_HELP[capture_variant]}")

    primary_is_video = capture_variant in VIDEO_VARIANTS
    media_a, media_b = find_inbox_pair(capture_variant)
    photo_path, photo_path_2 = assign_photo_roles(
        media_a,
        media_b,
        face_name=args.face,
        env_name=args.env,
        primary_is_video=primary_is_video,
    )

    if primary_is_video:
        validate_video(photo_path)
        validate_image(photo_path_2)
    else:
        validate_image(photo_path)
        validate_image(photo_path_2)

    photo_bytes = photo_path.read_bytes()
    photo_bytes_2 = photo_path_2.read_bytes()

    if hashlib.sha256(photo_bytes).hexdigest() == hashlib.sha256(photo_bytes_2).hexdigest():
        _log("ERROR", f"'{photo_path.name}' and '{photo_path_2.name}' are byte-for-byte identical.")
        _log(
            "ERROR",
            "Need TWO DIFFERENT files: a face close-up AND an environment shot, "
            "not the same file twice.",
        )
        sys.exit(1)

    primary_label = "Face close-up VIDEO (primary)" if primary_is_video else "Face close-up (photo 1)"
    _log(
        "OK",
        f"{primary_label}: {photo_path.name} ({len(photo_bytes)} bytes)",
    )
    _log(
        "OK",
        f"Environment (photo 2): {photo_path_2.name} ({len(photo_bytes_2)} bytes) "
        "— whole screen/device in surroundings",
    )

    seed_image = prompt_seed_image(args.seed_image)
    _log("OK", f"Seed image: {seed_image}")
    camera_used = prompt_if_missing(args.camera, "Camera/phone used to take the capture (e.g. 'iPhone 15 Pro')")
    device_photographed = prompt_device(args.device)
    date = args.date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

    print(
        "\nWhich visual cues are clearly visible across your capture? "
        "(report honestly — it's fine if none apply; keystone/glare/moire "
        "are more common on the environment shot)"
    )
    moire = prompt_bool_if_missing(args.moire, "  Moiré / pixel grid interference")
    glare = prompt_bool_if_missing(args.glare, "  Screen glare hotspots")
    keystone = prompt_bool_if_missing(args.keystone, "  Perspective / keystone distortion")
    gamma = prompt_bool_if_missing(args.gamma, "  Gamma / contrast shift")
    edge_crop = prompt_bool_if_missing(args.edge_crop, "  Screen edge / bezel / crop cues")

    # Move (not copy) both files out of inbox/ into staged/
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    primary_tag = "face_video" if primary_is_video else "face"
    staged_name = f"{date}_{primary_tag}_{photo_path.stem}{photo_path.suffix}"
    staged_name_2 = f"{date}_env_{photo_path_2.stem}{photo_path_2.suffix}"
    staged_path = STAGED_DIR / staged_name
    staged_path_2 = STAGED_DIR / staged_name_2
    shutil.move(str(photo_path), str(staged_path))
    shutil.move(str(photo_path_2), str(staged_path_2))
    _log("OK", f"Moved media to {staged_path} and {staged_path_2}")

    data = {
        "ready": True,
        "photo_path": str(staged_path.resolve()),
        "photo_path_2": str(staged_path_2.resolve()),
        "seed_image": seed_image,
        "date": date,
        "camera_used": camera_used,
        "device_photographed": device_photographed,
        "capture_variant": capture_variant,
        "primary_media": "video" if primary_is_video else "photo",
        "moire_pixel_grid": moire,
        "screen_glare_hotspots": glare,
        "perspective_keystone_distortion": keystone,
        "gamma_contrast_shift": gamma,
        "edge_crop_cues": edge_crop,
    }
    queue_existing_pending_capture()

    with open(SCREEN_REPLAY_JSON, "w") as f:
        json.dump(data, f, indent=2)

    _log("DONE", f"Wrote {SCREEN_REPLAY_JSON} with ready=true.")
    _log(
        "DONE",
        "Nothing else to do — your miner process will pick this up and submit both "
        "files on its next validator query, then flip ready back to false automatically.",
    )
    _log(
        "DONE",
        "Want to submit again? Drop a NEW pair (face close-up + environment) in "
        "inbox/, pick a capture_variant, and re-run. Never resubmit the same "
        "capture twice. Queued pending captures are preserved automatically.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
