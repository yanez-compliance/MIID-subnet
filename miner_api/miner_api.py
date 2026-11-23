"""Check if one picture breaks a detector.

Put the picture in the variation/ folder, then run:

    python miner_api.py

It asks two things: which detector, and which picture. No seed. No prompt.
If variation/ has one picture, that picture is used.

Skip the questions with flags. A file name is enough:

    python miner_api.py --face gamma
    python miner_api.py --screen gamma

  --face     deepfake detection
  --screen   passive liveness

For the exact grade (seed image + the challenge prompt), run exact_grade.py.
"""

import argparse
import base64
from datetime import datetime
from pathlib import Path

import bittensor
import requests

# Edit these variables directly in the file
wallet_name = "miner"
wallet_hotkey = "m"
api_base_url = "https://api.yanez.ai:5000"
variation_dir = Path(__file__).parent / "variation"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(directory: Path) -> list:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ask(prompt: str) -> str:
    return input(prompt).strip().strip('"').strip("'")


def resolve_image(value: str, folder: Path) -> Path:
    """Find a picture by path, by file name, or by the name without the extension."""
    text = (value or "").strip().strip('"').strip("'")
    if not text:
        raise FileNotFoundError("No image given.")
    candidates = [
        Path(text),
        Path(__file__).parent / text,
        folder / text,
        folder / Path(text).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    stem = Path(text).stem.lower()
    matches = [
        path for path in list_images(folder)
        if path.stem.lower() == stem or path.name.lower() == text.lower()
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"Could not find {text} in {folder}")


def choose_detector() -> str:
    while True:
        print("\nWhich detector do you want to check?")
        print("  1) face    deepfake detection")
        print("  2) screen  passive liveness")
        choice = ask("Enter 1 or 2: ").lower().replace(" ", "_").replace("-", "_")
        if choice in {"1", "face", "face_grade"}:
            return "face"
        if choice in {"2", "screen", "screen_replay", "screen_replay_grade"}:
            return "screen"
        print("Type 1 for deepfake, or 2 for passive liveness.")


def choose_image() -> Path:
    images = list_images(variation_dir)
    if len(images) == 1:
        print(f"\nUsing the picture in variation/: {images[0].name}")
        return images[0]
    while True:
        print("\nWhich picture do you want to check?")
        print("Put it in the variation/ folder, then pick it here.")
        if images:
            for index, path in enumerate(images, start=1):
                print(f"  {index}) {path.name}")
            print("Enter a number, a file name like gamma, or a full path.")
        else:
            print("variation/ is empty. Paste a full path, or add a picture and run again.")
        choice = ask("> ")
        if choice.isdigit() and images:
            number = int(choice)
            if 1 <= number <= len(images):
                return images[number - 1]
            print("That number is not in the list.")
            continue
        try:
            return resolve_image(choice, variation_dir)
        except FileNotFoundError:
            print(f"Could not find that file in variation/.")


def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def sign_message(wallet: bittensor.Wallet) -> str:
    timestamp = datetime.now()
    signed_message = (
        f"<Bytes>On {timestamp} {timestamp.astimezone().tzname()} API Request</Bytes>"
    )
    signature = wallet.hotkey.sign(data=signed_message)
    return (
        f"{signed_message}\n"
        f"\tSigned by: {wallet.hotkey.ss58_address}\n"
        f"\tSignature: {signature.hex()}"
    )


def print_result(body: dict, detector: str) -> None:
    name = "deepfake detection" if detector == "face" else "passive liveness"
    broke = body.get("breaks")
    if broke is True:
        broke_text = f"yes — it broke {name}"
    elif broke is False:
        broke_text = f"no — it did not break {name}"
    else:
        broke_text = "the response did not say"
    print("\nResult")
    print(f"  Broke:   {broke_text}")
    print(f"  Score:   {body.get('score', 'missing')}  (from -5 to 5)")
    print(f"  Comment: {body.get('comment', '')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check if one picture breaks deepfake detection or passive liveness.",
    )
    parser.add_argument("image_name", nargs="?", help="Picture in variation/, for example gamma.")
    parser.add_argument("--face", action="store_true", help="Check deepfake detection.")
    parser.add_argument("--screen", action="store_true", help="Check passive liveness.")
    parser.add_argument("--image", help="Same as the picture name. A name in variation/ is enough.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.face and args.screen:
        raise SystemExit("Pick one detector: --face or --screen.")

    print("Break check. This sends the picture in variation/ only.")
    print("For the exact grade, with the seed and the prompt, run exact_grade.py.")

    if args.face:
        detector = "face"
    elif args.screen:
        detector = "screen"
    else:
        detector = choose_detector()

    image_text = args.image or args.image_name
    if image_text:
        try:
            image_path = resolve_image(image_text, variation_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        image_path = choose_image()

    endpoint = "face_grade" if detector == "face" else "screen_replay_grade"
    api_url = f"{api_base_url.rstrip('/')}/{endpoint}"
    print(f"\nChecking {image_path.name}")
    print(f"  Call: {api_url}")

    wallet = bittensor.Wallet(name=wallet_name, hotkey=wallet_hotkey)
    response = requests.post(
        api_url,
        json={
            "image": encode_image_b64(image_path),
            "signature": sign_message(wallet),
        },
    )
    try:
        body = response.json()
    except ValueError:
        print(f"\nThe server did not return JSON ({response.status_code}).")
        print(response.text)
        return
    if response.status_code != 200:
        print(f"\nRequest failed ({response.status_code}).")
        print(body)
        return
    print_result(body, detector)


if __name__ == "__main__":
    main()
