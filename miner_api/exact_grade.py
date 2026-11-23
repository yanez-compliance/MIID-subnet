"""Get the exact grade for one face variation or one screen replay.

Put the variation in variation/ and the seed in seed/, then run:

    python exact_grade.py

The terminal asks for anything still missing: face or screen, the picture,
the seed, and the prompt. One file in a folder is used automatically.
Fill in case.txt to skip the questions, or pass flags:

    python exact_grade.py --face gamma --seed original --type 2 --intensity medium
    python exact_grade.py --screen replay --seed original --cue1 --cue2

  --face / --screen     which grade
  gamma, replay         picture in variation/ (the name is enough)
  --seed original       seed in seed/
  --type 1-5            face prompt: which variation
  --intensity light     face prompt: light, medium, or far (or 1, 2, 3)
  --cue1 ... --cue5     screen prompt: the cues the challenge asked for

Face prompt: variation type and intensity.
Screen prompt: the cues.

To only check if a picture breaks a detector, run miner_api.py.
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
seed_dir = Path(__file__).parent / "seed"
case_path = Path(__file__).parent / "case.txt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

FACE_TYPES = [
    ("background_in", "indoor background"),
    ("background_out", "outdoor background"),
    ("lighting_edit+expression_edit", "lighting and expression"),
    ("lighting_edit+pose_edit", "lighting and pose"),
    ("pose_edit+expression_edit", "pose and expression"),
]
INTENSITIES = [
    ("light", "small change"),
    ("medium", "clear change"),
    ("far", "strong change"),
]
CUES = [
    ("moire_pixel_grid", "moire / pixel grid"),
    ("screen_glare_hotspots", "screen glare"),
    ("perspective_keystone_distortion", "perspective / keystone"),
    ("gamma_contrast_shift", "gamma / contrast shift"),
    ("edge_crop_cues", "edge, crop, or bezel"),
]

FACE_TYPE_ALIASES = {
    "indoor": "background_in",
    "outdoor": "background_out",
    "background in": "background_in",
    "background out": "background_out",
    "lighting+expression": "lighting_edit+expression_edit",
    "lighting and expression": "lighting_edit+expression_edit",
    "lighting+pose": "lighting_edit+pose_edit",
    "lighting and pose": "lighting_edit+pose_edit",
    "pose+expression": "pose_edit+expression_edit",
    "pose and expression": "pose_edit+expression_edit",
}
for index, (key, label) in enumerate(FACE_TYPES, start=1):
    FACE_TYPE_ALIASES[str(index)] = key
    FACE_TYPE_ALIASES[key] = key
    FACE_TYPE_ALIASES[label] = key

INTENSITY_ALIASES = {}
for index, (key, label) in enumerate(INTENSITIES, start=1):
    INTENSITY_ALIASES[str(index)] = key
    INTENSITY_ALIASES[key] = key
    INTENSITY_ALIASES[label] = key

CUE_ALIASES = {
    "moire": "moire_pixel_grid",
    "glare": "screen_glare_hotspots",
    "perspective": "perspective_keystone_distortion",
    "keystone": "perspective_keystone_distortion",
    "gamma": "gamma_contrast_shift",
    "contrast": "gamma_contrast_shift",
    "edge": "edge_crop_cues",
    "bezel": "edge_crop_cues",
}
for index, (key, label) in enumerate(CUES, start=1):
    CUE_ALIASES[str(index)] = key
    CUE_ALIASES[key] = key
    CUE_ALIASES[label] = key


def load_case(path: Path) -> dict:
    """Read the fill-in file. Blank values stay empty strings."""
    values = {
        "grade": "",
        "image": "",
        "seed": "",
        "variation_type": "",
        "intensity": "",
        "cues": "",
    }
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            values[key] = value.strip().strip('"').strip("'")
    return values


def save_case(path: Path, values: dict) -> None:
    """Write the answers back so the next run can just be started."""
    text = f"""# Exact grade. Run:  python exact_grade.py
#
# Needs the picture, the seed, and the prompt.
# Face prompt: variation type and intensity.
# Screen prompt: the cues.
# To only check if a picture breaks a detector, run miner_api.py.

# face    or    screen
grade = {values['grade']}

# The picture you made.
image = {values['image']}

# The original seed face.
seed = {values['seed']}

# Face prompt. Leave blank for a screen replay.
# 1 background_in  2 background_out
# 3 lighting_edit+expression_edit  4 lighting_edit+pose_edit
# 5 pose_edit+expression_edit
variation_type = {values['variation_type']}

# Face prompt. light, medium, or far
intensity = {values['intensity']}

# Screen prompt. Leave blank for a face.
# 1 moire_pixel_grid  2 screen_glare_hotspots
# 3 perspective_keystone_distortion  4 gamma_contrast_shift
# 5 edge_crop_cues
cues = {values['cues']}
"""
    path.write_text(text, encoding="utf-8")


def normalize_grade(value: str) -> str:
    raw = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if raw in {"1", "face", "face_grade"}:
        return "face"
    if raw in {"2", "screen", "screen_replay", "screen_replay_grade"}:
        return "screen"
    return ""


def list_images(directory: Path) -> list:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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
    raise FileNotFoundError(f"Could not find {text} in {folder.name}/")


def ask(prompt: str) -> str:
    return input(prompt).strip().strip('"').strip("'")


def choose_grade() -> str:
    while True:
        print("\nWhat are you grading?")
        print("  1) face    a face variation. Prompt: variation type and intensity.")
        print("  2) screen  a screen replay. Prompt: the cues.")
        grade = normalize_grade(ask("Enter 1 or 2: "))
        if grade:
            return grade
        print("Type 1 for a face variation, or 2 for a screen replay.")


def only_picture(folder: Path, value: str, label: str) -> str:
    """Use the named file, or the only file in the folder. Otherwise leave it blank."""
    if (value or "").strip():
        try:
            return str(resolve_image(value, folder))
        except FileNotFoundError:
            print(f"Could not find {label}: {value}")
            return ""
    images = list_images(folder)
    if len(images) == 1:
        print(f"Using the only picture in {folder.name}/: {images[0].name}")
        return str(images[0])
    return ""


def choose_image(prompt: str, folder: Path) -> str:
    images = list_images(folder)
    while True:
        print(f"\n{prompt}")
        print(f"Put it in the {folder.name}/ folder, then pick it here.")
        if images:
            for index, path in enumerate(images, start=1):
                print(f"  {index}) {path.name}")
            print("Enter a number, a file name, or a full path.")
        else:
            print(f"{folder.name}/ is empty. Paste a full path, or add a picture and run again.")
        choice = ask("> ")
        if choice.isdigit() and images:
            number = int(choice)
            if 1 <= number <= len(images):
                return str(images[number - 1])
            print("That number is not in the list.")
            continue
        try:
            return str(resolve_image(choice, folder))
        except FileNotFoundError:
            print(f"Could not find that file in {folder.name}/.")


def choose_face_type() -> str:
    while True:
        print("\nFace prompt: what variation did the validator ask for?")
        for index, (key, label) in enumerate(FACE_TYPES, start=1):
            print(f"  {index}) {label}")
        choice = ask("Enter 1-5: ").lower()
        if choice in FACE_TYPE_ALIASES:
            return FACE_TYPE_ALIASES[choice]
        print("Pick a number from 1 to 5.")


def choose_intensity() -> str:
    while True:
        print("\nFace prompt: how strong was the edit?")
        for index, (key, label) in enumerate(INTENSITIES, start=1):
            print(f"  {index}) {key}  — {label}")
        choice = ask("Enter 1, 2, or 3: ").lower()
        if choice in INTENSITY_ALIASES:
            return INTENSITY_ALIASES[choice]
        print("Type light, medium, or far. Or 1, 2, or 3.")


def choose_cues() -> str:
    while True:
        print("\nScreen prompt: which cues was this replay supposed to show?")
        print("Pick every cue the challenge named, usually two.")
        for index, (_key, label) in enumerate(CUES, start=1):
            print(f"  {index}) {label}")
        raw = ask("Enter the numbers, separated by commas: ")
        keys = parse_cues(raw)
        if keys:
            return ",".join(keys)
        print("Pick at least one cue. Example: 1, 2")


def parse_cues(raw: str) -> list:
    keys = []
    for part in (raw or "").replace(";", ",").split(","):
        token = part.strip().lower().replace(" ", "_").replace("-", "_")
        if not token:
            continue
        key = CUE_ALIASES.get(token) or CUE_ALIASES.get(part.strip().lower())
        if key is None or key in keys:
            continue
        keys.append(key)
    return keys


def lookup_face_type(value: str) -> str:
    raw = " ".join((value or "").strip().lower().replace(" + ", "+").split())
    return FACE_TYPE_ALIASES.get(raw, "")


def complete_case(values: dict) -> dict:
    """Use whatever is already filled in. Ask for the picture, the seed, and the prompt."""
    filled = dict(values)
    asked = False

    grade = normalize_grade(filled["grade"])
    if not grade:
        asked = True
        print("\nExact grade. case.txt does not say face or screen yet.")
        print("You can fill that file and run again, or answer here.")
        grade = choose_grade()
    filled["grade"] = grade

    image = only_picture(variation_dir, filled["image"], "the variation")
    if not image:
        asked = True
        image = choose_image("Which file is the variation you made?", variation_dir)
    filled["image"] = image

    seed = only_picture(seed_dir, filled["seed"], "the seed")
    if not seed:
        asked = True
        seed = choose_image("Which file is the seed? That is the original face, before your edit.", seed_dir)
    filled["seed"] = seed

    if Path(filled["image"]).resolve() == Path(filled["seed"]).resolve():
        print("\nThe graded picture and the seed are the same file.")
        confirm = ask("Use them anyway? y/n: ").lower()
        if confirm not in {"y", "yes"}:
            filled["image"] = choose_image("Which file is the variation you made?", variation_dir)
            filled["seed"] = choose_image("Which file is the seed?", seed_dir)
            asked = True

    if grade == "face":
        variation = lookup_face_type(filled["variation_type"])
        if not variation:
            asked = True
            variation = choose_face_type()
        intensity = INTENSITY_ALIASES.get(filled["intensity"].strip().lower(), "")
        if not intensity:
            asked = True
            intensity = choose_intensity()
        filled["variation_type"] = variation
        filled["intensity"] = intensity
        filled["cues"] = ""
    else:
        cues = parse_cues(filled["cues"])
        if not cues:
            asked = True
            filled["cues"] = choose_cues()
        else:
            filled["cues"] = ",".join(cues)
        filled["variation_type"] = ""
        filled["intensity"] = ""

    if asked:
        save_case(case_path, filled)
        print(f"\nSaved your answers to {case_path.name}. Next time you can just run this again.")
    return filled


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


def print_result(body: dict, grade: str) -> None:
    detector = "deepfake detection" if grade == "face" else "passive liveness"
    broke = body.get("breaks")
    if broke is True:
        broke_text = f"yes — it broke {detector}"
    elif broke is False:
        broke_text = f"no — it did not break {detector}"
    else:
        broke_text = "the response did not say"
    print("\nResult")
    print(f"  Broke:   {broke_text}")
    print(f"  Score:   {body.get('score', 'missing')}  (from -5 to 5)")
    print(f"  Comment: {body.get('comment', '')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact grade: variation, seed, and the challenge prompt.",
    )
    parser.add_argument("image_name", nargs="?", help="Variation in variation/, for example gamma.")
    parser.add_argument("--face", action="store_true", help="Grade a face variation.")
    parser.add_argument("--screen", action="store_true", help="Grade a screen replay.")
    parser.add_argument("--image", help="Same as the variation name.")
    parser.add_argument("--seed", help="Seed in seed/. The name is enough.")
    parser.add_argument("--type", dest="variation_type", help="Face prompt: 1-5 or a variation name.")
    parser.add_argument("--intensity", help="Face prompt: light, medium, far, or 1, 2, 3.")
    parser.add_argument("--cue", action="append", default=[], help="Screen prompt cue. Repeat for more than one.")
    parser.add_argument("--cue1", action="store_true", help="Moiré / pixel grid.")
    parser.add_argument("--cue2", action="store_true", help="Screen glare.")
    parser.add_argument("--cue3", action="store_true", help="Perspective / keystone.")
    parser.add_argument("--cue4", action="store_true", help="Gamma / contrast shift.")
    parser.add_argument("--cue5", action="store_true", help="Edge, crop, or bezel.")
    return parser.parse_args()


def apply_cli(values: dict, args) -> dict:
    """Flags win over case.txt. A flag also picks face or screen when that is obvious."""
    filled = dict(values)
    if args.face and args.screen:
        raise SystemExit("Pick one: --face or --screen.")
    if args.face:
        filled["grade"] = "face"
    elif args.screen:
        filled["grade"] = "screen"
    image = args.image or args.image_name
    if image:
        filled["image"] = image
    if args.seed:
        filled["seed"] = args.seed
    if args.variation_type:
        filled["variation_type"] = args.variation_type
    if args.intensity:
        filled["intensity"] = args.intensity
    cue_parts = list(args.cue or [])
    for number, flag in enumerate(("cue1", "cue2", "cue3", "cue4", "cue5"), start=1):
        if getattr(args, flag):
            cue_parts.append(str(number))
    if cue_parts:
        filled["cues"] = ",".join(cue_parts)
    if not normalize_grade(filled["grade"]):
        if cue_parts and not args.variation_type:
            filled["grade"] = "screen"
        elif args.variation_type or args.intensity:
            filled["grade"] = "face"
    return filled


def main():
    args = parse_args()
    print("Exact grade. variation/ is the picture you made. seed/ is the original face.")
    print("To only check if a picture breaks a detector, run miner_api.py.")
    values = complete_case(apply_cli(load_case(case_path), args))
    endpoint = "face_grade" if values["grade"] == "face" else "screen_replay_grade"
    api_url = f"{api_base_url.rstrip('/')}/{endpoint}"
    image_path = resolve_image(values["image"], variation_dir)
    seed_path = resolve_image(values["seed"], seed_dir)

    print(f"\nGrading {image_path.name}")
    print(f"  Seed: {seed_path.name}")
    if values["grade"] == "face":
        print(f"  Prompt: {values['variation_type']}, {values['intensity']}")
    else:
        print(f"  Prompt: {values['cues']}")
    print(f"  Call: {api_url}")

    wallet = bittensor.Wallet(name=wallet_name, hotkey=wallet_hotkey)
    payload = {
        "image": encode_image_b64(image_path),
        "seed_image": encode_image_b64(seed_path),
        "signature": sign_message(wallet),
    }
    if values["grade"] == "face":
        payload["variation_type"] = values["variation_type"]
        payload["intensity"] = values["intensity"]
    else:
        payload["cues"] = parse_cues(values["cues"])

    response = requests.post(api_url, json=payload)
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
    print_result(body, values["grade"])


if __name__ == "__main__":
    main()
