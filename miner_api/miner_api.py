import base64
import requests
from datetime import datetime
from pathlib import Path

import bittensor

# Edit these variables directly in the file
wallet_name = "miner"  # Your wallet name
wallet_hotkey = "m"  # Your hotkey name
api_base_url = "http://98.90.28.118:5001"  # API base URL (no endpoint path)
image_dir = Path(__file__).parent / "image"  # Folder containing the image to send

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Available endpoints on the API
ENDPOINTS = {
    "1": "is_live",  # Screen replay / liveness check
    "2": "is_ai",    # Face variations / AI-generated face detection
}


def choose_endpoint() -> str:
    """Prompt the user to pick which endpoint to call."""
    while True:
        print("\nWhich check do you want to run?")
        print("  1) is_live  - Screen replay / liveness check")
        print("  2) is_ai    - Face variations / AI-generated face detection")
        choice = input("Enter 1 or 2 (or 'is_live'/'is_ai'): ").strip().lower()

        if choice in ENDPOINTS:
            return ENDPOINTS[choice]
        if choice in ENDPOINTS.values():
            return choice
        print("Invalid selection. Please try again.")


def get_image_path(directory: Path) -> Path:
    """Return the first image file found in the given directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    images = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No image found in {directory}. "
            f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return images[0]


def encode_image_b64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded contents."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sign_message(wallet: bittensor.Wallet, message_text: str = "API Request") -> str:
    """
    Signs a message using the specified wallet.

    Args:
        wallet (bittensor.Wallet): The wallet object to use for signing.
        message_text (str): The message you want to sign.

    Returns:
        str: The combined signature contents (message + signature info).
    """
    # Use the wallet's hotkey for signing
    keypair = wallet.hotkey

    # Generate a timestamped message
    timestamp = datetime.now()
    timezone_name = timestamp.astimezone().tzname()
    signed_message = f"<Bytes>On {timestamp} {timezone_name} {message_text}</Bytes>"

    # Sign the message
    signature = keypair.sign(data=signed_message)

    # Construct the output in the expected format
    signature_text = (
        f"{signed_message}\n"
        f"\tSigned by: {keypair.ss58_address}\n"
        f"\tSignature: {signature.hex()}"
    )

    return signature_text


def main():
    endpoint = choose_endpoint()
    api_url = f"{api_base_url}/{endpoint}"

    image_path = get_image_path(image_dir)
    print(f"Sending image: {image_path.name} -> {api_url}")

    wallet = bittensor.Wallet(name=wallet_name, hotkey=wallet_hotkey)
    
    # Sign message
    signature = sign_message(wallet)
    
    # Prepare payload
    payload = {
        "image": encode_image_b64(image_path),
        "signature": signature,
    }

    # Send request
    response = requests.post(api_url, json=payload)
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    main()


