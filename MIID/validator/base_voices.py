# MIID/validator/base_voices.py
#
# Voice challenge: fetch a reference WAV from the MIID API for miners to clone.

import os
import time
import bittensor as bt
from typing import Optional, Tuple

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def fetch_voice_from_api(wallet) -> Optional[Tuple[str, str, str, str]]:
    """Fetch a single reference voice clip from the Flask API.

    Uses a signed POST to ``/voice/<hotkey>`` and returns one WAV
    in memory for the current forward pass.

    Args:
        wallet: Bittensor wallet used to sign the request.

    Returns:
        Tuple of (filename, base64_encoded_wav, language, transcript)
        or None on failure. language is ``"en"`` or ``"es"``.
    """
    if not REQUESTS_AVAILABLE:
        bt.logging.warning("requests library not available. Cannot fetch voice from API.")
        return None

    try:
        from MIID.utils.sign_message import sign_message

        hotkey = wallet.hotkey
        hotkey_address = hotkey.ss58_address

        message_to_sign = (
            f"Hotkey: {hotkey} \n timestamp: {time.time()} \n request: base_voices"
        )
        signed_contents = sign_message(wallet, message_to_sign, output_file=None)

        server_url = os.environ.get("MIID_IMAGES_SERVER", "http://52.44.186.20:5000")
        base_url = server_url.rstrip("/")
        url = f"{base_url}/voice/{hotkey_address}"
        payload = {"signature": signed_contents}

        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            bt.logging.warning(
                f"Voice API returned {response.status_code}: {response.text[:200]}"
            )
            return None

        data = response.json()
        item = data.get("voice") or {}

        filename = item.get("filename")
        b64 = item.get("data_base64")
        language = (item.get("language") or "en").lower()
        transcript = item.get("transcript") or ""

        if not filename or not b64:
            bt.logging.warning("API voice entry missing filename or data_base64")
            return None

        if language not in ("en", "es"):
            bt.logging.warning(f"Unexpected voice language '{language}'; defaulting to 'en'")
            language = "en"

        bt.logging.debug(f"Fetched voice from API: {filename} lang={language}")
        return filename, b64, language, transcript

    except Exception as e:
        bt.logging.error(f"Error fetching voice from API: {e}")
        return None
