# MIID/miner/voice_generator.py
#
# Optional voice-clone hook for miners.
#
# Default behavior: return None (no voice submission). Image mining still works.
# To opt in, replace generate_voice_clone() with your TTS / voice-clone model:
#   - Input: reference WAV (base_wav_bytes) + prompt text (target_text / target_words)
#   - Output: WAV bytes that keep the same speaker identity and speak the prompt
#
# Prompt text is already joined for you as target_text, e.g.
#   "three orange six two nine apple table one"
# (same as " ".join(target_words)).

import base64
import hashlib
from typing import List, Optional

import bittensor as bt


def decode_base_voice(base64_voice: str) -> bytes:
    """Decode a base64-encoded WAV into raw bytes."""
    return base64.b64decode(base64_voice)


def generate_voice_clone(
    base_wav_bytes: bytes,
    target_words: List[str],
    language: str,
    target_text: str = "",
) -> Optional[bytes]:
    """Generate a voice-cloned WAV speaking the target prompt.

    Default: returns None so the miner sends no voice submission.
    Implement your own model here when you want to participate in voice.

    Args:
        base_wav_bytes: Reference speaker WAV bytes from the validator.
        target_words: Token list the generated audio should contain.
        language: ``"en"`` or ``"es"``.
        target_text: Ready-to-speak prompt string (preferred for TTS).
            Falls back to ``" ".join(target_words)`` if empty.

    Returns:
        Generated WAV bytes, or None to skip voice (default).
    """
    prompt = (target_text or " ".join(target_words or [])).strip()
    bt.logging.info(
        "Voice: generate_voice_clone not implemented — skipping voice submission "
        f"(lang={language}, prompt={prompt!r}). "
        "Implement this function to opt in."
    )
    return None


def hash_voice_bytes(wav_bytes: bytes) -> str:
    """SHA256 hex digest of WAV bytes (same role as image_hash)."""
    return hashlib.sha256(wav_bytes).hexdigest()
