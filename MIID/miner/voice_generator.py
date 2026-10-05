# MIID/miner/voice_generator.py
#
# Voice clone generation stub for miners.
#
# TODO: Plug in your TTS / voice-clone model here. The validator sends a
# reference WAV (~30s, English or Spanish) plus target_words; you must return
# a WAV that preserves speaker identity and speaks those words.
#
# Suggested approach (miner-chosen):
#   - XTTS / Tortoise / YourTTS / OpenVoice / CosyVoice / etc.
#   - Condition on base_wav_bytes as the speaker reference
#   - Synthesize target_words joined into a short phrase in `language`

import base64
import hashlib
import struct
from typing import List, Optional

import bittensor as bt


def decode_base_voice(base64_voice: str) -> bytes:
    """Decode a base64-encoded WAV into raw bytes."""
    return base64.b64decode(base64_voice)


def _make_silent_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Minimal silent PCM WAV (16-bit mono) used by the blank stub."""
    n_samples = int(duration_sec * sample_rate)
    data_size = n_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + (b"\x00\x00" * n_samples)


def generate_voice_clone(
    base_wav_bytes: bytes,
    target_words: List[str],
    language: str,
) -> Optional[bytes]:
    """Generate a voice-cloned WAV speaking ``target_words``.

    STUB: returns a short silent WAV (or a copy of the reference if present)
    so the encrypt/upload path can run. Replace this body with a real model.

    Args:
        base_wav_bytes: Reference speaker WAV bytes from the validator.
        target_words: Words the generated audio must contain.
        language: ``"en"`` or ``"es"``.

    Returns:
        Generated WAV bytes, or None on failure.
    """
    words_preview = " ".join(target_words[:8])
    bt.logging.warning(
        "Voice stub: generate_voice_clone is not wired to a real model. "
        f"language={language}, words=[{words_preview}]. "
        "TODO: plug in your TTS/voice-clone model."
    )

    # Prefer returning the reference clip so downstream identity stubs have
    # real audio to hash; fall back to silence if the reference is empty.
    if base_wav_bytes and len(base_wav_bytes) > 44:
        return base_wav_bytes

    return _make_silent_wav_bytes(duration_sec=1.0)


def hash_voice_bytes(wav_bytes: bytes) -> str:
    """SHA256 hex digest of WAV bytes (same role as image_hash)."""
    return hashlib.sha256(wav_bytes).hexdigest()
