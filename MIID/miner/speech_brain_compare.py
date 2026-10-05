# MIID/miner/speech_brain_compare.py
#
# Speaker identity comparison stub using SpeechBrain (planned).
#
# Intended model (when wired):
#   speechbrain/spkrec-ecapa-voxceleb
#   — ECAPA-TDNN speaker embeddings; cosine similarity between reference
#     and generated WAVs, analogous to AdaFace for faces.
#
# Prerequisites (when implementing for real):
#   pip install speechbrain torchaudio
#   # model downloads on first use from HuggingFace
#
# For now this module always returns True so the miner encrypt/upload path
# can run end-to-end. UAV post-grading will judge identity later.

from typing import Optional

import bittensor as bt


# Planned SpeechBrain model id (document only until stub is filled)
SPEECHBRAIN_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

# Default cosine-similarity threshold once real embeddings are wired
# (AdaFace uses 0.7; voice ECAPA scores sit a bit lower so default is 0.6)
DEFAULT_MIN_SIMILARITY = 0.6


def validate_voice_identity(
    base_wav: bytes,
    generated_wav: bytes,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    model: Optional[object] = None,
) -> bool:
    """Check that generated audio preserves the reference speaker identity.

    STUB: always returns True. Replace with SpeechBrain ECAPA embeddings
    and cosine similarity >= ``min_similarity``.

    Args:
        base_wav: Reference speaker WAV bytes.
        generated_wav: Miner-generated WAV bytes.
        min_similarity: Minimum cosine similarity (default 0.6; unused in stub).
        model: Optional preloaded encoder (unused in stub).

    Returns:
        True if identity is considered preserved.
    """
    bt.logging.warning(
        "SpeechBrain stub: validate_voice_identity always returns True. "
        f"TODO: load {SPEECHBRAIN_MODEL_ID} and compare embeddings "
        f"(threshold={min_similarity})."
    )
    if not base_wav or not generated_wav:
        bt.logging.warning("SpeechBrain stub: empty wav bytes; still returning True")
    return True
