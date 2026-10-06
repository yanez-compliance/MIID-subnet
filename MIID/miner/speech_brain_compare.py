# MIID/miner/speech_brain_compare.py
#
# Optional speaker-identity check for miners who implement voice generation.
#
# Same role as AdaFace for faces: if you generate a voice clone, you can gate
# it here before encrypt/upload. Default miner path does not generate voice,
# so this is unused until you opt in.
#
# Suggested model when implementing:
#   speechbrain/spkrec-ecapa-voxceleb
#   pip install speechbrain torchaudio
#
# Default thresholds (once wired):
#   module default 0.6; miner call site often uses 0.4 (same pattern as AdaFace).

from typing import Optional

import bittensor as bt


SPEECHBRAIN_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MIN_SIMILARITY = 0.6


def validate_voice_identity(
    base_wav: bytes,
    generated_wav: bytes,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    model: Optional[object] = None,
) -> bool:
    """Check that generated audio preserves the reference speaker identity.

    Default: returns True (no model loaded). Wire SpeechBrain ECAPA when you
    want a local pre-check before upload; UAV post-grading is the real score.

    Args:
        base_wav: Reference speaker WAV bytes.
        generated_wav: Miner-generated WAV bytes.
        min_similarity: Minimum cosine similarity (default 0.6).
        model: Optional preloaded encoder.

    Returns:
        True if identity is considered preserved.
    """
    if not base_wav or not generated_wav:
        bt.logging.warning("Voice identity: empty wav bytes")
        return False
    # No model downloaded by default (same idea as AdaFace being optional until set up).
    bt.logging.debug(
        f"Voice identity: no SpeechBrain model loaded; accepting submission "
        f"(planned model={SPEECHBRAIN_MODEL_ID}, threshold={min_similarity})"
    )
    return True
