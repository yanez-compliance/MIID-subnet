# MIID/validator/voice_variations.py
#
# Phase 5 Cycle 2: five fixed voice variations. The emotion is not scaled.
# Each clip gets a random background-noise level, the same way a face slot
# gets a random intensity.

import random
from typing import Dict, List


VOICE_VARIATION_ORDER = (
    "neutral",
    "surprised",
    "excited",
    "angry",
    "whisper",
)

VOICE_VARIATION_TYPES = {
    "neutral": {
        "description": "Plain read of the challenge text in the reference speaker's voice",
        "detail": (
            "Speak the challenge text in a clear, conversational delivery. "
            "Keep the reference speaker. Do not add an emotion."
        ),
    },
    "surprised": {
        "description": "Same words and speaker, delivered with surprise",
        "detail": (
            "Speak the same challenge text as the same speaker, with a sudden pitch rise "
            "and a surprised onset. Do not change the words."
        ),
    },
    "excited": {
        "description": "Same words and speaker, delivered excited",
        "detail": (
            "Speak the same challenge text as the same speaker, faster and brighter, "
            "with the energy held up. This is not the same as surprise."
        ),
    },
    "angry": {
        "description": "Same words and speaker, delivered angry",
        "detail": (
            "Speak the same challenge text as the same speaker, louder and pressed, "
            "with harder consonants. This is not the same as excited."
        ),
    },
    "whisper": {
        "description": "Same words and speaker, whispered",
        "detail": (
            "Speak the same challenge text as the same speaker, breathy and quiet, "
            "with very little pitch. The words must still be intelligible."
        ),
    },
}

NOISE_LEVELS = {
    "none": {
        "label": "No background noise",
        "detail": "No background noise. The voice is clean, as in a quiet room.",
    },
    "light": {
        "label": "Light background noise",
        "detail": (
            "Light background noise under the voice, such as a quiet room tone or distant chatter. "
            "The words stay easy to hear."
        ),
    },
    "high": {
        "label": "High background noise",
        "detail": (
            "High background noise under the voice, such as a busy room, street, or wind. "
            "The words stay intelligible."
        ),
    },
}

ALL_NOISE_LEVELS = list(NOISE_LEVELS.keys())


def build_standard_voice_variations() -> List[Dict[str, str]]:
    """Five voice clips, always the same types, each with a random noise level.

    Returns:
        List of dicts with type, noise, description, and detail.
        Detail is the prompt the miner should follow: style plus noise.
    """
    variations = []
    for var_type in VOICE_VARIATION_ORDER:
        noise = random.choice(ALL_NOISE_LEVELS)
        type_info = VOICE_VARIATION_TYPES[var_type]
        noise_info = NOISE_LEVELS[noise]
        variations.append({
            "type": var_type,
            "noise": noise,
            "description": type_info["description"],
            "detail": f"{type_info['detail']} {noise_info['detail']}",
        })
    return variations
