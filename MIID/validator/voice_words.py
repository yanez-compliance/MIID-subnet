# MIID/validator/voice_words.py
#
# Target passcode text for voice-clone challenges (English / Spanish).
#
# Each challenge draws tokens a synthetic voice model must speak:
#   - 5 digit words (e.g. "three", "seis") — never zero
#   - 10 words from a language word pool
# The 15 tokens are shuffled into one sequence, then joined into a single
# TTS prompt string. Clip length is not prescribed — miners speak the prompt
# at a natural pace. Only "en" / "es".
#
# Public helpers:
#   generate(language)            -> KaraokeChallenge (.text, .words)
#   generate_target_words(lang)   -> list of spoken tokens for VoiceRequest.target_words

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# --- counts ---
KARAOKE_DIGIT_COUNT = 5
KARAOKE_WORD_COUNT = 10


class SpeechLanguage(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"


ENGLISH_DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ENGLISH_WORDS = [
    "blue",
    "chair",
    "apple",
    "table",
    "green",
    "house",
    "water",
    "paper",
    "clock",
    "window",
    "garden",
    "orange",
    "market",
    "river",
    "silver",
    "book",
    "door",
    "light",
    "music",
    "phone",
    "school",
    "street",
    "summer",
    "train",
    "yellow",
    "bread",
    "cloud",
    "flower",
    "forest",
    "mountain",
]

SPANISH_DIGITS = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
SPANISH_WORDS = [
    "azul",
    "silla",
    "mano",
    "mesa",
    "verde",
    "casa",
    "agua",
    "papel",
    "reloj",
    "ventana",
    "jardín",
    "naranja",
    "mercado",
    "río",
    "plata",
    "libro",
    "puerta",
    "luz",
    "música",
    "teléfono",
    "escuela",
    "calle",
    "verano",
    "tren",
    "amarillo",
    "pan",
    "nube",
    "flor",
    "bosque",
    "montaña",
]

POOLS: dict[SpeechLanguage, tuple[list[str], list[str]]] = {
    SpeechLanguage.ENGLISH: (ENGLISH_DIGITS, ENGLISH_WORDS),
    SpeechLanguage.SPANISH: (SPANISH_DIGITS, SPANISH_WORDS),
}


@dataclass
class KaraokeChallenge:
    """Shuffled spoken tokens ready for a TTS / voice-clone model."""

    words: list[str]
    language: SpeechLanguage

    @property
    def text(self) -> str:
        """Single prompt string for a synthetic voice maker."""
        return " ".join(self.words)

    @property
    def token_count(self) -> int:
        return len(self.words)

    def to_dict(self) -> dict:
        return {
            "language": self.language.value,
            "text": self.text,
            "words": list(self.words),
        }


def _parse_language(language: str) -> SpeechLanguage:
    key = (language or "en").strip().lower()
    aliases = {
        "en": SpeechLanguage.ENGLISH,
        "english": SpeechLanguage.ENGLISH,
        "es": SpeechLanguage.SPANISH,
        "spanish": SpeechLanguage.SPANISH,
    }
    return aliases.get(key, SpeechLanguage.ENGLISH)


def generate(
    language: SpeechLanguage | str,
    rng: Optional[random.Random] = None,
) -> KaraokeChallenge:
    """Draw a passcode: 5 digit-words + 10 words, then shuffle into TTS text."""
    rng = rng or random.Random()
    if not isinstance(language, SpeechLanguage):
        language = _parse_language(str(language))
    digits, words = POOLS[language]

    tokens = rng.sample(digits, KARAOKE_DIGIT_COUNT) + rng.sample(words, KARAOKE_WORD_COUNT)
    rng.shuffle(tokens)
    return KaraokeChallenge(words=tokens, language=language)


def generate_target_words(language: str, count: int = 15) -> List[str]:
    """Spoken tokens for ``VoiceRequest.target_words``.

    Always draws the standard set (5 digit-words + 10 words = 15 tokens).
    ``count`` is accepted for call-site compatibility but ignored.
    """
    _ = count  # fixed size
    return list(generate(language).words)
