# MIID/validator/voice_words.py
#
# Target passcode text for voice-clone challenges (English / Spanish).
#
# Each challenge draws tokens a synthetic voice model must speak:
#   - 5 digit words (e.g. "three", "seis") — never zero
#   - 3 words from a language word pool
# The 8 tokens are shuffled into one sequence, then joined into a single
# TTS prompt string (e.g. "three orange six two nine apple table one").
#
# Timing helpers assume a short lead-in, then one token every few seconds at
# 16 kHz — useful for windowing audio during post-grading. Only "en" / "es".
#
# Public helpers:
#   generate(language)            -> KaraokeChallenge (.text, .words)
#   generate_target_words(lang)   -> list of spoken tokens for VoiceRequest.target_words

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# --- counts and timing ---
KARAOKE_DIGIT_COUNT = 5
KARAOKE_WORD_COUNT = 3
KARAOKE_LEAD_IN_SECONDS = 1
KARAOKE_SECONDS_PER_TOKEN = 2
SAMPLE_RATE = 16_000


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
    "jardin",
    "naranja",
    "mercado",
    "rio",
    "plata",
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
    lead_in_seconds: int = KARAOKE_LEAD_IN_SECONDS
    seconds_per_token: int = KARAOKE_SECONDS_PER_TOKEN

    @property
    def text(self) -> str:
        """Single prompt string for a synthetic voice maker."""
        return " ".join(self.words)

    @property
    def token_count(self) -> int:
        return len(self.words)

    def record_duration_seconds(self) -> int:
        return self.lead_in_seconds + self.token_count * self.seconds_per_token

    def window_start_sample(self, token_index: int, sample_rate: int = SAMPLE_RATE) -> int:
        return (self.lead_in_seconds + token_index * self.seconds_per_token) * sample_rate

    def window_end_sample(self, token_index: int, sample_rate: int = SAMPLE_RATE) -> int:
        return (self.lead_in_seconds + (token_index + 1) * self.seconds_per_token) * sample_rate

    def to_dict(self) -> dict:
        return {
            "language": self.language.value,
            "text": self.text,
            "words": list(self.words),
            "record_duration_seconds": self.record_duration_seconds(),
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
    """Draw a passcode: 5 digit-words + 3 words, then shuffle into TTS text."""
    rng = rng or random.Random()
    if not isinstance(language, SpeechLanguage):
        language = _parse_language(str(language))
    digits, words = POOLS[language]

    tokens = rng.sample(digits, KARAOKE_DIGIT_COUNT) + rng.sample(words, KARAOKE_WORD_COUNT)
    rng.shuffle(tokens)
    return KaraokeChallenge(words=tokens, language=language)


def generate_target_words(language: str, count: int = 8) -> List[str]:
    """Spoken tokens for ``VoiceRequest.target_words``.

    Always draws the standard set (5 digit-words + 3 words = 8 tokens).
    ``count`` is accepted for call-site compatibility but ignored.
    """
    _ = count  # fixed size
    return list(generate(language).words)
