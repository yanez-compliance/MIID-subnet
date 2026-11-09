# MIID/validator/voice_words.py
#
# Target passcode text for voice-clone challenges (English / Spanish).
#
# Each challenge draws tokens a synthetic voice model must speak:
#   - 2 digit words (e.g. "three", "seis") — never zero
#   - 10 words from a language word pool (word_pools/en.txt or es.txt)
# The 12 tokens are shuffled into one sequence. When a miner hotkey is
# supplied, the first 4 and last 4 characters of that SS58 address are
# each emitted as separate tokens (uppercase letters) and wrapped around
# the sequence, e.g.:
#   5 G U P libro escuela ... dos mesa K V D U
# so ASR / search can match characters individually. Clip length is not
# prescribed — miners speak the prompt at a natural pace. Only "en" / "es".
#
# Hotkey bookends are Latin alphanumeric (SS58 / Base58). They are language-
# agnostic: the same characters work for English and Spanish TTS.
#
# Public helpers:
#   generate(language, hotkey=...)  -> KaraokeChallenge (.text, .words)
#   with_hotkey(challenge, hotkey)  -> KaraokeChallenge with bookends
#   hotkey_bookends(hotkey)         -> (prefix4, suffix4)
#   generate_target_words(lang)     -> list of spoken tokens for VoiceRequest.target_words

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# --- counts ---
KARAOKE_DIGIT_COUNT = 2
KARAOKE_WORD_COUNT = 10
HOTKEY_PREFIX_LEN = 4
HOTKEY_SUFFIX_LEN = 4
# Base passcode size (no hotkey). With hotkey: +8 single-character tokens.
KARAOKE_BASE_TOKEN_COUNT = KARAOKE_DIGIT_COUNT + KARAOKE_WORD_COUNT
KARAOKE_HOTKEY_CHAR_COUNT = HOTKEY_PREFIX_LEN + HOTKEY_SUFFIX_LEN

_WORD_POOL_DIR = Path(__file__).resolve().parent / "word_pools"


class SpeechLanguage(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"


ENGLISH_DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
SPANISH_DIGITS = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]


def _load_word_pool(filename: str) -> List[str]:
    """Load a one-word-per-line pool; skip blanks and ``#`` comment lines."""
    path = _WORD_POOL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"karaoke word pool missing: {path}")

    words: List[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # One token per line; ignore accidental multi-word entries.
        token = line.split()[0].lower()
        if token in seen:
            continue
        seen.add(token)
        words.append(token)

    if len(words) < KARAOKE_WORD_COUNT:
        raise ValueError(
            f"karaoke word pool {path.name} has {len(words)} words; "
            f"need at least {KARAOKE_WORD_COUNT}"
        )
    return words


ENGLISH_WORDS = _load_word_pool("en.txt")
SPANISH_WORDS = _load_word_pool("es.txt")

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


def _char_tokens(chunk: str) -> List[str]:
    """Split a hotkey chunk into single-character tokens (letters uppercased)."""
    return [c.upper() if c.isalpha() else c for c in chunk]


def hotkey_bookends(hotkey: str) -> Tuple[List[str], List[str]]:
    """Return (first 4, last 4) characters of an SS58 miner hotkey as token lists.

    Letters are uppercased so search/ASR can match characters individually.
    Digits stay as-is. Shared by English and Spanish prompts.
    """
    hk = (hotkey or "").strip()
    min_len = HOTKEY_PREFIX_LEN + HOTKEY_SUFFIX_LEN
    if len(hk) < min_len:
        raise ValueError(
            f"hotkey must be at least {min_len} characters, got {len(hk)}"
        )
    return (
        _char_tokens(hk[:HOTKEY_PREFIX_LEN]),
        _char_tokens(hk[-HOTKEY_SUFFIX_LEN:]),
    )


def with_hotkey(challenge: KaraokeChallenge, hotkey: str) -> KaraokeChallenge:
    """Wrap a base passcode with per-character hotkey prefix and suffix tokens."""
    prefix, suffix = hotkey_bookends(hotkey)
    return KaraokeChallenge(
        words=prefix + list(challenge.words) + suffix,
        language=challenge.language,
    )


def generate(
    language: SpeechLanguage | str,
    rng: Optional[random.Random] = None,
    hotkey: Optional[str] = None,
) -> KaraokeChallenge:
    """Draw a passcode: 2 digit-words + 10 words, shuffle, optionally add hotkey chars.

    When ``hotkey`` is set, the spoken sequence is:
      ``<c1> <c2> <c3> <c4> <shuffled passcode...> <c5> <c6> <c7> <c8>``
    where c1..c4 are hotkey[:4] and c5..c8 are hotkey[-4:], one token each.
    """
    rng = rng or random.Random()
    if not isinstance(language, SpeechLanguage):
        language = _parse_language(str(language))
    digits, words = POOLS[language]

    tokens = rng.sample(digits, KARAOKE_DIGIT_COUNT) + rng.sample(words, KARAOKE_WORD_COUNT)
    rng.shuffle(tokens)
    challenge = KaraokeChallenge(words=tokens, language=language)
    if hotkey:
        challenge = with_hotkey(challenge, hotkey)
    return challenge


def generate_target_words(
    language: str,
    count: int = KARAOKE_BASE_TOKEN_COUNT,
    hotkey: Optional[str] = None,
) -> List[str]:
    """Spoken tokens for ``VoiceRequest.target_words``.

    Always draws the standard set (2 digit-words + 10 words = 12 tokens),
    plus 8 single-character hotkey bookends when ``hotkey`` is provided.
    ``count`` is accepted for call-site compatibility but ignored.
    """
    _ = count  # fixed size
    return list(generate(language, hotkey=hotkey).words)
