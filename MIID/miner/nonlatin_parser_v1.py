import regex
from typing import List
import bittensor as bt

LATIN_ONLY = regex.compile(r"^(?:\p{Script=Latin}|\p{Mark}|[\s'’.-])+$", flags=regex.UNICODE)

def is_latin_name(name: str) -> bool:
    # True if every visible character belongs to the Latin script (letters with accents included)
    # and common name punctuation/spacing.
    return bool(LATIN_ONLY.fullmatch(name))


# Lightweight script detection using Unicode script properties
_SCRIPT_PATTERNS = {
    "arabic": regex.compile(r"\p{Script=Arabic}+"),
    "cyrillic": regex.compile(r"\p{Script=Cyrillic}+"),
    "hebrew": regex.compile(r"\p{Script=Hebrew}+"),
    "greek": regex.compile(r"\p{Script=Greek}+"),
    "devanagari": regex.compile(r"\p{Script=Devanagari}+"),
    "thai": regex.compile(r"\p{Script=Thai}+"),
    # East Asian scripts
    "chinese": regex.compile(r"\p{Script=Han}+"),
    "japanese": regex.compile(r"(\p{Script=Hiragana}|\p{Script=Katakana}|\p{Script=Han})+"),
    "korean": regex.compile(r"\p{Script=Hangul}+"),
}


def _detect_script_label(name: str) -> str:
    """Detect the dominant script label used by the name.

    Returns labels like 'arabic', 'cyrillic', 'chinese', 'japanese', 'korean', etc.
    Returns 'latin' if already Latin, otherwise 'nonlatin' as a fallback.
    """
    if is_latin_name(name):
        return "latin"

    best_label = "nonlatin"
    best_count = 0
    for label, pattern in _SCRIPT_PATTERNS.items():
        count = 0
        for m in pattern.finditer(name):
            count += len(m.group(0))
        if count > best_count:
            best_count = count
            best_label = label
    return best_label


def transliterate_to_latin(name: str) -> str:
    """Transliterate a single non-Latin name to Latin letters.

    Input: non-Latin name; Output: Latin name only.
    Uses validator's LLM transliterator with a script hint; falls back to unidecode.
    """
    try:
        if not name:
            return ""
        if is_latin_name(name):
            return name

        script = _detect_script_label(name)
        bt.logging.info(f"Name: {name}, Script: {script}")
        try:
            from MIID.validator.reward import transliterate_name_with_llm  # type: ignore
            latin = transliterate_name_with_llm(name, script)
            return latin.strip()
        except Exception:
            try:
                from unidecode import unidecode
                return unidecode(name).strip()
            except Exception:
                return name
    except Exception:
        return name


def transliterate_list(names: List[str]) -> List[str]:
    """Transliterate a list of names to Latin using transliterate_to_latin."""
    if not names:
        return []
    return [transliterate_to_latin(n) for n in names]

