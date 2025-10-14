# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 YANEZ - MIID Team

from itertools import combinations
import random
import re
from typing import List, Dict, Tuple, Any, Set
import bittensor as bt
import Levenshtein
import jellyfish
import string

from MIID.miner.pool_generator import make_replacement_map
# List of rules that can be checked algorithmically
# These are basic heuristics to check if a variation follows a specific rule
# More complex rules would need more sophisticated checking methods
REPLACE_MAP = make_replacement_map

def uppercasing(variation, rng: random.Random):
    variation = variation.lower()
    def toggle_case(s:str, idxs:List[int])->str:
        arr=list(s)
        for i in idxs:
            ch=arr[i]
            arr[i]= ch.upper() if ch.islower() else ch.lower()
        return ''.join(arr)
    maxm = 1 << len(variation)
    mask = rng.randint(1, maxm)
    idxs=[i for i in range(len(variation)) if (mask >> i) & 1]
    uppercased = toggle_case(variation, idxs)
    return uppercased

def default_char_sampler(original: str, full_pool: str, rng: random.Random) -> str:
    """
    Sample any non-space character. Uses letters, digits, punctuation, and a few emojis.
    You can swap this out with your own sampler if you want broader Unicode.
    """
    # Get replacement map to filter out characters that might cause phonetic collisions
    repl_map = make_replacement_map(original.lower())
    
    # Build pool excluding characters that are phonetic replacements for each other
    # to avoid unintended phonetic similarities
    excluded_chars = set()
    for char, replacements in repl_map.items():
        excluded_chars.update(replacements)
        excluded_chars.update(char)
    # Create pool from ascii letters/digits/punctuation, excluding phonetic replacements
    pool = ''.join(ch for ch in full_pool if ch.lower() not in excluded_chars)
    
    # Fallback to basic alphanumeric if pool becomes too small
    if len(pool) < 3:
        pool = full_pool
    return pool

def replace_space_with_special_chars(original: str, rng: random.Random):

    """
    Returns a new string with:
      - no spaces
      - Levenshtein(new_string, original_without_spaces) < count(' ') + 2
    """

    """
    Replace every ' ' in original with ANY non-space character from `sampler`.
    - Output has no spaces
    - Distance to original.replace(' ', '') == number_of_spaces (<= spaces+1)
    - Output != original (guaranteed if original contains spaces)
    """
    if ' ' not in original:
        raise ValueError("Original must contain at least one space.")

    sample_chars = default_char_sampler(original, [c for c in string.ascii_lowercase if is_consonant(c)], rng)
    no_spaces = list(original.replace(' ', ''))
    spaces = original.count(' ')
    spins = spaces + 1
    variation = []
    while len(variation) < len(no_spaces):
        variation = list(no_spaces)
        for _ in range(spins):
            if not no_spaces:
                op = rng.choice(["insert", "no_action"])
            else:
                op = rng.choice(["no_action", "insert", "delete", "substitute"])
            if op == "no_action":
                pass
            elif op == "insert":
                pos = rng.randrange(len(variation) + 1)
                new_char = rng.choice(sample_chars)
                variation.insert(pos, new_char)
            elif op == "delete":
                pos = rng.randrange(len(variation))
                del variation[pos]
            elif op == "substitute":
                pos = rng.randrange(len(variation))
                current = variation[pos]
                new_char = rng.choice(sample_chars)
                while new_char == current:
                    new_char = rng.choice(sample_chars)
                variation[pos] = new_char
    return "".join(variation)


def replace_double_letter(original: str, rng: random.Random):
    """
    Generates a variation from original by replacing a double letter with a single one.
    This passes the is_double_letter_replaced() evaluator.
    """
    # Find double-letter positions
    double_positions = [
        i for i in range(len(original) - 1)
        if original[i] == original[i+1]
    ]
    if not double_positions:
        raise ValueError("Original must contain at least one double letter.")
    variation = list(original)
    idx = rng.choice(double_positions)
    del variation[idx]
    return uppercasing("".join(variation), rng)

def replace_vowel(original: str, rng: random.Random):

    """
    Generate a variation of `original` where at least one vowel is replaced
    with a different vowel, and at most one non-vowel character is changed.
    """
    vowels = "aeiou"

    # Find vowel positions in original
    vowel_positions = [i for i, ch in enumerate(original) if ch in vowels]
    if not vowel_positions:
        raise ValueError("Original must contain at least one vowel.")

    variation = list(original)

    # Replace at least one vowel with a different vowel
    vowel_change_count = rng.randint(1, len(vowel_positions))
    
    changed = [False] * len(variation)
    for _ in range(vowel_change_count):
        # allowed only substitute
        op = rng.choice(["substitute"])
        if op == "substitute":
            # take a random vowel position that is not changed
            idx = rng.choice([vowel_positions[i] for i in range(len(vowel_positions)) if not changed[vowel_positions[i]]])
            orig_vowel = variation[idx]
            other_vowels = [v for v in vowels if v != orig_vowel]
            variation[idx] = rng.choice(other_vowels)
            changed[idx] = True

    # Replace at most one action other than vowel -> vowel
    other_change_count = rng.randint(0, 1)
    for _ in range(other_change_count):
        # allowed only substitute
        op = rng.choice(["substitute"])
        if op == "substitute":
            # take a random position that is not changed
            choices = [i for i in range(len(variation)) if not changed[i]]
            if len(choices) == 0:
                break
            idx = rng.choice(choices)
            # take a random character that is not a vowel
            orig_char = variation[idx]
            if orig_char in vowels:
                other_chars = [c for c in string.ascii_lowercase if c not in vowels and c != orig_char]
            else:
                other_chars = [c for c in string.ascii_lowercase if c != orig_char]
            variation[idx] = rng.choice(other_chars)
            changed[idx] = True

    return uppercasing("".join(variation), rng)

def replace_consonant(original: str, rng: random.Random):

    """
    Generate a variation of `original` where at least one consonant is replaced
    with a different consonant, and at most one vowel is changed.
    """

    # Find consonant positions in original
    consonant_positions = [i for i, ch in enumerate(original) if is_consonant(ch)]
    if not consonant_positions:
        raise ValueError("Original must contain at least one consonant.")

    variation = list(original)

    # Replace at least one vowel with a different vowel
    consonant_change_count = rng.randint(1, len(consonant_positions))
    consonant_change_pos = rng.sample(consonant_positions, consonant_change_count)
    pool_consonants = default_char_sampler(
        original,
        [c for c in string.ascii_lowercase if is_consonant(c)],
        rng )
    vowels = "aeiou"
    for idx in consonant_change_pos:
        # allowed only substitute
        op = rng.choice(["substitute"])
        if op == "substitute":
            # take a random vowel position that is not changed
            # idx = rng.choice([consonant_positions[i] for i in range(len(consonant_positions)) if not changed[consonant_positions[i]]])
            orig_consonant = variation[idx]
            other_consonants = [c for c in pool_consonants if c.lower() != orig_consonant.lower()]
            variation[idx] = rng.choice(other_consonants)
    if rng.random() < 0.5:
        other_change_pos = rng.choice([i for i in range(len(original)) if i not in consonant_change_pos])
        # Replace at most one action other than consonant -> consonant
        # allowed only substitute
        op = rng.choice(["substitute"])
        if op == "substitute":
            # take a random position that is not changed
            # take a random character that is not a consonant
            orig_char = variation[other_change_pos]
            if is_consonant(orig_char):
                other_chars = [c for c in vowels if c.lower() != orig_char.lower()]
            else:
                other_chars = [c for c in pool_consonants + vowels if c != orig_char]
            variation[other_change_pos] = rng.choice(other_chars)

    return uppercasing("".join(variation), rng)

def swap_letters(original: str, rng: random.Random):

    """
    Create a variation by swapping exactly one pair of adjacent letters,
    ensuring variation != original and it passes is_letters_swapped().
    """
    if len(original) < 2:
        raise ValueError("Need at least 2 characters to swap.")

    # Only indices where swapping changes the string
    candidates = [i for i in range(len(original) - 1)
                  if original[i] != original[i + 1]]
    if not candidates:
        raise ValueError("All adjacent pairs are identical; any swap would be a no-op.")

    idx = rng.choice(candidates)

    variation = list(original)
    variation[idx], variation[idx + 1] = variation[idx + 1], variation[idx]
    variation = "".join(variation)

    return "".join(variation)

def is_consonant(char: str) -> bool:
    """Check if a character is a consonant (case-insensitive)"""
    vowels = 'aeiou'
    return char.isalpha() and char.lower() not in vowels

def swap_adjacent_consonants(original: str, rng: random.Random):
    """Check if two adjacent consonants are swapped."""
    """
    Create a variation by swapping exactly one pair of adjacent consonants.
    """
    # Find all pairs of adjacent consonants
    consonant_pairs = []
    for i in range(len(original) - 1):
        if is_consonant(original[i]) and is_consonant(original[i + 1]) and original[i] != original[i + 1]:
            consonant_pairs.append(i)
    if not consonant_pairs:
        raise ValueError("No adjacent consonants found to swap.")
    
    variation = list(original)
    idx = rng.choice(consonant_pairs)
    variation[idx], variation[idx + 1] = variation[idx + 1], variation[idx]
    variation = "".join(variation)
    return uppercasing("".join(variation), rng)

def remove_letter(original: str, rng: random.Random):
    """
    Generate a variation by deleting exactly one character from `original`.
    """
    if len(original) < 1:
        raise ValueError("Original must have at least 1 character.")

    idx = rng.randrange(len(original))  # position to delete

    variation = original[:idx] + original[idx+1:]

    return variation

def remove_vowel(original: str, rng: random.Random):

    """
    Generate a variation by removing exactly one vowel from the original string.
    Passes is_vowel_removed().
    """
    vowels = "aeiou"
    vowel_positions = [i for i, ch in enumerate(original) if ch in vowels]

    if not vowel_positions:
        raise ValueError("Original must contain at least one vowel.")

    idx = rng.choice(vowel_positions)  # pick a vowel to remove

    variation = original[:idx] + original[idx+1:]

    return uppercasing("".join(variation), rng)

def remove_consonant(original: str, rng: random.Random):
    """
    Generate a variation by removing exactly one consonant from the original string.
    Passes is_consonant_removed().
    """
    vowels = "aeiou"
    # Find all consonant positions (alphabetic but not vowel)
    consonant_positions = [
        i for i, ch in enumerate(original)
        if ch.isalpha() and ch not in vowels
    ]

    if not consonant_positions:
        raise ValueError("Original must contain at least one consonant.")

    idx = rng.choice(consonant_positions)  # choose consonant to remove

    variation = original[:idx] + original[idx+1:]

    return uppercasing("".join(variation), rng)

def replace_special_character(original: str, rng: random.Random):
    """Check if a special character is replaced with a different one"""
    """
    Replace at least one special character in `original` with a different special char,
    and optionally replace at most one non-special character.
    Passes is_special_character_replaced().
    """
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'

    # Find special character positions in original
    special_positions = [i for i, ch in enumerate(original) if ch in special_chars]
    if not special_positions:
        raise ValueError("Original must contain at least one special character.")

    variation = list(original)

    # Replace at least one special character with a different special character
    special_change_count = rng.randint(1, len(special_positions))
    
    changed = [False] * len(variation)
    for _ in range(special_change_count):
        # allowed only substitute
        op = rng.choice(["substitute"])
        if op == "substitute":
            # take a random special character position that is not changed
            idx = rng.choice([special_positions[i] for i in range(len(special_positions)) if not changed[special_positions[i]]])
            orig_special = variation[idx]
            other_specials = [c for c in special_chars if c != orig_special]
            variation[idx] = rng.choice(other_specials)
            changed[idx] = True

    # Replace at most one action other than special character -> special character
    other_change_count = rng.randint(0, 1)
    for _ in range(other_change_count):
        # allowed only substitute
        op = rng.choice(["substitute"])
        if op == "substitute":
            # take a random position that is not changed
            choices = [i for i in range(len(variation)) if not changed[i]]
            if len(choices) == 0:
                break
            idx = rng.choice(choices)
            # take a random character that is not a special character
            orig_char = variation[idx]
            if orig_char in special_chars:
                other_chars = [c for c in string.ascii_lowercase if c not in special_chars and c != orig_char]
            else:
                other_chars = [c for c in string.ascii_lowercase if c != orig_char]
            variation[idx] = rng.choice(other_chars)
            changed[idx] = True

    return "".join(variation)

def remove_random_special(original: str, rng: random.Random):
    """
    Generate a variation by removing exactly one special character from the original string.
    Passes is_random_special_removed().
    """
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'

    # Find all special char positions
    special_positions = [
        i for i, ch in enumerate(original) if ch in special_chars
    ]
    if not special_positions:
        raise ValueError("Original must contain at least one special character.")

    idx = rng.choice(special_positions)  # choose a special char to remove

    variation = original[:idx] + original[idx+1:]

    return variation

def remove_title(original: str, rng: random.Random):
    """Check if a title is removed from the name"""
    """
    Remove a leading title (from TITLES) and perform <=2 substitutions on the remainder.
    This guarantees: variation != original and
    Levenshtein(variation, stripped_original) <= 2 (since we only do substitutions).
    """

    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]

    matches = [t for t in titles if original.startswith(t + " ")]
    if not matches:
        raise ValueError("Original has no leading title from the allowed list.")

    title = max(matches, key=len)          # prefer longest match
    stripped = original[len(title) + 1:]   # after "<title><space>"

    var = list(stripped)
    num_edits = rng.randint(0, min(2, len(var)))
    for _ in range(num_edits):
        if not var:
            # Only insertion possible if name became empty
            op = "insert"
        else:
            op = rng.choice(["no_action", "insert", "delete", "substitute"])
        
        if op == "no_action":
            pass
        elif op == "insert":
            pos = rng.randrange(len(var) + 1)
            new_char = rng.choice(string.ascii_lowercase)
            var.insert(pos, new_char)
        elif op == "delete":
            pos = rng.randrange(len(var))
            del var[pos]
        elif op == "substitute":
            pos = rng.randrange(len(var))
            current = var[pos]
            pool = string.ascii_lowercase
            choices = [c for c in pool if c != current]
            var[pos] = rng.choice(choices)

    return uppercasing("".join(var), rng)


def abbreviate_name(original: str, rng: random.Random):
    """
    Abbreviates each part of the name so that:
    - variation != original
    - Each variation part is shorter than the original part
    - Each variation part is a prefix of the original part
    Passes is_name_abbreviated().
    """
    parts = original.split()

    variation_parts = []
    changed = False
    min_cut = 1
    too_short = False
    while changed == False and too_short == False:
        too_short = True
        for part in parts:
            if len(part) <= min_cut:
                # Can't shorten further; keep as is
                too_short = True
                break
                
            else:
                # Choose cut length between min_cut and len(part)-1
                cut_len = rng.randint(min_cut, len(part) - 1)
                abbrev = part[:cut_len]
                variation_parts.append(abbrev)
                if abbrev != part:
                    changed = True
                too_short = False
    if too_short == True:
        raise ValueError("Could not abbreviate any part (too short).")
    variation = " ".join(variation_parts)

    if len(variation_parts) == 1:
        return uppercasing(variation, rng)
    return variation


def remove_all_spaces(original: str, rng: random.Random):
    """
    Generate a variation by removing all spaces from the original string.
    Passes is_all_spaces_removed().
    """
    if ' ' not in original:
        raise ValueError("Original must contain at least one space.")

    variation = original.replace(' ', '')
    return variation


def duplicate_letter(original: str, rng: random.Random):
    """
    Generate a variation by duplicating exactly one letter from the original string.
    Passes is_letter_duplicated().
    """
    if len(original) < 1:
        raise ValueError("Original must have at least 1 character.")

    idx = rng.randrange(len(original))  # pick a position to duplicate

    variation = original[:idx] + original[idx] + original[idx:]
    
    return variation

def insert_random_letter(original: str, rng: random.Random):

    """
    Insert exactly one random ASCII letter at a random position.
    Passes is_random_letter_inserted().
    """
    # pos = rng.randrange(len(original) + 1)          # insertion index [0..len]
    # letter = rng.choice(string.ascii_lowercase)
    # variation = original[:pos] + letter + original[pos:]
    pos = rng.randrange(len(original))
    letter = rng.choice(string.ascii_lowercase)
    variation = original[:pos] + letter + original[pos:]
    return variation

def add_title(original: str, rng: random.Random):
    """
    Adds a title and performs up to `max_edits` random edit operations
    (insert, delete, substitute) on the name.
    Passes is_title_added().
    """
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]

    title = rng.choice(titles)

    # Start with the base name (original)
    name_chars = list(original)

    # Apply 0–max_edits edits
    num_edits = rng.randint(0, min(2, len(original)))
    for _ in range(num_edits):
        if not name_chars:
            # Only insertion possible if name became empty
            op = "insert"
        else:
            op = rng.choice(["no_action", "insert", "delete", "substitute"])
        
        if op == "no_action":
            pass
        elif op == "insert":
            pos = rng.randrange(len(name_chars) + 1)
            new_char = rng.choice(string.ascii_lowercase)
            name_chars.insert(pos, new_char)
        elif op == "delete":
            pos = rng.randrange(len(name_chars))
            del name_chars[pos]
        elif op == "substitute":
            pos = rng.randrange(len(name_chars))
            current = name_chars[pos]
            pool = string.ascii_lowercase
            choices = [c for c in pool if c != current]
            name_chars[pos] = rng.choice(choices)

    edited_name = "".join(name_chars)
    variation = f"{title} {edited_name}"

    return uppercasing(variation, rng)

def add_suffix(original: str, rng: random.Random):
    """
    Adds a title and performs up to `max_edits` random edit operations
    (insert, delete, substitute) on the name.
    Passes is_suffix_added().
    """

    suffixes = ["Jr.", "Sr.", "III", "IV", "V", "PhD", "MD", "Esq.", "Jr", "Sr"]
    suffix = rng.choice(suffixes)

    # Start with the base name (original)
    name_chars = list(original)

    # Apply 0–max_edits edits
    num_edits = rng.randint(0, min(2, len(original)))
    for _ in range(num_edits):
        if not name_chars:
            # Only insertion possible if name became empty
            op = "insert"
        else:
            op = rng.choice(["no_action", "insert", "delete", "substitute"])
        
        if op == "no_action":
            pass
        elif op == "insert":
            pos = rng.randrange(len(name_chars) + 1)
            new_char = rng.choice(string.ascii_lowercase)
            name_chars.insert(pos, new_char)
        elif op == "delete":
            pos = rng.randrange(len(name_chars))
            del name_chars[pos]
        elif op == "substitute":
            pos = rng.randrange(len(name_chars))
            current = name_chars[pos]
            pool = string.ascii_lowercase
            choices = [c for c in pool if c != current]
            name_chars[pos] = rng.choice(choices)

    edited_name = "".join(name_chars)
    variation = f"{edited_name} {suffix}"

    return uppercasing(variation, rng)

def only_initials(original: str, rng: random.Random):

    """
    Create an initials-only variation of `original` that passes is_initials_only().
    style: "dots" -> "A.B."
           "spaced" -> "A. B."
           "plain" -> "AB"
    """
    parts = [p for p in original.split() if p]  # ignore extra spaces
    if len(parts) < 2:
        raise ValueError("Original must contain at least two words for initials.")

    initials = [p[0] for p in parts]
    style = rng.choice(["dots", "spaced", "plain"])
    variation = ""
    if style == "dots":
        variation = ".".join(initials) + "."
    elif style == "spaced":
        variation = ". ".join(initials) + "."
    elif style == "plain":
        variation = "".join(initials)
    else:
        raise ValueError('style must be one of: "dots", "spaced", "plain"')

    return uppercasing(variation, rng)

def name_parts_permutations(original: str, rng: random.Random):
    """
    Generate a variation where the name parts are permuted but not identical in order.
    Passes is_name_parts_permutation().
    """
    parts = original.split()
    if len(parts) < 2:
        raise ValueError("Original must contain at least two parts to permute.")

    variation_parts = parts[:]
    # Keep shuffling until order is different from original
    max_retry = 100
    for _ in range(max_retry):
        rng.shuffle(variation_parts)
        if variation_parts != parts:
            break
    space_count = rng.randint(1, 10)
    variation = (" " * space_count).join(variation_parts)

    return variation

def initial_first_name(original: str, rng: random.Random):
    """
    Generate a variation where the first name is reduced to its initial.
    Passes is_first_name_initial().

    style:
      "spaced" -> "A. LastName"
      "compact" -> "A.LastName"
    """
    parts = original.split()
    if len(parts) < 2:
        raise ValueError("Original must contain at least two words for this transformation.")

    initial = parts[0][0]
    rest = " ".join(parts[1:])

    style = rng.choice(["spaced", "compact"])
    variation = ""
    if style == "spaced":
        variation = f"{initial}. {rest}"
    elif style == "compact":
        variation = f"{initial}.{rest}"
    else:
        raise ValueError('style must be either "spaced" or "compact"')

    return uppercasing(variation, rng)

# Map rule names to their evaluation functions
RULE_BASED_TRANSFORMATIONS = {
    "replace_spaces_with_random_special_characters": replace_space_with_special_chars,
    "replace_double_letters_with_single_letter": replace_double_letter,
    "replace_random_vowel_with_random_vowel": replace_vowel,
    "replace_random_consonant_with_random_consonant": replace_consonant,
    "replace_random_special_character_with_random_special_character": replace_special_character,
    "swap_random_letter": swap_letters,
    "swap_adjacent_consonants": swap_adjacent_consonants,  # Simplified, would need more specific check
    "delete_random_letter": remove_letter,
    "remove_random_vowel": remove_vowel,
    "remove_random_consonant": remove_consonant,
    "remove_random_special_character": remove_random_special,
    "remove_title": remove_title,
    "remove_all_spaces": remove_all_spaces,
    "duplicate_random_letter_as_double_letter": duplicate_letter,
    "insert_random_letter": insert_random_letter,
    "add_random_leading_title": add_title,
    "add_random_trailing_title": add_suffix,
    "shorten_name_to_initials": only_initials,
    "name_parts_permutations": name_parts_permutations,
    "initial_only_first_name": initial_first_name,
    "shorten_name_to_abbreviations": abbreviate_name
}

def swap_random_letter_swap_adjacent_consonants(original: str, rng: random.Random):
    """
    Swap a random pair of consonants in the original string.
    """
    return swap_adjacent_consonants(original, rng).lower()

def swap_random_letter_replace_random_vowel_with_random_vowel(original: str, rng: random.Random):
    """
    Swap a random pair of letters in the original string.
    """
    vowels = "aeiou"
    # Only indices where swapping changes the string
    candidates = [i for i in range(len(original) - 1)
                if original[i] != original[i + 1] and original[i] in vowels]
    if not candidates:
        raise ValueError("All adjacent pairs are identical or not vowels; any swap would be a no-op.")

    idx = rng.choice(candidates)

    variation = list(original)
    variation[idx], variation[idx + 1] = variation[idx + 1], variation[idx]
    variation = "".join(variation)
    
    return variation


def swap_random_letter_replace_random_consonant_with_random_consonant(original: str, rng: random.Random):
    # Only indices where swapping changes the string
    candidates = [i for i in range(len(original) - 1)
                if original[i] != original[i + 1] and is_consonant(original[i])]
    if not candidates:
        raise ValueError("All adjacent pairs are identical or not consonants; any swap would be a no-op.")

    idx = rng.choice(candidates)

    variation = list(original)
    variation[idx], variation[idx + 1] = variation[idx + 1], variation[idx]
    variation = "".join(variation)
            
    return variation

def swap_adjacent_consonants_replace_random_consonant_with_random_consonant(original: str, rng: random.Random):
    """
    Swap a random pair of consonants in the original string.
    """
    return swap_adjacent_consonants(original, rng)

def delete_random_letter_abbreviate_name(original: str, rng: random.Random):
    """
    Delete a random letter and abbreviate the name.
    """
    if len(original) < 1:
        raise ValueError(f"Name must has more than 2 characters")
    return original[:-1]

def delete_random_letter_replace_double_letter(original: str, rng: random.Random):
    """
    Generates a variation from original by replacing a double letter with a single one.
    This passes the is_double_letter_replaced() evaluator.
    """
    # Find double-letter positions
    double_positions = [
        i for i in range(len(original) - 1)
        if original[i] == original[i+1]
    ]
    if not double_positions:
        raise ValueError("Original must contain at least one double letter.")
    variation = list(original)
    idx = rng.choice(double_positions)
    del variation[idx]
    return "".join(variation)



def delete_random_letter_remove_consonant(original: str, rng: random.Random):
    """
    Generate a variation by removing exactly one consonant from the original string.
    Passes is_consonant_removed().
    """
    vowels = "aeiou"
    # Find all consonant positions (alphabetic but not vowel)
    consonant_positions = [
        i for i, ch in enumerate(original)
        if ch.isalpha() and ch not in vowels
    ]

    if not consonant_positions:
        raise ValueError("Original must contain at least one consonant.")

    idx = rng.choice(consonant_positions)  # choose consonant to remove

    variation = original[:idx] + original[idx+1:]
    return "".join(variation)

def delete_random_letter_remove_vowel(original: str, rng: random.Random):
    """
    Generate a variation by removing exactly one vowel from the original string.
    Passes is_vowel_removed().
    """
    vowels = "aeiou"
    vowel_positions = [i for i, ch in enumerate(original) if ch in vowels]

    if not vowel_positions:
        raise ValueError("Original must contain at least one vowel.")

    idx = rng.choice(vowel_positions)  # pick a vowel to remove

    variation = original[:idx] + original[idx+1:]

    return "".join(variation)

def delete_random_letter_remove_all_spaces(original: str, rng: random.Random):
    """
    Delete a random letter and remove all spaces.
    """
    if ' ' not in original or original.find(original) != original.rfind(original):
        raise ValueError("Original must contain only one space.")
    return original.replace(' ', '')

def replace_spaces_with_random_special_characters(original: str, rng: random.Random):
    """
    Replace all spaces with random special characters.
    """
    return original.replace(' ', rng.choice(string.ascii_lowercase))

def remove_random_consonant_abbreviate_name(original: str, rng: random.Random):
    """
    Delete a random letter and abbreviate the name.
    """
    if len(original) < 1:
        raise ValueError(f"Name must has more than 2 characters")

    if not is_consonant(original[-1]):
        raise ValueError("Original must end with a consonant.")
    return original[:-1]

def remove_random_consonant_replace_double_letter(original: str, rng: random.Random):
    """
    Generates a variation from original by replacing a double letter with a single one.
    This passes the is_double_letter_replaced() evaluator.
    """
    # Find double-letter positions
    double_positions = [
        i for i in range(len(original) - 1)
        if original[i] == original[i+1] and is_consonant(original[i])
    ]
    if not double_positions:
        raise ValueError("Original must contain at least one double letter with consonants.")
    variation = list(original)
    idx = rng.choice(double_positions)
    del variation[idx]
    return "".join(variation)

def replace_spaces_with_random_special_characters_replace_random_vowel_with_random_vowel(original: str, rng: random.Random):
    """
    Replace exactly one vowel with a different vowel (preserve case).
    """
    vowels = "aeiou"
    if ' ' not in original:
        raise ValueError("Original must contain at least one space.")
    space_replacement = rng.choice(string.ascii_lowercase)
    space_replaced = original[0:original.find(' ')] + space_replacement + original[original.find(' ')+1:]

    idxs = [i for i, c in enumerate(space_replaced) if c in vowels]
    if not idxs:
        raise ValueError("No vowels to replace.")
    vowel_replacement_pos = rng.choice(idxs)
    vowels = [c for c in vowels if c != space_replaced[vowel_replacement_pos]]
    vowel_replacement = rng.choice(vowels)
    result = space_replaced[0:vowel_replacement_pos] + vowel_replacement + space_replaced[vowel_replacement_pos+1:]
    return result

def replace_spaces_with_random_special_characters_replace_random_consonant_with_random_consonant(original: str, rng: random.Random):
    """
    Replace exactly one consonant with a different consonant (preserve case).
    """
    consonants = "bcdfghjklmnpqrstvwxyz"
    if ' ' not in original:
        raise ValueError("Original must contain at least one space.")
    space_replacement = rng.choice(string.ascii_lowercase)
    space_replaced = original[0:original.find(' ')] + space_replacement + original[original.find(' ')+1:]

    idxs = [i for i, c in enumerate(space_replaced) if c in consonants]
    if not idxs:
        raise ValueError("No consonants to replace.")
    consonant_replacement_pos = rng.choice(idxs)
    consonants = [c for c in consonants if c != space_replaced[consonant_replacement_pos]]
    consonant_replacement = rng.choice(consonants)
    result = space_replaced[0:consonant_replacement_pos] + consonant_replacement + space_replaced[consonant_replacement_pos+1:]
    return result
    
def replace_spaces_with_random_special_characters_replace_random_special_character_with_random_special_character(original: str, rng: random.Random):
    """
    Replace exactly one special character with a different special character (preserve case).
    """
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if ' ' not in original:
        raise ValueError("Original must contain at least one space.")
    space_replacement = rng.choice(string.ascii_lowercase)
    space_replaced = original[0:original.find(' ')] + space_replacement + original[original.find(' ')+1:]

    idxs = [i for i, c in enumerate(space_replaced) if c in special_chars]
    if not idxs:
        raise ValueError("No special characters to replace.")
    special_char_replacement_pos = rng.choice(idxs)
    special_chars = [c for c in special_chars if c != space_replaced[special_char_replacement_pos]]
    special_char_replacement = rng.choice(special_chars)
    result = space_replaced[0:special_char_replacement_pos] + special_char_replacement + space_replaced[special_char_replacement_pos+1:]
    return result


RULE_BASED_TRANSFORMATIONS_PAIR = {
    "swap_random_letter": {
        "swap_adjacent_consonants": swap_random_letter_swap_adjacent_consonants,
        "replace_random_vowel_with_random_vowel": swap_random_letter_replace_random_vowel_with_random_vowel,
        "replace_random_consonant_with_random_consonant": swap_random_letter_replace_random_consonant_with_random_consonant,
    },
    "swap_adjacent_consonants": {
        "replace_random_consonant_with_random_consonant": swap_adjacent_consonants_replace_random_consonant_with_random_consonant,
    },
    "delete_random_letter": {
        "remove_random_consonant": delete_random_letter_remove_consonant,
        "remove_random_vowel": delete_random_letter_remove_vowel,
        "shorten_name_to_abbreviations": delete_random_letter_abbreviate_name,
        "replace_double_letters_with_single_letter": delete_random_letter_replace_double_letter,
        "remove_all_spaces": delete_random_letter_remove_all_spaces,
        "replace_spaces_with_random_special_characters": delete_random_letter_remove_all_spaces
    },
    "remove_random_consonant": {
        "shorten_name_to_abbreviations": remove_random_consonant_abbreviate_name,
        "replace_double_letters_with_single_letter": remove_random_consonant_replace_double_letter,
    },
    "remove_all_spaces": {
        "replace_spaces_with_random_special_characters": remove_all_spaces
    },
    "insert_random_letter": {
        "duplicate_random_letter_as_double_letter": duplicate_letter
    },
    "replace_spaces_with_random_special_characters": {
        "replace_random_vowel_with_random_vowel": lambda original: replace_spaces_with_random_special_characters_replace_random_vowel_with_random_vowel,
        "replace_random_consonant_with_random_consonant": lambda original: replace_spaces_with_random_special_characters_replace_random_consonant_with_random_consonant,
        "replace_random_special_character_with_random_special_character": lambda original: replace_spaces_with_random_special_characters_replace_random_special_character_with_random_special_character,
    }
}

RULE_BASED_TRANSFORMATIONS_PAIR3 = {
    "swap_random_letter": {
        "swap_adjacent_consonants": {
            "replace_random_consonant_with_random_consonant": swap_random_letter_swap_adjacent_consonants
        },
    },
    "delete_random_letter": {   
        "remove_random_consonant": {
            "shorten_name_to_abbreviations": remove_random_consonant_abbreviate_name,
            "replace_double_letters_with_single_letter": remove_random_consonant_replace_double_letter,
        },
        "remove_all_spaces": {
            "replace_spaces_with_random_special_characters": delete_random_letter_remove_all_spaces
        },
   }
}


RULE_BASED_TRANSFORMATIONS_COMBINED = {}
for first, rule in RULE_BASED_TRANSFORMATIONS.items():
    RULE_BASED_TRANSFORMATIONS_COMBINED[first] = rule
for first, sencond_rule in RULE_BASED_TRANSFORMATIONS_PAIR.items():
    for second, rule in sencond_rule.items():
        sorted_list = sorted([first, second])
        RULE_BASED_TRANSFORMATIONS_COMBINED["_".join(sorted_list)] = rule

for first, sencond_rule in RULE_BASED_TRANSFORMATIONS_PAIR3.items():
    for second, third_rule in sencond_rule.items():
        for third, rule in third_rule.items():
            sorted_list = sorted(list([first, second, third]))
            RULE_BASED_TRANSFORMATIONS_COMBINED["_".join(sorted_list)] = rule
