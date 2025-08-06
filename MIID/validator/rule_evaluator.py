# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 YANEZ - MIID Team

import random
import re
from typing import List, Dict, Tuple, Any, Set
import bittensor as bt
import Levenshtein
import jellyfish

# List of rules that can be checked algorithmically
# These are basic heuristics to check if a variation follows a specific rule
# More complex rules would need more sophisticated checking methods

def is_space_replaced_with_special_chars(original: str, variation: str) -> bool:
    """Check if spaces in the original are replaced with special characters"""
    if ' ' not in original or ' ' in variation:
        return False
        
    if original == variation:
        return False

    # Get positions of spaces in original
    space_positions = [i for i, char in enumerate(original) if char == ' ']
    
    # Check if length is compatible (allowing for minor changes)
    if abs(len(variation) - len(original)) > len(space_positions):
        return False
    
    # Try to map the variation to the original with spaces removed
    original_no_spaces = original.replace(' ', '')
    
    # Use Levenshtein distance to check if the variation is close to original with spaces removed
    # but with special characters potentially at the space positions
    lev_distance = Levenshtein.distance(original_no_spaces, variation)
    
    # Allow some flexibility: each space can be replaced by a special char
    return lev_distance <= len(space_positions) + 1

def is_double_letter_replaced(original: str, variation: str) -> bool:
    """Check if a double letter in the original is replaced with a single letter"""
    original_lower = original.lower()
    
    # Early exit if no double letters are found
    if not any(original_lower[i] == original_lower[i+1] for i in range(len(original_lower) - 1)):
        return False
        
    variation_lower = variation.lower()
    if len(variation_lower) != len(original_lower) - 1:
        return False

    for i in range(len(original_lower) - 1):
        # Check for double alphabetic characters
        if original_lower[i] == original_lower[i + 1] and original_lower[i].isalpha():
            # Candidate string with one of the double letters removed
            candidate = original_lower[:i] + original_lower[i+1:]
            if candidate == variation_lower:
                return True

    return False

def is_vowel_replaced(original: str, variation: str) -> bool:
    """Check if some vowels are replaced with different vowels"""
    vowels = 'aeiou'
    
    original_lower = original.lower()
    if not any(char in vowels for char in original_lower):
        return False

    variation_lower = variation.lower()

    if original_lower == variation_lower or len(original_lower) != len(variation_lower):
        return False

    vowel_changes = 0
    other_changes = 0
    
    for i in range(len(original_lower)):
        if original_lower[i] != variation_lower[i]:
            if original_lower[i] in vowels and variation_lower[i] in vowels:
                vowel_changes += 1
            else:
                other_changes += 1
    
    return vowel_changes >= 1 and other_changes <= 1

def is_consonant(char: str) -> bool:
    """Check if a character is a consonant (case-insensitive)"""
    vowels = 'aeiou'
    return char.isalpha() and char.lower() not in vowels

def is_adjacent_consonants_swapped(original: str, variation: str) -> bool:
    """Check if two adjacent consonants are swapped."""
    original_lower = original.lower()

    if not any(is_consonant(original_lower[i]) and is_consonant(original_lower[i+1]) for i in range(len(original_lower)-1)):
        return False
        
    variation_lower = variation.lower()

    if len(original_lower) != len(variation_lower) or original_lower == variation_lower:
        return False

    for i in range(len(original_lower) - 1):
        if is_consonant(original_lower[i]) and is_consonant(original_lower[i+1]):
            test_str = list(original_lower)
            test_str[i], test_str[i+1] = test_str[i+1], test_str[i]
            if "".join(test_str) == variation_lower:
                return True
    return False

def is_consonant_replaced(original: str, variation: str) -> bool:
    """Check if some consonants are replaced with different consonants"""
    original_lower = original.lower()
    if not any(is_consonant(char) for char in original_lower):
        return False

    variation_lower = variation.lower()

    if original_lower == variation_lower or len(original_lower) != len(variation_lower):
        return False
    
    consonant_changes = 0
    other_changes = 0
    
    for i in range(len(original_lower)):
        if original_lower[i] != variation_lower[i]:
            if is_consonant(original_lower[i]) and is_consonant(variation_lower[i]):
                consonant_changes += 1
            else:
                other_changes += 1
    
    return consonant_changes >= 1 and other_changes <= 1

def is_letters_swapped(original: str, variation: str) -> bool:
    """Check if some adjacent letters are swapped"""
    if len(original) != len(variation) or original == variation:
        return False
    
    diffs = []
    for i in range(len(original)):
        if original[i] != variation[i]:
            diffs.append(i)
    
    if len(diffs) != 2 or abs(diffs[0] - diffs[1]) != 1:
        return False
    
    return (original[diffs[0]] == variation[diffs[1]] and
            original[diffs[1]] == variation[diffs[0]])

def is_letter_removed(original: str, variation: str) -> bool:
    """Check if a letter is removed"""
    if len(variation) != len(original) - 1:
        return False
    
    return Levenshtein.distance(original, variation) == 1

def is_vowel_removed(original: str, variation: str) -> bool:
    """Check if a vowel is removed"""
    vowels = 'aeiou'
    original_lower = original.lower()
    if not any(c in vowels for c in original_lower):
        return False

    if len(variation) != len(original) - 1:
        return False
    
    variation_lower = variation.lower()
    for i, char in enumerate(original_lower):
        if char in vowels:
            test = original_lower[:i] + original_lower[i+1:]
            if test == variation_lower:
                return True
    
    return False

def is_consonant_removed(original: str, variation: str) -> bool:
    """Check if a consonant is removed"""
    original_lower = original.lower()
    if not any(is_consonant(c) for c in original_lower):
        return False

    if len(variation) != len(original) - 1:
        return False
    
    variation_lower = variation.lower()
    for i, char in enumerate(original_lower):
        if is_consonant(char):
            test = original_lower[:i] + original_lower[i+1:]
            if test == variation_lower:
                return True
    
    return False

def is_special_character_replaced(original: str, variation: str) -> bool:
    """Check if a special character is replaced with a different one"""
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    if not any(c in special_chars for c in original):
        return False

    if original == variation or len(original) != len(variation):
        return False
    
    special_changes = 0
    other_changes = 0
    
    for i in range(len(original)):
        if original[i] != variation[i]:
            if original[i] in special_chars and variation[i] in special_chars:
                special_changes += 1
            else:
                other_changes += 1
    
    return special_changes >= 1 and other_changes <= 1

def is_random_special_removed(original: str, variation: str) -> bool:
    """Check if a special character is removed"""
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    if not any(c in special_chars for c in original):
        return False

    if len(variation) != len(original) - 1:
        return False
    
    for i, char in enumerate(original):
        if char in special_chars:
            test = original[:i] + original[i+1:]
            if test == variation:
                return True
    
    return False

def is_title_removed(original: str, variation: str) -> bool:
    """Check if a title is removed from the name"""
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
    
    original_lower = original.lower()
    if not any(original_lower.startswith(title.lower() + " ") for title in titles):
        return False
    
    variation_lower = variation.lower()
    if original_lower == variation_lower:
        return False

    for title in titles:
        if original_lower.startswith(title.lower() + " "):
            test = original_lower[len(title)+1:]
            if test == variation_lower:
                return True
            if Levenshtein.distance(variation_lower, test) <= 2:
                return True
    
    return False

def is_name_abbreviated(original: str, variation: str) -> bool:
    """Check if name parts are abbreviated (simple heuristic)"""
    if original == variation:
        return False
    original_parts = original.split()
    variation_parts = variation.split()
    
    if len(original_parts) < 2 or len(original_parts) != len(variation_parts):
        return False
    
    for orig, var in zip(original_parts, variation_parts):
        if len(var) >= len(orig) or not orig.startswith(var):
            return False
    
    return True

def is_all_spaces_removed(original: str, variation: str) -> bool:
    """Check if all spaces are removed"""
    if ' ' not in original:
        return False
    
    return variation == original.replace(' ', '')

def is_letter_duplicated(original: str, variation: str) -> bool:
    """Check if a letter is duplicated"""
    if len(variation) != len(original) + 1:
        return False
    
    for i, char in enumerate(original):
        test = original[:i] + char + original[i:]
        if test == variation:
            return True
    
    return False

def is_random_letter_inserted(original: str, variation: str) -> bool:
    """Check if a random letter is inserted"""
    if len(variation) != len(original) + 1:
        return False
    
    return Levenshtein.distance(original, variation) == 1

def is_title_added(original: str, variation: str) -> bool:
    """Check if a title is added to the name"""
    if original == variation:
        return False
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
    
    variation_lower = variation.lower()
    original_lower = original.lower()

    for title in titles:
        title_lower = title.lower()
        if variation_lower.startswith(title_lower + " "):
             # Exact match
            if variation_lower == (title_lower + " " + original_lower):
                return True
            # Match with manipulation
            if Levenshtein.distance(original_lower, variation_lower[len(title_lower)+1:]) <= 2:
                return True
    
    return False

def is_suffix_added(original: str, variation: str) -> bool:
    """Check if a suffix is added to the name"""
    if original == variation:
        return False
    suffixes = ["Jr.", "Sr.", "III", "IV", "V", "PhD", "MD", "Esq.", "Jr", "Sr"]
    
    original_lower = original.lower()
    variation_lower = variation.lower()
    
    for suffix in suffixes:
        suffix_lower = suffix.lower()
        # Check if suffix was added at end
        if variation_lower == original_lower + " " + suffix_lower:
            return True
        # Check if suffix was added with some name manipulation
        if variation_lower.endswith(" " + suffix_lower) and Levenshtein.distance(original_lower, variation_lower[:-len(suffix_lower)-1]) <= 2:
            return True
    
    return False

def is_initials_only(original: str, variation: str) -> bool:
    """Check if the name is converted to initials"""
    if original == variation:
        return False
    parts = original.split()
    
    if len(parts) < 2:
        return False
    
    variation_lower = variation.lower()
    initials = ".".join([p[0] for p in parts]).lower() + "."
    initials_spaced = ". ".join([p[0] for p in parts]).lower() + "."
    initials_no_dots = "".join([p[0] for p in parts]).lower()
    
    return variation_lower in [initials, initials_spaced, initials_no_dots]

def is_name_parts_permutation(original: str, variation: str) -> bool:
    """Check if the name parts are permuted"""
    original_parts = original.split()
    variation_parts = variation.split()
    
    if len(original_parts) < 2 or len(original_parts) != len(variation_parts):
        return False

    return sorted(original_parts) == sorted(variation_parts) and original_parts != variation_parts

def is_first_name_initial(original: str, variation: str) -> bool:
    """Check if first name is reduced to initial"""
    parts = original.split()
    
    if len(parts) < 2:
        return False
    
    variation_lower = variation.lower()
    test_variation = (parts[0][0] + ". " + " ".join(parts[1:])).lower()
    test_variation2 = (parts[0][0] + "." + " ".join(parts[1:])).lower()
    
    return variation_lower in [test_variation, test_variation2]

# Map rule names to their evaluation functions
RULE_EVALUATORS = {
    "replace_spaces_with_random_special_characters": is_space_replaced_with_special_chars,
    "replace_double_letters_with_single_letter": is_double_letter_replaced,
    "replace_random_vowel_with_random_vowel": is_vowel_replaced,
    "replace_random_consonant_with_random_consonant": is_consonant_replaced,
    "replace_random_special_character_with_random_special_character": is_special_character_replaced,
    "swap_random_letter": is_letters_swapped,
    "swap_adjacent_consonants": is_adjacent_consonants_swapped,
    "swap_adjacent_syllables": is_letters_swapped,   # Simplified, would need more specific check
    "delete_random_letter": is_letter_removed,
    "remove_random_vowel": is_vowel_removed,
    "remove_random_consonant": is_consonant_removed,
    "remove_random_special_character": is_random_special_removed,
    "remove_title": is_title_removed,
    "remove_all_spaces": is_all_spaces_removed,
    "duplicate_random_letter_as_double_letter": is_letter_duplicated,
    "insert_random_letter": is_random_letter_inserted,
    "add_random_leading_title": is_title_added,
    "add_random_trailing_title": is_suffix_added,
    "shorten_name_to_initials": is_initials_only,
    "name_parts_permutations": is_name_parts_permutation,
    "initial_only_first_name": is_first_name_initial,
    "shorten_name_to_abbreviations": is_name_abbreviated
}

def evaluate_rule_compliance(
    original_name: str,
    variations: List[str],
    rules: List[str]
) -> Tuple[Dict[str, List[str]], float]:
    """
    Evaluate which variations comply with which rules
    
    Args:
        original_name: The original name
        variations: List of name variations
        rules: List of rule names to check against
        
    Returns:
        Tuple containing:
        - Dictionary mapping rules to lists of compliant variations
        - Compliance ratio (compliant variations / total variations)
    """
    if not variations or not rules:
        return {}, 0.0

    # Initialize the result dictionary
    compliant_variations = {rule: [] for rule in rules}
    all_compliant_variations = set()
    
    # For each variation, check if it complies with any of the rules
    for variation in variations:
        for rule in rules:
            if rule in RULE_EVALUATORS:
                try:
                    if RULE_EVALUATORS[rule](original_name, variation):
                        compliant_variations[rule].append(variation)
                        all_compliant_variations.add(variation)
                except Exception as e:
                    bt.logging.warning(f"Error evaluating rule {rule} for {variation}: {str(e)}")
    
    # Calculate the compliance ratio
    compliance_ratio = len(all_compliant_variations) / len(variations) if variations else 0.0
    
    return compliant_variations, compliance_ratio