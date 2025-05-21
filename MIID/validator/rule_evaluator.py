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
    if ' ' not in original:
        return False
    
    # Get positions of spaces in original
    space_positions = [i for i, char in enumerate(original) if char == ' ']
    
    # Check if variation has no spaces but has special chars at space positions
    if ' ' in variation:
        return False
    
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
    # Find double letters in original
    double_letters = []
    for i in range(len(original) - 1):
        if original[i] == original[i+1]:
            double_letters.append(original[i])
    
    if not double_letters:
        return False
    
    # Check if variation has one fewer character and is otherwise similar
    if len(variation) != len(original) - 1:
        return False
    
    # Check if Levenshtein distance is appropriate for this transformation
    lev_distance = Levenshtein.distance(original, variation)
    return 1 <= lev_distance <= 2

def is_vowel_replaced(original: str, variation: str) -> bool:
    """Check if some vowels are replaced with different vowels"""
    vowels = 'aeiou'
    
    # Check if length is the same
    if len(original) != len(variation):
        return False
    
    # Count changes in vowels
    vowel_changes = 0
    other_changes = 0
    
    for i in range(len(original)):
        if original[i] != variation[i]:
            if original[i] in vowels and variation[i] in vowels:
                vowel_changes += 1
            else:
                other_changes += 1
    
    # Should have at least one vowel change and few other changes
    return vowel_changes >= 1 and other_changes <= 1

def is_consonant_replaced(original: str, variation: str) -> bool:
    """Check if some consonants are replaced with different consonants"""
    vowels = 'aeiou'
    
    # Check if length is the same
    if len(original) != len(variation):
        return False
    
    # Count changes in consonants
    consonant_changes = 0
    other_changes = 0
    
    for i in range(len(original)):
        if original[i] != variation[i]:
            if original[i] not in vowels and variation[i] not in vowels:
                consonant_changes += 1
            else:
                other_changes += 1
    
    # Should have at least one consonant change and few other changes
    return consonant_changes >= 1 and other_changes <= 1

def is_letters_swapped(original: str, variation: str) -> bool:
    """Check if some adjacent letters are swapped"""
    # Check if length is the same
    if len(original) != len(variation):
        return False
    
    # Count differences and detect swaps
    diffs = []
    for i in range(len(original)):
        if original[i] != variation[i]:
            diffs.append(i)
    
    # Should have exactly 2 differences for a swap
    if len(diffs) != 2:
        return False
    
    # Check if differences are adjacent
    if abs(diffs[0] - diffs[1]) != 1:
        return False
    
    # Check if it's a swap
    return (original[diffs[0]] == variation[diffs[1]] and
            original[diffs[1]] == variation[diffs[0]])

def is_letter_removed(original: str, variation: str) -> bool:
    """Check if a letter is removed"""
    # Check if variation is one character shorter
    if len(variation) != len(original) - 1:
        return False
    
    # Use Levenshtein to check if it's a single character removal
    return Levenshtein.distance(original, variation) == 1

def is_vowel_removed(original: str, variation: str) -> bool:
    """Check if a vowel is removed"""
    vowels = 'aeiou'
    
    # Check if variation is one character shorter
    if len(variation) != len(original) - 1:
        return False
    
    # Try removing each vowel and see if we get the variation
    for i, char in enumerate(original):
        if char in vowels:
            test = original[:i] + original[i+1:]
            if test == variation:
                return True
    
    return False

def is_consonant_removed(original: str, variation: str) -> bool:
    """Check if a consonant is removed"""
    vowels = 'aeiou'
    
    # Check if variation is one character shorter
    if len(variation) != len(original) - 1:
        return False
    
    # Try removing each consonant and see if we get the variation
    for i, char in enumerate(original):
        if char not in vowels and char.isalpha():
            test = original[:i] + original[i+1:]
            if test == variation:
                return True
    
    return False

def is_all_spaces_removed(original: str, variation: str) -> bool:
    """Check if all spaces are removed"""
    if ' ' not in original:
        return False
    
    return variation == original.replace(' ', '')

def is_letter_duplicated(original: str, variation: str) -> bool:
    """Check if a letter is duplicated"""
    # Check if variation is one character longer
    if len(variation) != len(original) + 1:
        return False
    
    # Try duplicating each letter and see if we get the variation
    for i, char in enumerate(original):
        test = original[:i] + char + original[i:]
        if test == variation:
            return True
    
    return False

def is_random_letter_inserted(original: str, variation: str) -> bool:
    """Check if a random letter is inserted"""
    # Check if variation is one character longer
    if len(variation) != len(original) + 1:
        return False
    
    # Use Levenshtein to check if it's a single character insertion
    return Levenshtein.distance(original, variation) == 1

def is_title_added(original: str, variation: str) -> bool:
    """Check if a title is added to the name"""
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
    
    for title in titles:
        # Check if title was added at beginning
        if variation.startswith(title + " " + original):
            return True
        # Check if title was added with some name manipulation
        if variation.startswith(title + " ") and Levenshtein.distance(original, variation[len(title)+1:]) <= 2:
            return True
    
    return False

def is_suffix_added(original: str, variation: str) -> bool:
    """Check if a suffix is added to the name"""
    suffixes = ["Jr.", "Sr.", "III", "IV", "V", "PhD", "MD", "Esq.", "Jr", "Sr"]
    
    for suffix in suffixes:
        # Check if suffix was added at end
        if variation == original + " " + suffix:
            return True
        # Check if suffix was added with some name manipulation
        if variation.endswith(" " + suffix) and Levenshtein.distance(original, variation[:-len(suffix)-1]) <= 2:
            return True
    
    return False

def is_initials_only(original: str, variation: str) -> bool:
    """Check if the name is converted to initials"""
    parts = original.split()
    
    if len(parts) < 2:
        return False
    
    # Check if variation consists of initials
    initials = ".".join([p[0] for p in parts]) + "."
    initials_spaced = ". ".join([p[0] for p in parts]) + "."
    initials_no_dots = "".join([p[0] for p in parts])
    
    return variation in [initials, initials_spaced, initials_no_dots]

def is_name_parts_permutation(original: str, variation: str) -> bool:
    """Check if the name parts are permuted"""
    original_parts = original.split()
    variation_parts = variation.split()
    
    if len(original_parts) != len(variation_parts) or len(original_parts) < 2:
        return False
    
    # Check if variation parts are a permutation of original parts
    return sorted(original_parts) == sorted(variation_parts) and original_parts != variation_parts

def is_first_name_initial(original: str, variation: str) -> bool:
    """Check if first name is reduced to initial"""
    parts = original.split()
    
    if len(parts) < 2:
        return False
    
    # Compare with original first name as initial
    test_variation = parts[0][0] + ". " + " ".join(parts[1:])
    test_variation2 = parts[0][0] + "." + " ".join(parts[1:])
    
    return variation in [test_variation, test_variation2]

# Map rule names to their evaluation functions
RULE_EVALUATORS = {
    "replace_spaces_with_random_special_characters": is_space_replaced_with_special_chars,
    "replace_double_letters_with_single_letter": is_double_letter_replaced,
    "replace_random_vowel_with_random_vowel": is_vowel_replaced,
    "replace_random_consonant_with_random_consonant": is_consonant_replaced,
    "swap_random_letter": is_letters_swapped,
    "swap_adjacent_consonants": is_letters_swapped,  # Simplified, would need more specific check
    "swap_adjacent_syllables": is_letters_swapped,   # Simplified, would need more specific check
    "delete_random_letter": is_letter_removed,
    "remove_random_vowel": is_vowel_removed,
    "remove_random_consonant": is_consonant_removed,
    "remove_all_spaces": is_all_spaces_removed,
    "duplicate_random_letter_as_double_letter": is_letter_duplicated,
    "insert_random_letter": is_random_letter_inserted,
    "add_random_leading_title": is_title_added,
    "add_random_trailing_title": is_suffix_added,
    "shorten_name_to_initials": is_initials_only,
    "name_parts_permutations": is_name_parts_permutation,
    "initial_only_first_name": is_first_name_initial
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