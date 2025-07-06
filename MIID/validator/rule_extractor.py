# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 YANEZ - MIID Team

import random
from typing import List, Dict, Tuple, Callable, Any


RULE_FUNCTIONS = {
    "character_replacement": [
        #"replace_spaces_with_random_special_characters", # need labels for this
        "replace_double_letters_with_single_letter", 
        "replace_random_vowel_with_random_vowel", # nice to have labels for this
        "replace_random_consonant_with_random_consonant", # nice to have labels for this
        #"replace_random_special_character_with_random_special_character"
    ],
    "character_swapping": [
        "swap_adjacent_consonants",
        "swap_adjacent_syllables",
        "swap_random_letter"
    ],
    "character_removal": [
        "delete_random_letter",
        "remove_random_vowel",
        "remove_random_consonant",
        #"remove_random_special_character", # need labels for this
        #"remove_title", # need labels for this
        #"remove_all_spaces" # need labels for this
    ],
    "character_insertion": [
        "duplicate_random_letter_as_double_letter",
        "insert_random_letter",
        #"add_random_leading_title", # need labels for this
        #"add_random_trailing_title" # need labels for this
    ],
    #"name_formatting": [
        #"initial_only_first_name", 
        #"shorten_name_to_initials", 
        #"shorten_name_to_abberivations" # need labels for this
    #],
    #"structure_change": [
        #"name_parts_permutations"
    #]
}

# Map rule names to their descriptions for inclusion in queries
RULE_DESCRIPTIONS = {
    #"replace_spaces_with_random_special_characters": "Replace spaces with special characters",
    "replace_double_letters_with_single_letter": "Replace double letters with a single letter",
    "replace_random_vowel_with_random_vowel": "Replace random vowels with different vowels",
    "replace_random_consonant_with_random_consonant": "Replace random consonants with different consonants",
    #"replace_random_special_character_with_random_special_character": "Replace special characters with different ones",
    "swap_adjacent_consonants": "Swap adjacent consonants",
    "swap_adjacent_syllables": "Swap adjacent syllables",
    "swap_random_letter": "Swap random adjacent letters",
    "delete_random_letter": "Delete a random letter",
    "remove_random_vowel": "Remove a random vowel",
    "remove_random_consonant": "Remove a random consonant",
    #"remove_random_special_character": "Remove a random special character",
    #"remove_title": "Remove title (Mr., Dr., etc.)",
    #"remove_all_spaces": "Remove all spaces",
    "duplicate_random_letter_as_double_letter": "Duplicate a random letter",
    "insert_random_letter": "Insert a random letter",
    #"add_random_leading_title": "Add a title prefix (Mr., Dr., etc.)",
    #"add_random_trailing_title": "Add a title suffix (Jr., PhD, etc.)",
    #"initial_only_first_name": "Use first name initial with last name",
    #"shorten_name_to_initials": "Convert name to initials",
    #"shorten_name_to_abberivations": "Abbreviate name parts",
    #"name_parts_permutations": "Reorder name parts"
}

def get_all_rule_categories() -> List[str]:
    """Get all available rule categories"""
    return list(RULE_FUNCTIONS.keys())

def get_rules_by_category(category: str) -> List[str]:
    """Get all rules in a specific category"""
    return RULE_FUNCTIONS.get(category, [])

def get_random_rules(n_rules: int = 3) -> List[str]:
    """
    Get a random selection of rules across categories
    
    Args:
        n_rules: Number of rules to select
        
    Returns:
        List of rule function names
    """
    all_rules = []
    for rules in RULE_FUNCTIONS.values():
        all_rules.extend(rules)
    
    # Ensure we don't ask for more rules than exist
    n_rules = min(n_rules, len(all_rules))
    
    return random.sample(all_rules, n_rules)

def get_rule_description(rule_name: str) -> str:
    """Get a human-readable description of a rule for query templates"""
    return RULE_DESCRIPTIONS.get(rule_name, f"Apply the {rule_name} transformation")

def format_rules_for_query(rules: List[str]) -> str:
    """Format a list of rules into a natural language instruction"""
    if not rules:
        return ""
    
    descriptions = [get_rule_description(rule) for rule in rules]
    
    if len(descriptions) == 1:
        return f"Additionally, generate variations that: {descriptions[0]}."
    
    formatted_rules = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
    return f"Additionally, generate variations that perform these transformations: {formatted_rules}."

def get_rule_template_and_metadata(rule_percentage: int = 30) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a template for rule-based variations and metadata to track them
    
    Args:
        rule_percentage: Percentage of variations that should follow rules
        
    Returns:
        Tuple containing rule template text and metadata dict
    """
    # Select 1-3 random rules
    n_rules = random.randint(1, 3)
    selected_rules = get_random_rules(n_rules)
    
    rule_template = format_rules_for_query(selected_rules)
    
    metadata = {
        "rule_percentage": rule_percentage,
        "selected_rules": selected_rules,
        "rule_descriptions": {rule: get_rule_description(rule) for rule in selected_rules}
    }
    
    return rule_template, metadata 