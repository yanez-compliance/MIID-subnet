# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np
from typing import List, Dict, Tuple, Any, Set
import bittensor as bt
import Levenshtein
import jellyfish
import os
import csv
import time
import traceback
import math
import random
import ollama
from datetime import datetime, timedelta
import re
import requests
from unidecode import unidecode
import geonamescache

# Import rule_evaluator for rule-based compliance checking
from MIID.validator.rule_evaluator import evaluate_rule_compliance
from MIID.validator.cheat_detection import (
    build_normalized_set,
    pairwise_similarity_metrics,
    hash_signature,
    overlap_coefficient,
    jaccard,
    detect_cheating_patterns,
)

def clean_transliteration_output(raw_response: str) -> str:
    """
    Extracts the transliterated name from LLM output, removes appended instructions,
    and any leading/trailing punctuation like '-'.
    """
    lines = raw_response.splitlines()
    transliterated = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip instruction/meta lines
        if any(keyword in line.lower() for keyword in [
            "output only", "do not", "end of output", "input", "latin script"
        ]):
            continue
        if re.search(r"[A-Za-zÀ-ÿ]", line):
            transliterated = line
            break

    # Remove known script words (case-insensitive)
    for word in ["latin", "cyrillic", "arabic", "chinese"]:
        transliterated = re.sub(rf"\b{word}\b", "", transliterated, flags=re.IGNORECASE)

    # Remove unwanted characters, keep letters, hyphens, apostrophes, spaces
    transliterated = re.sub(r"[^A-Za-zÀ-ÿ\s\-\']", "", transliterated)

    # Strip leading/trailing punctuation (like '-')
    transliterated = transliterated.strip(" -")

    # Collapse multiple spaces
    transliterated = re.sub(r"\s+", " ", transliterated).strip()
    
    return transliterated


def translate_unidecode(original_name):
    """
    Fallback transliteration function using unidecode.
    Converts to ASCII.
    
    Args:
        original_name: The original name to transliterate
        
    Returns:
        The transliterated name in ASCII
    """
    english_name = unidecode(original_name)
    return english_name


# Define the reward component weights globally
MIID_REWARD_WEIGHTS = {
    ##### Quality based similarity weights (phonetic and orthographic similarity)
    "similarity_weight": 0.60,       # Combined weight for phonetic and orthographic similarity (reduced from 0.60)
    "count_weight": 0.15,            # Weight for having the correct number of variations (reduced from 0.15)
    "uniqueness_weight": 0.10,       # Weight for having unique variations (unchanged)
    "length_weight": 0.15,           # Weight for reasonable length variations (reduced from 0.15)
    ##### Rule compliance weight
    "rule_compliance_weight": 0.20,  # NEW: Weight for rule-based compliance
    ##### First/last name weights
    # Weights for combining first/last name scores in calculate_variation_quality
    "first_name_weight": 0.3, 
    "last_name_weight": 0.7
}

def reward(query: int, response: int) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    bt.logging.info(
        f"In rewards, query val: {query}, response val: {response}, rewards val: {1.0 if response == query * 2 else 0}"
    )
    return 1.0 if response == query * 2 else 0


def calculate_phonetic_similarity(original_name: str, variation: str) -> float:
    """
    Calculate phonetic similarity between two strings using a randomized subset of phonetic algorithms.
    This makes it harder for miners to game the system by not knowing which algorithms will be used.
    The selection and weighting are deterministic for each original_name.

    We randomize the subset and weights of multiple phonetic algorithms (Soundex, Metaphone, NYSIIS)
    to reduce overfitting to any single encoding. Randomness is deterministically seeded per interpreter
    session using Python’s salted hash of `original_name`, which yields:
      • Reproducibility for the same name within a single run (same selection/weights each time)
      • Variation across different runs (fresh selections/weights after interpreter restart)

    This per-run determinism and cross-run variability are intentional to balance auditability and
    anti-gaming. If strict cross-run reproducibility becomes a requirement, we can switch to a stable
    digest (e.g., SHA-256) and/or a local RNG seeded from that digest.

    """
    # Define available phonetic algorithms
    algorithms = {
        "soundex": lambda x, y: jellyfish.soundex(x) == jellyfish.soundex(y),
        "metaphone": lambda x, y: jellyfish.metaphone(x) == jellyfish.metaphone(y),
        "nysiis": lambda x, y: jellyfish.nysiis(x) == jellyfish.nysiis(y),
        # Add more algorithms if needed
    }

    # Deterministically seed the random selection based on the original name
    random.seed(hash(original_name) % 10000)
    selected_algorithms = random.sample(list(algorithms.keys()), k=min(3, len(algorithms)))

    # Generate random weights that sum to 1.0
    weights = [random.random() for _ in selected_algorithms]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate the weighted phonetic score
    phonetic_score = sum(
        algorithms[algo](original_name, variation) * weight
        for algo, weight in zip(selected_algorithms, normalized_weights)
    )

    return float(phonetic_score)

def calculate_orthographic_similarity(original_name: str, variation: str) -> float:
    """
    Calculate orthographic similarity between two strings using Levenshtein distance.
    
    Args:
        original_name: The original name
        variation: The variation to compare against
        
    Returns:
        Orthographic similarity score between 0 and 1
    """
    try:
        # Use Levenshtein distance to compare
        distance = Levenshtein.distance(original_name, variation)
        max_len = max(len(original_name), len(variation))
        
        # Calculate orthographic similarity score (0-1)
        return 1.0 - (distance / max_len)
    except Exception as e:
        bt.logging.warning(f"Error calculating orthographic score: {str(e)}")
        return 0.0

def looks_like_address(address: str) -> bool:
    address = address.strip().lower()

    address_len = address.strip().replace(" ", "").replace(",", "")
    if len(address_len) < 25:
        return False
    if len(address_len) > 300:  # maximum length check
        return False

    if re.match(r"^[^a-zA-Z]*$", address):  # no letters at all
        return False
    if len(set(address)) < 5:  # all chars basically the same
        return False
        
    # Has at least one digit (street number)
    number_groups = re.findall(r"\d+", address)
    if len(number_groups) < 2:
        return False

    if address.count(",") < 2:
        return False
    
    # # Contains common address words or patterns
    # common_words = ["st", "street", "rd", "road", "ave", "avenue", "blvd", "boulevard", "drive", "ln", "lane", "plaza", "city", "platz", "straße", "straße", "way", "place", "square", "allee", "allee", "gasse", "gasse"]
    # # Also check for common patterns like "1-1-1" (Japanese addresses) or "Unter den" (German)
    # has_common_word = any(word in address for word in common_words)
    # has_address_pattern = re.search(r'\d+-\d+-\d+', address) or re.search(r'unter den|marienplatz|champs|place de', address)
    
    # if not (has_common_word or has_address_pattern):
    #     return False
    
    return True

def check_with_nominatim(address: str) -> bool:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json"}
        response = requests.get(url, params=params, headers={"User-Agent": "address-checker"}, timeout=5)
        return len(response.json()) > 0
    except requests.exceptions.Timeout:
        bt.logging.warning(f"API timeout for address: {address}")
        return "TIMEOUT"
    except:
        return False

def mapbox_verify_address(address: str) -> bool:
    """
    Return True if Mapbox confirms this is a valid, real-world address.
    Returns False if relevance is above 0.6 (indicating low confidence).
    """
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {"access_token": 'pk.eyJ1Ijoib21hci1hYmF6YSIsImEiOiJjbWdyMGRpZTAzMnAyMmxwcml0ZDR2OXdsIn0.04pTyGweTPBN2tRZ9XUFCA', "limit": 1, "autocomplete": "false"}
    headers = {"User-Agent": "address-verifier"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return False

        result = features[0]
        relevance = result.get("relevance", 0)
        bt.logging.info(f"Mapbox relevance for {address}: {relevance}")
        
        # If relevance is above 0.6, consider it a fail (low confidence)
        if relevance < 0.6:
            return False
            
        return True

    except requests.exceptions.Timeout:
        bt.logging.warning(f"Mapbox API timeout for address: {address}")
        return "TIMEOUT"
    except Exception as e:
        bt.logging.warning(f"Mapbox API error for address {address}: {str(e)}")
        return False


def extract_city_country(address: str) -> tuple:
    """
    Extract city and country from an address.
    Country is always the last part.
    City is found by checking each section from right to left (excluding country)
    and validating against geonames data to ensure it's a real city in the country.
    
    Examples:
    - "115 New Cavendish Street, London W1T 5DU, United Kingdom" -> ("London", "United Kingdom")
    - "223 William Street, Melbourne VIC 3000, Australia" -> ("Melbourne", "Australia")
    - "Rosenthaler Straße 1, 10119 Berlin, Germany" -> ("Berlin", "Germany")
    - "3 Upper Alma Road, Rosebank, Cape Town, 7700, South Africa" -> ("Cape Town", "South Africa")
    - "6 , Yemen" -> ("", "Yemen")  # No valid city found
    
    Args:
        address: The address to extract from
        
    Returns:
        Tuple of (city, country) - both strings, empty if not found
    """
    if not address:
        return "", ""

    address = address.lower()
    
    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 2:
        return "", ""
    
    country = parts[-1]
    
    # If no country found, return empty
    if not country:
        return "", ""

    # Check each section from right to left (excluding the country)
    for i in range(2, len(parts) + 1):
        candidate_index = -i
        if abs(candidate_index) > len(parts):
            break
        
        candidate_part = parts[candidate_index]
        if not candidate_part:
            continue
            
        words = candidate_part.split()
        
        # Try different combinations of words (1-2 words max)
        # Start with 2 words, then 1 word for better city matching
        for num_words in range(len(words)):
            current_word = words[num_words]

            # Try current word
            candidates = [current_word]

            # Also try current + previous (if exists)
            if num_words > 0:
                prev_plus_current = words[num_words - 1] + " " + words[num_words]
                candidates.append(prev_plus_current)

            for city_candidate in candidates:
                # Skip if contains numbers or is too short
                if any(char.isdigit() for char in city_candidate):
                    continue

                # Validate the city exists in the country
                if city_in_country(city_candidate, country):
                    return city_candidate, country

    return "", country

# Global cache for geonames data to avoid reloading
_geonames_cache = None
_cities_data = None
_countries_data = None

def get_geonames_data():
    """Get cached geonames data, loading it only once."""
    global _geonames_cache, _cities_data, _countries_data
    
    if _geonames_cache is None:
        bt.logging.info("Loading geonames data for the first time...")
        start_time = time.time()
        _geonames_cache = geonamescache.GeonamesCache()
        _cities_data = _geonames_cache.get_cities()
        _countries_data = _geonames_cache.get_countries()
        end_time = time.time()
        bt.logging.info(f"Geonames data loaded in {end_time - start_time:.2f} seconds")
    
    return _cities_data, _countries_data

def city_in_country(city_name: str, country_name: str) -> bool:
    """
    Check if a city is actually in the specified country using geonamescache.
    
    Args:
        city_name: Name of the city
        country_name: Name of the country
        
    Returns:
        True if city is in country, False otherwise
    """
    if not city_name or not country_name:
        return False
    
    try:
        cities, countries = get_geonames_data()
        
        city_name_lower = city_name.lower()
        country_name_lower = country_name.lower()
        
        # Find country code
        country_code = None
        for code, data in countries.items():
            if data.get('name', '').lower() == country_name_lower:
                country_code = code
                break
        
        if not country_code:
            return False
        
        # Only check cities that are actually in the specified country
        city_words = city_name_lower.split()
        
        for city_id, city_data in cities.items():
            # Skip cities not in the target country
            if city_data.get("countrycode", "") != country_code:
                continue
                
            city_data_name = city_data.get("name", "").lower()
            
            # Check exact match first
            if city_data_name == city_name_lower:
                return True
            # Check first word match
            elif len(city_words) >= 2 and city_data_name.startswith(city_words[0]):
                return True
            # Check second word match
            elif len(city_words) >= 2 and city_words[1] in city_data_name:
                return True
        
        return False
        
    except Exception as e:
        bt.logging.warning(f"Error checking city '{city_name}' in country '{country_name}': {str(e)}")
        return False

def validate_address_region(generated_address: str, seed_address: str) -> bool:
    """
    Validate that generated address has correct region from seed address.
    
    Args:
        generated_address: The generated address to validate
        seed_address: The seed address to match against
        
    Returns:
        True if region is valid, False otherwise
    """
    if not generated_address or not seed_address:
        return False
    
    # Extract city and country from both addresses
    gen_city, gen_country = extract_city_country(generated_address)
    seed_city, seed_country = seed_address.lower(), seed_address.lower()
    
    # If no city was extracted from generated address, it's an error
    if not gen_city:
        return False
    
    # If no country was extracted from generated address, it's an error
    if not gen_country:
        return False
    
    # Check if either city or country matches
    city_match = False
    if gen_city and seed_city:
        gen_words = gen_city.split()
        
        # Check exact match first
        if gen_city == seed_city:
            city_match = True
        # Check first word match
        elif len(gen_words) >= 1 and gen_words[0] in seed_city:
            city_match = True
        # Check second word match
        elif len(gen_words) >= 2 and gen_words[1] in seed_city:
            city_match = True
        # Check both words together
        elif len(gen_words) >= 2 and gen_words[0] in seed_city and gen_words[1] in seed_city:
            city_match = True
    
    country_match = gen_country and seed_country and gen_country == seed_country
    
    if not (city_match or country_match):
        return False
    
    # If we have both city and country, validate city is in country
    if gen_city and gen_country:
        return city_in_country(gen_city, gen_country)
    
    return True

def transliterate_name_with_llm(original_name: str, script: str, model_name: str = "tinyllama:latest") -> str:
    """
    Use LLM to transliterate a non-Latin name to Latin script for phonetic comparison.
    Tries tinyllama first, then falls back to llama3.1:latest.
    
    Args:
        original_name: The original non-Latin name to transliterate
        model_name: The Ollama model to use for transliteration (if specified, uses that first)
        
    Returns:
        The transliterated name in Latin script, or fallback transliteration if all models fail
    """
    llama_models = [
        "tinyllama:latest",
        "llama3.1:latest",
    ]

    if model_name in llama_models:
        models_to_try = [model_name] + [m for m in llama_models if m != model_name]
    else:
        models_to_try = llama_models

    for current_model in models_to_try:
        attempts = 0
        while attempts < 5:
            try:
                prompt = f"Transliterate this {script} name to Latin script, output only the name:\n{original_name}"

                response = ollama.generate(
                    model=current_model,
                    prompt=prompt,
                    options={
                        'temperature': 0.1,
                        'top_p': 0.9,
                        'max_tokens': 50,
                        'timeout': 30
                    }
                )

                raw_output = response['response'].strip()

                transliterated = clean_transliteration_output(raw_output)

                # Validate transliteration: must contain at least one letter
                if transliterated and re.match(r"^[A-Za-zÀ-ÿ\s\-\']+$", transliterated):
                    bt.logging.info(f"Successfully transliterated '{original_name}' to '{transliterated}' using {current_model}")
                    return transliterated
                else:
                    bt.logging.warning(f"Invalid transliteration result for '{original_name}' with {current_model}, attempt {attempts + 1}/5")
                    attempts += 1

            except Exception as e:
                bt.logging.error(f"Error in LLM transliteration for '{original_name}' with {current_model}, attempt {attempts + 1}/5: {str(e)}")
                attempts += 1

        bt.logging.warning(f"Model {current_model} failed 5 times for '{original_name}', trying next model")

    # Fallback
    bt.logging.info(f"All LLM attempts failed for '{original_name}', using fallback transliteration")
    fallback_result = translate_unidecode(original_name)
    bt.logging.info(f"Fallback transliteration result for '{original_name}': '{fallback_result}'")
    return fallback_result


def calculate_part_score(
    original_part: str,
    variations: List[str],
    phonetic_similarity: Dict[str, float],
    orthographic_similarity: Dict[str, float],
    expected_count: int
) -> Tuple[float, Dict]:
    """Calculate score and detailed metrics for a single part (first or last name)"""
    # bt.logging.info(f"\nCalculating part score for: {original_part}")
    # bt.logging.info(f"Number of variations: {len(variations)}")
    # bt.logging.info(f"Expected count: {expected_count}")
    
    if not variations:
        bt.logging.warning("No variations provided")
        return 0.0, {}
    
    # Define the boundaries for each similarity level with no overlaps
    # There is a gap so no code can be in 2 different bounds
    phonetic_boundaries = {
        "Light": (0.80, 1.00),   # High similarity range
        "Medium": (0.60, 0.79),  # Moderate similarity range
        "Far": (0.30, 0.59)      # Low similarity range
    }
    
    orthographic_boundaries = {
        "Light": (0.70, 1.00),   # High similarity range
        "Medium": (0.50, 0.69),  # Moderate similarity range
        "Far": (0.20, 0.49)      # Low similarity range
    }
    
    # 1. Check if count matches expected count with adaptive tolerance
    # Handle case where expected_count is 0 (100% rule-based scenario)
    if expected_count == 0:
        # If no variations are expected for non-rule-compliant part, give full score
        count_score = 1.0
        bt.logging.info(f"Count score: 1.0 (no non-rule variations expected)")
    else:
        # Tolerance increases with expected count to be more forgiving for larger sets
        base_tolerance = 0.2  # 20% base tolerance
        tolerance = base_tolerance + (0.05 * (expected_count // 10))  # Add 5% per 10 expected variations
        tolerance = min(tolerance, 0.4)  # Cap at 40% maximum tolerance
        
        tolerance_range = expected_count * tolerance
        actual_count = len(variations)
        lower_bound = max(1, expected_count - tolerance_range)  # Ensure at least 1 variation required
        upper_bound = expected_count + tolerance_range
        
        if lower_bound <= actual_count <= upper_bound:
            count_score = 1.0
            #bt.logging.info(f"Count score: 1.0 (within tolerance range: {lower_bound:.1f}-{upper_bound:.1f})")
        else:
            if actual_count < lower_bound:
                deviation = lower_bound - actual_count
                #bt.logging.warning(f"Too few variations: {actual_count} < {lower_bound:.1f}")
            else:
                deviation = actual_count - upper_bound
                #bt.logging.warning(f"Too many variations: {actual_count} > {upper_bound:.1f}")
            
            # Smoother penalty curve using exponential decay
            count_score = math.exp(-deviation / expected_count)
            #bt.logging.info(f"Count score: {count_score:.3f} (penalty for deviation: {deviation})")
    
    # 2. Enhanced uniqueness check with similarity clustering
    unique_variations = []
    for var in variations:
        # Check if this variation is too similar to any existing unique variation
        is_unique = True
        for unique_var in unique_variations:
            combined_similarity = (
                calculate_phonetic_similarity(var, unique_var) * 0.7 +
                calculate_orthographic_similarity(var, unique_var) * 0.3
            )
            if combined_similarity > 0.99:  # Very high similarity threshold
                is_unique = False
                #bt.logging.warning(f"Variation '{var}' is too similar to existing variation '{unique_var}'")
                break
        if is_unique:
            unique_variations.append(var)
    
    uniqueness_score = len(unique_variations) / len(variations) if variations else 0
    # if uniqueness_score < 1.0:
    #     bt.logging.warning(f"Found similar variations. Uniqueness score: {uniqueness_score:.3f}")
    # else:
        #bt.logging.info("All variations are sufficiently unique. Uniqueness score: 1.0")
    
    # 3. Improved length reasonableness with adaptive thresholds
    length_scores = []
    for var in unique_variations:
        original_len = len(original_part)
        var_len = len(var)
        
        # Adaptive threshold based on original name length
        min_ratio = 0.6 if original_len <= 5 else 0.7  # More forgiving for short names
        
        # Consider both absolute and relative length differences
        length_ratio = min(var_len / original_len, original_len / var_len)
        absolute_diff = abs(var_len - original_len)
        
        # Combine both metrics with smooth transition
        length_score = length_ratio * (1.0 - min(1.0, absolute_diff / original_len))
        length_scores.append(length_score)
        
        # if length_score < min_ratio:
        #     bt.logging.warning(
        #         f"Variation '{var}' (len={var_len}) has poor length score {length_score:.3f} "
        #         f"compared to original '{original_part}' (len={original_len})"
        #     )
    
    length_score = sum(length_scores) / len(length_scores) if length_scores else 0
    #bt.logging.info(f"Average length score: {length_score:.3f}")
    
    # Calculate similarity scores with improved distribution analysis
    phonetic_scores = []
    orthographic_scores = []
    
    for variation in unique_variations:
        p_score = calculate_phonetic_similarity(original_part, variation)
        o_score = calculate_orthographic_similarity(original_part, variation)
        
        phonetic_scores.append(p_score)
        orthographic_scores.append(o_score)
        
        # if p_score < 0.3 and o_score < 0.3:
        #     bt.logging.warning(
        #         f"Very low similarity for variation '{variation}': "
        #         f"phonetic={p_score:.3f}, orthographic={o_score:.3f}"
        #     )
    
    # Sort scores for distribution analysis
    phonetic_scores.sort()
    orthographic_scores.sort()
    
    # Calculate quality scores with improved distribution matching
    def calculate_distribution_quality(scores, boundaries, targets):
        quality = 0.0
        total_matched = 0
        
        for level, (lower, upper) in boundaries.items():
            target_percentage = targets.get(level, 0.0)
            if target_percentage == 0.0:
                continue
                
            # Count scores in this range
            count = sum(1 for score in scores if lower <= score <= upper)
            target_count = int(target_percentage * len(scores))
            
            if target_count > 0:
                # Calculate match quality with diminishing returns
                match_ratio = count / target_count
                #match_quality = 1.0 - math.exp(-match_ratio)  # Smooth curve
                # Diminishing returns after target
                # this gives 100% at target, then diminishing returns for exceeding
                if match_ratio <= 1.0:
                    match_quality = match_ratio  # Linear up to target
                else:    
                    match_quality = 1.0 - math.exp(-(match_ratio - 1.0))  
                quality += target_percentage * match_quality
                total_matched += count
                
                # bt.logging.info(
                #     f"{level} similarity: {count}/{target_count} variations "
                #     f"({match_quality:.3f} quality)"
                # )
        
        # Penalize unmatched variations
        unmatched = len(scores) - total_matched
        if unmatched > 0:
            penalty = 0.1 * (unmatched / len(scores))
            quality = max(0.0, quality - penalty)
            # bt.logging.warning(f"Penalty of {penalty:.3f} applied for {unmatched} unmatched variations")
        
        return quality
    
    phonetic_quality = calculate_distribution_quality(
        phonetic_scores, phonetic_boundaries, phonetic_similarity
    )
    orthographic_quality = calculate_distribution_quality(
        orthographic_scores, orthographic_boundaries, orthographic_similarity
    )
    
    # Calculate combined similarity score
    similarity_score = (phonetic_quality + orthographic_quality) / 2  # Average of both similarities
    #bt.logging.info(f"Similarity score: {similarity_score:.3f} (phonetic: {phonetic_quality:.3f}, orthographic: {orthographic_quality:.3f})")
    
    # Apply minimum similarity threshold to prevent gaming
    # If similarity is very low, severely reduce the score
    min_similarity_threshold = 0.2
    if similarity_score < min_similarity_threshold:
        #bt.logging.warning(f"Very low similarity score ({similarity_score:.3f}) detected - applying penalty")
        # Penalize low similarity scores
        similarity_score *= 0.1  # Keep only 10% of the similarity score
        #bt.logging.warning(f"Adjusted similarity score: {similarity_score:.3f}")
    #########################################################
    ### move this to config file
    # Calculate final quality score with all factors - updated weights from analysis
    # Use the globally defined weights
    similarity_weight = MIID_REWARD_WEIGHTS["similarity_weight"]
    count_weight = MIID_REWARD_WEIGHTS["count_weight"]
    uniqueness_weight = MIID_REWARD_WEIGHTS["uniqueness_weight"]
    length_weight = MIID_REWARD_WEIGHTS["length_weight"]
    
    final_score = (
        similarity_weight * similarity_score +
        count_weight * count_score +
        uniqueness_weight * uniqueness_score +
        length_weight * length_score
    )
    
    # # DETAILED LOGGING FOR DEBUGGING 0.0 SCORES
    # bt.logging.info(f"--- DETAILED SCORE CALCULATION FOR '{original_part}' ---")
    # bt.logging.info(f"Similarity Score: {similarity_score:.4f} (Weight: {similarity_weight}) -> Contributes: {similarity_weight * similarity_score:.4f}")
    # bt.logging.info(f"Count Score: {count_score:.4f} (Weight: {count_weight}) -> Contributes: {count_weight * count_score:.4f}")
    # bt.logging.info(f"Uniqueness Score: {uniqueness_score:.4f} (Weight: {uniqueness_weight}) -> Contributes: {uniqueness_weight * uniqueness_score:.4f}")
    # bt.logging.info(f"Length Score: {length_score:.4f} (Weight: {length_weight}) -> Contributes: {length_weight * length_score:.4f}")
    # bt.logging.info(f"FINAL PART SCORE for '{original_part}': {final_score:.4f}")
    # bt.logging.info(f"--- END DETAILED CALCULATION ---")
    
    if final_score == 0:
        bt.logging.warning(f"Zero score for '{original_part}'. Possible reasons:")
        if len(variations) == 0:
            bt.logging.warning("  - No variations provided")
        if similarity_score == 0:
            bt.logging.warning("  - All variations had very low similarity scores")
        if uniqueness_score == 0:
            bt.logging.warning("  - All variations were too similar to each other")
    
    # Just before returning the final_score, prepare the detailed metrics
    detailed_metrics = {
        "similarity": {
            "phonetic": float(phonetic_quality),
            "orthographic": float(orthographic_quality),
            "combined": float(similarity_score)
        },
        "count": {
            "actual": actual_count,
            "expected": expected_count,
            "score": float(count_score)
        },
        "uniqueness": {
            "unique_count": len(unique_variations),
            "total_count": len(variations),
            "score": float(uniqueness_score)
        },
        "length": {
            "score": float(length_score)
        },
        "variations": [{
            "variation": var,
            "phonetic_score": float(calculate_phonetic_similarity(original_part, var)),
            "orthographic_score": float(calculate_orthographic_similarity(original_part, var)),
            "length_ratio": float(len(var)) / float(len(original_part))
        } for var in variations]
    }
    
    return final_score, detailed_metrics


def calculate_part_score_phonetic_only(
    original_part: str,
    variations: List[str],
    phonetic_similarity: Dict[str, float],
    expected_count: int
) -> Tuple[float, Dict]:
    """Calculate score and detailed metrics for a single part (first or last name) using only phonetic similarity"""
    
    if not variations:
        bt.logging.warning("No variations provided")
        return 0.0, {}
    
    # Define the boundaries for phonetic similarity levels with no overlaps
    # There is a gap so no code can be in 2 different bounds
    phonetic_boundaries = {
        "Light": (0.80, 1.00),   # High similarity range
        "Medium": (0.60, 0.79),  # Moderate similarity range
        "Far": (0.30, 0.59)      # Low similarity range
    }
    
    # 1. Check if count matches expected count with adaptive tolerance
    # Handle case where expected_count is 0 (100% rule-based scenario)
    if expected_count == 0:
        # If no variations are expected for non-rule-compliant part, give full score
        count_score = 1.0
        bt.logging.info(f"Count score: 1.0 (no non-rule variations expected)")
    else:
        # Tolerance increases with expected count to be more forgiving for larger sets
        base_tolerance = 0.2  # 20% base tolerance
        tolerance = base_tolerance + (0.05 * (expected_count // 10))  # Add 5% per 10 expected variations
        tolerance = min(tolerance, 0.4)  # Cap at 40% maximum tolerance
        
        tolerance_range = expected_count * tolerance
        actual_count = len(variations)
        lower_bound = max(1, expected_count - tolerance_range)  # Ensure at least 1 variation required
        upper_bound = expected_count + tolerance_range
        
        if lower_bound <= actual_count <= upper_bound:
            count_score = 1.0
            #bt.logging.info(f"Count score: 1.0 (within tolerance range: {lower_bound:.1f}-{upper_bound:.1f})")
        else:
            if actual_count < lower_bound:
                deviation = lower_bound - actual_count
                #bt.logging.warning(f"Too few variations: {actual_count} < {lower_bound:.1f}")
            else:
                deviation = actual_count - upper_bound
                #bt.logging.warning(f"Too many variations: {actual_count} > {upper_bound:.1f}")
            
            # Smoother penalty curve using exponential decay
            count_score = math.exp(-deviation / expected_count)
            #bt.logging.info(f"Count score: {count_score:.3f} (penalty for deviation: {deviation})")
    
    # 2. Enhanced uniqueness check with phonetic similarity clustering
    unique_variations = []
    for var in variations:
        # Check if this variation is too similar to any existing unique variation
        is_unique = True
        for unique_var in unique_variations:
            phonetic_sim = calculate_phonetic_similarity(var, unique_var)
            if phonetic_sim > 0.99:  # Very high similarity threshold
                is_unique = False
                #bt.logging.warning(f"Variation '{var}' is too similar to existing variation '{unique_var}'")
                break
        if is_unique:
            unique_variations.append(var)
    
    uniqueness_score = len(unique_variations) / len(variations) if variations else 0
    
    # 3. Improved length reasonableness with adaptive thresholds
    length_scores = []
    for var in unique_variations:
        original_len = len(original_part)
        var_len = len(var)
        
        # Adaptive threshold based on original name length
        min_ratio = 0.6 if original_len <= 5 else 0.7  # More forgiving for short names
        
        # Consider both absolute and relative length differences
        length_ratio = min(var_len / original_len, original_len / var_len)
        absolute_diff = abs(var_len - original_len)
        
        # Combine both metrics with smooth transition
        length_score = length_ratio * (1.0 - min(1.0, absolute_diff / original_len))
        length_scores.append(length_score)
        
    
    length_score = sum(length_scores) / len(length_scores) if length_scores else 0
    #bt.logging.info(f"Average length score: {length_score:.3f}")
    
    # Calculate phonetic similarity scores with improved distribution analysis
    phonetic_scores = []
    
    for variation in unique_variations:
        p_score = calculate_phonetic_similarity(original_part, variation)
        phonetic_scores.append(p_score)
        
    # Sort scores for distribution analysis
    phonetic_scores.sort()
    
    # Calculate quality scores with improved distribution matching
    def calculate_distribution_quality(scores, boundaries, targets):
        quality = 0.0
        total_matched = 0
        
        for level, (lower, upper) in boundaries.items():
            target_percentage = targets.get(level, 0.0)
            if target_percentage == 0.0:
                continue
                
            # Count scores in this range
            count = sum(1 for score in scores if lower <= score <= upper)
            target_count = int(target_percentage * len(scores))
            
            if target_count > 0:
                # Calculate match quality with diminishing returns
                match_ratio = count / target_count
                #match_quality = 1.0 - math.exp(-match_ratio)  # Smooth curve
                # Diminishing returns after target
                # this gives 100% at target, then diminishing returns for exceeding
                if match_ratio <= 1.0:
                    match_quality = match_ratio  # Linear up to target
                else:    
                    match_quality = 1.0 - math.exp(-(match_ratio - 1.0))  
                quality += target_percentage * match_quality
                total_matched += count
                
                # bt.logging.info(
                #     f"{level} similarity: {count}/{target_count} variations "
                #     f"({match_quality:.3f} quality)"
                # )
        
        # Penalize unmatched variations
        unmatched = len(scores) - total_matched
        if unmatched > 0:
            penalty = 0.1 * (unmatched / len(scores))
            quality = max(0.0, quality - penalty)
            # bt.logging.warning(f"Penalty of {penalty:.3f} applied for {unmatched} unmatched variations")
        
        return quality
    
    phonetic_quality = calculate_distribution_quality(
        phonetic_scores, phonetic_boundaries, phonetic_similarity
    )
    
    # Use only phonetic similarity score
    similarity_score = phonetic_quality
    
    # Apply minimum similarity threshold to prevent gaming
    # If similarity is very low, severely reduce the score
    min_similarity_threshold = 0.2
    if similarity_score < min_similarity_threshold:
        similarity_score *= 0.1  # Keep only 10% of the similarity score

    similarity_weight = MIID_REWARD_WEIGHTS["similarity_weight"]
    count_weight = MIID_REWARD_WEIGHTS["count_weight"]
    uniqueness_weight = MIID_REWARD_WEIGHTS["uniqueness_weight"]
    length_weight = MIID_REWARD_WEIGHTS["length_weight"]
    
    final_score = (
        similarity_weight * similarity_score +
        count_weight * count_score +
        uniqueness_weight * uniqueness_score +
        length_weight * length_score
    )
    if final_score == 0:
        bt.logging.warning(f"Zero score for '{original_part}'. Possible reasons:")
        if len(variations) == 0:
            bt.logging.warning("  - No variations provided")
        if similarity_score == 0:
            bt.logging.warning("  - All variations had very low phonetic similarity scores")
        if uniqueness_score == 0:
            bt.logging.warning("  - All variations were too similar to each other")
    
    # Just before returning the final_score, prepare the detailed metrics
    detailed_metrics = {
        "similarity": {
            "phonetic": float(phonetic_quality),
            "combined": float(similarity_score)
        },
        "count": {
            "actual": actual_count,
            "expected": expected_count,
            "score": float(count_score)
        },
        "uniqueness": {
            "unique_count": len(unique_variations),
            "total_count": len(variations),
            "score": float(uniqueness_score)
        },
        "length": {
            "score": float(length_score)
        },
        "variations": [{
            "variation": var,
            "phonetic_score": float(calculate_phonetic_similarity(original_part, var)),
            "length_ratio": float(len(var)) / float(len(original_part))
        } for var in variations]
    }
    
    return final_score, detailed_metrics


def get_name_part_weights(name: str) -> dict:
    """Generate weights for different name parts based on name characteristics, with randomness."""
    random.seed(hash(name) % 10000)
    name_parts = name.split()
    if len(name_parts) < 2:
        return {"first_name_weight": 1.0, "last_name_weight": 0.0}
    lengths = [len(part) for part in name_parts]
    total_length = sum(lengths)
    weights = []
    for length in lengths:
        base_weight = length / total_length
        randomized_weight = base_weight * random.uniform(0.8, 1.2)  # 20% randomness
        weights.append(randomized_weight)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    return {
        "first_name_weight": normalized_weights[0],
        "last_name_weight": normalized_weights[1]
    }

def calculate_variation_quality(
    original_name: str,  # Full name as a string
    variations: List[str],
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    expected_count: int = 10,
    rule_based: Dict[str, Any] = None  # New parameter for rule-based metadata
) -> Tuple[float, Dict]:
    """
    Calculate the quality of execution vectors (name variations) for threat detection.
    Returns both the quality score and detailed metrics.
    """
    #bt.logging.info(f"\n{'='*50}")
    #bt.logging.info(f"Calculating variation quality for: {original_name}")
    #bt.logging.info(f"Total variations: {len(variations)}")
    #bt.logging.info(f"Expected count: {expected_count}")

    # Default similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}

    # First, calculate rule compliance to identify rule-based variations
    rule_compliance_score = 0.0
    rule_compliance_metrics = {}
    rule_compliant_variations = set()
    target_percentage = 0.0
    effective_target_rules = []

    if rule_based and "selected_rules" in rule_based:
        #bt.logging.info("\nCalculating rule-based compliance score:")
        target_rules = rule_based.get("selected_rules", [])
        
        # The pre-filtering logic has been moved to evaluate_rule_compliance
        effective_target_rules = target_rules

        if effective_target_rules:
            target_percentage = rule_based.get("rule_percentage", 30) / 100.0  # Convert to fraction
            rule_compliance_score, rule_compliance_metrics = calculate_rule_compliance_score(
                original_name,
                variations,
                effective_target_rules,
                target_percentage
            )
            if "rules_satisfied_by_variation" in rule_compliance_metrics:
                rule_compliant_variations = set(rule_compliance_metrics["rules_satisfied_by_variation"].keys())
    else:
        bt.logging.info("No rule-based requirements specified")

    # Separate variations into rule-compliant and non-rule-compliant
    non_rule_compliant_variations = [
        var for var in variations if var not in rule_compliant_variations
    ]
    
    # Split the original name into first and last name
    name_parts = original_name.split()
    part_weights = get_name_part_weights(original_name)
    if len(name_parts) < 2:
        first_name = original_name
        last_name = None
        #bt.logging.warning(f"Could not split name '{original_name}' into first and last name")
    else:
        first_name = name_parts[0]
        last_name = name_parts[-1]
        #bt.logging.info(f"Using first name: '{first_name}', last name: '{last_name}'")
    
    # Process NON-RULE-COMPLIANT variations for base quality score
    first_name_variations = []
    last_name_variations = []
    
    for variation in non_rule_compliant_variations:
        parts = variation.split()
        if len(parts) >= 2:
            first_name_variations.append(parts[0])
            last_name_variations.append(parts[-1])
        elif len(parts) == 1:
            # If variation is a single word and we expect two names,
            # this should be considered a lower quality variation
            if last_name:
                bt.logging.warning(f"Single word variation '{variation}' for full name '{original_name}'")
                # Only use it for first name with a penalty
                first_name_variations.append(parts[0])
            else:
                # If original is also single word, use normally
                first_name_variations.append(parts[0])
        else:
            bt.logging.warning(f"Empty variation found for '{original_name}'")

    # Adjust expected count for non-rule-compliant part
    expected_base_count = expected_count * (1.0 - target_percentage)

    #bt.logging.info(f"First name variations (non-rule): {len(first_name_variations)}")
    #if last_name:
    #    bt.logging.info(f"Last name variations (non-rule): {len(last_name_variations)}")
    
    # Calculate score for first name (non-rule-compliant part)
    #bt.logging.info("\nCalculating first name score (non-rule-compliant):")
    first_name_score, first_metrics = calculate_part_score(
        first_name,
        first_name_variations,
        phonetic_similarity,
        orthographic_similarity,
        expected_base_count
    )
    
    # Calculate score for last name if available
    last_name_score = 0.0
    last_metrics = {}
    if last_name:
        #bt.logging.info("\nCalculating last name score (non-rule-compliant):")
        last_name_score, last_metrics = calculate_part_score(
            last_name,
            last_name_variations,
            phonetic_similarity,
            orthographic_similarity,
            expected_base_count
        )
        
        # Apply penalty for missing last names in non-rule-compliant variations
        if len(last_name_variations) < len(non_rule_compliant_variations):
            missing_ratio = (len(non_rule_compliant_variations) - len(last_name_variations)) / len(non_rule_compliant_variations) if len(non_rule_compliant_variations) > 0 else 0
            last_name_score *= (1.0 - missing_ratio)
            bt.logging.warning(f"Applied missing last name penalty (non-rule): {missing_ratio:.2f}")

    # Combine first/last name scores for the base_score
    if last_name:
        base_score = (
            part_weights.get("first_name_weight", MIID_REWARD_WEIGHTS["first_name_weight"]) * first_name_score +
            part_weights.get("last_name_weight", MIID_REWARD_WEIGHTS["last_name_weight"]) * last_name_score
        )
    else:
        # If no last name, use only first name score
        base_score = first_name_score
        
    # If no non-rule variations were expected, the base_score is not penalized.
    # This is separated from the case where they were expected but not provided.
    if expected_base_count == 0:
        base_score = 1.0 # Or some other neutral value, 1.0 seems fair to not penalize.
    
    # Apply rule compliance to final score using weights from the global config
    rule_compliance_weight = MIID_REWARD_WEIGHTS["rule_compliance_weight"]
    
    # If rules were requested but none were applicable to this name, adjust weights
    # to base the score entirely on similarity.
    if rule_based and "selected_rules" in rule_based and not effective_target_rules:
        bt.logging.debug(f"⚖️ No rules applicable for '{original_name}', adjusting weights. Base score will be final score.")
        base_weight = 1.0
        rule_compliance_weight = 0.0
    else:
        base_weight = 1.0 - rule_compliance_weight

    # The base_score already contains the other weighted components (length, count, uniqueness)
    # So we need to scale it down to make room for the rule compliance component
    final_score = (base_weight * base_score) + (rule_compliance_weight * rule_compliance_score)

    # Prepare detailed metrics
    detailed_metrics = {
        "first_name": {
            "score": float(first_name_score),
            "metrics": first_metrics
        },
        "base_score": float(base_score),
        "final_score": float(final_score),
        "variation_count": len(variations),
        "non_rule_compliant_variations_count": len(non_rule_compliant_variations),
        "rule_compliant_variations_count": len(rule_compliant_variations)
    }
    
    if last_name:
        detailed_metrics["last_name"] = {
            "score": float(last_name_score),
            "metrics": last_metrics
        }
        
    if rule_based:
        detailed_metrics["rule_compliance"] = {
            "score": float(rule_compliance_score),
            "metrics": rule_compliance_metrics
        }

    # bt.logging.info(f"\nFinal score breakdown for '{original_name}':")
    # bt.logging.info(f"  - First name score (non-rule): {first_name_score:.3f}")
    # if last_name:
    #     bt.logging.info(f"  - Last name score (non-rule): {last_name_score:.3f}")
    # bt.logging.info(f"  - Base similarity score (non-rule): {base_score:.3f}")
    # if rule_based:
    #     bt.logging.info(f"  - Rule compliance score: {rule_compliance_score:.3f} (weight: {rule_compliance_weight:.2f})")
    # bt.logging.info(f"  - Final score: {final_score:.3f}")
    
    if final_score == 0:
        bt.logging.warning(f"Zero final score for '{original_name}'. Possible reasons:")
        if first_name_score == 0 and len(non_rule_compliant_variations) > 0:
            bt.logging.warning("  - Zero first name score on non-rule variations")
        if last_name and last_name_score == 0 and len(last_name_variations) > 0:
            bt.logging.warning("  - Zero last name score on non-rule variations")
        if len(variations) == 0:
            bt.logging.warning("  - No variations provided")
        if rule_based and rule_compliance_score == 0 and len(rule_compliant_variations) > 0:
            bt.logging.warning("  - Zero rule compliance score on rule-compliant variations")
    
    #bt.logging.info(f"{'='*50}\n")
    return final_score, base_score, detailed_metrics


def calculate_variation_quality_phonetic_only(
    original_name: str,  # Full name as a string
    variations: List[str],
    phonetic_similarity: Dict[str, float] = None,
    expected_count: int = 10
) -> Tuple[float, float, Dict]:
    """
    Calculate the quality of execution vectors (name variations) using ONLY phonetic similarity.
    No rule-based scoring, no orthographic similarity - just phonetic similarity for both first and last names.
    """

    # Default phonetic similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}

    # Split the original name into first and last name
    name_parts = original_name.split()
    part_weights = get_name_part_weights(original_name)
    if len(name_parts) < 2:
        first_name = original_name
        last_name = None
    else:
        first_name = name_parts[0]
        last_name = name_parts[-1]
    
    # Process ALL variations for phonetic-only scoring
    first_name_variations = []
    last_name_variations = []
    
    for variation in variations:
        parts = variation.split()
        if len(parts) >= 2:
            first_name_variations.append(parts[0])
            last_name_variations.append(parts[-1])
        elif len(parts) == 1:
            # If variation is a single word and we expect two names,
            # this should be considered a lower quality variation
            if last_name:
                bt.logging.warning(f"Single word variation '{variation}' for full name '{original_name}'")
                # Only use it for first name with a penalty
                first_name_variations.append(parts[0])
            else:
                # If original is also single word, use normally
                first_name_variations.append(parts[0])
        else:
            bt.logging.warning(f"Empty variation found for '{original_name}'")

    # Calculate phonetic-only score for first name
    first_name_score, first_metrics = calculate_part_score_phonetic_only(
        first_name,
        first_name_variations,
        phonetic_similarity,
        expected_count
    )
    
    # Calculate phonetic-only score for last name if available
    last_name_score = 0.0
    last_metrics = {}
    if last_name:
        #bt.logging.info("\nCalculating last name score (phonetic-only):")
        last_name_score, last_metrics = calculate_part_score_phonetic_only(
            last_name,
            last_name_variations,
            phonetic_similarity,
            expected_count
        )
        
        # Apply penalty for missing last names in variations
        if len(last_name_variations) < len(variations):
            missing_ratio = (len(variations) - len(last_name_variations)) / len(variations) if len(variations) > 0 else 0
            last_name_score *= (1.0 - missing_ratio)
            bt.logging.warning(f"Applied missing last name penalty: {missing_ratio:.2f}")

    # Combine first/last name scores for the final score
    if last_name:
        final_score = (
            part_weights.get("first_name_weight", MIID_REWARD_WEIGHTS["first_name_weight"]) * first_name_score +
            part_weights.get("last_name_weight", MIID_REWARD_WEIGHTS["last_name_weight"]) * last_name_score
        )
        base_score = final_score  # Same as final score since no rule compliance
    else:
        # If no last name, use only first name score
        final_score = first_name_score
        base_score = final_score

    # Prepare detailed metrics
    detailed_metrics = {
        "first_name": {
            "score": float(first_name_score),
            "metrics": first_metrics
        },
        "final_score": float(final_score),
        "base_score": float(base_score),
        "variation_count": len(variations),
        "scoring_method": "phonetic_only"
    }
    
    if last_name:
        detailed_metrics["last_name"] = {
            "score": float(last_name_score),
            "metrics": last_metrics
        }
    
    if final_score == 0:
        bt.logging.warning(f"Zero final score for '{original_name}'. Possible reasons:")
        if first_name_score == 0 and len(first_name_variations) > 0:
            bt.logging.warning("  - Zero first name phonetic score")
        if last_name and last_name_score == 0 and len(last_name_variations) > 0:
            bt.logging.warning("  - Zero last name phonetic score")
        if len(variations) == 0:
            bt.logging.warning("  - No variations provided")

    return final_score, base_score, detailed_metrics


def _calculate_similarity_and_penalties(responses: list, uids: list, seed_names: list, detailed_metrics: list, rewards: np.ndarray) -> tuple:
    """
    Calculate similarity between miner responses and determine penalties for duplication.
    Also, calculate penalties for excessive use of special characters.

    Args:
        responses: A list of response objects from miners.
        uids: A list of miner UIDs.
        seed_names: A list of seed names for which variations were requested.
        detailed_metrics: A list of dictionaries with detailed metrics for each miner.
        rewards: The current rewards for each miner.

    Returns:
        A tuple containing the updated rewards and detailed_metrics list.
    """
    
    bt.logging.info(f"\n{'='*60}")
    bt.logging.info(f"🔍 CHEATING DETECTION ANALYSIS")
    bt.logging.info(f"{'='*60}")
    bt.logging.info(f"Analyzing {len(uids)} miners for cheating patterns...")
    
    # Convert IdentitySynapse objects to dictionaries for cheat detection
    response_dicts = []
    for response in responses:
        if hasattr(response, 'variations') and response.variations:
            response_dicts.append(response.variations)
        else:
            response_dicts.append({})
    
    cheating_results = detect_cheating_patterns(response_dicts, uids, rewards, seed_names)

    duplication_penalties = cheating_results["duplication_penalties"]
    signature_penalties = cheating_results["signature_penalties"]
    collusion_penalties = cheating_results["collusion_penalties"]
    special_char_penalties = cheating_results["special_char_penalties"]
    address_duplication_penalties = cheating_results["address_duplication_penalties"]
    special_char_ratios = cheating_results["special_char_ratios"]
    special_char_counts = cheating_results["special_char_counts"]
    total_variations_counts = cheating_results["total_variations_counts"]
    
    # Analyze penalty patterns for summary
    penalized_miners = []
    collusion_groups = []
    duplication_pairs = []
    signature_duplicates = []
    special_char_offenders = []
    address_duplication_offenders = []
    
    # Collect penalty statistics
    for i, uid in enumerate(uids):
        total_penalty = 0
        penalties_applied = []
        
        if collusion_penalties[i] > 0:
            total_penalty += collusion_penalties[i]
            penalties_applied.append(f"collusion({collusion_penalties[i]:.2f})")
            collusion_groups.append(uid)
            
        if duplication_penalties[i] > 0:
            total_penalty += duplication_penalties[i]
            penalties_applied.append(f"duplication({duplication_penalties[i]:.2f})")
            duplication_pairs.append(uid)
            
        if signature_penalties[i] > 0:
            total_penalty += signature_penalties[i]
            penalties_applied.append(f"signature({signature_penalties[i]:.2f})")
            signature_duplicates.append(uid)
            
        if special_char_penalties[i] > 0:
            total_penalty += special_char_penalties[i]
            penalties_applied.append(f"special_chars({special_char_penalties[i]:.2f})")
            special_char_offenders.append(uid)
            
        if address_duplication_penalties[i] > 0:
            total_penalty += address_duplication_penalties[i]
            penalties_applied.append(f"address_duplication({address_duplication_penalties[i]:.2f})")
            address_duplication_offenders.append(uid)
            
        if total_penalty > 0:
            penalized_miners.append((uid, total_penalty, penalties_applied))
    
    # Log summary of findings
    bt.logging.info(f"\n📊 CHEATING DETECTION SUMMARY:")
    bt.logging.info(f"  • Total miners analyzed: {len(uids)}")
    bt.logging.info(f"  • Miners with penalties: {len(penalized_miners)}")
    bt.logging.info(f"  • Honest miners: {len(uids) - len(penalized_miners)}")
    
    if penalized_miners:
        bt.logging.info(f"\n🚨 CHEATING PATTERNS DETECTED:")
        
        if collusion_groups:
            bt.logging.info(f"  • Collusion groups: {len(collusion_groups)} miners")
            bt.logging.info(f"    Miners: {', '.join(str(uid) for uid in collusion_groups)}")
            
        if duplication_pairs:
            bt.logging.info(f"  • Duplication pairs: {len(duplication_pairs)} miners")
            bt.logging.info(f"    Miners: {', '.join(str(uid) for uid in duplication_pairs)}")
            
        if signature_duplicates:
            bt.logging.info(f"  • Signature duplicates: {len(signature_duplicates)} miners")
            bt.logging.info(f"    Miners: {', '.join(str(uid) for uid in signature_duplicates)}")
            
        if special_char_offenders:
            bt.logging.info(f"  • Special character abuse: {len(special_char_offenders)} miners")
            bt.logging.info(f"    Miners: {', '.join(str(uid) for uid in special_char_offenders)}")
            
        if address_duplication_offenders:
            bt.logging.info(f"  • Address duplication: {len(address_duplication_offenders)} miners")
            bt.logging.info(f"    Miners: {', '.join(str(uid) for uid in address_duplication_offenders)}")
            
        bt.logging.info(f"\n⚠️  PENALTY BREAKDOWN:")
        for uid, total_penalty, penalties in penalized_miners:
            bt.logging.info(f"  • Miner {uid}: {total_penalty:.3f} total penalty [{', '.join(penalties)}]")
    else:
        bt.logging.info(f"\n✅ NO CHEATING DETECTED")
        bt.logging.info(f"  All {len(uids)} miners appear to be honest")
    
    bt.logging.info(f"\n{'='*60}")
    
    # Apply penalties and update metrics
    updated_rewards = np.copy(rewards)
    num_miners = len(responses)
    for i in range(num_miners):
        uid = uids[i]
        total_penalty = 0

        # Always record special character observability metrics
        if i < len(detailed_metrics):
            miner_metrics = detailed_metrics[i]
            miner_metrics.setdefault('penalties', {})
            miner_metrics['penalties']['special_chars_ratio'] = float(special_char_ratios[i])
            miner_metrics['penalties']['special_char_variations_count'] = int(special_char_counts[i])
            miner_metrics['penalties']['total_variations_count'] = int(total_variations_counts[i])

        # Collusion Penalty
        if collusion_penalties[i] > 0:
            penalty_amount = collusion_penalties[i]
            total_penalty += penalty_amount
            if i < len(detailed_metrics):
                miner_metrics = detailed_metrics[i]
                miner_metrics.setdefault('penalties', {})
                miner_metrics['penalties']['collusion'] = penalty_amount

        # Duplication Penalty
        if duplication_penalties[i] > 0:
            penalty_amount = duplication_penalties[i]
            total_penalty += penalty_amount
            if i < len(detailed_metrics):
                miner_metrics = detailed_metrics[i]
                miner_metrics.setdefault('penalties', {})
                miner_metrics['penalties']['duplication'] = penalty_amount

        # Signature (exact-copy) Penalty
        if signature_penalties[i] > 0:
            penalty_amount = signature_penalties[i]
            total_penalty += penalty_amount
            if i < len(detailed_metrics):
                miner_metrics = detailed_metrics[i]
                miner_metrics.setdefault('penalties', {})
                miner_metrics['penalties']['signature_copy'] = penalty_amount

        # Special Character Penalty
        if special_char_penalties[i] > 0:
            penalty_amount = special_char_penalties[i]
            total_penalty += penalty_amount
            if i < len(detailed_metrics):
                miner_metrics = detailed_metrics[i]
                miner_metrics.setdefault('penalties', {})
                miner_metrics['penalties']['special_chars'] = penalty_amount

        # Address Duplication Penalty
        if address_duplication_penalties[i] > 0:
            penalty_amount = address_duplication_penalties[i]
            total_penalty += penalty_amount
            if i < len(detailed_metrics):
                miner_metrics = detailed_metrics[i]
                miner_metrics.setdefault('penalties', {})
                miner_metrics['penalties']['post_address_duplication'] = penalty_amount

        # Apply combined penalty
        if total_penalty > 0:
            total_penalty = min(total_penalty, 1.0) # Cap total penalty
            current_reward = updated_rewards[i]
            penalized_reward = current_reward * (1.0 - total_penalty)
            updated_rewards[i] = penalized_reward
            if i < len(detailed_metrics):
                # Record post-processing penalties separately and combined with pre-penalties
                miner_metrics = detailed_metrics[i]
                miner_metrics.setdefault('penalties', {})
                miner_metrics['penalties']['post_total_penalty'] = float(total_penalty)
                pre_total_penalty = float(miner_metrics['penalties'].get('total_penalty', 0.0))
                # miner_metrics['penalties']['overall_total_penalty'] = (1.0 - pre_total_penalty) * (1.0 - total_penalty)
                miner_metrics['penalties']['overall_total_penalty'] = 1.0 - ((1.0 - pre_total_penalty) * (1.0 - total_penalty))
                detailed_metrics[i]['final_reward'] = penalized_reward

    bt.logging.info(f"✅ Cheating detection and penalty application completed")
    bt.logging.info(f"{'='*60}\n")
    
    return updated_rewards, detailed_metrics


def _grade_dob_variations(variations: Dict[str, List[List[str]]], seed_dob: List[str], miner_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade DOB variations based on the required criteria.
    Each seed name gets its own score based on variations found, then scores are averaged.
    """
    
    # Required day ranges (max days for each category)
    ranges = [1, 3, 30, 90, 365]  # ±1, ±3, ±30, ±90, ±365 days
    total_ranges = len(ranges) + 1  # +1 for year_month
    
    # Store detailed breakdown in results (do the work once)
    detailed_breakdown = {
        "seed_dobs": [],
        "variations_by_name": {},
        "category_classifications": {}
    }
    
    # Track individual scores for each name
    name_scores = []
    all_found_ranges = set()
    
    for name_idx, name in enumerate(variations.keys()):
        if name not in variations or len(variations[name]) < 1 or name_idx >= len(seed_dob):
            continue
            
        if not seed_dob[name_idx]:
            continue
            
        # Extract DOB variations for this name
        all_variations = variations[name]
        dob_variations = [var[1] for var in all_variations if len(var) > 1]  # Include empty strings for proper counting
        
        if not dob_variations:
            continue
        
        # Check for duplicates within this name's DOB variations
        unique_dobs = set()
        duplicates_in_name = 0
        duplicate_details = []
        
        for dob_var in dob_variations:
            if dob_var and dob_var in unique_dobs:
                duplicates_in_name += 1
                duplicate_details.append(dob_var)
            elif dob_var:
                unique_dobs.add(dob_var)
        
        # Category classification for this name (do the work once)
        name_found_ranges = set()
        try:
            seed_date = datetime.strptime(seed_dob[name_idx], "%Y-%m-%d")
            categories = {}
            
            for dob_var in dob_variations:
                if not dob_var:
                    continue
                    
                try:
                    # Try full date format first
                    var_date = datetime.strptime(dob_var, "%Y-%m-%d")
                    day_diff = abs((var_date - seed_date).days)
                    
                    # Classify this variation
                    if day_diff <= 1:
                        category = "±1 day"
                        name_found_ranges.add(1)
                        all_found_ranges.add(1)
                    elif day_diff <= 3:
                        category = "±3 days"
                        name_found_ranges.add(3)
                        all_found_ranges.add(3)
                    elif day_diff <= 30:
                        category = "±30 days"
                        name_found_ranges.add(30)
                        all_found_ranges.add(30)
                    elif day_diff <= 90:
                        category = "±90 days"
                        name_found_ranges.add(90)
                        all_found_ranges.add(90)
                    elif day_diff <= 365:
                        category = "±365 days"
                        name_found_ranges.add(365)
                        all_found_ranges.add(365)
                    else:
                        category = "Outside range"
                        
                except ValueError:
                    # Try year-month only format
                    try:
                        year_month = datetime.strptime(dob_var, "%Y-%m")
                        if (seed_date.year == year_month.year and 
                            seed_date.month == year_month.month):
                            category = "Year+Month only"
                            name_found_ranges.add("year_month")
                            all_found_ranges.add("year_month")
                        else:
                            category = "Invalid year-month"
                    except ValueError:
                        category = "Invalid format"
                
                if category not in categories:
                    categories[category] = []
                categories[category].append(dob_var)
            
            # Calculate individual score for this name based on variations found
            name_score = len(name_found_ranges) / total_ranges if total_ranges > 0 else 0.0
            name_scores.append(name_score)
            
            # Store category classifications and score
            detailed_breakdown["category_classifications"][name] = {
                "categories": categories,
                "score": name_score
            }
                
        except ValueError:
            detailed_breakdown["category_classifications"][name] = {
                "error": "Invalid seed DOB format",
                "score": 0.0
            }
            # Add 0 score for invalid seed DOB
            name_scores.append(0.0)
    
    # Calculate overall score as average of individual name scores
    if name_scores:
        overall_score = sum(name_scores) / len(name_scores)
    else:
        overall_score = 0.0
    
    return {
        "overall_score": overall_score,
        "found_ranges": list(all_found_ranges),
        "total_ranges": total_ranges,
        "detailed_breakdown": detailed_breakdown
    }

def _grade_address_variations(variations: Dict[str, List[List[str]]], seed_addresses: List[str], miner_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Grade address variations - check all with heuristics, one random with API, and region validation."""
    if not seed_addresses or not any(seed_addresses):
        return {"overall_score": 1.0}
    
    # Collect all addresses with their corresponding seed addresses
    all_addresses = []
    address_seed_mapping = []
    for name_idx, name in enumerate(variations.keys()):
        if name in variations and len(variations[name]) >= 1 and name_idx < len(seed_addresses):
            # Extract address variations (index 2 of each [name_var, dob_var, address_var] array)
            all_variations = variations[name]
            address_variations = [var[2] for var in all_variations if len(var) > 2]  # Include empty strings for proper counting
            if address_variations and seed_addresses[name_idx]:
                valid_addrs = [addr for addr in address_variations if addr and addr.strip()]
                all_addresses.extend(valid_addrs)
                # Map each address to its corresponding seed address
                seed_addr = seed_addresses[name_idx]
                address_seed_mapping.extend([seed_addr] * len(valid_addrs))
    
    if not all_addresses:
        return {"overall_score": 0.0}
    
    # Store detailed breakdown in results (do the work once)
    address_breakdown = {
        "seed_addresses": [],
        "variations_by_name": {},
        "validation_results": {},
        "api_validation": {}
    }
    
    # Strict validation in order:
    # 1. If looks_like_address is false -> return 0
    # 2. If country or city are not in seed -> return 0  
    # 3. If API passes -> full score
    
    # Process each name and validate addresses (do the work once)
    heuristic_perfect = True
    region_matches = 0
    api_validated_addresses = []
    
    for name_idx, name in enumerate(variations.keys()):
        if name not in variations or len(variations[name]) < 1 or name_idx >= len(seed_addresses):
            continue
            
        if not seed_addresses[name_idx]:
            continue
            
        # Extract address variations for this name
        all_variations = variations[name]
        address_variations = [var[2] for var in all_variations if len(var) > 2 and var[2]]
        
        if not address_variations:
            continue
        
        # Store validation results for each address (do the work once)
        validation_results = []
        for i, addr in enumerate(address_variations):
            if not addr or not addr.strip():
                validation_results.append({
                    "address": addr,
                    "looks_like_address": False,
                    "region_match": False,
                    "passed_validation": False,
                    "status": "EMPTY/INVALID"
                })
                heuristic_perfect = False
                continue
                
            # Step 1: Check if looks like address
            looks_like = looks_like_address(addr)
            
            # Step 2: Check if country or city matches seed
            region_match = False
            if looks_like:
                region_match = validate_address_region(addr, seed_addresses[name_idx])
            
            # Track validation results
            passed_validation = looks_like and region_match
            if not looks_like or not region_match:
                heuristic_perfect = False
            
            if passed_validation:
                api_validated_addresses.append(addr)
                region_matches += 1
            
            validation_results.append({
                "address": addr,
                "looks_like_address": looks_like,
                "region_match": region_match,
                "passed_validation": passed_validation,
                "status": "PASSED" if passed_validation else "FAILED"
            })
        
        address_breakdown["validation_results"][name] = validation_results
    
    # If first 2 steps fail, return 0 immediately (no API call needed)
    if not heuristic_perfect:
        address_breakdown["api_validation"] = {
            "api_result": False,
            "total_eligible_addresses": 0,
            "api_attempts": [],
            "reason": "Heuristic validation failed - no API call made"
        }
        
        return {
            "overall_score": 0.0,
            "heuristic_perfect": False,
            "api_result": False,
            "region_matches": region_matches,
            "total_addresses": len(all_addresses),
            "base_score": 0.0,
            "detailed_breakdown": address_breakdown
        }
    
    # Only call API if all addresses passed first 2 checks - now validates with both APIs
    api_result = False
    api_attempts = []
    nominatim_successful_calls = 0
    nominatim_timeout_calls = 0
    nominatim_failed_calls = 0
    mapbox_successful_calls = 0
    mapbox_timeout_calls = 0
    mapbox_failed_calls = 0
    total_calls = 0
    
    if api_validated_addresses:
        # Randomly choose up to 4 different addresses for API validation (2 for each API)
        max_addresses = min(4, len(api_validated_addresses))
        chosen_addresses = random.sample(api_validated_addresses, max_addresses)
        
        # Split addresses for each API (2 each)
        nominatim_addresses = chosen_addresses[:2]
        mapbox_addresses = chosen_addresses[2:4] if len(chosen_addresses) > 2 else chosen_addresses[:2]
        
        # Try Nominatim API (2 calls)
        for i, addr in enumerate(nominatim_addresses):
            result = check_with_nominatim(addr)
            api_attempts.append({
                "address": addr,
                "api": "nominatim",
                "result": result,
                "attempt": i + 1
            })
            
            if result == "TIMEOUT":
                nominatim_timeout_calls += 1
            elif result == True:
                nominatim_successful_calls += 1
            else:
                nominatim_failed_calls += 1
            
            # Wait 1 second between API calls to prevent rate limiting
            if i < len(nominatim_addresses) - 1:
                time.sleep(1.0)
        
        # Wait between different APIs
        if nominatim_addresses and mapbox_addresses:
            time.sleep(1.0)
        
        # Try Mapbox API (2 calls)
        for i, addr in enumerate(mapbox_addresses):
            result = mapbox_verify_address(addr)
            api_attempts.append({
                "address": addr,
                "api": "mapbox",
                "result": result,
                "attempt": i + 1
            })
            
            if result == "TIMEOUT":
                mapbox_timeout_calls += 1
            elif result == True:
                mapbox_successful_calls += 1
            else:
                mapbox_failed_calls += 1
            
            # Wait 1 second between API calls to prevent rate limiting
            if i < len(mapbox_addresses) - 1:
                time.sleep(1.0)
        
        # Set final result based on individual results from both APIs
        total_calls = len(chosen_addresses)
        total_successful = nominatim_successful_calls + mapbox_successful_calls
        total_timeouts = nominatim_timeout_calls + mapbox_timeout_calls
        total_failed = nominatim_failed_calls + mapbox_failed_calls
        
        if total_failed > 0:
            api_result = "FAILED"  # Any failure = 0.0 score
        elif total_timeouts > 0:
            api_result = "TIMEOUT"  # All pass but timeouts = -0.2 per timeout
        else:
            api_result = "SUCCESS"  # All pass without timeouts = perfect score
    
    # Scoring based on individual API results
    if api_result == "FAILED":
        base_score = 0.0  # Any failure = 0.0 score
    elif api_result == "TIMEOUT":
        # Calculate penalty: -0.2 per timeout
        timeout_penalty = total_timeouts * 0.2
        base_score = max(0.0, 1.0 - timeout_penalty)  # Ensure score doesn't go below 0
    elif api_result == "SUCCESS":
        base_score = 1.0  # Perfect - all addresses passed without timeouts
    else:
        base_score = 0.0  # Default fallback
    
    # Store API validation results
    address_breakdown["api_validation"] = {
        "api_result": api_result,
        "total_eligible_addresses": len(api_validated_addresses),
        "api_attempts": api_attempts,
        "nominatim_successful_calls": nominatim_successful_calls,
        "nominatim_timeout_calls": nominatim_timeout_calls,
        "nominatim_failed_calls": nominatim_failed_calls,
        "mapbox_successful_calls": mapbox_successful_calls,
        "mapbox_timeout_calls": mapbox_timeout_calls,
        "mapbox_failed_calls": mapbox_failed_calls,
        "total_successful_calls": total_successful,
        "total_timeout_calls": total_timeouts,
        "total_failed_calls": total_failed,
        "total_calls": total_calls
    }
    
    return {
        "overall_score": base_score,
        "heuristic_perfect": heuristic_perfect,
        "api_result": api_result,
        "region_matches": region_matches,
        "total_addresses": len(all_addresses),
        "detailed_breakdown": address_breakdown
    }


def get_name_variation_rewards(
    self,
    seed_names: List[str],
    seed_dob: List[str],
    seed_addresses: List[str],
    seed_script: List[str],
    responses: List[Dict[str, List[List[str]]]],
    uids: List[int],
    variation_count: int = 10,
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    rule_based: Dict[str, Any] = None  # New parameter for rule-based metadata
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Calculate rewards for execution vectors (name variations) that simulate threat scenarios.
    
    This function evaluates how well miners generate execution vectors (name variations)
    that could be used to bypass identity screening systems. The evaluation considers:
    1. Adherence to the specified threat scenario parameters
    2. Quality and diversity of execution vectors
    3. Effectiveness as potential bypass methods based on similarity metrics
    4. Compliance with requested rule-based transformations
    5. Compliance with requested DOB and address variations
    
    Args:
        seed_names: Original identity names to generate variations for
        responses: List of execution vector responses from miners
        uids: List of UIDs corresponding to responses
        variation_count: Expected number of execution vectors per identity
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        rule_based: Dictionary containing rule-based requirements
        
    Returns:
        Tuple containing:
        - Array of rewards for each miner based on execution vector quality
        - List of dictionaries containing detailed scoring metrics for each miner
    """
    # Default similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
        
    # Boundaries for different similarity levels (ranges instead of thresholds)
    phonetic_boundaries = {
        "Light": (0.8, 1.0),  # High similarity range
        "Medium": (0.6, 0.8), # Moderate similarity range
        "Far": (0.3, 0.6)     # Low similarity range
    }
    
    orthographic_boundaries = {
        "Light": (0.7, 1.0),  # High similarity range
        "Medium": (0.5, 0.7), # Moderate similarity range
        "Far": (0.2, 0.5)     # Low similarity range
    }
    
    # Run ID is now managed at the forward pass level, not in reward function
    
    rewards = np.zeros(len(responses))
    detailed_metrics = []  # Store detailed metrics for each miner
    
    # Log rule-based requirements if provided
    # if rule_based:
    #     #bt.logging.info(f"Rule-based requirements: {rule_based}")
    #     #bt.logging.info(f"Target rules: {rule_based.get('selected_rules', [])}")
    #     #bt.logging.info(f"Target rule-based percentage: {rule_based.get('rule_percentage', 30)}%")
    # else:
    #     #bt.logging.info("No rule-based requirements specified")
    #     pass
    
    # Validate that responses and uids have the same length
    if len(responses) != len(uids):
        bt.logging.error(f"CRITICAL: Length mismatch between responses ({len(responses)}) and uids ({len(uids)})")
        raise ValueError(f"Length mismatch: responses={len(responses)}, uids={len(uids)}")
    
    bt.logging.info(f"Processing rewards for {len(responses)} miners with UIDs: {uids}")
    
    # Pre-transliterate non-Latin seed names once for all miners
    bt.logging.info("Pre-transliterating non-Latin seed names...")
    start_time = time.time()
    transliterated_seed_names = {}
    for name, script in zip(seed_names, seed_script):
        if script != "latin":
            bt.logging.info(f"Transliterating non-Latin name: '{name}' (script: {script})")
            transliterated_name = transliterate_name_with_llm(name, script)
            transliterated_seed_names[name] = transliterated_name
            bt.logging.info(f"Transliterated '{name}' to '{transliterated_name}'")
        else:
            transliterated_seed_names[name] = name  # Keep Latin names as-is
    end_time = time.time()
    bt.logging.info(f"Transliteration complete in {end_time - start_time:.2f} seconds")
    bt.logging.info(f"Transliteration complete. Transliterated names: {transliterated_seed_names}")
    
    # Process each miner's response
    for i, (response, uid) in enumerate(zip(responses, uids)):
        #bt.logging.info(f"\n{'='*50}")
        #bt.logging.info(f"Processing miner {uid}")
        
        # Initialize metrics dictionary for this miner
        miner_metrics = {
            "quality_scores": {},
            "penalties": {
                "extra_names": 0.0,
                "missing_names": 0.0,
                "insufficient_addresses": 0.0,
                "insufficient_dob": 0.0,
                "total_penalty": 0.0
            },
            "completeness_multiplier": 1.0,
            "name_metrics": {},
            "invalid_names": [],
            "missing_names": []
        }
        
        # Correctly access the variations from the response object
        variations = response.variations if hasattr(response, 'variations') else {}
        
        if not variations:
            bt.logging.warning(f"Miner {uid} returned invalid or empty response")
            rewards[i] = 0.0
            # Correctly set metrics for invalid/empty response
            miner_metrics["penalties"]["missing_names"] = 1.0  # All names are missing
            miner_metrics["penalties"]["total_penalty"] = 1.0 
            miner_metrics["completeness_multiplier"] = 0.0
            miner_metrics["average_quality"] = 0.0
            miner_metrics["final_reward"] = 0.0
            detailed_metrics.append(miner_metrics)
            continue
            
        quality_scores = []
        base_scores = []
        extra_names_penalty = 0.0

        # Calculate penalty for unexpected names (extra variations)
        invalid_names = set(variations.keys()) - set(seed_names)
        if invalid_names:
            bt.logging.warning(f"Miner {uid} provided variations for unexpected names: {invalid_names}")
            # 10% penalty per extra name, up to 70% max
            extra_penalty = min(0.7, len(invalid_names) * 0.1)
            bt.logging.info(f"Extra penalty: {extra_penalty}")
            extra_names_penalty = float(extra_penalty)
            miner_metrics["invalid_names"] = list(invalid_names)
        
        # Safety check: ensure all variations have the correct format [name, dob, address]
        for name, vars_list in variations.items():
            if not vars_list:
                bt.logging.warning(f"Miner {uid} provided empty variations for {name}")
                continue
            
            # Check if each variation has at least 3 elements (name, dob, address)
            for j, var in enumerate(vars_list):
                if not isinstance(var, (list, tuple)) or len(var) < 3:
                    bt.logging.warning(f"Miner {uid} provided incomplete variation {j} for {name}: expected [name, dob, address], got {var}")
                    # Pad with empty strings if needed
                    if isinstance(var, (list, tuple)):
                        while len(var) < 3:
                            var.append("")
                    else:
                        # If it's not a list/tuple, replace it with a properly formatted one
                        vars_list[j] = [str(var) if var else "", "", ""]
        
        # Penalty for too many variations per name, DOB, and addresses
        for name, vars_list in variations.items():
            
            # Extract name, DOB, and address variations from the structure once
            # vars_list is [[name_var, dob_var, address_var], [name_var, dob_var, address_var], ...]
            name_variations = [var[0] for var in vars_list if len(var) > 0]  # Include empty strings for proper counting
            dob_variations = [var[1] for var in vars_list if len(var) > 1]  # Include empty strings for proper counting
            address_variations = [var[2] for var in vars_list if len(var) > 2]  # Include empty strings for proper counting
            
            if variation_count > 0:
                allowed_with_grace = int(variation_count * 1.2)  # 20% grace, rounded down
                
                # Check names for variation count
                if len(name_variations) > allowed_with_grace:
                    too_many = len(name_variations) - allowed_with_grace
                    penalty_too_many = too_many * 0.05  # 5% per extra
                    # bt.logging.info(f"Too many name variations for {name}: {too_many} extra → penalty {penalty_too_many}")
                    extra_names_penalty += penalty_too_many
                
                # Check DOB for variation count
                if len(dob_variations) > allowed_with_grace:
                    too_many = len(dob_variations) - allowed_with_grace
                    penalty_too_many = too_many * 0.05  # 5% per extra
                    # bt.logging.info(f"Too many DOB variations for {name}: {too_many} extra → penalty {penalty_too_many}")
                    extra_names_penalty += penalty_too_many
                
                # Check addresses for variation count
                if len(address_variations) > allowed_with_grace:
                    too_many = len(address_variations) - allowed_with_grace
                    penalty_too_many = too_many * 0.05  # 5% per extra
                    # bt.logging.info(f"Too many address variations for {name}: {too_many} extra → penalty {penalty_too_many}")
                    extra_names_penalty += penalty_too_many
        
            # Normalize DOB variations for duplicate detection
            def normalize_dob(dob_str):
                """Normalize DOB string by removing extra spaces and standardizing format"""
                if not dob_str:
                    return ""
                # Remove all spaces and convert to lowercase
                normalized = dob_str.replace(" ", "").replace("-", "").replace("/", "").replace(".", "").lower()
                return normalized
            
            # Normalize address variations for duplicate detection
            def normalize_address(addr_str):
                """Normalize address string by removing extra spaces and standardizing format"""
                if not addr_str:
                    return ""
                # Remove extra spaces, convert to lowercase, and standardize common separators
                normalized = " ".join(addr_str.split()).lower()
                # Replace common separators with spaces
                normalized = normalized.replace(",", " ").replace(";", " ").replace("-", " ")
                # Remove multiple spaces
                normalized = " ".join(normalized.split())
                return normalized
            
            # Penalty for duplicate variations - names
            duplicates_names = len(name_variations) - len(set(name_variations))
            if duplicates_names > 0:
                penalty_duplicates = duplicates_names * 0.05  # e.g. 5% penalty per duplicate
                # bt.logging.info(f"Duplicate variations for {name}: {duplicates_names} duplicates → penalty {penalty_duplicates}")
                extra_names_penalty += penalty_duplicates
            
            # Penalty for duplicate variations - DOB (with normalization)
            dob_duplicates_penalty = 0.0
            if dob_variations:  # Check if DOB list exists and is not empty
                normalized_dobs = [normalize_dob(dob) for dob in dob_variations if dob]  # Filter out empty strings
                duplicates_dob = len(normalized_dobs) - len(set(normalized_dobs))
                if duplicates_dob > 0:
                    penalty_duplicates = duplicates_dob * 0.05  # e.g. 5% penalty per duplicate
                    # bt.logging.info(f"Duplicate DOB variations for {name}: {duplicates_dob} duplicates → penalty {penalty_duplicates}")
                    dob_duplicates_penalty += penalty_duplicates
            
            dob_duplicates_penalty = min(dob_duplicates_penalty, 0.1)  # Max 10% penalty
            # Penalty for duplicate variations - addresses (with normalization)
            address_duplicates_penalty = 0.0
            if address_variations:  # Check if address list exists and is not empty
                normalized_addresses = [normalize_address(addr) for addr in address_variations if addr]  # Filter out empty strings
                duplicates_addresses = len(normalized_addresses) - len(set(normalized_addresses))
                if duplicates_addresses > 0:
                    penalty_duplicates = duplicates_addresses * 0.05  # e.g. 5% penalty per duplicate
                    # bt.logging.info(f"Duplicate address variations for {name}: {duplicates_addresses} duplicates → penalty {penalty_duplicates}")
                    address_duplicates_penalty += penalty_duplicates
            
            address_duplicates_penalty = min(address_duplicates_penalty, 0.2)  # Max 20% penalty

            extra_names_penalty = extra_names_penalty + dob_duplicates_penalty + address_duplicates_penalty

        # Optionally cap at 1.0 total
        extra_names_penalty = min(extra_names_penalty, 1.0)
        miner_metrics["penalties"]["extra_names"] = extra_names_penalty
        miner_metrics["penalties"]["extra_names_breakdown"] = {
            "dob_duplicates": dob_duplicates_penalty,
            "address_duplicates": address_duplicates_penalty
        }
    
        # Calculate penalty for missing names
        missing_names = set(seed_names) - set(variations.keys())
        if missing_names:
            bt.logging.warning(f"Miner {uid} missing variations for names: {missing_names}")
            # 20% penalty per missing name, up to 90% max
            missing_penalty = min(0.9, len(missing_names) * 0.2)
            #bt.logging.info(f"Missing penalty: {missing_penalty}")
            miner_metrics["penalties"]["missing_names"] = float(missing_penalty)
            miner_metrics["missing_names"] = list(missing_names)
        
        # Calculate penalty for insufficient address variations
        insufficient_addresses_penalty = 0.0
        insufficient_addresses = []
        
        if variation_count > 0:
            min_required = max(1, int(variation_count * 0.8))  # At least 80% of expected variations
            
            for name, vars_list in variations.items():
                # Extract address variations from the structure
                address_variations = [var[2] for var in vars_list if len(var) > 2]  # Include empty strings for proper counting
                address_count = len(address_variations)  # Count non-empty addresses
                if address_count < min_required:
                    insufficient_count = min_required - address_count
                    insufficient_addresses.append(f"{name}: {address_count}/{min_required}")
                    # 10% penalty per missing address variation, up to 50% max per name
                    penalty_per_name = min(0.5, insufficient_count * 0.1)
                    insufficient_addresses_penalty += penalty_per_name
                    bt.logging.warning(f"Miner {uid} insufficient address variations for {name}: {address_count}/{min_required} → penalty {penalty_per_name}")
        
        # Cap the insufficient addresses penalty
        insufficient_addresses_penalty = min(insufficient_addresses_penalty, 0.2)  # Max 20% penalty
        miner_metrics["penalties"]["insufficient_addresses"] = float(insufficient_addresses_penalty)
        
        # Calculate penalty for insufficient DOB variations
        insufficient_dob_penalty = 0.0
        insufficient_dob = []
        
        if variation_count > 0:
            min_required = max(1, int(variation_count * 0.8))  # At least 80% of expected variations
            
            for name, vars_list in variations.items():
                # Extract DOB variations from the structure
                dob_variations = [var[1] for var in vars_list if len(var) > 1]  # Include empty strings for proper counting
                dob_count = len(dob_variations)  # Count non-empty DOBs
                if dob_count < min_required:
                    insufficient_count = min_required - dob_count
                    insufficient_dob.append(f"{name}: {dob_count}/{min_required}")
                    # 10% penalty per missing DOB variation, up to 50% max per name
                    penalty_per_name = min(0.5, insufficient_count * 0.1)
                    insufficient_dob_penalty += penalty_per_name
                    bt.logging.warning(f"Miner {uid} insufficient DOB variations for {name}: {dob_count}/{min_required} → penalty {penalty_per_name}")
        
        # Cap the insufficient DOB penalty
        insufficient_dob_penalty = min(insufficient_dob_penalty, 0.1)  # Max 10% penalty
        miner_metrics["penalties"]["insufficient_dob"] = float(insufficient_dob_penalty)
        
        # Calculate total penalty and completeness multiplier
        total_penalty = min(0.9, miner_metrics["penalties"]["extra_names"] + miner_metrics["penalties"]["missing_names"] + miner_metrics["penalties"]["insufficient_addresses"] + miner_metrics["penalties"]["insufficient_dob"])
        completeness_multiplier = max(0.1, 1.0 - total_penalty)
        miner_metrics["penalties"]["total_penalty"] = float(total_penalty)
        miner_metrics["completeness_multiplier"] = float(completeness_multiplier)

        # Add rule-based metrics fields
        if rule_based:
            miner_metrics["rule_compliance"] = {
                "overall_score": 0.0,
                "by_name": {}
            }
        
        # Process each seed name
        for name, script in zip(seed_names, seed_script):
            if name not in variations or not variations[name]:
                continue
            
            # Skip non-Latin names - they will have their own judging part
            if script != "latin":
                continue

            # Convert base name to lowercase, only grading lowercase names
            base_name = name.lower()
            
            # Get variations for this name (use original name as key)
            # variations[name] is a list of [name_var, dob_var, address_var] arrays
            # We need to extract just the name variations (index 0 of each array)
            all_variations = variations[name]
            name_variations = [var[0] for var in all_variations if len(var) > 0]  # Include empty strings for proper counting
            name_metrics = {
                "variations": [],
                "quality_score": 0.0,
                "uniqueness_score": 0.0,
                "count_score": 0.0,
                "length_score": 0.0,
                "phonetic_scores": [],
                "orthographic_scores": []
            }
            
            # Calculate quality score
            try:
                quality, base_score, name_detailed_metrics = calculate_variation_quality(
                    base_name,
                    name_variations,
                    phonetic_similarity=phonetic_similarity,
                    orthographic_similarity=orthographic_similarity,
                    expected_count=variation_count,
                    rule_based=rule_based  # Pass rule-based metadata
                )
                quality_scores.append(quality)
                base_scores.append(base_score)
                miner_metrics["name_metrics"][name] = name_detailed_metrics
                
                # Extract rule compliance metrics if available
                if rule_based and "rule_compliance" in name_detailed_metrics:
                    miner_metrics["rule_compliance"]["by_name"][name] = name_detailed_metrics["rule_compliance"]
            except Exception as e:
                bt.logging.error(f"Error calculating quality for miner {uid}, name '{base_name}': {str(e)}")
                traceback.print_exc()

        start_time = time.time()
        # Process each non-Latin seed name using phonetic-only scoring with pre-transliterated names
        for name, script in zip(seed_names, seed_script):
            if name not in variations or not variations[name]:
                continue

            if script == "latin":
                continue
                
            # Get variations for this name
            # variations[name] is a list of [name_var, dob_var, address_var] arrays
            # We need to extract just the name variations (index 0 of each array)
            all_variations = variations[name]
            name_variations = [var[0] for var in all_variations if len(var) > 0]  # Include empty strings for proper counting
            
            # Use pre-transliterated name
            transliterated_name = transliterated_seed_names[name]
            
            # Use phonetic-only scoring on transliterated name
            try:
                quality, base_score, name_detailed_metrics = calculate_variation_quality_phonetic_only(
                    transliterated_name,
                    name_variations,
                    phonetic_similarity=phonetic_similarity,
                    expected_count=variation_count
                )
                # Add transliteration info to metrics
                name_detailed_metrics["transliteration"] = {
                    "original_name": name,
                    "transliterated_name": transliterated_name,
                    "transliteration_success": True
                }
            except Exception as e:
                bt.logging.error(f"Error calculating phonetic-only quality for miner {uid}, name '{name}': {str(e)}")
                traceback.print_exc()
                continue
            
            # Add scoring method info to metrics
            name_detailed_metrics["scoring_method"] = "phonetic_only_with_llm_transliteration"
            
            quality_scores.append(quality)
            base_scores.append(base_score)
            miner_metrics["name_metrics"][name] = name_detailed_metrics
        end_time = time.time()
        bt.logging.info(f"Phonetic-only scoring time: {end_time - start_time:.2f} seconds")
        
        # Calculate overall rule compliance score if applicable
        if rule_based and quality_scores and any("rule_compliance" in miner_metrics["name_metrics"].get(name, {}) for name in seed_names):
            rule_scores = [
                miner_metrics["name_metrics"].get(name, {}).get("rule_compliance", {}).get("score", 0.0)
                for name in seed_names if name in miner_metrics["name_metrics"]
            ]
            if rule_scores:
                miner_metrics["rule_compliance"]["overall_score"] = float(sum(rule_scores) / len(rule_scores))
                #bt.logging.info(f"Overall rule compliance score: {miner_metrics['rule_compliance']['overall_score']:.3f}")
        
        # Grade DOB variations before final reward calculation
        dob_grading_score = _grade_dob_variations(variations, seed_dob, miner_metrics)
        miner_metrics["dob_grading"] = dob_grading_score
        
        # Grade address variations before final reward calculation
        start_time = time.time()
        address_grading_score = _grade_address_variations(variations, seed_addresses, miner_metrics)
        miner_metrics["address_grading"] = address_grading_score
        end_time = time.time()
        bt.logging.info(f"Address grading time: {end_time - start_time:.2f} seconds")
        
        # Calculate final reward incorporating DOB and address grading
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            avg_base_score = sum(base_scores) / len(base_scores)
            
            # Separate weights for each component
            quality_weight = 0.3
            dob_weight = 0.1
            address_weight = 0.6
            
            # Calculate each component separately
            quality_component = avg_quality * quality_weight
            
            # DOB component
            if dob_grading_score["overall_score"] == 0.0 and any(seed_dob):
                dob_component = 0.0  # 0 points for missing DOB variations
            else:
                dob_component = dob_grading_score["overall_score"] * dob_weight
            
            # Address component
            if address_grading_score["overall_score"] == 0.0 and any(seed_addresses):
                address_component = 0.0  # 0 points for missing address variations
            else:
                address_component = address_grading_score["overall_score"] * address_weight
            
            # Final quality is sum of all components
            final_quality = quality_component + dob_component + address_component
            
            rewards[i] = final_quality * completeness_multiplier
            miner_metrics["average_base_score"] = float(avg_base_score)
            miner_metrics["average_quality"] = float(avg_quality)
            miner_metrics["identity_quality"] = float(final_quality)
            miner_metrics["final_reward"] = float(rewards[i])
        else:
            rewards[i] = 0.0
            miner_metrics["average_quality"] = 0.0
            miner_metrics["identity_quality"] = 0.0
            miner_metrics["final_reward"] = 0.0
        
        # # # ADD DETAILED LOGGING HERE to debug 0.0 scores
        # # bt.logging.info(f"--- DEBUGGING REWARD FOR MINER {uid} ---")
        # # bt.logging.info(f"Seed names with variations: {[name for name in seed_names if name in variations and variations[name]]}")
        # # bt.logging.info(f"Quality scores per name: {quality_scores}")
        # # bt.logging.info(f"Average quality score: {miner_metrics['average_quality']:.4f}")
        # # bt.logging.info(f"Completeness multiplier (1.0 - penalty): {completeness_multiplier:.4f}")
        # # bt.logging.info(f"Final reward (avg_quality * completeness_multiplier): {rewards[i]:.4f}")
        # # bt.logging.info(f"--- END DEBUGGING REWARD FOR MINER {uid} ---")
        
        # #bt.logging.info(f"Miner {uid} final Score: {rewards[i]}")
        # #bt.logging.info(f"Miner {uid} penalties: {miner_metrics['penalties']}")
        # if 'rule_compliance' in miner_metrics:
        #     bt.logging.info(f"Miner {uid} rule compliance: {miner_metrics['rule_compliance']['overall_score']}")
        # else:
        #     bt.logging.info(f"Miner {uid} rule compliance: 0.0")
        bt.logging.info(f"Miner {uid} Base quality scores: {quality_scores}")
        bt.logging.info(f"Miner {uid} average quality: {miner_metrics['average_quality']}")
        bt.logging.info(f"Miner {uid} completeness multiplier: {miner_metrics['completeness_multiplier']}")
        bt.logging.info(f"Miner {uid} DOB score: {miner_metrics.get('dob_grading', {}).get('overall_score', 0.0)}")
        bt.logging.info(f"Miner {uid} Address score: {miner_metrics.get('address_grading', {}).get('overall_score', 0.0)}")
        bt.logging.info(f"Miner {uid} final Score: {miner_metrics['final_reward']}")
        detailed_metrics.append(miner_metrics)
        
    # After initial rewards are calculated, apply penalties for high similarity between miners
    # Process seed names for cheating detection using pre-transliterated names
    processed_seed_names = []
    for name, script in zip(seed_names, seed_script):
        if script != "latin":
            # Use pre-transliterated name
            processed_seed_names.append(transliterated_seed_names[name])
        else:
            # Use name as-is without script suffix processing
            processed_seed_names.append(name)
    
    bt.logging.info(f"Processed seed names for cheating detection: {processed_seed_names}")
    
    try:
        rewards, detailed_metrics = _calculate_similarity_and_penalties(
            responses, uids, processed_seed_names, detailed_metrics, rewards
        )
    except Exception as e:
        bt.logging.error(f"Error in similarity and penalty calculation: {str(e)}")
        bt.logging.warning("Using rewards without similarity penalties as fallback")
        # Keep the original rewards without applying similarity penalties
        # detailed_metrics would remain as calculated before the penalty step
    
    # Final verification: ensure rewards array length matches UIDs length
    if len(rewards) != len(uids):
        bt.logging.error(f"CRITICAL: Final rewards length ({len(rewards)}) does not match UIDs length ({len(uids)})")
        raise ValueError(f"Final length mismatch: rewards={len(rewards)}, uids={len(uids)}")
    
    bt.logging.info(f"Successfully calculated rewards for {len(rewards)} miners")
    bt.logging.debug(f"Final rewards: {rewards}")
    bt.logging.debug(f"Final UIDs: {uids}")
    
    return rewards, detailed_metrics


def calculate_rule_compliance_score(
    original_name: str,
    variations: List[str],
    target_rules: List[str],
    target_percentage: float = 0.3
) -> Tuple[float, Dict]:
    """
    Calculate how well the variations comply with the target rules.
    
    Args:
        original_name: The original name
        variations: List of name variations
        target_rules: List of rules that should be followed
        target_percentage: Percentage of variations that should comply with rules
        
    Returns:
        Tuple containing:
        - Compliance score (0-1)
        - Dictionary with detailed metrics
    """
    # bt.logging.info(f"\nCalculating rule compliance for '{original_name}'")
    # bt.logging.info(f"Target rules: {target_rules}")
    # bt.logging.info(f"Target percentage: {target_percentage * 100:.1f}%")
    
    if not variations or not target_rules:
        bt.logging.warning("No variations or no target rules provided for rule compliance calculation.")
        return 0.0, {
            "compliant_variations_by_rule": {},
            "rules_satisfied_by_variation": {},
            "compliance_ratio_overall_variations": 0.0,
            "overall_compliant_unique_variations_count": 0,
            "expected_compliant_variations_count": 0,
            "quantity_score": 0.0,
            "rule_diversity_factor": 0.0,
            "num_target_rules_met": 0,
            "total_target_rules": len(target_rules) if target_rules else 0,
            "score": 0.0
        }
    
    # Evaluate rule compliance
    # compliant_variations_by_rule is Dict[str (rule_name), List[str (variation)]]
    compliant_variations_by_rule, compliance_ratio_from_evaluator = evaluate_rule_compliance(
        original_name, 
        variations, 
        target_rules
    )
    
    # Check if no rules were possible for this name structure
    # If target_rules were provided but evaluate_rule_compliance returned empty results,
    # it means no rules were applicable to this name structure
    if target_rules and not compliant_variations_by_rule:
        bt.logging.debug(f"⚠️ No rules were applicable for '{original_name}' with target rules: {target_rules}")
        return 1.0, {
            "compliant_variations_by_rule": {},
            "rules_satisfied_by_variation": {},
            "compliance_ratio_overall_variations": 0.0,
            "overall_compliant_unique_variations_count": 0,
            "expected_compliant_variations_count": 0,
            "quantity_score": 1.0,
            "rule_diversity_factor": 1.0,
            "num_target_rules_met": 0,
            "total_target_rules": len(target_rules),
            "score": 1.0
        }
    
    # Create a dictionary to map each compliant variation to the list of rules it satisfied
    rules_satisfied_by_variation = {}
    for rule, rule_compliant_variations_list in compliant_variations_by_rule.items():
        for variation in rule_compliant_variations_list:
            if variation not in rules_satisfied_by_variation:
                rules_satisfied_by_variation[variation] = []
            # Ensure no duplicate rules (though unlikely with current evaluators)
            if rule not in rules_satisfied_by_variation[variation]:
                 rules_satisfied_by_variation[variation].append(rule)

    # Count unique variations that satisfied at least one rule (from the target_rules)
    overall_compliant_count = len(rules_satisfied_by_variation)
    expected_compliant_count = max(1, int(len(variations) * target_percentage))
    
    # bt.logging.info(f"Found {overall_compliant_count} unique variations complying with at least one target rule (expected ~{expected_compliant_count} based on target percentage)")
    
    # for rule, variations_list in compliant_variations_by_rule.items():
    #     # This logging shows all rules returned by evaluate_rule_compliance, which should be the target_rules
    #     bt.logging.info(f"Rule '{rule}': {len(variations_list)} variations matched")
    
    # Calculate the quantity-based compliance score
    ratio_of_actual_to_expected = overall_compliant_count / expected_compliant_count if expected_compliant_count > 0 else 0.0
    
    quantity_score = 0.0
    if ratio_of_actual_to_expected <= 0.0:
        quantity_score = 0.0
    elif ratio_of_actual_to_expected <= 1.0:  # At or below target
        quantity_score = ratio_of_actual_to_expected
    else:  # Above target - apply a gentler penalty
        quantity_score = max(0.5, 1.5 - 0.5 * ratio_of_actual_to_expected)
    
    #bt.logging.info(f"Overall compliance ratio vs target: {ratio_of_actual_to_expected:.2f}, Quantity-based score: {quantity_score:.2f}")

    # Calculate rule diversity factor
    num_target_rules_met = 0
    rule_diversity_factor = 0.0

    if not target_rules: # No specific rules targeted, so diversity is maximal or not applicable.
        rule_diversity_factor = 1.0
    elif overall_compliant_count == 0: # No variations complied with any target rule.
        # This case should have been handled earlier, but just in case
        rule_diversity_factor = 0.0
        num_target_rules_met = 0
    else:
        # Count how many of the *effective_rules* were satisfied by at least one variation.
        # compliant_variations_by_rule.keys() contains only the rules that were actually evaluated
        # (after filtering out impossible rules in evaluate_rule_compliance)
        satisfied_effective_rules = set()
        for rule_name, compliant_vars_for_rule_list in compliant_variations_by_rule.items():
            if compliant_vars_for_rule_list:  # Rule was satisfied by at least one variation
                satisfied_effective_rules.add(rule_name)
        num_target_rules_met = len(satisfied_effective_rules)
        
        # Calculate diversity based on effective rules (rules that were actually possible to apply)
        effective_rules_count = len(compliant_variations_by_rule)
        if effective_rules_count > 0:
            rule_diversity_factor = num_target_rules_met / effective_rules_count
        else:
            # No effective rules means no rules were possible for this name structure
            rule_diversity_factor = 1.0

    #bt.logging.info(f"Met {num_target_rules_met} out of {len(compliant_variations_by_rule)} effective rules. Rule diversity factor: {rule_diversity_factor:.2f}")

    # Final score combines quantity and diversity
    final_score = quantity_score * rule_diversity_factor
    # bt.logging.info(f"Final rule compliance score (quantity * diversity): {final_score:.2f}")
    
    return final_score, {
        "compliant_variations_by_rule": compliant_variations_by_rule,
        "rules_satisfied_by_variation": rules_satisfied_by_variation,
        "compliance_ratio_overall_variations": compliance_ratio_from_evaluator, # Ratio of variations that matched any rule to total variations
        "overall_compliant_unique_variations_count": overall_compliant_count,
        "expected_compliant_variations_count": expected_compliant_count,
        "quantity_score": float(quantity_score),
        "rule_diversity_factor": float(rule_diversity_factor),
        "num_target_rules_met": num_target_rules_met,
        "total_target_rules": len(target_rules),
        "score": float(final_score) # This is the score based on meeting the target rule_percentage and diversity
    }
