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
from typing import List, Dict, Tuple, Any
import bittensor as bt
import Levenshtein
import jellyfish
import os
import csv
import time
import traceback
import math
import random

# Import rule_evaluator for rule-based compliance checking
from MIID.validator.rule_evaluator import evaluate_rule_compliance

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
        
        if p_score < 0.3 and o_score < 0.3:
            bt.logging.warning(
                f"Very low similarity for variation '{variation}': "
                f"phonetic={p_score:.3f}, orthographic={o_score:.3f}"
            )
    
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
    # bt.logging.info(f"Similarity score: {similarity_score:.3f} (phonetic: {phonetic_quality:.3f}, orthographic: {orthographic_quality:.3f})")
    
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
    
    # Detailed logging of final score components
    # bt.logging.info(f"\nFinal score breakdown for '{original_part}':")
    # bt.logging.info(f"  - Similarity ({similarity_weight*100:.0f}%):")
    # bt.logging.info(f"    * Phonetic: {phonetic_quality:.3f}")
    # bt.logging.info(f"    * Orthographic: {orthographic_quality:.3f}")
    # bt.logging.info(f"    * Combined: {similarity_score:.3f}")
    # bt.logging.info(f"  - Count ({count_weight*100:.0f}%): {count_score:.3f}")
    # bt.logging.info(f"  - Uniqueness ({uniqueness_weight*100:.0f}%): {uniqueness_score:.3f}")
    # bt.logging.info(f"  - Length ({length_weight*100:.0f}%): {length_score:.3f}")
    # bt.logging.info(f"  - Total score: {final_score:.3f}")
    
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
    if len(name_parts) > 2:
        return {
            "first_name_weight": normalized_weights[0],
            "middle_name_weight": sum(normalized_weights[1:-1]),
            "last_name_weight": normalized_weights[-1]
        }
    else:
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
        
        # Filter out rules that are impossible for the given name structure
        for rule in target_rules:
            if rule in ('name_parts_permutations', 'initial_only_first_name') and len(original_name.split()) < 2:
                bt.logging.debug(f"Skipping impossible rule '{rule}' for single-part name '{original_name}'")
                continue
            effective_target_rules.append(rule)

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
        
    # If no non-rule variations were expected or provided, base_score is not penalized
    if expected_base_count == 0 or len(non_rule_compliant_variations) == 0:
        base_score = 1.0 # Or some other neutral value, 1.0 seems fair to not penalize.
    
    # Apply rule compliance to final score using weights from the global config
    rule_compliance_weight = MIID_REWARD_WEIGHTS["rule_compliance_weight"]
    
    # If rules were requested but none were applicable to this name, adjust weights
    # to base the score entirely on similarity.
    if rule_based and "selected_rules" in rule_based and not effective_target_rules:
        bt.logging.debug(f"No rules applicable for '{original_name}', adjusting weights. Base score will be final score.")
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
    
    bt.logging.info(f"{'='*50}\n")
    return final_score, detailed_metrics


def get_name_variation_rewards(
    self,
    seed_names: List[str],
    responses: List[Dict[str, List[str]]],
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
    if rule_based:
        bt.logging.info(f"Rule-based requirements: {rule_based}")
        bt.logging.info(f"Target rules: {rule_based.get('selected_rules', [])}")
        bt.logging.info(f"Target rule-based percentage: {rule_based.get('rule_percentage', 30)}%")
    else:
        bt.logging.info("No rule-based requirements specified")
    
    # Process each miner's response
    for i, (response, uid) in enumerate(zip(responses, uids)):
        bt.logging.info(f"\n{'='*50}")
        bt.logging.info(f"Processing miner {uid}")
        
        # Initialize metrics dictionary for this miner
        miner_metrics = {
            "quality_scores": {},
            "penalties": {
                "extra_names": 0.0,
                "missing_names": 0.0,
                "total_penalty": 0.0
            },
            "completeness_multiplier": 1.0,
            "name_metrics": {},
            "invalid_names": [],
            "missing_names": []
        }
        
        if not hasattr(response, 'variations') or not response.variations:
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
            
        variations = response.variations
        quality_scores = []
        
        # Calculate penalty for unexpected names (extra variations)
        invalid_names = set(variations.keys()) - set(seed_names)
        if invalid_names:
            bt.logging.warning(f"Miner {uid} provided variations for unexpected names: {invalid_names}")
            # 10% penalty per extra name, up to 70% max
            extra_penalty = min(0.7, len(invalid_names) * 0.1)
            bt.logging.info(f"Extra penalty: {extra_penalty}")
            miner_metrics["penalties"]["extra_names"] = float(extra_penalty)
            miner_metrics["invalid_names"] = list(invalid_names)
            
        # Calculate penalty for missing names
        missing_names = set(seed_names) - set(variations.keys())
        if missing_names:
            bt.logging.warning(f"Miner {uid} missing variations for names: {missing_names}")
            # 20% penalty per missing name, up to 90% max
            missing_penalty = min(0.9, len(missing_names) * 0.2)
            bt.logging.info(f"Missing penalty: {missing_penalty}")
            miner_metrics["penalties"]["missing_names"] = float(missing_penalty)
            miner_metrics["missing_names"] = list(missing_names)
        
        # Calculate total penalty and completeness multiplier
        total_penalty = min(0.9, miner_metrics["penalties"]["extra_names"] + miner_metrics["penalties"]["missing_names"])
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
        for name in seed_names:
            if name not in variations or not variations[name]:
                continue
                
            # Get variations for this name
            name_variations = variations[name]
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
                quality, name_detailed_metrics = calculate_variation_quality(
                    name,
                    name_variations,
                    phonetic_similarity=phonetic_similarity,
                    orthographic_similarity=orthographic_similarity,
                    expected_count=variation_count,
                    rule_based=rule_based  # Pass rule-based metadata
                )
                quality_scores.append(quality)
                miner_metrics["name_metrics"][name] = name_detailed_metrics
                
                # Extract rule compliance metrics if available
                if rule_based and "rule_compliance" in name_detailed_metrics:
                    miner_metrics["rule_compliance"]["by_name"][name] = name_detailed_metrics["rule_compliance"]
            except Exception as e:
                bt.logging.error(f"Error calculating quality for miner {uid}, name '{name}': {str(e)}")
                traceback.print_exc()
        
        # Calculate overall rule compliance score if applicable
        if rule_based and quality_scores and any("rule_compliance" in miner_metrics["name_metrics"].get(name, {}) for name in seed_names):
            rule_scores = [
                miner_metrics["name_metrics"].get(name, {}).get("rule_compliance", {}).get("score", 0.0)
                for name in seed_names if name in miner_metrics["name_metrics"]
            ]
            if rule_scores:
                miner_metrics["rule_compliance"]["overall_score"] = float(sum(rule_scores) / len(rule_scores))
                #bt.logging.info(f"Overall rule compliance score: {miner_metrics['rule_compliance']['overall_score']:.3f}")
        
        # Calculate final reward
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            rewards[i] = avg_quality * completeness_multiplier
            miner_metrics["average_quality"] = float(avg_quality)
            miner_metrics["final_reward"] = float(rewards[i])
        else:
            rewards[i] = 0.0
            miner_metrics["average_quality"] = 0.0
            miner_metrics["final_reward"] = 0.0
        
        #bt.logging.info(f"Miner {uid} final Score: {rewards[i]}")
        #bt.logging.info(f"Miner {uid} penalties: {miner_metrics['penalties']}")
        bt.logging.info(f"Miner {uid} rule compliance: {miner_metrics['rule_compliance']['overall_score']}")
        bt.logging.info(f"Miner {uid} Base quality scores: {quality_scores}")
        bt.logging.info(f"Miner {uid} average quality: {miner_metrics['average_quality']}")
        bt.logging.info(f"Miner {uid} completeness multiplier: {miner_metrics['completeness_multiplier']}")
        bt.logging.info(f"Miner {uid} final Score: {miner_metrics['final_reward']}")
        detailed_metrics.append(miner_metrics)
        
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
        quantity_score = max(0.0, 1.5 - 0.5 * ratio_of_actual_to_expected)
    
    bt.logging.info(f"Overall compliance ratio vs target: {ratio_of_actual_to_expected:.2f}, Quantity-based score: {quantity_score:.2f}")

    # Calculate rule diversity factor
    num_target_rules_met = 0
    rule_diversity_factor = 0.0

    if not target_rules: # No specific rules targeted, so diversity is maximal or not applicable.
        rule_diversity_factor = 1.0
    elif overall_compliant_count == 0: # No variations complied with any target rule.
        rule_diversity_factor = 0.0
        num_target_rules_met = 0
    else:
        # Count how many of the *target_rules* were satisfied by at least one variation.
        # compliant_variations_by_rule.keys() should be a subset of target_rules if evaluate_rule_compliance is strict.
        satisfied_target_rules = set()
        for rule_name, compliant_vars_for_rule_list in compliant_variations_by_rule.items():
            if rule_name in target_rules and compliant_vars_for_rule_list:
                satisfied_target_rules.add(rule_name)
        num_target_rules_met = len(satisfied_target_rules)
        rule_diversity_factor = num_target_rules_met / len(target_rules) if len(target_rules) > 0 else 1.0

    bt.logging.info(f"Met {num_target_rules_met} out of {len(target_rules)} target rules. Rule diversity factor: {rule_diversity_factor:.2f}")

    # Final score combines quantity and diversity
    final_score = quantity_score * rule_diversity_factor
    bt.logging.info(f"Final rule compliance score (quantity * diversity): {final_score:.2f}")
    
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
