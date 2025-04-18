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
from typing import List, Dict
import bittensor as bt
import Levenshtein
import jellyfish
import os
import csv
import time
import traceback


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
    Calculate phonetic similarity between two strings using Soundex.
    
    Args:
        original_name: The original name
        variation: The variation to compare against
        
    Returns:
        Phonetic similarity score between 0 and 1
    """
    try:
        soundex_original = jellyfish.soundex(original_name)
        soundex_variation = jellyfish.soundex(variation)
        
        # Calculate phonetic similarity score (0-1)
        if soundex_original == soundex_variation:
            return 1.0
        else:
            # Use Levenshtein distance as a fallback
            return 1.0 - (Levenshtein.distance(soundex_original, soundex_variation) / 
                         max(len(soundex_original), len(soundex_variation)))
    except Exception as e:
        bt.logging.warning(f"Error calculating phonetic score: {str(e)}")
        return 0.0

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

def calculate_variation_quality(
    original_name: str, 
    variations: List[str],
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    expected_count: int = 10  # Add parameter for expected variation count
) -> float:
    """
    Calculate the quality of execution vectors (name variations) for threat detection.
    
    This function evaluates how effective a set of name variations would be as execution
    vectors in an identity screening bypass attempt. It considers multiple factors:
    
    Args:
        original_name: The original identity name to compare against
        variations: List of execution vector variations to evaluate
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        expected_count: Expected number of execution vectors
        
    Returns:
        Quality score between 0 and 1 indicating effectiveness as bypass vectors
    """
    # Default similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
    
    # Define the boundaries for each similarity level
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
    
    # 1. Check if count matches expected count with 20% tolerance
    tolerance = 0.2  # 20% tolerance
    tolerance_range = expected_count * tolerance
    
    # Calculate how far outside the tolerance range we are
    actual_count = len(variations)
    lower_bound = expected_count - tolerance_range
    upper_bound = expected_count + tolerance_range
    
    if lower_bound <= actual_count <= upper_bound:
        # Within tolerance range - perfect score
        count_score = 1.0
    else:
        # Outside tolerance range - calculate penalty
        if actual_count < lower_bound:
            deviation = lower_bound - actual_count
        else:  # actual_count > upper_bound
            deviation = actual_count - upper_bound
            
        # Calculate score with penalty for deviation beyond tolerance
        count_score = 1.0 - min(1.0, deviation / expected_count)
    
    # 2. Check for uniqueness - penalize duplicates
    unique_variations = list(set(variations))
    uniqueness_score = len(unique_variations) / len(variations) if variations else 0
    
    # 3. Calculate length reasonableness
    length_scores = []
    for var in unique_variations:
        # Penalize variations that are too short or too long compared to original
        original_len = len(original_name)
        var_len = len(var)
        
        # Ideal length is within 30% of original length
        length_ratio = min(var_len / original_len, original_len / var_len)
        length_scores.append(length_ratio)
    
    length_score = sum(length_scores) / len(length_scores) if length_scores else 0
    
    # Calculate phonetic similarity scores for each variation
    phonetic_scores = []
    for variation in variations:
        phonetic_score = calculate_phonetic_similarity(original_name, variation)
        phonetic_scores.append(phonetic_score)
    
    # Calculate orthographic similarity scores for each variation
    orthographic_scores = []
    for variation in variations:
        orthographic_score = calculate_orthographic_similarity(original_name, variation)
        orthographic_scores.append(orthographic_score)
    
    # Sort scores to analyze distribution
    phonetic_scores.sort()
    orthographic_scores.sort()
    
    # Calculate quality score based on distribution of scores
    phonetic_quality = 0.0
    orthographic_quality = 0.0
    
    # Count how many scores fall into each range (for phonetic)
    phonetic_counts = {}
    for level, (lower, upper) in phonetic_boundaries.items():
        # Count scores that fall within this range
        if level == "Light":  # Special case for Light (includes upper bound)
            count = sum(1 for score in phonetic_scores if lower <= score <= upper)
        else:
            count = sum(1 for score in phonetic_scores if lower <= score < upper)
        phonetic_counts[level] = count
    
    # For each desired level, calculate how well we match the target
    for level, percentage in phonetic_similarity.items():
        # Calculate how many scores should be in this range
        target_count = int(percentage * len(variations))
        actual_count = phonetic_counts.get(level, 0)
        
        # Calculate match quality for this level
        if target_count > 0:
            match_quality = min(actual_count / target_count, 1.0)
            phonetic_quality += percentage * match_quality
    
    # Repeat for orthographic similarity - count by range
    orthographic_counts = {}
    for level, (lower, upper) in orthographic_boundaries.items():
        # Count scores that fall within this range
        if level == "Light":  # Special case for Light (includes upper bound)
            count = sum(1 for score in orthographic_scores if lower <= score <= upper)
        else:
            count = sum(1 for score in orthographic_scores if lower <= score < upper)
        orthographic_counts[level] = count
    
    # For each desired level, calculate match quality
    for level, percentage in orthographic_similarity.items():
        # Calculate how many scores should be in this range
        target_count = int(percentage * len(variations))
        actual_count = orthographic_counts.get(level, 0)
        
        # Calculate match quality for this level
        if target_count > 0:
            match_quality = min(actual_count / target_count, 1.0)
            orthographic_quality += percentage * match_quality
    
    # Calculate final quality score with all factors
    # Weight factors according to importance
    similarity_weight = 0.6  # Combined weight for phonetic and orthographic similarity
    count_weight = 0.05       # Weight for having the correct number of variations
    uniqueness_weight = 0.3  # Weight for having unique variations
    length_weight = 0.05      # Weight for reasonable length variations
    
    similarity_score = (phonetic_quality + orthographic_quality) / 2  # Average of both similarities
    
    final_score = (
        similarity_weight * similarity_score +
        count_weight * count_score +
        uniqueness_weight * uniqueness_score +
        length_weight * length_score
    )
    
    return final_score


def get_name_variation_rewards(
    self,
    seed_names: List[str],
    responses: List[Dict[str, List[str]]],
    uids: List[int],
    variation_count: int = 10,
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None
) -> np.ndarray:
    """
    Calculate rewards for execution vectors (name variations) that simulate threat scenarios.
    
    This function evaluates how well miners generate execution vectors (name variations)
    that could be used to bypass identity screening systems. The evaluation considers:
    1. Adherence to the specified threat scenario parameters
    2. Quality and diversity of execution vectors
    3. Effectiveness as potential bypass methods based on similarity metrics
    
    Args:
        seed_names: Original identity names to generate variations for
        responses: List of execution vector responses from miners
        uids: List of UIDs corresponding to responses
        variation_count: Expected number of execution vectors per identity
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        
    Returns:
        Array of rewards for each miner based on execution vector quality
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
    
    # Generate a unique run ID
    run_id = f"run_{int(time.time())}"
    
    # Save variations to CSV
    try:
        save_variations_to_csv(
            self,
            seed_names,
            responses,
            uids,
            run_id
        )
    except Exception as e:
        bt.logging.error(f"Error calling save_variations_to_csv: {str(e)}")
        traceback.print_exc()
    
    rewards = np.zeros(len(responses))
    
    # Process each miner's response
    for i, (response, uid) in enumerate(zip(responses, uids)):
        if not hasattr(response, 'variations') or not response.variations:
            bt.logging.warning(f"Miner {uid} returned invalid or empty response")
            rewards[i] = 0.0
            continue
            
        variations = response.variations
        quality_scores = []
        
        # Initialize penalties
        extra_penalty = 0.0
        missing_penalty = 0.0
        
        # Calculate penalty for unexpected names (extra variations)
        invalid_names = set(variations.keys()) - set(seed_names)
        if invalid_names:
            bt.logging.warning(f"Miner {uid} provided variations for unexpected names: {invalid_names}")
            # 10% penalty per extra name, up to 70% max
            extra_penalty = min(0.7, len(invalid_names) * 0.1)
            
        # Calculate penalty for missing names
        missing_names = set(seed_names) - set(variations.keys())
        if missing_names:
            bt.logging.warning(f"Miner {uid} missing variations for names: {missing_names}")
            # 20% penalty per missing name, up to 90% max
            missing_penalty = min(0.9, len(missing_names) * 0.2)
        
        # Calculate total penalty (additive) with cap of 0.9
        total_penalty = min(0.9, extra_penalty + missing_penalty)
        
        # Calculate completeness multiplier (minimum 0.1)
        completeness_multiplier = max(0.1, 1.0 - total_penalty)
        
        # Process each seed name
        for name in seed_names:
            if name not in variations or not variations[name]:
                bt.logging.warning(f"Miner {uid} did not provide variations for '{name}'")
                continue
                
            # Get variations for this name
            name_variations = variations[name]
            
            # Calculate scores for this name's variations
            try:
                # Calculate quality score for all variations of this name
                quality = calculate_variation_quality(
                    name, 
                    name_variations,
                    phonetic_similarity=phonetic_similarity,
                    orthographic_similarity=orthographic_similarity,
                    expected_count=variation_count  # Pass the expected count
                )
                quality_scores.append(quality)
            except Exception as e:
                bt.logging.error(f"Error calculating quality for miner {uid}, name '{name}': {str(e)}")
                traceback.print_exc()
        
        # Calculate average quality across all names
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            # Apply completeness multiplier to final score
            rewards[i] = avg_quality * completeness_multiplier
        else:
            rewards[i] = 0.0
                
    return rewards


def save_variations_to_csv(
    self,
    seed_names: List[str],
    responses: List,
    uids: List[int],
    run_id: str
):
    """
    Save name variations from miners to a CSV file.
    
    Args:
        self: The validator object
        seed_names: Original names that variations were generated for
        responses: List of response objects from miners
        uids: List of UIDs corresponding to responses
        run_id: Unique identifier for this validation run
    """
    try:
        # Determine the database directory with fallbacks
        database_dir = None
        
        # Try to use logging_dir from config if it exists
        try:
            if hasattr(self, 'config') and hasattr(self.config, 'logging') and hasattr(self.config.logging, 'logging_dir'):
                base_dir = self.config.logging.logging_dir
                database_dir = os.path.join(base_dir, "Database")
                bt.logging.info(f"Using config logging directory: {base_dir}")
        except Exception as e:
            bt.logging.warning(f"Could not access config.logging.logging_dir: {str(e)}")
        
        # Fallback 1: Use current working directory
        if not database_dir:
            database_dir = os.path.join(os.getcwd(), "Database")
            bt.logging.info(f"Using current working directory fallback: {os.getcwd()}")
        
        # Create directory
        os.makedirs(database_dir, exist_ok=True)
        
        # Test if directory is writable
        if not os.access(database_dir, os.W_OK):
            # Fallback 2: Use home directory if current dir is not writable
            home_dir = os.path.expanduser("~")
            database_dir = os.path.join(home_dir, "Database")
            os.makedirs(database_dir, exist_ok=True)
            bt.logging.info(f"Using home directory fallback: {database_dir}")
            
            # Final check if this is writable
            if not os.access(database_dir, os.W_OK):
                bt.logging.error(f"Cannot write to any database directory. Aborting CSV save.")
                return
        
        # Create or append to the CSV file
        timestamp = int(time.time())
        csv_filename = os.path.join(database_dir, f"name_variations_log.csv")
        
        bt.logging.info(f"Saving variations to CSV file: {csv_filename}")
        
        # Check if file exists to decide whether to write headers
        file_exists = os.path.isfile(csv_filename)
        
        # Debug information to validate if responses have variations
        variation_count_per_miner = {}
        for i, (response, uid) in enumerate(zip(responses, uids)):
            if hasattr(response, 'variations') and response.variations:
                count = sum(len(variations) for variations in response.variations.values())
                variation_count_per_miner[uid] = count
            else:
                variation_count_per_miner[uid] = 0
        
        bt.logging.info(f"Variation counts by miner: {variation_count_per_miner}")
        
        # Prepare CSV file for logging variations
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'run_id', 'seed_name', 'variation', 'miner_uid', 'phonetic_score', 'orthographic_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if the file is new
            if not file_exists:
                writer.writeheader()
                bt.logging.info(f"Created new CSV file with headers")
            
            variation_count = 0
            # Process each miner's response
            for response, uid in zip(responses, uids):
                if not hasattr(response, 'variations') or not response.variations:
                    bt.logging.warning(f"Miner {uid} returned invalid or empty response")
                    continue
                    
                variations = response.variations
                
                # Process each seed name
                for name in seed_names:
                    if name not in variations or not variations[name]:
                        bt.logging.warning(f"Miner {uid} did not provide variations for '{name}'")
                        continue
                        
                    # Get variations for this name
                    name_variations = variations[name]
                    
                    # Log individual scores for each variation
                    for variation in name_variations:
                        # Calculate scores using the same functions as in reward calculation
                        phonetic_score = calculate_phonetic_similarity(name, variation)
                        orthographic_score = calculate_orthographic_similarity(name, variation)
                        
                        # Log to CSV
                        writer.writerow({
                            'timestamp': timestamp,
                            'run_id': run_id,
                            'seed_name': name,
                            'variation': variation,
                            'miner_uid': uid,
                            'phonetic_score': phonetic_score,
                            'orthographic_score': orthographic_score
                        })
                        variation_count += 1
        
        bt.logging.info(f"Successfully saved {variation_count} variations to {csv_filename}")
        
    except Exception as e:
        bt.logging.error(f"Error saving variations to CSV: {str(e)}")
        traceback.print_exc()
