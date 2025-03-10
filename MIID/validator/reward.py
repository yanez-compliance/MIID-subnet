# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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


# def get_rewards(
#     self,
#     query: int,
#     responses: List[float],
# ) -> np.ndarray:
#     """
#     Returns an array of rewards for the given query and responses.

#     Args:
#     - query (int): The query sent to the miner.
#     - responses (List[float]): A list of responses from the miner.

#     Returns:
#     - np.ndarray: An array of rewards for the given query and responses.
#     """
#     # Get all the reward results by iteratively calling your reward() function.

#     return np.array([reward(query, response) for response in responses])


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
    phonetic_thresholds: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    orthographic_thresholds: Dict[str, float] = None
) -> float:
    """
    Calculate the quality of name variations based on phonetic and orthographic similarity.
    
    Args:
        original_name: The original name to compare against
        variations: List of name variations to evaluate
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        phonetic_thresholds: Thresholds for different phonetic similarity levels
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_thresholds: Thresholds for different orthographic similarity levels
        
    Returns:
        Quality score between 0 and 1
    """
    # Default similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
    
    # Default thresholds if none provided
    if phonetic_thresholds is None:
        phonetic_thresholds = {
            "Light": 0.8,
            "Medium": 0.6,
            "Far": 0.4
        }
    
    if orthographic_thresholds is None:
        orthographic_thresholds = {
            "Light": 0.7,
            "Medium": 0.5,
            "Far": 0.3
        }
    
    # Calculate phonetic similarity scores for each variation
    phonetic_scores = [calculate_phonetic_similarity(original_name, variation) for variation in variations]
    
    # Calculate orthographic similarity scores for each variation
    orthographic_scores = [calculate_orthographic_similarity(original_name, variation) for variation in variations]
    
    # Evaluate how well the variations match the requested similarity distributions
    phonetic_quality = 0.0
    orthographic_quality = 0.0
    
    # For each phonetic similarity level, check if the right percentage of variations meet the threshold
    for level, percentage in phonetic_similarity.items():
        threshold = phonetic_thresholds.get(level, 0.6)  # Default to Medium if level not found
        target_count = int(len(variations) * percentage)
        if target_count == 0:
            continue
            
        # Count variations that meet this threshold
        actual_count = sum(1 for score in phonetic_scores if score >= threshold)
        
        # Calculate quality for this level
        level_quality = min(1.0, actual_count / target_count)
        phonetic_quality += level_quality * percentage
    
    # For each orthographic similarity level, check if the right percentage of variations meet the threshold
    for level, percentage in orthographic_similarity.items():
        threshold = orthographic_thresholds.get(level, 0.6)  # Default to Medium if level not found
        target_count = int(len(variations) * percentage)
        if target_count == 0:
            continue
            
        # Count variations that meet this threshold
        actual_count = sum(1 for score in orthographic_scores if score >= threshold)
        
        # Calculate quality for this level
        level_quality = min(1.0, actual_count / target_count)
        orthographic_quality += level_quality * percentage
    
    # Combine phonetic and orthographic quality (equal weighting)
    overall_quality = (phonetic_quality + orthographic_quality) / 2.0
    
    return overall_quality


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
    Calculate rewards for name variation responses.
    
    Args:
        seed_names: Original names to generate variations for
        responses: List of response objects from miners
        uids: List of UIDs corresponding to responses
        variation_count: Expected number of variations per name
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        
    Returns:
        Array of rewards for each miner
    """
    # Default similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
        
    # Thresholds for different similarity levels
    phonetic_thresholds = {
        "Light": 0.8,  # High similarity
        "Medium": 0.6, # Moderate similarity
        "Far": 0.3     # Low similarity
    }
    
    orthographic_thresholds = {
        "Light": 0.7,  # High similarity
        "Medium": 0.5, # Moderate similarity
        "Far": 0.2     # Low similarity
    }
    
    # Save variations to CSV
    try:
        save_variations_to_csv(
            seed_names,
            responses,
            uids,
            phonetic_thresholds,
            orthographic_thresholds
        )
    except Exception as e:
        bt.logging.error(f"Error saving variations to CSV: {str(e)}")
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
                    phonetic_similarity,
                    phonetic_thresholds,
                    orthographic_similarity,
                    orthographic_thresholds
                )
                quality_scores.append(quality)
            except Exception as e:
                bt.logging.error(f"Error calculating quality for miner {uid}, name '{name}': {str(e)}")
                traceback.print_exc()
        
        # Calculate average quality across all names
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            rewards[i] = avg_quality
        else:
            rewards[i] = 0.0
                
    return rewards


def save_variations_to_csv(
    seed_names: List[str],
    responses: List,
    uids: List[int],
    phonetic_thresholds: Dict[str, float],
    orthographic_thresholds: Dict[str, float]
):
    """
    Save name variations from miners to a CSV file.
    
    Args:
        seed_names: Original names that variations were generated for
        responses: List of response objects from miners
        uids: List of UIDs corresponding to responses
        phonetic_thresholds: Thresholds for different phonetic similarity levels
        orthographic_thresholds: Thresholds for different orthographic similarity levels
    """
    # Create Database directory if it doesn't exist
    database_dir = "/Database/"
    os.makedirs(database_dir, exist_ok=True)
    
    # Create or append to the CSV file
    timestamp = int(time.time())
    csv_filename = os.path.join(database_dir, f"name_variations_log.csv")
    
    # Check if file exists to decide whether to write headers
    file_exists = os.path.isfile(csv_filename)
    
    # Prepare CSV file for logging variations
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'seed_name', 'variation', 'miner_uid', 'phonetic_score', 'orthographic_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        
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
                        'seed_name': name,
                        'variation': variation,
                        'miner_uid': uid,
                        'phonetic_score': phonetic_score,
                        'orthographic_score': orthographic_score
                    })
    
    bt.logging.info(f"Saved name variations data to {csv_filename}")
