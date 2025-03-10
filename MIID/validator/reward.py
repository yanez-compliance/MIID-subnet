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


def calculate_variation_quality(
    original_name: str, 
    variations: List[str],
    phonetic_similarity: Dict[str, float] = None,
    phonetic_thresholds: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    orthographic_thresholds: Dict[str, float] = None
) -> float:
    """
    Calculate the quality of name variations based on specified similarity requirements.
    
    Args:
        original_name: The original name
        variations: List of name variations
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        phonetic_thresholds: Dictionary mapping similarity levels to threshold values
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_thresholds: Dictionary mapping similarity levels to threshold values
        
    Returns:
        Float between 0 and 1 representing the quality of variations
    """
    if not variations:
        return 0.0
    
    # Set default values if not provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    
    if phonetic_thresholds is None:
        phonetic_thresholds = {
            "Light": 0.8,
            "Medium": 0.6,
            "Far": 0.4
        }
    
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
    
    if orthographic_thresholds is None:
        orthographic_thresholds = {
            "Light": 0.8,
            "Medium": 0.6,
            "Far": 0.4
        }
    
    # Calculate phonetic similarity scores for each variation
    phonetic_scores = []
    for variation in variations:
        # Use Soundex or other phonetic algorithm to compare
        try:
            soundex_original = jellyfish.soundex(original_name)
            soundex_variation = jellyfish.soundex(variation)
            
            # Calculate phonetic similarity score (0-1)
            if soundex_original == soundex_variation:
                phonetic_score = 1.0
            else:
                # Use Levenshtein distance as a fallback
                phonetic_score = 1.0 - (Levenshtein.distance(soundex_original, soundex_variation) / 
                                      max(len(soundex_original), len(soundex_variation)))
            
            phonetic_scores.append(phonetic_score)
        except Exception as e:
            bt.logging.warning(f"Error calculating phonetic score: {str(e)}")
            phonetic_scores.append(0.0)
    
    # Calculate orthographic similarity scores for each variation
    orthographic_scores = []
    for variation in variations:
        try:
            # Use Levenshtein distance to compare
            distance = Levenshtein.distance(original_name, variation)
            max_len = max(len(original_name), len(variation))
            
            # Calculate orthographic similarity score (0-1)
            orthographic_score = 1.0 - (distance / max_len)
            orthographic_scores.append(orthographic_score)
        except Exception as e:
            bt.logging.warning(f"Error calculating orthographic score: {str(e)}")
            orthographic_scores.append(0.0)
    
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
    
    This function evaluates the quality of name variations provided by miners,
    considering:
    1. Whether the miner responded at all
    2. Whether the miner provided variations for all requested names
    3. The quality of the variations based on the requested parameters
    
    Args:
    - seed_names: List of original names
    - responses: List of dictionaries mapping names to variations
    - uids: List of miner UIDs
    - variation_count: Number of variations requested
    - phonetic_similarity: Dictionary mapping similarity levels to percentages
                          (e.g., {"Light": 0.3, "Medium": 0.5, "Far": 0.2})
    - orthographic_similarity: Dictionary mapping similarity levels to percentages
                              (e.g., {"Light": 0.4, "Medium": 0.4, "Far": 0.2})
    
    Returns:
    - np.ndarray: Array of rewards for each miner
    """
    rewards = np.zeros(len(responses))
    
    # Set default values if not provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
    
    # Convert similarity levels to numerical thresholds
    phonetic_thresholds = {
        "Light": 0.8,  # High similarity threshold
        "Medium": 0.6,
        "Far": 0.4     # Low similarity threshold
    }
    
    orthographic_thresholds = {
        "Light": 0.8,  # High similarity threshold
        "Medium": 0.6,
        "Far": 0.4     # Low similarity threshold
    }
    
    # Log the evaluation parameters
    phonetic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in phonetic_similarity.items()])
    orthographic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in orthographic_similarity.items()])
    
    bt.logging.info(f"Evaluating responses with parameters: variation_count={variation_count}, " +
                  f"phonetic similarity: {phonetic_spec}, " +
                  f"orthographic similarity: {orthographic_spec}")
    
    for i, (response, uid) in enumerate(zip(responses, uids)):
        if response is None:
            bt.logging.warning(f"Miner {uid} returned None response")
            continue
            
        # Calculate reward for this miner
        miner_reward = 0.0
        
        for name in seed_names:
            if name not in response:
                bt.logging.warning(f"Miner {uid} missing variations for name: {name}")
                continue
                
            variations = response[name]
            
            # Check if the miner provided the requested number of variations
            variation_ratio = min(1.0, len(variations) / variation_count)
            
            # Calculate quality of variations based on the requested parameters
            quality = calculate_variation_quality(
                name, 
                variations, 
                phonetic_similarity=phonetic_similarity,
                phonetic_thresholds=phonetic_thresholds,
                orthographic_similarity=orthographic_similarity,
                orthographic_thresholds=orthographic_thresholds
            )
            
            # Combine the metrics
            name_reward = variation_ratio * quality
            miner_reward += name_reward / len(seed_names)
        
        rewards[i] = miner_reward
    
    return rewards
