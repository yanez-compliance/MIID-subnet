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


def get_rewards(
    self,
    query: int,
    responses: List[float],
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.

    return np.array([reward(query, response) for response in responses])


def calculate_variation_quality(original_name: str, variations: List[str]) -> float:
    """
    Calculate the quality of name variations based on:
    1. Uniqueness (no duplicates)
    2. Orthographic similarity to original (using Levenshtein distance)
    3. Phonetic similarity to original (using Soundex)
    4. Reasonable length
    
    The function checks that:
    - At least 5 variations are orthographically similar (Levenshtein) ## TODO: change later based on requested number of variations
    - At least 5 variations are phonetically similar (Soundex) ## TODO: change later based on requested number of variations
    - All variations are unique
    - All variations have reasonable length
    
    Args:
    - original_name: The original name
    - variations: List of name variations
    
    Returns:
    - float: Quality score between 0 and 1
    """
    # If no variations, return 0
    if not variations:
        return 0.0
    
    # Check for duplicates
    unique_variations = list(set(variations))
    uniqueness_score = len(unique_variations) / len(variations)
    
    # Convert original name to lowercase for comparison
    original_name_lower = original_name.lower()
    
    # Get Soundex code for the original name
    original_soundex = jellyfish.soundex(original_name_lower)
    
    # Lists to store similarity scores
    levenshtein_scores = []
    soundex_matches = []
    
    # Calculate similarity scores for each variation
    for var in unique_variations:
        var_lower = var.lower()
        
        # Calculate Levenshtein distance (orthographic similarity)
        distance = Levenshtein.distance(original_name_lower, var_lower)
        max_len = max(len(original_name_lower), len(var_lower))
        
        # Convert distance to similarity (1 = identical, 0 = completely different)
        levenshtein_similarity = 1 - (distance / max_len)
        
        # We want variations that are similar but not identical
        # Ideal similarity is around 0.7-0.8
        adjusted_levenshtein = 1.0 - abs(0.75 - levenshtein_similarity)
        levenshtein_scores.append((var, adjusted_levenshtein))
        
        # Calculate Soundex similarity (phonetic similarity)
        var_soundex = jellyfish.soundex(var_lower)
        
        # Check if Soundex codes match (indicating phonetic similarity)
        soundex_match = 1.0 if var_soundex == original_soundex else 0.0
        soundex_matches.append((var, soundex_match))
    
    # Sort variations by Levenshtein similarity (orthographic)
    levenshtein_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Count how many variations have good orthographic similarity (top 5)
    orthographic_count = min(5, len(levenshtein_scores))
    orthographic_score = sum([score for _, score in levenshtein_scores[:orthographic_count]]) / 5
    
    # Count how many variations have matching Soundex (phonetic similarity)
    phonetic_matches = [score for _, score in soundex_matches if score > 0]
    phonetic_count = len(phonetic_matches)
    phonetic_score = min(phonetic_count / 5, 1.0)  # Normalize to max of 1.0
    
    # Calculate length reasonableness
    length_scores = []
    for var in unique_variations:
        # Penalize variations that are too short or too long compared to original
        original_len = len(original_name)
        var_len = len(var)
        
        # Ideal length is within 30% of original length
        length_ratio = min(var_len / original_len, original_len / var_len)
        length_scores.append(length_ratio)
    
    length_score = sum(length_scores) / len(length_scores) if length_scores else 0
    
    # Log detailed scores for debugging
    bt.logging.debug(f"Original name: {original_name}")
    bt.logging.debug(f"Uniqueness score: {uniqueness_score}")
    bt.logging.debug(f"Orthographic score: {orthographic_score} ({orthographic_count} good matches)")
    bt.logging.debug(f"Phonetic score: {phonetic_score} ({phonetic_count} Soundex matches)")
    bt.logging.debug(f"Length score: {length_score}")
    
    # Combine scores with weights
    # - 25% for uniqueness
    # - 30% for orthographic similarity
    # - 30% for phonetic similarity
    # - 15% for reasonable length
    final_score = (
        0.25 * uniqueness_score +
        0.30 * orthographic_score +
        0.30 * phonetic_score +
        0.15 * length_score
    )
    
    bt.logging.debug(f"Final quality score: {final_score}")
    
    return final_score


def get_name_variation_rewards(
    self,
    seed_names: List[str],
    responses: List[Dict[str, List[str]]],
    uids: List[int]
) -> np.ndarray:
    """
    Calculate rewards for name variation responses.
    
    This function evaluates the quality of name variations provided by miners,
    considering:
    1. Whether the miner responded at all
    2. Whether the miner provided variations for all requested names
    3. The quality of the variations (uniqueness, orthographic similarity, 
       phonetic similarity, and reasonable length)
    
    Args:
    - seed_names: List of original names
    - responses: List of dictionaries mapping names to variations
    - uids: List of miner UIDs
    
    Returns:
    - np.ndarray: Array of rewards for each miner
    """
    rewards = np.zeros(len(responses))
    bt.logging.info(f"######################### Responses: {responses}")
    bt.logging.info(f"######################### UIDs: {uids}")
    bt.logging.info(f"######################### Seed names: {seed_names}")
    for i, (response, uid) in enumerate(zip(responses, uids)):
        if response is None:
            bt.logging.warning(f"+++++++++++++++++ Miner {uid} returned None response")
            continue
            
        # Calculate reward for this miner
        miner_reward = 0.0
        
        for name in seed_names:
            if name not in response:
                bt.logging.warning(f"+++++++++++++++++ Miner {uid} missing variations for name: {name}")
                continue
                
            variations = response[name]
            bt.logging.info(f"+++++++++++++++++ Variations: {variations}")
            bt.logging.info(f"+++++++++++++++++ Name: {name}")
            # Check if we have enough variations
            if not variations or len(variations) < 5:
                bt.logging.warning(f"+++++++++++++++++ Miner {uid} returned too few variations for {name}: {len(variations)}")
                continue
                
            # Reward for number of variations (up to 10)
            variation_count = min(len(variations), 10)
            count_reward = variation_count / 10.0
            bt.logging.info(f"+++++++++++++++++ Count reward: {count_reward}")
            # Reward for quality of variations
            quality_reward = calculate_variation_quality(name, variations)
            bt.logging.info(f"+++++++++++++++++ Quality reward: {quality_reward}")
            # Combine rewards
            name_reward = 0.2 * count_reward + 0.8 * quality_reward
            miner_reward += name_reward / len(seed_names)
            bt.logging.info(f"+++++++++++++++++ Miner reward: {miner_reward}")
        rewards[i] = miner_reward
        bt.logging.info(f"+++++++++++++++++ Miner {uid} received reward: {miner_reward:.4f}")
        
    return rewards
