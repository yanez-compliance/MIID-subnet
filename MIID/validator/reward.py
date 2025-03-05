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


def get_name_variation_rewards(
    self,
    seed_names: List[str],
    responses: List[Dict[str, List[str]]],
    uids: List[int]
) -> np.ndarray:
    """
    Calculate rewards for name variation responses.
    
    Args:
    - seed_names: List of original names
    - responses: List of dictionaries mapping names to variations
    - uids: List of miner UIDs
    
    Returns:
    - np.ndarray: Array of rewards for each miner
    """
    rewards = np.zeros(len(responses))
    
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
            
            # Check if we have enough variations
            if not variations or len(variations) < 5:
                bt.logging.warning(f"Miner {uid} returned too few variations for {name}: {len(variations)}")
                continue
                
            # Reward for number of variations (up to 10)
            variation_count = min(len(variations), 10)
            count_reward = variation_count / 10.0
            
            # Reward for quality of variations
            quality_reward = calculate_variation_quality(name, variations)
            
            # Combine rewards
            name_reward = 0.5 * count_reward + 0.5 * quality_reward
            miner_reward += name_reward / len(seed_names)
            
        rewards[i] = miner_reward
        
    return rewards


def calculate_variation_quality(original_name: str, variations: List[str]) -> float:
    """
    Calculate the quality of name variations based on:
    1. Uniqueness (no duplicates)
    2. Similarity to original (using Levenshtein distance)
    3. Reasonable length
    
    Args:
    - original_name: The original name
    - variations: List of name variations
    
    Returns:
    - float: Quality score between 0 and 1
    """
    # Check for duplicates
    unique_variations = set(variations)
    uniqueness_score = len(unique_variations) / len(variations) if variations else 0
    
    # Calculate similarity scores
    similarity_scores = []
    for var in unique_variations:
        # Normalize Levenshtein distance to a similarity score between 0 and 1
        # We want variations that are similar but not identical
        distance = Levenshtein.distance(original_name.lower(), var.lower())
        max_len = max(len(original_name), len(var))
        
        # Convert distance to similarity (1 = identical, 0 = completely different)
        similarity = 1 - (distance / max_len)
        
        # We want variations that are similar but not identical
        # Ideal similarity is around 0.7-0.8
        adjusted_similarity = 1.0 - abs(0.75 - similarity)
        similarity_scores.append(adjusted_similarity)
    
    # Average similarity score
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    # Combine scores (equal weights)
    return 0.5 * uniqueness_score + 0.5 * avg_similarity
