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

"""
Validator Forward Module

This module implements the forward function for the name variation validator.
The forward function is responsible for:
1. Selecting random miners to query
2. Generating a list of names to request variations for
3. Sending the request to the selected miners
4. Evaluating the responses and updating miner scores
5. Saving the results for analysis

The module uses the NameVariationRequest protocol to communicate with miners,
and the get_name_variation_rewards function to evaluate the quality of the
responses.
"""

import time
import bittensor as bt
import json
import os
import random
from faker import Faker
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
import ollama

from MIID.protocol import IdentityRequest
from MIID.validator.reward import get_name_variation_rewards
from MIID.utils.uids import get_random_uids

EPOCH_MIN_TIME = 200 # seconds

# Constants for query generation
SIMILARITY_LEVELS = ["Light", "Medium", "Far"]
DEFAULT_VARIATION_COUNT = 10
DEFAULT_ORTHOGRAPHIC_SIMILARITY = "Medium"
DEFAULT_PHONETIC_SIMILARITY = "Medium"
DEFAULT_LLM_MODEL = "llama3.1:latest"


async def dendrite_with_retries(dendrite: bt.dendrite, axons: list, synapse, deserialize: bool, timeout: float, cnt_attempts=7):
    """
    Send requests to miners with automatic retry logic for failed connections.
    
    Args:
        dendrite: The dendrite object to use for communication
        axons: List of axons to query
        synapse: The synapse object containing the request
        deserialize: Whether to deserialize the response
        timeout: Timeout for each request in seconds
        cnt_attempts: Number of retry attempts for failed connections
        
    Returns:
        List of responses from miners
    """
    res = [None] * len(axons)
    idx = list(range(len(axons)))
    axons_copy = axons.copy()
    
    # Create a default empty response with required fields
    synapse_fields = {}
    if hasattr(synapse, 'names'):
        synapse_fields['names'] = synapse.names or []
    if hasattr(synapse, 'query_template'):
        synapse_fields['query_template'] = synapse.query_template or ""
    
    # Function to create a default response
    def create_default_response():
        try:
            # Create a new instance with the required fields
            if isinstance(synapse, bt.Synapse):
                # For IdentityRequest, we need to provide the required fields
                if synapse.__class__.__name__ == 'IdentityRequest':
                    return synapse.__class__(
                        names=synapse_fields.get('names', []),
                        query_template=synapse_fields.get('query_template', ""),
                        variations={}  # Empty variations dictionary
                    )
                # For other synapse types, try to create with minimal fields
                return synapse.__class__(**synapse_fields)
            else:
                # Fallback for non-synapse objects
                return type(synapse)()
        except Exception as e:
            bt.logging.error(f"Error creating default response: {str(e)}")
            # Last resort: return the original synapse with empty variations
            if hasattr(synapse, 'variations'):
                synapse.variations = {}
            return synapse
    
    for attempt in range(cnt_attempts):
        try:
            # Log retry information
            if attempt > 0:
                bt.logging.info(f"Retry attempt {attempt+1}/{cnt_attempts} for {len(axons_copy)} axons")
                
            # For later attempts, increase timeout dramatically
            current_timeout = timeout * (1 + (attempt * 1.0))  # Double multiplier from 0.5 to 1.0
            
            bt.logging.info(f"Sending dendrite request with timeout {current_timeout:.1f}s to {len(axons_copy)} axons")
            
            # Diagnostic logging before the call
            for i, axon in enumerate(axons_copy):
                bt.logging.debug(f"Axon {i}: {axon.hotkey} at {axon.ip}:{axon.port}")
            
            # Perform the dendrite call
            start_time = time.time()
            responses = await dendrite(
                axons=axons_copy,
                synapse=synapse,
                deserialize=deserialize,
                timeout=current_timeout
            )
            call_duration = time.time() - start_time
            
            bt.logging.info(f"Dendrite call returned after {call_duration:.1f}s (timeout was {current_timeout:.1f}s)")
            
            new_idx = []
            new_axons = []
            
            # Log some diagnostic info
            response_types = {}
            for resp in responses:
                resp_type = type(resp).__name__
                response_types[resp_type] = response_types.get(resp_type, 0) + 1
            bt.logging.info(f"Response types: {response_types}")
            
            for i, response in enumerate(responses):
                # Check if response is None or has connection errors
                if response is None:
                    bt.logging.warning(f"Received None response from axon {axons_copy[i]}")
                    if attempt == cnt_attempts - 1:
                        # On last attempt, use a default empty response
                        res[idx[i]] = create_default_response()
                    else:
                        # Add to retry list
                        new_idx.append(idx[i])
                        new_axons.append(axons_copy[i])
                # Check for broken pipe or connection errors (status code 422)
                elif hasattr(response, 'dendrite') and response.dendrite is not None and \
                     hasattr(response.dendrite, 'status_code') and \
                     response.dendrite.status_code is not None and \
                     int(response.dendrite.status_code) != 200:  # Check for any non-200 code
                    status_code = int(response.dendrite.status_code)
                    bt.logging.warning(f"Received error status {status_code} from axon {axons_copy[i]}")
                    if attempt == cnt_attempts - 1:
                        res[idx[i]] = response
                        bt.logging.warning(f"Failed to get response from axon {axons_copy[i]} after {cnt_attempts} attempts")
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_copy[i])
                else:
                    # Valid response
                    res[idx[i]] = response
            
            if len(new_idx):
                bt.logging.info(f'Found {len(new_idx)} connections to retry (attempt {attempt+1}/{cnt_attempts})')
            else:
                bt.logging.info(f'All miners responded successfully on attempt {attempt+1}')
                break
            
            idx = new_idx
            axons_copy = new_axons
            
            # Increase wait time between retries substantially
            retry_wait = 5 * (attempt + 1)  # 5s, 10s, 15s, etc.
            bt.logging.info(f"Waiting {retry_wait}s before retry attempt {attempt+2}")
            await asyncio.sleep(retry_wait)
            
        except Exception as e:
            bt.logging.error(f"Error in dendrite call: {str(e)}")
            # If this is the last attempt, fill remaining slots with empty responses
            if attempt == cnt_attempts - 1:
                for i in range(len(idx)):
                    if res[idx[i]] is None:
                        res[idx[i]] = create_default_response()
            await asyncio.sleep(2 * (attempt + 1)* 30)  # Use async sleep with increasing wait time
    
    # Ensure all responses are filled
    for i, r in enumerate(res):
        if r is None:
            bt.logging.warning(f"No response received for axon {axons[i]}")
            # Create an empty response to avoid None values
            res[i] = create_default_response()
            
    # Log final status
    valid_responses = sum(1 for r in res if hasattr(r, 'variations') and r.variations)
    bt.logging.info(f"Dendrite call completed with {valid_responses}/{len(res)} valid responses")
    
    return res

async def generate_complex_query(
    model_name: str,
    variation_count: int = 10,
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    use_default: bool = False  # Add new flag parameter
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a complex query template for name variations using an LLM.
    
    Args:
        model_name: Name of the LLM model to use
        variation_count: Number of variations to request per name
        phonetic_similarity: Dictionary mapping similarity levels to percentages
        orthographic_similarity: Dictionary mapping similarity levels to percentages
        use_default: If True, skip complex query generation and use default template
        
    Returns:
        Tuple of (query_template, labels)
    """
    # Default similarity preferences if none provided
    if phonetic_similarity is None:
        phonetic_similarity = {"Medium": 1.0}
    if orthographic_similarity is None:
        orthographic_similarity = {"Medium": 1.0}
    
    # Create the labels dictionary from the parameters
    labels = {
        "variation_count": variation_count,
        "phonetic_similarity": phonetic_similarity,
        "orthographic_similarity": orthographic_similarity
    }
    
    # If use_default flag is True, skip LLM and use default template
    if use_default:
        bt.logging.info("Using default query template (skipping complex query generation)")
        default_template = f"Give me {variation_count} comma separated alternative spellings of the name {{name}}. Include a mix of phonetically similar and orthographically similar variations. Provide only the names."
        return default_template, labels
    
    # Format the similarity specifications for the prompt
    phonetic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in phonetic_similarity.items()])
    orthographic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in orthographic_similarity.items()])
    
    bt.logging.info(f"Generating query with: {variation_count} variations, " +
                  f"phonetic similarity: {phonetic_spec}, " +
                  f"orthographic similarity: {orthographic_spec}")
    
    # Define the prompt with specific parameters
    prompt = f"""Generate a complex name variation query for a name variation system with these exact specifications:
    1. Request exactly {variation_count} variations for each name
    2. For phonetic similarity, require: {phonetic_spec}
    3. For orthographic similarity, require: {orthographic_spec}
    
    Format as a natural language query that explicitly states all requirements.
    """

    try:
        # Generate the query using Ollama
        response = ollama.generate(model=model_name, prompt=prompt)
        query_template = response['response'].strip()
        bt.logging.info(f"Generated query template: {query_template}")
        
        return query_template, labels
        
    except Exception as e:
        bt.logging.error(f"Error generating complex query: {str(e)}")
        # Fallback to a simple query template and default labels
        simple_template = f"Give me {variation_count} comma separated alternative spellings of the name {{name}}. Include a mix of phonetically similar and orthographically similar variations. Provide only the names."
        return simple_template, labels

async def timed_dendrite(dendrite, axons, synapse, deserialize, timeout, uid_map):
    """
    Track response times for dendrite calls.
    
    Args:
        dendrite: The dendrite object
        axons: List of axons to query
        synapse: The synapse object
        deserialize: Whether to deserialize responses
        timeout: Timeout in seconds
        uid_map: Mapping from index to UID
        
    Returns:
        List of responses
    """
    response_times = {}
    start_times = {i: time.time() for i in range(len(axons))}
    
    try:
        responses = await dendrite(
            axons=axons, 
            synapse=synapse, 
            deserialize=deserialize, 
            timeout=timeout
        )
        end_time = time.time()
        
        for i, response in enumerate(responses):
            uid = uid_map[i]
            response_time = end_time - start_times[i]
            response_times[uid] = response_time
            bt.logging.debug(f"Miner {uid} responded in {response_time:.2f}s")
        
        # Log response times
        bt.logging.info(f"Response times: " + 
                      ", ".join([f"UID {uid}: {response_times[uid]:.2f}s" for uid in uid_map.values()]))
        
        return responses
    except Exception as e:
        bt.logging.error(f"Error in timed_dendrite: {str(e)}")
        # Create default responses for all axons
        return [create_default_response() for _ in range(len(axons))]

async def forward(self):
    """
    The forward function is called by the validator every time step.

    This function is responsible for:
    1. Selecting a random set of miners to query
    2. Generating a complex query using an LLM
    3. Generating a list of random names to request variations for
    4. Sending the request to the selected miners with retry logic
    5. Evaluating the quality of the variations returned by each miner
    6. Updating the scores of the miners based on their performance
    7. Saving the results for analysis
    
    Returns:
        The result of the forward function from the MIID.validator module
    """
    request_start = time.time()
    # Get random UIDs to query
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # Convert to a list if you need to add more UIDs
    miner_uids = miner_uids.tolist()  # Convert NumPy array to Python list
    
    # # Add miner_uid 1 to the list for testing purposes if it exists --->(commented out)
    if 1 not in miner_uids and 1 in self.metagraph.uids:
        miner_uids.append(1)
    
    bt.logging.info(f"Selected {len(miner_uids)} miners to query: {miner_uids}")
    
    # Initialize Ollama with the same approach as in miner.py
    self.model_name = getattr(self.config, 'model_name', None)
    if self.model_name is None:
        self.model_name = DEFAULT_LLM_MODEL
        bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
    
    bt.logging.info(f"Using LLM model: {self.model_name}")
    
    # Check if Ollama is available
    try:
        # Check if model exists locally first
        models = ollama.list().get('models', [])
        model_exists = any(model.get('name') == self.model_name for model in models)
        
        if model_exists:
            bt.logging.info(f"Model {self.model_name} already pulled")
        else:
            # Model not found locally, pull it
            bt.logging.info(f"Pulling model {self.model_name}...")
            ollama.pull(self.model_name)
            
        # Set up query parameters - randomly select different configurations
        # for each validation round to test miners on various tasks
        
        # 1. Determine variation count (between 5-15)
        variation_count = random.randint(5, 15)
        
        # 2. Set up phonetic similarity distribution
        phonetic_config = random.choice([
            # Balanced distribution
            {"Light": 0.33, "Medium": 0.34, "Far": 0.33},
            # Focus on Light similarity
            {"Light": 0.6, "Medium": 0.3, "Far": 0.1},
            # Focus on Medium similarity
            {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
            # Focus on Far similarity
            {"Light": 0.1, "Medium": 0.3, "Far": 0.6},
            # Only Light similarity
            {"Light": 1.0},
            # Only Medium similarity
            {"Medium": 1.0},
            # 50% Light, 50% Medium (no Far)
            {"Light": 0.5, "Medium": 0.5},
            # 70% Light, 30% Medium (no Far)
            {"Light": 0.7, "Medium": 0.3},
            # 30% Light, 70% Medium (no Far)
            {"Light": 0.3, "Medium": 0.7},
        ])
        
        # 3. Set up orthographic similarity distribution
        orthographic_config = random.choice([
            # Balanced distribution
            {"Light": 0.33, "Medium": 0.34, "Far": 0.33},
            # Focus on Light similarity
            {"Light": 0.6, "Medium": 0.3, "Far": 0.1},
            # Focus on Medium similarity
            {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
            # Focus on Far similarity
            {"Light": 0.1, "Medium": 0.3, "Far": 0.6},
            # Only Light similarity
            {"Light": 1.0},
            # Only Medium similarity
            {"Medium": 1.0},
            # 50% Light, 50% Medium (no Far)
            {"Light": 0.5, "Medium": 0.5},
            # 70% Light, 30% Medium (no Far)
            {"Light": 0.7, "Medium": 0.3},
            # 30% Light, 70% Medium (no Far)
            {"Light": 0.3, "Medium": 0.7},
        ])
        
        # TEMPORARILY FORCE DEFAULT QUERY:
        use_default_query = True  # Override config setting
        
        # Generate a complex query template
        query_template, query_labels = await generate_complex_query(
            model_name=self.model_name,
            variation_count=variation_count,
            phonetic_similarity=phonetic_config,
            orthographic_similarity=orthographic_config,
            use_default=use_default_query
        )
        bt.logging.info(f"@@@@@@@@@@@@@\nGenerated query template: {query_template}\n@@@@@@@@@@@@@")
        bt.logging.debug(f"Variation count: {variation_count}")
        bt.logging.debug(f"Phonetic similarity: {phonetic_config}")
        bt.logging.debug(f"Orthographic similarity: {orthographic_config}")
        bt.logging.debug(f"Query labels: {query_labels}")
        
    except Exception as e:
        bt.logging.error(f"Error with Ollama: {str(e)}")
        bt.logging.error("Make sure Ollama is installed and running on this machine")
        bt.logging.error("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        bt.logging.error("Start Ollama: ollama serve")
        
        # Fallback to a simple query template and default labels
        variation_count = 10
        phonetic_config = {"Medium": 1.0}
        orthographic_config = {"Medium": 1.0}
        
        query_template = f"Give me {variation_count} comma separated alternative spellings of the name {{name}}. Include a mix of phonetically similar and orthographically similar variations. Provide only the names."
        query_labels = {
            "variation_count": variation_count,
            "phonetic_similarity": phonetic_config,
            "orthographic_similarity": orthographic_config
        }
    
    # Generate random names using Faker
    fake = Faker()
    
    # Create a list to store the generated names
    seed_names = []
    
    # Ensure name_variation config exists
    if not hasattr(self.config, 'name_variation') or self.config.name_variation is None:
        bt.logging.warning("name_variation config not found, creating it now")
        self.config.name_variation = bt.config()
        self.config.name_variation.sample_size = 5
    
    # Ensure sample_size exists and has a valid value
    sample_size = getattr(self.config.name_variation, 'sample_size', 5)
    if sample_size is None:
        sample_size = 5
        
    bt.logging.info(f"Using name variation sample size: {sample_size}")
    
    # Generate the required number of unique names
    while len(seed_names) < sample_size:
        # Randomly choose between first_name and last_name
        if random.choice([True, False]):
            name = fake.first_name().lower()
        else:
            name = fake.last_name().lower()
        
        # Ensure the name is unique and not too long or too short
        if name not in seed_names and 3 <= len(name) <= 12:
            seed_names.append(name)
    
    bt.logging.info(f"Generated {len(seed_names)} random names: {seed_names}")
    
    # Calculate timeout based on the number of names and complexity
    base_timeout = getattr(self.config.neuron, 'timeout', 120)  # Double from 60 to 120 seconds
    # More generous allocation - especially for LLM operations
    adaptive_timeout = base_timeout + (len(seed_names) * 20) + (query_labels['variation_count'] * 10)
    adaptive_timeout = min(600, max(120, adaptive_timeout))  # Min 120s, max 600s (10 minutes)
    bt.logging.info(f"Using adaptive timeout of {adaptive_timeout} seconds for {len(seed_names)} names")
    
    # Prepare the synapse for the request
    request_synapse = IdentityRequest(
        names=seed_names,
        query_template=query_template,
        variations={}  # Initialize with empty variations
    )
    
    # Query the network with retry logic
    if use_default_query:
        bt.logging.info(f"Querying {len(miner_uids)} miners with default query template")
    else:
        bt.logging.info(f"Querying {len(miner_uids)} miners with complex query")    
    
    start_time = time.time()
    
    # Initialize all_responses to collect responses from all batches
    all_responses = []
    
    # Process miners in batches to avoid overwhelming the network
    batch_size = 3  # Further reduce from 5 to just 3 miners per batch
    total_batches = (len(miner_uids) + batch_size - 1) // batch_size
    
    for i in range(0, len(miner_uids), batch_size):
        batch_uids = miner_uids[i:i+batch_size]
        batch_axons = [self.metagraph.axons[uid] for uid in batch_uids]
        
        bt.logging.info(f"Processing batch {i//batch_size + 1}/{total_batches} with {len(batch_uids)} miners")
        batch_start_time = time.time()
        
        # Use dendrite_with_retries to ensure we get responses
        batch_responses = await dendrite_with_retries(
            dendrite=self.dendrite,
            axons=batch_axons,
            synapse=request_synapse,
            deserialize=True,
            timeout=adaptive_timeout,
            cnt_attempts=7
        )
        
        batch_duration = time.time() - batch_start_time
        bt.logging.info(f"Batch {i//batch_size + 1} completed in {batch_duration:.1f}s")
        
        # More detailed response validation
        for i, response in enumerate(batch_responses):
            uid = batch_uids[i]
            if not hasattr(response, 'variations'):
                bt.logging.warning(f"Miner {uid} returned response without 'variations' attribute: {type(response)}")
            elif response.variations is None:
                bt.logging.warning(f"Miner {uid} returned response with None variations")
            elif not response.variations:
                bt.logging.warning(f"Miner {uid} returned empty variations dictionary")
            else:
                # Count variations
                total_variations = sum(len(vars) for vars in response.variations.values())
                bt.logging.info(f"Miner {uid} returned {len(response.variations)} names with {total_variations} total variations")
        
        # Add batch responses to all_responses
        all_responses.extend(batch_responses)
        
        # Sleep between batches to allow miners to recover and process new requests
        if i + batch_size < len(miner_uids):
            sleep_time = 20  # Increase from 10 to 20 seconds between batches
            bt.logging.info(f"Sleeping for {sleep_time}s before next batch")
            await asyncio.sleep(sleep_time)
    
    end_time = time.time()
    bt.logging.info(f"Query completed in {end_time - start_time:.2f} seconds")
    
    # Log the results
    bt.logging.info(f"Received name variation responses for {len(all_responses)} miners")
    
    # Check for empty or invalid responses
    valid_responses = 0
    for i, response in enumerate(all_responses):
        if hasattr(response, 'variations') and response.variations:
            valid_responses += 1
        else:
            bt.logging.warning(f"Miner {miner_uids[i]} returned invalid or empty response")
    
    bt.logging.info(f"Received {valid_responses} valid responses out of {len(all_responses)}")
    
    # Score the responses with the extracted labels
    rewards = get_name_variation_rewards(
        self, 
        seed_names, 
        all_responses, 
        miner_uids,
        variation_count=query_labels['variation_count'],
        phonetic_similarity=query_labels['phonetic_similarity'],
        orthographic_similarity=query_labels['orthographic_similarity']
    )
    
    # Update the validator's internal scores
    self.update_scores(rewards, miner_uids)
    ## print the rewards and the miner uids
    bt.logging.info(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nRewards: {rewards}\nMiner UIDs: {miner_uids}\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    # Save the results for analysis
    # Create a unique timestamp for this run
    timestamp = int(time.time())
    
    # Create a directory for validator results if it doesn't exist
    results_dir = os.path.join(self.config.logging.logging_dir, "validator_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a run-specific directory
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the query and responses to a JSON file
    results = {
        "timestamp": timestamp,
        "seed_names": seed_names,
        "query_template": query_template,
        "query_labels": query_labels,
        "responses": {},
        "rewards": {}
    }
    
    # Add the responses and rewards for each miner
    for i, uid in enumerate(miner_uids):
        if i < len(all_responses):
            # Convert the response to a serializable format
            miner_response = {}
            if hasattr(all_responses[i], 'variations') and all_responses[i].variations is not None:
                miner_response = all_responses[i].variations
            
            # Add to the results
            results["responses"][str(uid)] = miner_response
            results["rewards"][str(uid)] = float(rewards[i]) if i < len(rewards) else 0.0
    
    # Save to JSON file
    json_path = os.path.join(run_dir, f"results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    bt.logging.info(f"Saved validator results to: {json_path}")
    
    # Set weights based on the updated scores
    self.set_weights()
    
    # Add minimum processing time to avoid overwhelming the network
    
    request_end = time.time()
    if request_end - request_start < EPOCH_MIN_TIME:
        bt.logging.info(f"Finished too fast, sleeping for {EPOCH_MIN_TIME - (request_end - request_start)} seconds")
        time.sleep(EPOCH_MIN_TIME - (request_end - request_start))

    bt.logging.info("All batches processed, waiting for any remaining responses...")
    await asyncio.sleep(30)  # Extra 30 second wait at the end
    
    # Return success
    return True
