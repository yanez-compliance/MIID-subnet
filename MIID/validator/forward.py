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

from MIID.protocol import NameVariationRequest
from MIID.validator.reward import get_name_variation_rewards
from MIID.utils.uids import get_random_uids

EPOCH_MIN_TIME = 200 # seconds

async def dendrite_with_retries(dendrite: bt.dendrite, axons: list, synapse, deserialize: bool, timeout: float, cnt_attempts=3):
    """
    Send requests to miners with automatic retry logic and progressive timeouts.
    
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
    # Extract the required fields from the original synapse
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
                # For NameVariationRequest, we need to provide the required fields
                if synapse.__class__.__name__ == 'NameVariationRequest':
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
        # Increase timeout with each retry
        current_timeout = timeout * (1 + attempt * 0.5)  # 1x, 1.5x, 2x original timeout
        bt.logging.info(f"Attempt {attempt+1}/{cnt_attempts} with timeout {current_timeout:.1f}s")
        
        try:
            responses = await dendrite(
                axons=axons_copy,
                synapse=synapse,
                deserialize=deserialize,
                timeout=current_timeout
            )
            
            new_idx = []
            new_axons = []
            
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
                     int(response.dendrite.status_code) == 422:
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
                bt.logging.info(f'Found {len(new_idx)} synapses with broken connections, retrying them')
            else:
                break
            
            idx = new_idx
            axons_copy = new_axons
            
        except Exception as e:
            bt.logging.error(f"Error in dendrite call: {str(e)}")
            # If this is the last attempt, fill remaining slots with empty responses
            if attempt == cnt_attempts - 1:
                for i in range(len(idx)):
                    if res[idx[i]] is None:
                        res[idx[i]] = create_default_response()
            time.sleep(1)  # Brief pause before retry
    
    # Ensure all responses are filled
    for i, r in enumerate(res):
        if r is None:
            bt.logging.warning(f"No response received for axon {axons[i]}")
            # Create an empty response to avoid None values
            res[i] = create_default_response()
            
    return res

async def forward(self):
    """
    The forward function is called by the validator every time step.
    
    This function is responsible for:
    1. Selecting a random set of miners to query
    2. Generating a list of random names to request variations for
    3. Sending the request to the selected miners with retry logic
    4. Evaluating the quality of the variations returned by each miner
    5. Updating the scores of the miners based on their performance
    6. Saving the results for analysis
    
    Returns:
        The result of the forward function from the MIID.validator module
    """
    # Get random UIDs to query
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    
    # Convert to a list if you need to add more UIDs
    miner_uids = miner_uids.tolist()  # Convert NumPy array to Python list
    
    # Add miner_uid 1 to the list for testing purposes if it exists
    if 1 not in miner_uids and 1 in self.metagraph.uids:
        miner_uids.append(1)
    
    bt.logging.info(f"Selected {len(miner_uids)} miners to query: {miner_uids}")
    
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
    
    # Query template
    query_template = "Give me 10 comma separated alternative spellings of the name {name}. 5 of them should sound similar to the original name and 5 should be orthographically similar. Provide only the names."
    
    # Prepare the synapse for the request
    request_synapse = NameVariationRequest(
        names=seed_names,
        query_template=query_template,
        variations={}  # Initialize with empty variations
    )
    
    # Calculate timeout based on the number of names to process
    base_timeout = getattr(self.config.neuron, 'timeout', 30)
    # Add 10 seconds per name to process
    adaptive_timeout = base_timeout + (len(seed_names) * 10)
    bt.logging.info(f"Using adaptive timeout of {adaptive_timeout} seconds")
    
    # Before sending the main request, check if miners are responsive with a simple ping
    bt.logging.info("Sending ping to check miner responsiveness")
    ping_synapse = bt.Synapse()  # Use a simple synapse for ping
    ping_responses = await dendrite_with_retries(
        dendrite=self.dendrite,
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=ping_synapse,
        deserialize=True,
        timeout=10,  # Short timeout for ping
        cnt_attempts=1
    )

    # Filter out unresponsive miners
    responsive_uids = []
    for i, response in enumerate(ping_responses):
        if response is not None and hasattr(response, 'dendrite') and response.dendrite.status_code == 200:
            responsive_uids.append(miner_uids[i])

    if len(responsive_uids) < len(miner_uids):
        bt.logging.warning(f"Only {len(responsive_uids)}/{len(miner_uids)} miners responded to ping")
        # Continue with only responsive miners
        miner_uids = responsive_uids
    
    # Query the network with retry logic
    bt.logging.info(f"Querying {len(miner_uids)} miners with retry logic")
    start_time = time.time()
    
    # Add timing information to track how long each miner takes to respond
    response_times = {}
    
    # Define the timed_dendrite function inside the forward function
    async def timed_dendrite(dendrite, axons, synapse, deserialize, timeout, uid_map):
        start_times = {i: time.time() for i in range(len(axons))}
        responses = await dendrite(axons=axons, synapse=synapse, deserialize=deserialize, timeout=timeout)
        end_time = time.time()
        
        for i, response in enumerate(responses):
            uid = uid_map[i]
            response_time = end_time - start_times[i]
            response_times[uid] = response_time
            bt.logging.debug(f"Miner {uid} responded in {response_time:.2f}s")
        
        return responses
    
    # Initialize all_responses to collect responses from all batches
    all_responses = []
    
    # Process miners in batches with pauses between batches
    batch_size = 10  # Smaller batch size to reduce load
    for i in range(0, len(miner_uids), batch_size):
        batch_uids = miner_uids[i:i+batch_size]
        batch_axons = [self.metagraph.axons[uid] for uid in batch_uids]
        uid_map = {j: batch_uids[j] for j in range(len(batch_uids))}
        
        bt.logging.info(f"Processing batch {i//batch_size + 1} with {len(batch_uids)} miners")
        
        # Query this batch with timing
        batch_responses = await timed_dendrite(
            dendrite=self.dendrite,
            axons=batch_axons,
            synapse=request_synapse,
            deserialize=True,
            timeout=adaptive_timeout,
            uid_map=uid_map
        )
        
        # Add batch responses to all_responses
        all_responses.extend(batch_responses)
        
        # Log response times for this batch
        bt.logging.info(f"Response times for batch {i//batch_size + 1}: " + 
                       ", ".join([f"UID {uid}: {response_times[uid]:.2f}s" for uid in batch_uids]))
        
        bt.logging.info(f"Completed batch {i//batch_size + 1}, received {len(batch_responses)} responses")
        
        # Sleep between batches to allow miners to recover and process new requests
        if i + batch_size < len(miner_uids):
            sleep_time = 5  # 5 seconds between batches
            bt.logging.info(f"Sleeping for {sleep_time}s before next batch")
            await asyncio.sleep(sleep_time)
    
    end_time = time.time()
    bt.logging.info(f"Query completed in {end_time - start_time:.2f} seconds")
    
    # Log the results
    bt.logging.info(f"Received name variation responses for {len(all_responses)} miners")
    
    # Check for empty or invalid responses
    valid_count = sum(1 for r in all_responses if hasattr(r, 'variations') and r.variations)
    min_valid_responses = max(1, int(len(miner_uids) * 0.2))  # At least 20% of miners should respond

    if valid_count < min_valid_responses:
        bt.logging.warning(f"Only received {valid_count} valid responses out of {len(miner_uids)}. Retrying with longer timeout.")
        
        # Try again with a much longer timeout for all miners that didn't respond properly
        retry_uids = []
        retry_axons = []
        for i, response in enumerate(all_responses):
            if not (hasattr(response, 'variations') and response.variations):
                retry_uids.append(miner_uids[i])
                retry_axons.append(self.metagraph.axons[miner_uids[i]])
        
        if retry_axons:
            bt.logging.info(f"Retrying {len(retry_axons)} miners with extended timeout")
            retry_responses = await dendrite_with_retries(
                dendrite=self.dendrite,
                axons=retry_axons,
                synapse=request_synapse,
                deserialize=True,
                timeout=adaptive_timeout * 2,  # Double the timeout
                cnt_attempts=2
            )
            
            # Replace the failed responses with retry responses
            for i, uid in enumerate(retry_uids):
                original_idx = miner_uids.index(uid)
                all_responses[original_idx] = retry_responses[i]
            
            # Recount valid responses
            valid_count = sum(1 for r in all_responses if hasattr(r, 'variations') and r.variations)
            bt.logging.info(f"After retry: {valid_count} valid responses")
    
    # Score the responses
    rewards = get_name_variation_rewards(self, seed_names, all_responses, miner_uids)
    
    # Update the validator's internal scores
    self.update_scores(rewards, miner_uids)
    
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
    
    # Add more detailed logging throughout the process
    bt.logging.info(f"Request details: {len(seed_names)} names, {len(miner_uids)} miners")
    for i, response in enumerate(all_responses):
        status = getattr(response.dendrite, 'status_code', None) if hasattr(response, 'dendrite') else None
        bt.logging.debug(f"Miner {miner_uids[i]} response status: {status}")
    
    # Add minimum processing time to avoid overwhelming the network
    min_time = 60  # 60 seconds minimum processing time
    if end_time - start_time < min_time:
        sleep_time = min_time - (end_time - start_time)
        bt.logging.info(f"Completed too quickly, sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    # Return success
    return True
