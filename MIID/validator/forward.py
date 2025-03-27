# forward.py

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
6. Uploading the same results to the external endpoint (Flask-based) via HTTP POST
"""

import time
import bittensor as bt
import json
import os
import random
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple

from MIID.protocol import IdentitySynapse
from MIID.validator.reward import get_name_variation_rewards
from MIID.utils.uids import get_random_uids
from MIID.utils import sign_message
from MIID.validator.query_generator import QueryGenerator

# Import your new upload_data function here
from MIID.utils.misc import upload_data

EPOCH_MIN_TIME = 360  # seconds

async def dendrite_with_retries(dendrite: bt.dendrite, axons: list, synapse: IdentitySynapse,
                                deserialize: bool, timeout: float, cnt_attempts=3):
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
    axons_for_retry = axons.copy()
    
    def create_default_response():
        return IdentitySynapse(
            names=synapse.names,
            query_template=synapse.query_template,
            variations={}
        )
    
    for attempt in range(cnt_attempts):
        responses = await dendrite(
            axons=axons_for_retry,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout * (1 + attempt * 1.0)
        )
        
        new_idx = []
        new_axons = []
        
        for i, response in enumerate(responses):
            bt.logging.info(f"#########################################Response {i}: {response}#########################################")
            bt.logging.info(f"#########################################Response type: {type(response)}#########################################")
            
            if isinstance(response, dict):
                # Got the variations dictionary directly
                complete_response = IdentitySynapse(
                    names=synapse.names,
                    query_template=synapse.query_template,
                    variations=response
                )
                res[idx[i]] = complete_response
            
            elif hasattr(response, 'dendrite'):
                # Check status code
                if (response.dendrite.status_code is not None and 
                    int(response.dendrite.status_code) == 422):
                    if attempt == cnt_attempts - 1:
                        res[idx[i]] = response
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_for_retry[i])
                else:
                    res[idx[i]] = response
            
            else:
                # If the response has variations attribute, treat it as a valid response
                if hasattr(response, 'variations'):
                    res[idx[i]] = response
                else:
                    # Retry or assign default
                    if attempt == cnt_attempts - 1:
                        res[idx[i]] = create_default_response()
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_for_retry[i])
        
        if not new_idx:
            break
        
        idx = new_idx
        axons_for_retry = new_axons
        time.sleep(5 * (attempt + 1))
    
    # Fill any remaining None
    for i, r in enumerate(res):
        if r is None:
            res[i] = create_default_response()
    
    return res


async def forward(self):
    """
    The forward function is called by the validator every time step.

    This function:
      1. Selects a random set of miners to query
      2. Generates a complex query using an LLM
      3. Generates a list of random names to request variations for
      4. Sends the request to the selected miners with retry logic
      5. Evaluates the quality of the variations returned by each miner
      6. Updates the scores of the miners based on their performance
      7. Saves the results for analysis
      8. Uploads the generated JSON results to an external endpoint
    """

    request_start = time.time()
    
    bt.logging.info("Updating and querying available uids")

    # 1) Get random UIDs to query
    available_axon_size = len(self.metagraph.axons) - 1  # Exclude self
    miner_selection_size = min(available_axon_size, self.config.neuron.sample_size)
    miner_uids = get_random_uids(self, k=miner_selection_size)
    axons = [self.metagraph.axons[uid] for uid in miner_uids]

    bt.logging.info(f"#########################################Miner axons: {axons}#########################################")
    bt.logging.info(f"#########################################Miner selection size: {miner_selection_size}#########################################")
    bt.logging.info(f"#########################################Available axon size: {available_axon_size}#########################################")

    miner_uids = miner_uids.tolist()
    bt.logging.info(f"#########################################Selected {len(miner_uids)} miners to query: {miner_uids}#########################################")

    # 2) Initialize the query generator
    query_generator = QueryGenerator(self.config)

    # 3) Generate the query, names, and labels
    start_time = time.time()
    seed_names, query_template, query_labels = await query_generator.build_queries()
    end_time = time.time()
    bt.logging.info(f"Time to generate challenges: {int(end_time - start_time)}s")

    # 4) Calculate adaptive timeout
    base_timeout = self.config.neuron.timeout
    adaptive_timeout = base_timeout + (len(seed_names) * 20) + (query_labels['variation_count'] * 10)
    adaptive_timeout = min(600, max(120, adaptive_timeout))  # clamp [120, 600]
    bt.logging.info(f"Using adaptive timeout of {adaptive_timeout} seconds for {len(seed_names)} names")

    # 5) Prepare the synapse
    request_synapse = IdentitySynapse(
        names=seed_names,
        query_template=query_template,
        variations={},
        timeout=adaptive_timeout
    )

    if query_generator.use_default_query:  
        bt.logging.info(f"Querying {len(miner_uids)} miners with default query template")
    else:
        bt.logging.info(f"Querying {len(miner_uids)} miners with complex query")

    bt.logging.info(f"#########################################Request synapse: {request_synapse}#########################################")
    time.sleep(3)

    # 6) Query the network in batches
    start_time = time.time()
    all_responses = []
    batch_size = 5
    total_batches = (len(miner_uids) + batch_size - 1) // batch_size
    
    for i in range(0, len(miner_uids), batch_size):
        batch_uids = miner_uids[i:i+batch_size]
        batch_axons = [self.metagraph.axons[uid] for uid in batch_uids]
        
        bt.logging.info(f"#########################################Batch uids: {batch_uids}#########################################")
        time.sleep(300)  # Large sleep; adjust as desired

        bt.logging.info(f"Processing batch {i//batch_size + 1}/{total_batches} with {len(batch_uids)} miners")
        batch_start_time = time.time()
        
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

        for idx_resp, response in enumerate(batch_responses):
            uid = batch_uids[idx_resp]
            if not hasattr(response, 'variations'):
                bt.logging.warning(f"Miner {uid} returned response without 'variations' attribute.")
            elif response.variations is None:
                bt.logging.warning(f"Miner {uid} returned None in 'variations'.")
            elif not response.variations:
                bt.logging.warning(f"Miner {uid} returned empty variations dictionary.")
            else:
                total_variations = sum(len(v) for v in response.variations.values())
                bt.logging.info(f"Miner {uid} returned {len(response.variations)} names with {total_variations} total variations.")
        
        all_responses.extend(batch_responses)
        
        if i + batch_size < len(miner_uids):
            sleep_time = 20
            bt.logging.info(f"Sleeping for {sleep_time}s before next batch")
            await asyncio.sleep(sleep_time)
    
    end_time = time.time()
    bt.logging.info(f"Query completed in {end_time - start_time:.2f} seconds")

    # 7) Compute rewards
    bt.logging.info(f"Received name variation responses for {len(all_responses)} miners")
    valid_responses = 0
    for i, response in enumerate(all_responses):
        if hasattr(response, 'variations') and response.variations:
            valid_responses += 1
        else:
            bt.logging.warning(f"Miner {miner_uids[i]} returned invalid or empty response")
    
    bt.logging.info(f"Received {valid_responses} valid responses out of {len(all_responses)}")

    rewards = get_name_variation_rewards(
        self, 
        seed_names,
        all_responses, 
        miner_uids,
        variation_count=query_labels['variation_count'],
        phonetic_similarity=query_labels['phonetic_similarity'],
        orthographic_similarity=query_labels['orthographic_similarity']
    )

    self.update_scores(rewards, miner_uids)
    bt.logging.info(f"REWARDS: {rewards}  for MINER UIDs: {miner_uids}")

    # 8) Save results locally
    timestamp = int(time.time())
    results_dir = os.path.join(self.config.logging.logging_dir, "validator_results")
    os.makedirs(results_dir, exist_ok=True)
    
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    results = {
        "timestamp": timestamp,
        "seed_names": seed_names,
        "query_template": query_template,
        "query_labels": query_labels,
        "responses": {},
        "rewards": {}
    }
    
    for i, uid in enumerate(miner_uids):
        if i < len(all_responses):
            miner_response = {}
            if hasattr(all_responses[i], 'variations') and all_responses[i].variations is not None:
                miner_response = all_responses[i].variations
            results["responses"][str(uid)] = miner_response
            results["rewards"][str(uid)] = float(rewards[i]) if i < len(rewards) else 0.0
    
    json_path = os.path.join(run_dir, f"results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    bt.logging.info(f"Saved validator results to: {json_path}")

    # 9) Upload to external endpoint (moved to a separate utils function)
    # Adjust endpoint URL/hotkey if needed
    results_json_string = json.dumps(results, sort_keys=True)
    endpoint_base = "http://127.0.0.1:5000/upload_data" ## MIID server
    hotkey = self.wallet.hotkey
    signed_contents = sign_message(self.wallet, results_json_string, output_file=None)
    results["signature"] = signed_contents

    upload_data(endpoint_base, hotkey, results) 

    # 10) Set weights and enforce min epoch time
    self.set_weights()
    request_end = time.time()
    if request_end - request_start < EPOCH_MIN_TIME:
        bt.logging.info(f"Finished quickly; sleeping for {EPOCH_MIN_TIME - (request_end - request_start)}s")
        time.sleep(EPOCH_MIN_TIME - (request_end - request_start))

    bt.logging.info("All batches processed, waiting 30 more seconds...")
    await asyncio.sleep(30)
    
    return True
