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

from MIID.protocol import NameVariationRequest
from MIID.validator.reward import get_name_variation_rewards
from MIID.utils.uids import get_random_uids


async def forward(self):
    """
    The forward function is called by the validator every time step.
    
    This function is responsible for:
    1. Selecting a random set of miners to query
    2. Generating a list of random names to request variations for
    3. Sending the request to the selected miners
    4. Evaluating the quality of the variations returned by each miner
    5. Updating the scores of the miners based on their performance
    6. Saving the results for analysis
    
    The results of each query are saved to a file for analysis, and the
    scores of the miners are updated based on their performance.
    
    Returns:
        The result of the forward function from the MIID.validator module
    """
    # Get random UIDs to query
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    
    # Convert to a list if you need to add more UIDs
    miner_uids = miner_uids.tolist()  # Convert NumPy array to Python list
    
    # Add miner_uid 1 to the list for testing purposes
    if 1 not in miner_uids and 1 in self.metagraph.uids:
        miner_uids.append(1)  # Now this will work since miner_uids is a list
    
    bt.logging.info(f"Selected {len(miner_uids)} miners to query: {miner_uids}")
    
    # Generate random names using Faker
    fake = Faker()
    
    # Create a list to store the generated names
    seed_names = []
    
    # Ensure name_variation config exists
    if not hasattr(self.config, 'name_variation'):
        bt.logging.warning("name_variation config not found, creating it now")
        self.config.name_variation = bt.config()
        self.config.name_variation.sample_size = 5
    
    # Ensure sample_size exists and has a valid value
    if not hasattr(self.config.name_variation, 'sample_size') or self.config.name_variation.sample_size is None:
        bt.logging.warning("sample_size is None, setting default value of 5")
        self.config.name_variation.sample_size = 5

    sample_size = self.config.name_variation.sample_size
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
    
    # Query the network
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=NameVariationRequest(
            names=seed_names,
            query_template=query_template
        ),
        deserialize=True,
    )
    
    # Log the results
    bt.logging.info(f"Received name variation responses for {len(responses)} miners")
    
    # Score the responses
    rewards = get_name_variation_rewards(self, seed_names, responses, miner_uids)
    
    # Update the validator's internal scores
    # This is what set_weights() will use
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
        if i < len(responses):
            # Convert the response to a serializable format
            miner_response = {}
            if hasattr(responses[i], 'variations') and responses[i].variations is not None:
                miner_response = responses[i].variations
            
            # Add to the results
            results["responses"][str(uid)] = miner_response
            results["rewards"][str(uid)] = float(rewards[i]) if i < len(rewards) else 0.0
    
    # Save to JSON file
    json_path = os.path.join(run_dir, f"results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    bt.logging.info(f"Saved validator results to: {json_path}")
    
    # Set weights based on the updated scores
    # This uses the internal self.scores that we updated with update_scores()
    self.set_weights()
    
    # Return success
    return True
