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

import time
import bittensor as bt
import json
import os
import random

from MIID.protocol import NameVariationRequest
from MIID.validator.reward import get_name_variation_rewards
from MIID.utils.uids import get_random_uids


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """
    # Get random UIDs to query
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    
    # Generate a list of random names to query
    # In a real implementation, you might want to load these from a file or database
    seed_names = random.sample([
        "gillbert", "jamehriah", "joana", "wynnfred", "camille", 
        "michael", "sarah", "robert", "elizabeth", "david"
    ], self.config.name_variation.sample_size)  # Select random names
    
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
    
    bt.logging.info(f"Scored name variation responses: {rewards}")
    self.update_scores(rewards, miner_uids)
    
    # Save the responses to a file for analysis
    if hasattr(self, 'config') and hasattr(self.config, 'neuron') and hasattr(self.config.neuron, 'logging_dir'):
        output_dir = self.config.neuron.logging_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save responses to a JSON file
        output_file = os.path.join(output_dir, f"name_variations_{self.step}.json")
        with open(output_file, 'w') as f:
            # Convert responses to a serializable format
            serialized_responses = {}
            for uid, response in zip(miner_uids, responses):
                if response is not None:
                    serialized_responses[str(uid)] = response
            
            json.dump(serialized_responses, f, indent=4)
    
    time.sleep(5)
