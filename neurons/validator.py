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
Name Variation Validator Module

This module implements a Bittensor validator that queries miners for alternative
spellings of names. The validator sends requests to miners with a list of names
and a query template, then evaluates the quality of the variations returned by
each miner.

The validator follows these steps:
1. Select a random set of miners to query
2. Generate a list of random names to request variations for
3. Send the request to the selected miners
4. Evaluate the quality of the variations returned by each miner
5. Update the scores of the miners based on their performance

The evaluation considers factors such as:
- The number of variations provided
- The uniqueness of the variations
- The similarity of the variations to the original name

This validator helps incentivize miners to provide high-quality name variations
that can be used for various applications such as identity verification, name
matching, and data augmentation.
"""

import time
import traceback

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from MIID.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from MIID.validator import forward
from MIID.validator.reward import get_name_variation_rewards
import ollama


class Validator(BaseValidatorNeuron):
    """
    Validator neuron for the name variation protocol.
    
    This validator sends requests to miners to generate alternative spellings for names,
    and rewards miners based on the quality of their responses. It evaluates miners on:
    
    1. Responsiveness - whether they respond to requests at all
    2. Completeness - whether they provide variations for all requested names
    3. Quantity - the number of variations provided (up to a reasonable limit)
    4. Quality - the uniqueness and similarity of variations to the original names
    
    The validator maintains a scoring system that rewards miners who consistently
    provide high-quality name variations, which helps to improve the overall
    quality of the network.
    """

    DEFAULT_LLM_MODEL = "llama3.1:latest"

    def __init__(self, config=None):
        """
        Initialize the Name Variation Validator.
        
        This sets up the validator with configuration for the name variation protocol,
        including how many names to sample in each query.
        
        Args:
            config: Configuration object for the validator
        """
        bt.logging.info("Initializing Validator")

        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        
        # # Make sure self.config exists
        # if not hasattr(self, 'config') or self.config is None:
        #     bt.logging.warning("self.config is None, creating a new config object")
        #     self.config = bt.config()
        
        # # Configuration for the name variation protocol
        # # Create the name_variation config object if it doesn't exist
        # if not hasattr(self.config, 'name_variation') or self.config.name_variation is None:
        #     bt.logging.warning("name_variation config is None, creating a new config object")
        #     self.config.name_variation = bt.config()
        
        # # Explicitly create the sample_size attribute with a default value
        # # This is a more direct approach that should work regardless of the config object's implementation
        # self.config.name_variation.sample_size = 5
        
        # # Log the configuration to verify it's set correctly
        # bt.logging.info(f"Name variation sample size: {self.config.name_variation.sample_size}")

        # Initialize Ollama with the same approach as in miner.py
        if hasattr(self.config, 'neuron') and hasattr(self.config.neuron, 'ollama_model_name'):
            self.model_name = self.config.neuron.ollama_model_name
        else:
            self.model_name = self.DEFAULT_LLM_MODEL
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
        
        bt.logging.info(f"Using LLM model: {self.model_name}")
        
        # Check if Ollama is available
        try:
            # Check if model exists locally first
            models = ollama.list().get('models', [])
            bt.logging.info(f"Models: {models}")
            model_exists = any(model.model == self.model_name for model in models)
            
            if model_exists:
                bt.logging.info(f"Model {self.model_name} already pulled")
            else:
                # Model not found locally, pull it
                bt.logging.info(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
        except Exception as e:
            bt.logging.error(f"Error initializing Ollama: {e}")
            raise e
        
        bt.logging.info("Ollama initialized")
        bt.logging.info(f"Using LLM model: {self.model_name}")
        bt.logging.info("Finished initializing Validator")
        bt.logging.info("----------------------------------")
        time.sleep(100)

    async def forward(self):
        """
        Validator forward pass.
        
        This method is called periodically to query miners and evaluate their responses.
        The forward pass consists of:
        1. Generating a list of random names
        2. Querying the miners for name variations
        3. Rewarding the miners based on the quality of their responses
        
        The results of each query are saved to a file for analysis, and the
        scores of the miners are updated based on their performance.
        
        Returns:
            The result of the forward function from the MIID.validator module
        """
        try:
            res = await forward(self)
            return res
        except Exception as e:
            bt.logging.error("Got error in forward function")
            bt.logging.info(traceback.format_exc())
            return None


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Validator running... {time.time()}")
            time.sleep(60)
