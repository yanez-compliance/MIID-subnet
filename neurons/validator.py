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

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from MIID.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from MIID.validator import forward
from MIID.validator.reward import get_name_variation_rewards


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

    def __init__(self, config=None):
        """
        Initialize the Name Variation Validator.
        
        This sets up the validator with configuration for the name variation protocol,
        including how many names to sample in each query.
        
        Args:
            config: Configuration object for the validator
        """
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        
        # Configuration for the name variation protocol
        if not hasattr(self.config, 'name_variation'):
            self.config.name_variation = bt.config()
            # Number of names to sample per query
            # This determines how many names will be sent to miners in each request
            self.config.name_variation.sample_size = 5
            
        # Ensure required libraries are installed
        try:
            from faker import Faker
            import Levenshtein
            import jellyfish
            bt.logging.info("All required libraries are available")
        except ImportError as e:
            bt.logging.error(f"Required library not installed: {e}")
            bt.logging.error("Please install required libraries with: pip install faker python-Levenshtein jellyfish")
            raise ImportError("Required libraries missing. Install with: pip install faker python-Levenshtein jellyfish")

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
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Validator running... {time.time()}")
            time.sleep(5)
