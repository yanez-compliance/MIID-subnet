# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


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
    Validator neuron for the name variation protocol. This validator sends requests to miners
    to generate alternative spellings for names, and rewards miners based on the quality of
    their responses.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        
        # You might want to add configuration for the name variation protocol
        if not hasattr(self.config, 'name_variation'):
            self.config.name_variation = bt.config()
            self.config.name_variation.sample_size = 5  # Number of names to sample per query

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating a list of names
        - Querying the miners for name variations
        - Rewarding the miners based on the quality of their responses
        """
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Validator running... {time.time()}")
            time.sleep(5)
