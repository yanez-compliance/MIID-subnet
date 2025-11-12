# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2025 Yanez

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

import typing
import bittensor as bt
import json
from typing import List, Dict, Optional
from typing_extensions import TypedDict

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class IdentitySynapse(bt.Synapse):
    """
    Protocol for requesting identity variations from miners.
    
    Attributes:
    - identity: List of identity arrays, each containing [name, dob, address]
    - query_template: Template string for the LLM prompt with {name} placeholder
    - variations: Optional dictionary containing the generated variations for each name
                 Each name maps to a list of [name_variation, dob_variation, address_variation] arrays
    """
    
    # Required request input, filled by sending dendrite caller
    identity: List[List[str]]  # Each inner list contains [name, dob, address]
    query_template: str  # Base template used for all identities (backward compatibility)
    query_templates: Optional[Dict[str, str]] = None  # Optional per-identity templates (identity name -> template)
    
    timeout: float = 120.0
    
    # Optional request output, filled by receiving axon
    # Phase 3: support extended structure with UAV data while maintaining backward compatibility.
    # Define structured types for clarity
    
    
    # TypedDicts for UAV structure
    class UAVData(TypedDict):
        address: str
        label: str
        latitude: typing.Optional[float]
        longitude: typing.Optional[float]

    class SeedData(TypedDict):
        variations: List[List[str]]
        uav: 'IdentitySynapse.UAVData'

    variations: Optional[Dict[str, typing.Union[List[List[str]], 'IdentitySynapse.SeedData']]] = None
    process_time: Optional[float] = None  # <-- add this
    
    def deserialize(self) -> Dict[str, typing.Union[List[List[str]], 'IdentitySynapse.SeedData']]:
        """
        Deserialize the variations output.
        
        Returns:
        - Dict[str, List[List[str]]]: Dictionary mapping each name to its list of [name, dob, address] variations
        """
        return self.variations
