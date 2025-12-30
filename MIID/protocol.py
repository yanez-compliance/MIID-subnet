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
from typing import List, Dict, Optional, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

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


# =============================================================================
# Phase 4: Image Variation Types
# =============================================================================

class ImageRequest(BaseModel):
    """Phase 4: Image variation request from validator to miner.

    Contains the base image and parameters for generating variations.
    The miner will generate variations, encrypt them with drand timelock,
    upload to S3, and return S3 references.
    """
    base_image: str  # Base64 encoded image
    image_filename: str  # Original filename for reference
    variation_types: List[str] = Field(
        default_factory=lambda: ["pose", "expression", "lighting", "background"]
    )  # Types of variations requested
    target_drand_round: int  # Drand round when decryption becomes possible
    reveal_timestamp: int  # Unix timestamp when reveal occurs
    requested_variations: int = 3  # Number of variations to generate (3-5)
    challenge_id: Optional[str] = None  # Unique identifier for this challenge

    class Config:
        # Allow arbitrary types for flexibility
        arbitrary_types_allowed = True


class S3Submission(BaseModel):
    """Phase 4: Miner's S3 submission response.

    Contains references to encrypted images uploaded to S3.
    The actual images are NOT included - only S3 paths, hashes, and signatures.
    Post-validation will download and decrypt after drand reveal.
    """
    s3_key: str  # Path to encrypted file in S3 bucket
    image_hash: str  # SHA256 hash of the original (unencrypted) image
    signature: str  # Wallet signature proving ownership
    variation_type: str  # Which variation type this submission addresses

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# End Phase 4 Types
# =============================================================================


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
    query_template: str

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

    # ==========================================================================
    # Phase 4: Image Variation Fields
    # ==========================================================================
    # Request: Validator → Miner (base image + parameters)
    image_request: Optional[ImageRequest] = None

    # Response: Miner → Validator (S3 references, NOT actual images)
    s3_submissions: Optional[List[S3Submission]] = None
    # ==========================================================================
    
    def deserialize(self) -> Dict[str, typing.Union[List[List[str]], 'IdentitySynapse.SeedData']]:
        """
        Deserialize the variations output.
        
        Returns:
        - Dict[str, List[List[str]]]: Dictionary mapping each name to its list of [name, dob, address] variations
        """
        return self.variations
