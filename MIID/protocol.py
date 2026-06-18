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
from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Phase 4: Image Variation Types
# =============================================================================

class VariationRequest(BaseModel):
    """Phase 4: Single variation request with type and intensity.

    Specifies what kind of variation to generate and at what intensity level.
    Used as a guideline for miners; post-validation will judge compliance.
    """
    type: str        # Variation type: pose_edit, lighting_edit, expression_edit, background_edit, screen_replay
    intensity: str   # Intensity level: light, medium, far
    description: str = ""   # Human-readable description of the type
    detail: str = ""        # Intensity-specific detail/guideline

    class Config:
        arbitrary_types_allowed = True


class ImageRequest(BaseModel):
    """Phase 4: Image variation request from validator to miner.

    Contains the base image and parameters for generating variations.
    The miner will generate variations, encrypt them with drand timelock,
    upload to S3, and return S3 references.
    """
    base_image: str           # Base64 encoded image
    image_filename: str       # Original filename for reference
    variation_requests: List[VariationRequest] = Field(
        default_factory=list
    )  # Specific variation requests with type + intensity
    target_drand_round: int   # Drand round when decryption becomes possible
    reveal_timestamp: int     # Unix timestamp when reveal occurs
    challenge_id: Optional[str] = None  # Unique identifier for this challenge

    class Config:
        # Allow arbitrary types for flexibility
        arbitrary_types_allowed = True

    @property
    def requested_variations(self) -> int:
        """Number of variations requested (derived from variation_requests)."""
        return len(self.variation_requests)

    @property
    def variation_types(self) -> List[str]:
        """List of variation type names."""
        return [v.type for v in self.variation_requests]


class S3Submission(BaseModel):
    """Phase 4: Miner's S3 submission response.

    Contains references to encrypted images uploaded to S3.
    The actual images are NOT included - only S3 paths, hashes, and signatures.
    Post-validation will download and decrypt after drand reveal.

    SECURITY: path_signature prevents malicious miners from writing to other
    miners' S3 paths. The path_signature is derived from the miner's private
    key and can be verified during post-validation.
    """
    s3_key: str           # Path to encrypted file in S3 bucket
    image_hash: str       # SHA256 hash of the original (unencrypted) image
    signature: str        # Wallet signature proving ownership
    variation_type: str   # Which variation type this submission addresses
    path_signature: str   # Unique path component: sign(challenge_id:miner_hotkey)[:16]

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Main Synapse: Image Variation Protocol
# =============================================================================

class IdentitySynapse(bt.Synapse):
    """
    Protocol for requesting face image variations from miners.

    The validator sends a base image with variation parameters; the miner
    generates encrypted image variations, uploads them to S3, and returns
    S3 submission references.

    Attributes:
        image_request:   Validator → Miner. Base image + variation parameters.
        s3_submissions:  Miner → Validator. S3 paths + hashes (no actual images).
        process_time:    Optional timing metadata attached by the validator.
    """

    timeout: float = 120.0

    # Request (validator → miner)
    image_request: Optional[ImageRequest] = None

    # Response (miner → validator)
    s3_submissions: Optional[List[S3Submission]] = None

    # Timing metadata (attached by the validator dendrite)
    process_time: Optional[float] = None

    def deserialize(self) -> Optional[List[S3Submission]]:
        """Deserialize the miner's S3 submission response."""
        return self.s3_submissions
