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
    type: str        # Variation type: pose_edit, lighting_edit, expression_edit, background_in, background_out, combined (type+type), screen_replay
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

    Live challenge: one base_image (per-round face) plus six synthetic FLUX
    VariationRequests — five face edits and one screen_replay (device named
    + ≥2 visual cues in the variation text). Screen_replay is graded as a
    KAV slot but the remote API ignores its score; miners should aim to
    break Passive Liveness v3 (/is_live).

    # --- PAUSED: fixed IOTD seeds + real physical screen-replay ---
    # Uncomment the fields below (and matching validator/miner code) to
    # restore sending today's/tomorrow's IOTD and real-capture instructions.
    # Previously carried three images:
    #   1. base_image — per-round face for synthetic FLUX variations
    #   2. daily_seed_image — today's IOTD (screen-replay seed)
    #   3. tomorrow_seed_image — tomorrow's IOTD (sent early)
    """
    base_image: str           # Base64 encoded image
    image_filename: str       # Original filename for reference
    variation_requests: List[VariationRequest] = Field(
        default_factory=list
    )  # Specific variation requests with type + intensity
    target_drand_round: int   # Drand round when decryption becomes possible
    reveal_timestamp: int     # Unix timestamp when reveal occurs
    challenge_id: Optional[str] = None  # Unique identifier for this challenge

    # # --- PAUSED: fixed IOTD / real screen-replay (restore later) ---
    # # Today's image-of-the-day (screen-replay seed; same for every miner/validator that UTC day)
    # daily_seed_image: Optional[str] = None       # Base64 encoded today's IOTD
    # daily_seed_filename: Optional[str] = None    # Filename of today's IOTD
    # daily_seed_date: Optional[str] = None        # UTC date YYYY-MM-DD for today's IOTD
    #
    # # Tomorrow's image-of-the-day (sent early so miners can prepare overnight)
    # tomorrow_seed_image: Optional[str] = None    # Base64 encoded tomorrow's IOTD
    # tomorrow_seed_filename: Optional[str] = None # Filename of tomorrow's IOTD
    # tomorrow_seed_date: Optional[str] = None     # UTC date YYYY-MM-DD for tomorrow's IOTD
    #
    # real_screen_replay_instructions: Optional[str] = None  # Human-readable task instructions

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


# # --- PAUSED: real screen-replay UAV metadata (restore later) ---
# class ScreenReplayUAV(BaseModel):
#     """Miner-reported metadata for a real screen-replay capture (UAV-style).
#
#     Attached to the S3Submission whose variation_type == "screen_replay".
#     Describes ONE capture event that is proven with TWO media files of the
#     same seed-on-screen moment — (1) face close-up (photo or video) and
#     (2) environment photo (see S3Submission.s3_key / s3_key_angle2 below);
#     this carries the extra claims manual review needs: which seed image was
#     used, which capture_variant was chosen, when/how it was captured, and a
#     true/false checklist for each known visual cue (see
#     SCREEN_REPLAY_VISUAL_CUES in MIID/validator/image_variations.py). All
#     five cues are always reported — a real capture may show none, some, or
#     all of them.
#     """
#     seed_image: str            # Filename of the fixed daily seed used
#     date: str                  # Capture date, "YYYY-MM-DD" (UTC)
#     camera_used: str           # Camera/device used to take the photo/video
#     device_photographed: str   # Device the seed was displayed on (phone/tablet/laptop/monitor/tv)
#
#     # Which variety track this submission uses (see SCREEN_REPLAY_CAPTURE_VARIANTS).
#     # Defaults to seed_unchanged for older submissions that omit the field.
#     # Six options: 3 photo (seed_unchanged, seed_smiling, seed_eyes_closed) +
#     # 3 video (seed_video_blinking, seed_video_smiling, seed_video_smile_and_blink).
#     capture_variant: str = "seed_unchanged"
#
#     # Cue checklist — one bool per cue key in SCREEN_REPLAY_VISUAL_CUES
#     moire_pixel_grid: bool               # Interference pattern from screen subpixels
#     screen_glare_hotspots: bool          # Specular reflections on the display surface
#     perspective_keystone_distortion: bool  # Geometric distortion from off-angle capture
#     gamma_contrast_shift: bool           # Colour/brightness characteristics of display capture
#     edge_crop_cues: bool                 # Screen borders, bezel reflections, or cropping
#
#     class Config:
#         arbitrary_types_allowed = True


class S3Submission(BaseModel):
    """Phase 4: Miner's S3 submission response.

    Contains references to encrypted images uploaded to S3.
    The actual images are NOT included - only S3 paths, hashes, and signatures.
    Post-validation will download and decrypt after drand reveal.

    SECURITY: path_signature prevents malicious miners from writing to other
    miners' S3 paths. The path_signature is derived from the miner's private
    key and can be verified during post-validation.

    Live challenge: every variation type (including synthetic screen_replay)
    uses only the primary fields. Dual-file real screen-replay (angle2 +
    ScreenReplayUAV) is paused — uncomment those fields below to restore.
    """
    s3_key: str           # Path to encrypted file in S3 bucket
    image_hash: str       # SHA256 hash of the original (unencrypted) media
    signature: str        # Wallet signature proving ownership
    variation_type: str   # Which variation type this submission addresses
    path_signature: str   # Unique path component: sign(challenge_id:miner_hotkey)[:16]

    # # --- PAUSED: real screen-replay dual-file + UAV (restore later) ---
    # # screen_replay ONLY — wider environment shot of the same capture.
    # # image_hash_angle2 must differ from image_hash (two distinct files,
    # # not the same file uploaded twice) or the submission is rejected as a
    # # malformed/duplicate screen-replay.
    # s3_key_angle2: Optional[str] = None
    # image_hash_angle2: Optional[str] = None
    # signature_angle2: Optional[str] = None
    #
    # # Only populated when variation_type == "screen_replay" (real path)
    # screen_replay_uav: Optional[ScreenReplayUAV] = None

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Voice Clone Challenge Types
# =============================================================================

class VoiceRequest(BaseModel):
    """Voice clone request from validator to miner.

    Contains a ~30s reference WAV (English or Spanish) and the text the miner
    should speak in that speaker identity. Graded via UAV post-validation only
    (no KAV). Miners may skip voice and return an empty voice_s3_submissions list.
    """
    base_voice: str              # Base64 encoded reference WAV (~30s)
    voice_filename: str          # Original filename for reference
    language: str                # "en" | "es"
    target_words: List[str] = Field(default_factory=list)  # Tokens to speak
    target_text: str = ""        # Same tokens joined for TTS, e.g. "three apple ..."
    reference_transcript: str = ""  # What the reference clip says (from API)
    target_drand_round: int      # Drand round when decryption becomes possible
    reveal_timestamp: int        # Unix timestamp when reveal occurs
    challenge_id: Optional[str] = None  # Unique identifier for this challenge

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Main Synapse: Image Variation + Voice Protocol
# =============================================================================

class IdentitySynapse(bt.Synapse):
    """
    Protocol for requesting face image variations and/or voice clones from miners.

    The validator sends a base image with variation parameters and/or a voice
    request; the miner generates encrypted media, uploads to S3, and returns
    S3 submission references.

    Attributes:
        image_request:          Validator → Miner. Base image + variation parameters.
        s3_submissions:         Miner → Validator. Image S3 paths + hashes.
        voice_request:          Validator → Miner. Reference voice + target words.
        voice_s3_submissions:   Miner → Validator. Voice S3 paths + hashes.
        process_time:           Optional timing metadata attached by the validator.
    """

    timeout: float = 120.0

    # Request (validator → miner)
    image_request: Optional[ImageRequest] = None
    voice_request: Optional[VoiceRequest] = None

    # Response (miner → validator)
    s3_submissions: Optional[List[S3Submission]] = None
    voice_s3_submissions: Optional[List[S3Submission]] = None

    # Timing metadata (attached by the validator dendrite)
    process_time: Optional[float] = None

    def deserialize(self) -> Optional[List[S3Submission]]:
        """Deserialize the miner's S3 submission response."""
        return self.s3_submissions
