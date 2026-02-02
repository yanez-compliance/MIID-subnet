# MIID/miner/image_generator.py
#
# Phase 4: Image variation generator for miners.
# Uses FLUX-based generation from generate_variations.py when configured
# (HF token + model). See MIID.miner.generate_variations module docstring for setup.

import base64
import hashlib
import io
import bittensor as bt
from typing import List, Dict
from PIL import Image

from MIID.miner.generate_variations import generate_variations as generate_variations_flux
from MIID.miner.ada_face_compare import validate_single_variation


def decode_base_image(base64_image: str) -> Image.Image:
    """Decode a Base64 encoded image to a PIL Image.

    Args:
        base64_image: Base64 encoded image string

    Returns:
        PIL Image object
    """
    image_bytes = base64.b64decode(base64_image)
    return Image.open(io.BytesIO(image_bytes))


def encode_image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Encode a PIL Image to bytes.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Image as bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def calculate_image_hash(image_bytes: bytes) -> str:
    """Calculate SHA256 hash of image bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(image_bytes).hexdigest()


def generate_variations(
    base_image: Image.Image,
    variation_requests: List,
) -> List[Dict]:
    """Generate image variations from a base image using FLUX (see generate_variations.py).

    Flow: validator sends image_request with variation_requests (each has type + intensity)
    -> miner passes base_image + variation_requests here -> FLUX gets base image and
    the correct prompt per request (type + intensity from IMAGE_VARIATION_PROMPTS)
    -> one variation image per request. We add image_bytes and image_hash for S3/submission.

    Args:
        base_image: PIL Image of the base face (decoded from image_request.base_image).
        variation_requests: List of validator variation requests; each has .type and .intensity
            (e.g. protocol.VariationRequest, or dict with "type"/"intensity").
            Order is preserved: first request -> first result.

    Returns:
        List of dicts, each containing:
            - image: PIL Image object
            - variation_type: str - which type this is (matches request order)
            - image_bytes: bytes - raw image data
            - image_hash: str - SHA256 hash for verification
    """
    if not variation_requests:
        return []

    # FLUX-based generation: one (type, intensity) per request -> one prompt -> one image
    raw_results = generate_variations_flux(
        base_image,
        variation_requests,
    )

    variations = []
    for item in raw_results:
        variation_image = item["image"]
        var_type = item["variation_type"]
        image_bytes = encode_image_to_bytes(variation_image)
        image_hash = calculate_image_hash(image_bytes)

        variations.append({
            "image": variation_image,
            "variation_type": var_type,
            "image_bytes": image_bytes,
            "image_hash": image_hash
        })

        bt.logging.debug(
            f"Generated {var_type} variation, hash: {image_hash[:16]}..."
        )

    bt.logging.info(f"Generated {len(variations)} variations")
    return variations


def validate_variation(
    variation: Dict,
    base_image: Image.Image,
    min_similarity: float = 0.7
) -> bool:
    """Validate that a variation maintains face identity using AdaFace.

    Args:
        variation: Variation dict from generate_variations (must have "image" key).
        base_image: Original base image (PIL Image).
        min_similarity: Minimum AdaFace cosine similarity threshold (default 0.7).

    Returns:
        True if variation maintains face identity, False otherwise.
    """
    return validate_single_variation(
        base_image,
        variation["image"],
        min_similarity=min_similarity,
    )
