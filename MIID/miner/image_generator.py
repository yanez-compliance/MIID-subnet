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
    """Generate ONE combined image with ALL variations applied using FLUX.

    IMPORTANT: This function now generates ONE SINGLE IMAGE with all variation_requests
    applied simultaneously (e.g., background changed + accessory added + pose modified),
    NOT separate images for each variation type.

    Flow: validator sends image_request with variation_requests (each has type + intensity)
    -> miner passes base_image + variation_requests here -> FLUX gets base image and
    a COMBINED prompt with ALL variations -> ONE output image with all modifications.

    Args:
        base_image: PIL Image of the base face (decoded from image_request.base_image).
        variation_requests: List of validator variation requests; each has .type and .intensity
            (e.g. protocol.VariationRequest, or dict with "type"/"intensity").
            ALL variations will be applied to ONE output image.

    Returns:
        List with ONE dict containing:
            - image: PIL Image object (with ALL variations applied)
            - variation_type: "combined" (indicates all variations in one image)
            - image_bytes: bytes - raw image data
            - image_hash: str - SHA256 hash for verification
    """
    if not variation_requests:
        return []

    # FLUX-based generation: ALL requests combined -> one prompt -> ONE image
    # The generate_variations_flux function should combine all prompts
    raw_results = generate_variations_flux(
        base_image,
        variation_requests,
    )

    # Take only the first result (should be the combined image)
    # If generate_variations_flux returns multiple images, we need to update it
    if not raw_results:
        return []
    
    # Use the first (and should be only) result
    combined_result = raw_results[0] if isinstance(raw_results, list) else raw_results
    variation_image = combined_result["image"]
    
    image_bytes = encode_image_to_bytes(variation_image)
    image_hash = calculate_image_hash(image_bytes)

    # Create variation_type string that describes all variations
    variation_types = [req.type if hasattr(req, 'type') else req.get('type', 'unknown') 
                      for req in variation_requests]
    combined_type = "combined"  # Simple label, or could be: "+".join(variation_types)

    result = {
        "image": variation_image,
        "variation_type": combined_type,
        "image_bytes": image_bytes,
        "image_hash": image_hash
    }

    bt.logging.info(
        f"Generated ONE combined image with {len(variation_requests)} variations applied: "
        f"{', '.join(variation_types)}, hash: {image_hash[:16]}..."
    )

    # Return list with ONE element (for compatibility with existing code)
    return [result]


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
