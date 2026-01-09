# MIID/validator/base_images.py
#
# Phase 4: Base image loading module for image variation challenges.
# Loads base images from the local folder for sending to miners.

import os
import base64
import random
import bittensor as bt
from pathlib import Path
from typing import List, Tuple, Optional


# Base images folder (relative to validator module)
BASE_IMAGES_DIR = Path(__file__).parent / "base_images"

# Supported image extensions
SUPPORTED_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg"]


def load_random_base_image() -> Tuple[str, str]:
    """Load a random base image from the local folder.

    Selects a random image from the base_images directory,
    reads it, and encodes it as Base64 for transmission.

    Returns:
        Tuple of (filename, base64_encoded_image)

    Raises:
        FileNotFoundError: If base_images folder doesn't exist or is empty
    """
    if not BASE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Base images folder not found: {BASE_IMAGES_DIR}")

    # Get all image files
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(list(BASE_IMAGES_DIR.glob(ext)))

    if not image_files:
        raise FileNotFoundError(f"No images found in {BASE_IMAGES_DIR}")

    # Select random image
    selected_file = random.choice(image_files)

    # Read and encode
    with open(selected_file, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    bt.logging.debug(f"Loaded base image: {selected_file.name} ({len(image_bytes)} bytes)")

    return selected_file.name, base64_image


def load_all_base_images() -> List[Tuple[str, str]]:
    """Load all base images from the local folder.

    Useful for batch processing or when you need access to all images.

    Returns:
        List of (filename, base64_encoded_image) tuples
    """
    if not BASE_IMAGES_DIR.exists():
        bt.logging.warning(f"Base images folder not found: {BASE_IMAGES_DIR}")
        return []

    images = []
    for ext in SUPPORTED_EXTENSIONS:
        for img_path in BASE_IMAGES_DIR.glob(ext):
            try:
                with open(img_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                images.append((img_path.name, base64_image))
            except Exception as e:
                bt.logging.warning(f"Failed to load image {img_path.name}: {e}")

    bt.logging.info(f"Loaded {len(images)} base images from {BASE_IMAGES_DIR}")
    return images


def get_image_count() -> int:
    """Get the number of available base images.

    Returns:
        Number of images in the base_images directory
    """
    if not BASE_IMAGES_DIR.exists():
        return 0

    count = 0
    for ext in SUPPORTED_EXTENSIONS:
        count += len(list(BASE_IMAGES_DIR.glob(ext)))

    return count


def validate_base_images_folder() -> Tuple[bool, str]:
    """Validate that the base_images folder is properly configured.

    Checks:
    - Folder exists
    - Contains at least 1 image
    - Images are readable

    Returns:
        Tuple of (is_valid, message)
    """
    if not BASE_IMAGES_DIR.exists():
        return False, f"Base images folder does not exist: {BASE_IMAGES_DIR}"

    image_count = get_image_count()
    if image_count == 0:
        return False, f"No images found in {BASE_IMAGES_DIR}"

    # Try to load one image to verify readability
    try:
        _, _ = load_random_base_image()
    except Exception as e:
        return False, f"Failed to read images: {e}"

    return True, f"Base images folder valid: {image_count} images available"


def load_specific_image(filename: str) -> Optional[Tuple[str, str]]:
    """Load a specific image by filename.

    Args:
        filename: Name of the image file to load

    Returns:
        Tuple of (filename, base64_encoded_image) or None if not found
    """
    image_path = BASE_IMAGES_DIR / filename

    if not image_path.exists():
        bt.logging.warning(f"Image not found: {filename}")
        return None

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        return filename, base64_image
    except Exception as e:
        bt.logging.error(f"Failed to load image {filename}: {e}")
        return None
