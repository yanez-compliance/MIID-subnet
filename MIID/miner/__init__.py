# MIID/miner/__init__.py
#
# Phase 4: Miner modules for image variation generation.
# These imports are optional -- if the Phase 4 packages (diffusers,
# opencv-python, etc.) are not installed, the miner still works for
# name-variation tasks; image-variation requests are skipped at runtime.

try:
    from .image_generator import decode_base_image, generate_variations
    from .drand_encrypt import encrypt_for_drand, encrypt_image_for_drand
    from .s3_upload import upload_to_s3
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False

__all__ = [
    "PHASE4_AVAILABLE",
    "decode_base_image",
    "generate_variations",
    "encrypt_for_drand",
    "encrypt_image_for_drand",
    "upload_to_s3",
]
