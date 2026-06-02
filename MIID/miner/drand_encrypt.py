# MIID/miner/drand_encrypt.py
#
# Phase 4: Drand timelock encryption for miners.
# Encrypts image data with drand timelock so it can only be
# decrypted after the target drand round is released.

import bittensor as bt
from typing import Optional


# Use bittensor's built-in timelock (backed by bittensor_drand) so we don't
# need the separate `timelock` Python package. Available on bittensor>=9.
try:
    from bittensor import timelock as _bt_timelock
    from bittensor_drand import get_latest_round as _get_latest_round
    TLOCK_AVAILABLE = True
except ImportError:
    TLOCK_AVAILABLE = False
    # Don't log warning at import time - will be logged when used


# Drand Quicknet public key for encryption
# See: https://docs.drand.love/docs/cryptography/quicknet
DRAND_QUICKNET_PK = "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"


def encrypt_for_drand(data: bytes, target_round: int) -> bytes:
    """Encrypt data with drand timelock.

    Uses Identity-Based Encryption (IBE) where the drand round
    serves as the identity. The data can only be decrypted after
    the drand beacon releases the signature for the target round.

    Args:
        data: Raw bytes to encrypt
        target_round: Drand round number when decryption becomes possible

    Returns:
        Encrypted bytes (tlock ciphertext)

    Raises:
        ImportError: If timelock package is not installed
        Exception: If encryption fails
    """
    if not TLOCK_AVAILABLE:
        raise ImportError(
            "bittensor.timelock not available. "
            "Upgrade bittensor to >=9 (which ships bittensor_drand)."
        )

    # bittensor.timelock takes a relative `n_blocks`; convert from the
    # absolute target_round supplied by the validator. block_time=3.0
    # (drand QuickNet period) makes 1 block == 1 drand round.
    n_blocks = max(1, int(target_round) - int(_get_latest_round()))
    ciphertext, _reveal_round = _bt_timelock.encrypt(
        data, n_blocks=n_blocks, block_time=3.0
    )

    bt.logging.debug(
        f"Encrypted {len(data)} bytes for drand round {target_round}"
    )

    return ciphertext


def decrypt_with_drand(encrypted_data: bytes, drand_signature: bytes = None) -> bytes:
    """Decrypt timelock-encrypted data using drand signature.

    This can only succeed after the target round's signature is available.

    Args:
        encrypted_data: Tlock-encrypted ciphertext
        drand_signature: Unused; bittensor.timelock fetches the signature
            itself. Kept for backwards compatibility.

    Returns:
        Decrypted original data

    Raises:
        ImportError: If bittensor.timelock is not available
        Exception: If decryption fails
    """
    if not TLOCK_AVAILABLE:
        raise ImportError(
            "bittensor.timelock not available. "
            "Upgrade bittensor to >=9 (which ships bittensor_drand)."
        )

    plaintext = _bt_timelock.decrypt(encrypted_data, no_errors=False)
    if plaintext is None:
        raise RuntimeError("Decryption failed (drand round not yet revealed?)")

    bt.logging.debug(f"Decrypted {len(plaintext)} bytes")

    return plaintext


def encrypt_image_for_drand(
    image_bytes: bytes,
    target_round: int
) -> Optional[bytes]:
    """Encrypt image bytes with drand timelock.

    Wrapper around encrypt_for_drand with error handling.

    Args:
        image_bytes: Raw image bytes to encrypt
        target_round: Drand round when decryption becomes possible

    Returns:
        Encrypted bytes, or None if encryption fails
    """
    try:
        return encrypt_for_drand(image_bytes, target_round)
    except ImportError as e:
        bt.logging.error(f"Timelock not available: {e}")
        return None
    except Exception as e:
        bt.logging.error(f"Drand encryption failed: {e}")
        return None


def is_timelock_available() -> bool:
    """Check if timelock encryption is available.

    Returns:
        True if timelock package is installed and working
    """
    return TLOCK_AVAILABLE


def validate_encrypted_data(encrypted_data: bytes) -> bool:
    """Basic validation of encrypted data format.

    Args:
        encrypted_data: Bytes that should be tlock ciphertext

    Returns:
        True if data appears to be valid tlock ciphertext
    """
    if not encrypted_data:
        return False

    # Tlock ciphertext has a minimum size
    if len(encrypted_data) < 100:
        return False

    # TODO: Add more detailed validation if needed
    return True
