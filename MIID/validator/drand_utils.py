# MIID/validator/drand_utils.py
#
# Phase 4: Drand utilities for timelock encryption.
# Calculates target drand rounds for delayed reveal of encrypted images.

import time
import requests
import bittensor as bt
from typing import Tuple, Optional


# Drand Quicknet configuration (3-second periods)
# See: https://docs.drand.love/docs/cryptography/quicknet
DRAND_QUICKNET_URL = "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
DRAND_QUICKNET_PK = "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"

# Fallback values if API is unavailable
# Quicknet genesis: August 23, 2023, 00:09:27 UTC
FALLBACK_GENESIS = 1692803367
FALLBACK_PERIOD = 3  # 3 seconds per round


def get_drand_info() -> Tuple[int, int]:
    """Fetch current drand chain info from the API.

    Returns:
        Tuple of (genesis_time, period_seconds)

    Raises:
        Exception: If API call fails and we need to use fallback
    """
    try:
        response = requests.get(f"{DRAND_QUICKNET_URL}/info", timeout=5)
        response.raise_for_status()
        info = response.json()
        return info["genesis_time"], info["period"]
    except Exception as e:
        bt.logging.warning(f"Failed to fetch drand info, using fallback: {e}")
        return FALLBACK_GENESIS, FALLBACK_PERIOD


def calculate_target_round(delay_seconds: int) -> Tuple[int, int]:
    """Calculate the drand round for a future reveal time.

    The target round is calculated based on the current time plus
    the specified delay. This is used to determine when encrypted
    images should become decryptable.

    Args:
        delay_seconds: Number of seconds from now until reveal

    Returns:
        Tuple of (target_round, reveal_timestamp)

    Example:
        >>> target_round, reveal_time = calculate_target_round(300)
        >>> print(f"Images will be decryptable at round {target_round}")
        >>> print(f"Reveal time: {reveal_time} (Unix timestamp)")
    """
    try:
        genesis, period = get_drand_info()
    except Exception:
        genesis = FALLBACK_GENESIS
        period = FALLBACK_PERIOD

    target_time = int(time.time()) + delay_seconds
    target_round = (target_time - genesis) // period + 1
    reveal_timestamp = genesis + (target_round - 1) * period

    bt.logging.debug(
        f"Calculated drand target: round={target_round}, "
        f"reveal_time={reveal_timestamp}, delay={delay_seconds}s"
    )

    return target_round, reveal_timestamp


def get_current_round() -> int:
    """Get the current drand round number.

    Returns:
        Current round number
    """
    try:
        response = requests.get(f"{DRAND_QUICKNET_URL}/public/latest", timeout=5)
        response.raise_for_status()
        return response.json()["round"]
    except Exception as e:
        bt.logging.warning(f"Failed to fetch current round: {e}")
        # Estimate based on current time
        genesis, period = FALLBACK_GENESIS, FALLBACK_PERIOD
        return (int(time.time()) - genesis) // period + 1


def wait_for_round(target_round: int, timeout: int = 600) -> Optional[bytes]:
    """Wait for a specific drand round and return the signature.

    This function blocks until the target round is available or timeout.
    Used during post-validation to decrypt timelock-encrypted images.

    Args:
        target_round: The drand round to wait for
        timeout: Maximum seconds to wait (default: 10 minutes)

    Returns:
        The drand signature as bytes, or None if timeout/error
    """
    start_time = time.time()

    # First check if the round is already available
    try:
        response = requests.get(f"{DRAND_QUICKNET_URL}/public/{target_round}", timeout=5)
        if response.status_code == 200:
            signature = bytes.fromhex(response.json()["signature"])
            bt.logging.info(f"Round {target_round} already available")
            return signature
    except Exception:
        pass

    # Calculate expected time for target round
    genesis, period = get_drand_info()
    target_time = genesis + (target_round - 1) * period
    wait_seconds = max(0, target_time - int(time.time()))

    bt.logging.info(f"Waiting {wait_seconds}s for drand round {target_round}")

    # Wait until target time (with buffer)
    while time.time() < target_time + 2:  # +2s buffer
        if time.time() - start_time > timeout:
            bt.logging.error(f"Timeout waiting for drand round {target_round}")
            return None
        time.sleep(1)

    # Fetch the signature with retries
    for attempt in range(5):
        try:
            response = requests.get(
                f"{DRAND_QUICKNET_URL}/public/{target_round}",
                timeout=5
            )
            if response.status_code == 200:
                signature = bytes.fromhex(response.json()["signature"])
                bt.logging.info(f"Got signature for round {target_round}")
                return signature
        except Exception as e:
            bt.logging.warning(f"Attempt {attempt + 1} failed: {e}")

        time.sleep(2)

    bt.logging.error(f"Failed to get signature for round {target_round}")
    return None


def get_round_signature(round_number: int) -> Optional[bytes]:
    """Fetch the signature for a specific round (non-blocking).

    Args:
        round_number: The drand round number

    Returns:
        The signature as bytes, or None if not yet available
    """
    try:
        response = requests.get(
            f"{DRAND_QUICKNET_URL}/public/{round_number}",
            timeout=5
        )
        if response.status_code == 200:
            return bytes.fromhex(response.json()["signature"])
        return None
    except Exception as e:
        bt.logging.warning(f"Failed to get round {round_number} signature: {e}")
        return None


def is_round_available(round_number: int) -> bool:
    """Check if a drand round is available (non-blocking).

    Args:
        round_number: The drand round to check

    Returns:
        True if the round's signature is available
    """
    return get_round_signature(round_number) is not None


def calculate_reveal_buffer(timeout_seconds: float) -> int:
    """Calculate appropriate reveal delay based on request timeout.

    The reveal should happen after all miners have had time to submit.
    We add a buffer to ensure all responses are in before reveal.

    Args:
        timeout_seconds: The synapse timeout for miner responses

    Returns:
        Recommended delay in seconds for drand reveal
    """
    # Add 60 second buffer after timeout expires
    buffer = 60
    return int(timeout_seconds + buffer)
