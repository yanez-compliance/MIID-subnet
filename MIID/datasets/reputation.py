"""Reputation snapshot cache and reward-allocation decay for upload_data."""

import json
import os
import threading
from datetime import datetime

from MIID.datasets.config import REPUTATION_SNAPSHOT_PATH
from MIID.utils.validator_requests import MAX_ALLOCATION_BATCH

# Global reputation snapshot cache (thread-safe)
CURRENT_REP_SNAPSHOT = {
    "version": None,
    "generated_at": None,
    "miners": {},
}
_snapshot_lock = threading.Lock()

# UAV decay applied once per pending allocation against the honored snapshot.
DECAY_PER_ALLOCATION = 0.05


def load_reputation_snapshot():
    """
    Load reputation snapshot from JSON file into memory.

    Called at Flask startup and can be called to reload the snapshot.
    Thread-safe using _snapshot_lock.
    """
    global CURRENT_REP_SNAPSHOT

    if not os.path.exists(REPUTATION_SNAPSHOT_PATH):
        print(
            f"[WARNING] Reputation snapshot not found at {REPUTATION_SNAPSHOT_PATH}. "
            "Using empty snapshot."
        )
        return

    try:
        with open(REPUTATION_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        with _snapshot_lock:
            CURRENT_REP_SNAPSHOT = {
                "version": data.get("version"),
                "generated_at": data.get("generated_at"),
                "miners": data.get("miners", {}),
            }
        print(
            f"[INFO] Loaded reputation snapshot version: "
            f"{CURRENT_REP_SNAPSHOT['version']} with "
            f"{len(CURRENT_REP_SNAPSHOT['miners'])} miners"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load reputation snapshot: {e}")


def validate_reward_allocations(data):
    """Validate the batch before persisting or invoking the database callback."""
    batch = data.get("reward_allocation", {})
    if not isinstance(batch, dict):
        raise ValueError("reward_allocation must be a JSON object")
    allocations = batch.get("allocations", [])
    if not isinstance(allocations, list) or len(allocations) > MAX_ALLOCATION_BATCH:
        raise ValueError(f"Expected at most {MAX_ALLOCATION_BATCH} reward allocations")
    if batch.get("miners") or (allocations and batch.get("enabled") is False):
        raise ValueError("Expected reward allocations in the allocations array")
    if batch.get("pending_count", len(allocations)) != len(allocations):
        raise ValueError("pending_count does not match the allocation batch")

    seen = set()
    for event in allocations:
        if not isinstance(event, dict) or not isinstance(event.get("timestamp"), str):
            raise ValueError("Each allocation needs a valid cycle timestamp")
        timestamp = event["timestamp"]
        try:
            parsed = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            raise ValueError("Invalid allocation cycle timestamp") from None
        if parsed.strftime("%Y-%m-%d_%H-%M-%S") != timestamp or timestamp in seen:
            raise ValueError("Invalid or duplicate allocation cycle timestamp")
        seen.add(timestamp)

        if "rep_snapshot_version" not in event:
            raise ValueError("Each allocation needs a reputation snapshot version")
        version = event["rep_snapshot_version"]
        if version is not None and (not isinstance(version, str) or not version.strip()):
            raise ValueError("Invalid reputation snapshot version")
        miners = event.get("miners")
        if not isinstance(miners, list) or any(not isinstance(miner, dict) for miner in miners):
            raise ValueError("Each allocation needs a miner metrics list")
    return allocations


def process_reward_allocation(allocations, get_snapshot, reward_allocation):
    """Apply validated events and propagate database failures to the caller."""
    with _snapshot_lock:
        snapshot = get_snapshot()
        if not isinstance(snapshot, dict) or not snapshot.get("version"):
            raise RuntimeError("Database did not return a versioned reputation snapshot")
        miners = snapshot["miners"]
        if not isinstance(miners, list):
            raise RuntimeError("Database did not return a reputation miner list")

        if allocations and miners:
            decay = DECAY_PER_ALLOCATION * len(allocations)
            for miner in miners:
                miner["rep_score"] = max(0.0, miner.get("rep_score", 0.0) - decay)

            acknowledgement = reward_allocation(snapshot)
            if not isinstance(acknowledgement, dict) or not (
                acknowledgement.get("message") == "Reward allocation successful"
                or acknowledgement.get("status") == "success"
            ):
                raise RuntimeError("Database did not acknowledge the reputation write")

        return {
            "snapshot_version": snapshot["version"],
            "generated_at": snapshot.get("generated_at"),
            "rep_cache": {
                miner["hotkey"]: {
                    "rep_score": miner.get("rep_score", 0.0),
                    "rep_tier": miner.get("rep_tier", "Unknown"),
                }
                for miner in miners
                if miner.get("hotkey")
            },
        }
