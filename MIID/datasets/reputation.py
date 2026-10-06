"""Reputation snapshot cache and reward-allocation decay for upload_data."""

import json
import os
import threading

from MIID.datasets.config import REPUTATION_SNAPSHOT_PATH

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


def process_reward_allocation(hotkey, data, get_snapshot, reward_allocation):
    """
    Fetch current snapshot, optionally apply decay from reward_allocation,
    and build the rep_cache response fields.

    Must be called under _snapshot_lock by the caller, or this function
    acquires the lock itself.

    Returns:
        dict with keys: snapshot_version, generated_at, rep_cache
    """
    snapshot_version = None
    generated_at = None
    rep_cache = {}
    current_snapshot = None

    with _snapshot_lock:
        try:
            current_snapshot = get_snapshot()

            snapshot_version = current_snapshot.get("version")
            generated_at = current_snapshot.get("generated_at")

            miners_list = current_snapshot.get("miners", [])
            for miner in miners_list:
                hotkey_key = miner.get("hotkey")
                if hotkey_key:
                    rep_cache[hotkey_key] = {
                        "rep_score": miner.get("rep_score", 0),
                        "rep_tier": miner.get("rep_tier", "Unknown"),
                    }

            print(
                f"[INFO] Retrieved snapshot from get_snapshot(): "
                f"version={snapshot_version}, miners={len(rep_cache)}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to get snapshot from get_snapshot(): {e}")

        reward_allocation_data = data.get("reward_allocation")
        if reward_allocation_data and current_snapshot:
            try:
                allocations = reward_allocation_data.get("allocations", [])

                # Legacy "miners" field (backwards compatibility)
                if not allocations and reward_allocation_data.get("miners"):
                    allocations = [
                        {
                            "timestamp": reward_allocation_data.get(
                                "rep_snapshot_version"
                            ),
                            "rep_snapshot_version": reward_allocation_data.get(
                                "rep_snapshot_version"
                            ),
                            "miners": reward_allocation_data.get("miners"),
                        }
                    ]

                # Decay every miner in the honored snapshot once per pending
                # allocation (stack DECAY_PER_ALLOCATION per cycle).
                decay_counts = {}
                snapshot_miners_list = current_snapshot.get("miners", [])
                for _allocation in allocations:
                    for snapshot_miner in snapshot_miners_list:
                        miner_hotkey = snapshot_miner.get("hotkey")
                        if not miner_hotkey:
                            continue
                        decay_counts[miner_hotkey] = (
                            decay_counts.get(miner_hotkey, 0) + 1
                        )

                snapshot_miners_dict = {
                    miner.get("hotkey"): miner for miner in snapshot_miners_list
                }

                updated_hotkeys = []
                for miner_hotkey, decay_times in decay_counts.items():
                    if miner_hotkey in snapshot_miners_dict:
                        snapshot_miner = snapshot_miners_dict[miner_hotkey]
                        current_score = snapshot_miner.get("rep_score", 0.0)
                        snapshot_miner["rep_score"] = max(
                            0.0,
                            current_score - (DECAY_PER_ALLOCATION * decay_times),
                        )
                        if miner_hotkey in rep_cache:
                            rep_cache[miner_hotkey]["rep_score"] = snapshot_miner[
                                "rep_score"
                            ]
                        updated_hotkeys.append(miner_hotkey)

                updated_miners_count = len(updated_hotkeys)

                if updated_miners_count > 0:
                    reward_allocation(current_snapshot)
                    print(
                        f"[INFO] {hotkey} Applied decay (-{DECAY_PER_ALLOCATION}) to "
                        f"{updated_miners_count} miner(s) and sent to database "
                        f"via reward_allocation()"
                    )
                else:
                    print(
                        f"[INFO] {hotkey} No miners to update in reward allocation"
                    )

            except Exception as e:
                print(f"[ERROR] Failed to process reward_allocation: {e}")

    return {
        "snapshot_version": snapshot_version,
        "generated_at": generated_at,
        "rep_cache": rep_cache,
    }
