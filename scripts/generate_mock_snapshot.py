#!/usr/bin/env python3
"""
Generate a mock reputation snapshot for testing.

This script creates a `reputation_snapshot.json` file that Flask loads at startup.
It can be run manually or scheduled via cron for periodic updates.

Usage:
    python scripts/generate_mock_snapshot.py [--output /path/to/output.json]

The snapshot follows the structure defined in reputation-policy-v1.md:
- version: ISO timestamp of snapshot generation
- generated_at: ISO timestamp
- miners: dict mapping hotkey -> {rep_score, rep_tier}

Tier boundaries (from reputation-policy-v1.md):
- Diamond: rep_score >= 50.0
- Gold: 10.0 - 49.999
- Silver: 2.0 - 9.999
- Bronze: > 1.0 - 1.999
- Neutral: 0.70 - 1.00
- Watch: 0.10 - 0.699
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


# Default output path (matches REPUTATION_SNAPSHOT_PATH in config.py)
DEFAULT_OUTPUT_PATH = "/data/MIID_data/reputation_snapshot.json"


# Tier boundaries from reputation-policy-v1.md
def get_tier(rep_score: float) -> str:
    """
    Determine tier based on rep_score.

    Args:
        rep_score: Reputation score (0.10 - 9999.0)

    Returns:
        Tier string (Diamond/Gold/Silver/Bronze/Neutral/Watch)
    """
    if rep_score >= 50.0:
        return "Diamond"
    elif rep_score >= 10.0:
        return "Gold"
    elif rep_score >= 2.0:
        return "Silver"
    elif rep_score > 1.0:
        return "Bronze"
    elif rep_score >= 0.70:
        return "Neutral"
    else:
        return "Watch"


def generate_snapshot(miners: dict, output_path: str) -> dict:
    """
    Generate a reputation snapshot.

    Args:
        miners: Dict mapping hotkey -> rep_score (tier will be computed)
        output_path: Path to write the snapshot JSON

    Returns:
        The generated snapshot dict
    """
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    snapshot = {
        "version": now,
        "generated_at": now,
        "miners": {}
    }

    for hotkey, rep_score in miners.items():
        rep_tier = get_tier(rep_score)
        snapshot["miners"][hotkey] = {
            "rep_score": round(rep_score, 2),
            "rep_tier": rep_tier
        }

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write snapshot
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

    print(f"[INFO] Generated snapshot with {len(miners)} miners at {output_path}")
    print(f"[INFO] Snapshot version: {now}")

    # Print tier distribution
    tier_counts = {}
    for miner_data in snapshot["miners"].values():
        tier = miner_data["rep_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"[INFO] Tier distribution: {tier_counts}")

    return snapshot


def main():
    parser = argparse.ArgumentParser(
        description="Generate a mock reputation snapshot for testing"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for snapshot JSON (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--from-file", "-f",
        help="Read miner hotkeys and scores from a JSON file (format: {hotkey: rep_score, ...})"
    )
    args = parser.parse_args()

    if args.from_file:
        # Read miners from file
        with open(args.from_file, 'r') as f:
            miners = json.load(f)
        print(f"[INFO] Loaded {len(miners)} miners from {args.from_file}")
    else:
        # Generate sample miners with various tiers for testing
        # In production, this would query the metagraph or database
        miners = {
            # Diamond tier (rep_score >= 50.0)
            "5DiamondMiner1ExampleHotkeyForTestingPurposes111": 55.0,
            "5DiamondMiner2ExampleHotkeyForTestingPurposes222": 52.5,

            # Gold tier (10.0 - 49.999)
            "5GoldMiner1ExampleHotkeyForTestingPurposes1111111": 25.0,
            "5GoldMiner2ExampleHotkeyForTestingPurposes2222222": 15.5,

            # Silver tier (2.0 - 9.999)
            "5SilverMiner1ExampleHotkeyForTestingPurposes11111": 5.0,
            "5SilverMiner2ExampleHotkeyForTestingPurposes22222": 3.2,

            # Bronze tier (> 1.0 - 1.999)
            "5BronzeMiner1ExampleHotkeyForTestingPurposes11111": 1.5,
            "5BronzeMiner2ExampleHotkeyForTestingPurposes22222": 1.2,

            # Neutral tier (0.70 - 1.00) - baseline for new miners
            "5NeutralMiner1ExampleHotkeyForTestingPurpose1111": 1.0,
            "5NeutralMiner2ExampleHotkeyForTestingPurpose2222": 0.85,

            # Watch tier (0.10 - 0.699) - miners under scrutiny
            "5WatchMiner1ExampleHotkeyForTestingPurposes11111": 0.5,
            "5WatchMiner2ExampleHotkeyForTestingPurposes22222": 0.25,
        }
        print("[INFO] Using sample miners for testing")
        print("[INFO] To use real miners, provide --from-file with a JSON mapping hotkey -> rep_score")

    generate_snapshot(miners, args.output)


if __name__ == "__main__":
    main()
