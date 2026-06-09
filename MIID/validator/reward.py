# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Reward Module — Face Image Variation Scoring

KAV (Known Address Variation / online image quality score):
  - Computed by calling an external grading API that evaluates the submitted
    face image variations for quality, identity preservation, and variation
    compliance.
  - Currently a TODO placeholder returning 0.0 scores until the API is ready.

UAV (Unknown Address Variation / reputation score):
  - Computed from the validator's reputation snapshot (rep_data) obtained from
    the MIID server on each successful upload.
  - Handled entirely by apply_reputation_rewards() — unchanged from Phase 3.

Reward allocation:
  - KAV weight: 10% of kept rewards  (--neuron.kav_weight, default 0.10)
  - UAV weight: 90% of kept rewards  (--neuron.uav_weight, default 0.90)
  - Burn fraction: 65%               (--neuron.burn_fraction, default 0.65)
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import bittensor as bt


# =============================================================================
# KAV: Image Variation Grading (via external API)
# =============================================================================

def get_image_variation_rewards(
    self,
    challenge_id: str,
    s3_submissions_by_miner: Dict[str, Any],
    miner_uids: List[int],
    selected_variations: List[Dict],
    skip_burn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Compute KAV rewards for each miner based on their image variation submissions.

    Currently calls an external grading API (TODO — not yet implemented).
    Returns placeholder 0.0 scores for all miners until the API is ready.

    The function preserves the same return signature as the former
    get_name_variation_rewards() so that forward.py and apply_reputation_rewards()
    are unaffected.

    Args:
        self:                    Validator instance (for metagraph, config, wallet).
        challenge_id:            Unique ID for the current challenge round.
        s3_submissions_by_miner: Dict mapping str(uid) → submission metadata dict.
        miner_uids:              Ordered list of miner UIDs being scored.
        selected_variations:     List of requested variation specs (type, intensity…).
        skip_burn:               When True, the burn UID is NOT appended to the
                                 returned arrays (burn is applied later in
                                 apply_reputation_rewards).

    Returns:
        Tuple of:
            rewards       – np.ndarray of KAV quality scores, one per miner.
                            If skip_burn=False, the burn UID reward is appended.
            uids          – np.ndarray of corresponding UIDs.
            detailed_metrics – List of per-miner metric dicts for logging.
    """
    # TODO: Call external grading API to score each miner's image variations.
    #
    # Expected API contract (to be defined):
    #   POST /grade_variations
    #   Body: {
    #       "challenge_id": challenge_id,
    #       "selected_variations": selected_variations,
    #       "submissions": s3_submissions_by_miner,
    #   }
    #   Response: {
    #       "<uid>": { "kav_score": float(0-1), "details": {...} },
    #       ...
    #   }
    #
    # Until the API is implemented, every miner receives a 0.0 KAV score.
    # The UAV (reputation) component in apply_reputation_rewards() will still
    # operate normally using the rep_data snapshot from the MIID server.

    bt.logging.info(
        f"[TODO] get_image_variation_rewards: API grading not yet implemented. "
        f"Returning 0.0 KAV scores for {len(miner_uids)} miners."
    )

    rewards = np.zeros(len(miner_uids), dtype=float)
    detailed_metrics: List[Dict] = []

    for i, uid in enumerate(miner_uids):
        hotkey = self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else "unknown"
        submission = s3_submissions_by_miner.get(str(uid), {})
        submission_count = submission.get("submission_count", 0)

        detailed_metrics.append({
            "uid": uid,
            "miner_hotkey": hotkey,
            "kav_score": 0.0,
            "submission_count": submission_count,
            "api_graded": False,  # Will be True once the API is implemented
            "note": "TODO: API grading not yet implemented",
        })

    if not skip_burn:
        # Append burn UID 59 with its allocated reward weight
        # (burn_fraction is applied inside apply_reputation_rewards when skip_burn=True;
        #  here we use a default burn fraction of 0.65 for the KAV-only path)
        burn_fraction = getattr(self.config.neuron, 'burn_fraction', 0.65)
        rewards = np.append(rewards, burn_fraction)
        uids = np.append(np.array(miner_uids), BURN_UID)
    else:
        bt.logging.info("100% burn occurred (no miners qualified), skipping configured burn application")
        uids = np.array(miner_uids)

    return rewards, uids, detailed_metrics


# =============================================================================
# Reputation-Weighted Reward System (Phase 4 - Cycle 4)
#
# Configurable weights (via --neuron.kav_weight, --neuron.uav_weight):
#   - Passed to apply_reputation_rewards() from forward.py using getattr()
#
# Policy-based constants (hardcoded, rarely change):
#   - TIER_MULTIPLIERS, BURN_UID
#   - Normalization uses continuous function based on actual rep_score (no tier-based clamping)
#
# New Miner Behavior:
#   - Miners not in rep_data (snapshot) get ZERO UAV rewards
#   - They only receive KAV (online quality) rewards
#   - Must contribute validated UAVs to build reputation first
# =============================================================================

# Tier multipliers for reputation weighting (policy-based, rarely change)
TIER_MULTIPLIERS = {
    "Platinum": 1.25,
    "Diamond":  1.15,
    "Gold":     1.10,
    "Silver":   1.05,
    "Bronze":   1.02,
    "Neutral":  1.00,
    "Watch":    0.90,
}

# Normalization ranges (for reference - actual normalization uses continuous function)
# rep_score decays over time; normalization is based on actual score, not tier label
# Score ranges and their normalized output:
#   0.0  - 0.1   → 0.00 - 0.50  (decayed/Watch)
#   0.1  - 5.0   → 0.50 - 1.00  (Neutral)
#   5.0  - 15.0  → 1.00 - 1.20  (Bronze)
#   15.0 - 30.0  → 1.20 - 1.50  (Silver)
#   30.0 - 50.0  → 1.50 - 1.80  (Gold)
#   50.0 - 200.0 → 1.80 - 2.00  (Diamond)
#   200.0+       → 2.00 - 3.00  (Platinum)

# Burn UID (hardcoded in existing codebase)
BURN_UID = 59


def normalize_rep_score(rep_score: float, rep_tier: Optional[str] = None) -> float:
    """
    Normalize rep_score to reward-friendly range (0.0 - 3.0).

    Uses actual rep_score value for normalization - NO clamping to tier boundaries.
    As rep_score decays, normalized value decreases proportionally.
    When rep_score reaches 0, normalized value is 0 (zero UAV).

    Args:
        rep_score: Raw reputation score from DB (can decay from any value to 0)
        rep_tier: Tier string (unused for normalization, kept for API compatibility)

    Returns:
        Normalized rep_score in [0.0, 3.0].
    """
    # Zero or negative score = zero normalized (zero UAV)
    if rep_score <= 0:
        return 0.0

    # Continuous normalization based on actual rep_score (no tier-based clamping)
    # Score ranges map to normalized ranges:
    #   0.0  - 0.1   → 0.00 - 0.50  (Watch range, but use actual score)
    #   0.1  - 5.0   → 0.50 - 1.00  (Neutral range)
    #   5.0  - 15.0  → 1.00 - 1.20  (Bronze range)
    #   15.0 - 30.0  → 1.20 - 1.50  (Silver range)
    #   30.0 - 50.0  → 1.50 - 1.80  (Gold range)
    #   50.0 - 200.0 → 1.80 - 2.00  (Diamond range)
    #   200.0+       → 2.00 - 3.00  (Platinum range)

    if rep_score < 0.1:
        # Below Watch floor: linear from 0→0.1 maps to 0→0.50
        normalized = rep_score * (0.50 / 0.1)
    elif rep_score <= 5.0:
        # Neutral range: 0.1→5.0 maps to 0.50→1.00
        normalized = 0.50 + (rep_score - 0.1) * (1.00 - 0.50) / (5.0 - 0.1)
    elif rep_score <= 15.0:
        # Bronze range: 5.0→15.0 maps to 1.00→1.20
        normalized = 1.00 + (rep_score - 5.0) * (1.20 - 1.00) / (15.0 - 5.0)
    elif rep_score <= 30.0:
        # Silver range: 15.0→30.0 maps to 1.20→1.50
        normalized = 1.20 + (rep_score - 15.0) * (1.50 - 1.20) / (30.0 - 15.0)
    elif rep_score <= 50.0:
        # Gold range: 30.0→50.0 maps to 1.50→1.80
        normalized = 1.50 + (rep_score - 30.0) * (1.80 - 1.50) / (50.0 - 30.0)
    elif rep_score <= 200.0:
        # Diamond range: 50.0→200.0 maps to 1.80→2.00
        normalized = 1.80 + (rep_score - 50.0) * (2.00 - 1.80) / (200.0 - 50.0)
    else:
        # Platinum range: 200.0→9999.0 maps to 2.00→3.00
        normalized = 2.00 + (rep_score - 200.0) * (3.00 - 2.00) / (9999.0 - 200.0)
        normalized = min(3.00, normalized)  # Cap at 3.00

    return round(normalized, 3)


def apply_reputation_rewards(
    kav_rewards: np.ndarray,
    uids: List[int],
    rep_data: Dict[str, Dict],
    metagraph,
    burn_fraction: float = 0.65,
    kav_weight: float = 0.10,
    uav_weight: float = 0.90,
    kav_metrics: List[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Apply reputation weighting to KAV rewards, combine with UAV, and apply burn.

    Pipeline:
    1. Calculate KAV portions (kav_weight × Q) and UAV portions (uav_weight × R×T)
       separately for each miner.
    2. Rescale KAV and UAV portions independently so they hit exact target totals:
         KAV target = keep_fraction × kav_weight   (e.g. 0.35 × 0.10 = 0.035)
         UAV target = keep_fraction × uav_weight   (e.g. 0.35 × 0.90 = 0.315)
    3. Combine rescaled portions per miner and append burn UID 59.

    This ensures that after burn, exactly kav_weight% of kept rewards go to KAV
    and uav_weight% go to UAV, regardless of the raw score distributions.

    Args:
        kav_rewards: KAV quality scores (Q) from get_name_variation_rewards()
        uids: List of miner UIDs
        rep_data: Dict mapping hotkey -> {rep_score, rep_tier} from Flask
        metagraph: Bittensor metagraph for hotkey lookup
        burn_fraction: Fraction to burn (default 0.65 for Cycle 2)
        kav_weight: Weight for KAV online quality (default 0.10 = 10%)
        uav_weight: Weight for UAV reputation-based (default 0.90 = 90%)
        kav_metrics: Optional detailed metrics from KAV calculation

    Returns:
        Tuple of:
            - final_rewards: np.ndarray including burn UID (ready for set_weights)
            - final_uids: np.ndarray including burn UID
            - combined_metrics: List of dicts with full breakdown per miner
    """
    # Separate arrays for KAV and UAV portions
    kav_portions = np.zeros(len(uids))
    uav_portions = np.zeros(len(uids))
    combined_metrics = []

    # --- Step 1: Calculate KAV and UAV portions separately ---
    new_miner_count = 0
    for i, uid in enumerate(uids):
        hotkey = metagraph.hotkeys[uid]
        Q = kav_rewards[i]  # KAV quality score

        # Get reputation from snapshot
        rep = rep_data.get(hotkey)

        # New miners (not in rep_data) get ZERO UAV rewards - KAV only
        # They must first contribute validated UAVs to build reputation
        if rep is None:
            new_miner_count += 1
            rep_score = 0.0
            rep_tier = "New"
            R_norm = 0.0
            T = 0.0
            uav_reward = 0.0
        else:
            rep_score = rep.get("rep_score", 1.0)
            rep_tier = rep.get("rep_tier", "Neutral")

            # If rep_score has decayed to 0 or below, they get ZERO UAV
            # Tier stays as-is (e.g., "Diamond") but no UAV rewards
            # They only get KAV portion based on online quality
            if rep_score <= 0:
                R_norm = 0.0
                T = 0.0
                uav_reward = 0.0
            else:
                # Normalize rep_score to reward-friendly range (0.0 - 3.0)
                # This prevents Diamond miners from dominating emissions
                R_norm = normalize_rep_score(rep_score, rep_tier)
                # Get tier multiplier
                T = TIER_MULTIPLIERS.get(rep_tier, 1.0)
                # UAV reward = R_norm × T (using NORMALIZED rep_score)
                uav_reward = R_norm * T

        # Calculate portions separately (before rescaling)
        kav_portion_raw = kav_weight * Q
        uav_portion_raw = uav_weight * uav_reward
        kav_portions[i] = kav_portion_raw
        uav_portions[i] = uav_portion_raw

        # Build metrics - merge KAV details with reputation metrics
        kav_info = kav_metrics[i] if kav_metrics and i < len(kav_metrics) else {}

        is_new_miner = rep is None
        metric_entry = {
            "uid": uid,
            "miner_hotkey": hotkey,
            "is_new_miner": is_new_miner,  # True if miner not in rep_data (zero UAV)
            # KAV details
            "quality_score": float(Q),
            "kav_portion_raw": float(kav_portion_raw),
            # UAV details (raw + normalized)
            "rep_score": float(rep_score),           # Raw score from policy (can decay to 0; 0 for new miners)
            "rep_score_normalized": float(R_norm),   # Normalized for rewards (0.0 - 3.0), 0 for new miners
            "rep_tier": rep_tier,                    # "New" for new miners
            "tier_multiplier": float(T),
            "uav_reward": float(uav_reward),
            "uav_portion_raw": float(uav_portion_raw),
        }

        # # Merge KAV detailed metrics (variations, similarity scores, etc.) if available
        # if kav_info:
        #     metric_entry["kav_details"] = kav_info

        combined_metrics.append(metric_entry)

    # --- Step 2: Calculate totals and rescale separately to guarantee proportions ---
    keep_fraction = 1.0 - burn_fraction
    total_kav = np.sum(kav_portions)
    total_uav = np.sum(uav_portions)

    # Determine burn mode and target totals based on what's available
    # Edge cases: burn unused portions
    # 1. No KAV and no UAV -> burn 100%
    # 2. No UAV -> burn = burn_fraction + (uav_weight * keep_fraction) = 0.65 + 0.315 = 0.965
    # 3. No KAV -> burn = burn_fraction + (kav_weight * keep_fraction) = 0.65 + 0.035 = 0.685
    # 4. Both present -> normal burn = burn_fraction = 0.65

    if total_kav == 0 and total_uav == 0:
        burn_mode = "100_percent"
        target_kav_total = 0.0
        target_uav_total = 0.0
        applied_burn = 1.0
    elif total_uav == 0:
        burn_mode = "no_uav"
        target_kav_total = keep_fraction * kav_weight  # e.g., 0.35 * 0.10 = 0.035
        target_uav_total = 0.0
        applied_burn = burn_fraction + (uav_weight * keep_fraction)  # 0.65 + 0.315 = 0.965
    elif total_kav == 0:
        burn_mode = "no_kav"
        target_kav_total = 0.0
        target_uav_total = keep_fraction * uav_weight  # e.g., 0.35 * 0.90 = 0.315
        applied_burn = burn_fraction + (kav_weight * keep_fraction)  # 0.65 + 0.035 = 0.685
    else:
        burn_mode = "configured"
        target_kav_total = keep_fraction * kav_weight  # e.g., 0.35 * 0.10 = 0.035
        target_uav_total = keep_fraction * uav_weight  # e.g., 0.35 * 0.90 = 0.315
        applied_burn = burn_fraction

    # Rescale KAV and UAV portions separately to their target totals
    if total_kav > 0 and target_kav_total > 0:
        final_kav_portions = kav_portions * (target_kav_total / total_kav)
    else:
        final_kav_portions = np.zeros_like(kav_portions)

    if total_uav > 0 and target_uav_total > 0:
        final_uav_portions = uav_portions * (target_uav_total / total_uav)
    else:
        final_uav_portions = np.zeros_like(uav_portions)

    # Combine rescaled portions per miner
    final_rewards = final_kav_portions + final_uav_portions

    # Update metrics with final (post-burn) values
    for i, metric in enumerate(combined_metrics):
        final_kav = float(final_kav_portions[i])
        final_uav = float(final_uav_portions[i])
        combined = float(final_rewards[i])
        
        metric["kav_portion"] = final_kav
        metric["uav_portion"] = final_uav
        metric["combined_reward"] = combined
        # final_reward will be set after burn UID is added to ensure it matches final_rewards array
        
        # Calculate contributions (avoid division by zero)
        kav_contribution = final_kav / combined if combined > 0 else 0
        uav_contribution = final_uav / combined if combined > 0 else 0
        metric["kav_contribution"] = float(kav_contribution)
        metric["uav_contribution"] = float(uav_contribution)
        
        # Burn info (set after burn_mode is determined)
        metric["burn_fraction"] = float(burn_fraction)
        metric["burn_fraction_applied"] = float(applied_burn)
        metric["burn_mode"] = burn_mode

    # --- Step 3: Add burn UID ---
    final_rewards = np.append(final_rewards, applied_burn)
    final_uids = np.append(np.array(uids), BURN_UID)

    # Update metrics with final_reward values from final_rewards array (after burn UID added)
    # This ensures metrics match exactly what's in the final_rewards array
    for i, metric in enumerate(combined_metrics):
        metric["final_reward"] = float(final_rewards[i])

    bt.logging.info(
        f"Applied reputation rewards for {len(uids)} miners ({new_miner_count} new, KAV-only). "
        f"KAV: {kav_weight}, UAV: {uav_weight}, Burn: {burn_fraction}. "
        f"Burn UID {BURN_UID}: {applied_burn:.2%} (mode={burn_mode})"
    )

    return final_rewards, final_uids, combined_metrics
