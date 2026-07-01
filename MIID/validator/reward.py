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
  - Computed by calling an external grading API (GRADING_API_URL /grade) that
    evaluates submitted face image variations for quality, identity
    preservation, and variation compliance.
  - The API payload mirrors the format used by validator_api_test.py:
      { "signature": <signed message>, "phase4_image_data": { ... } }
  - Miners are ranked by avg validation score (primary) and avg identity
    preservation (tiebreaker); top 50 with avg_identity_preservation >= 0.6
    receive an exponential-decay blended reward.

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
import time
from typing import List, Dict, Tuple, Any, Optional
import bittensor as bt
import requests
from datetime import datetime
from MIID.utils.sign_message import sign_message
from MIID.validator.image_variations import format_variation_requirements


# =============================================================================
# KAV: Image Variation Grading (via external API)
# =============================================================================

# External grading API endpoint — grades submitted face image variations.
# Update this URL to match the current grading server before deployment.
GRADING_API_URL = "http://98.90.28.118:5000/grade"
GRADING_RETRY_ATTEMPTS = 3
GRADING_RETRY_DELAY_SECONDS = 60

# Tiny epsilon added to validation score to break ties using identity_preservation.
# e.g. two miners both averaging 0.6 validation will be ordered by their avg_ip.
_IDENTITY_TIEBREAK_EPS = 1e-5


def _apply_blended_rank_cap_with_quality(
    rewards: np.ndarray,
    detailed_metrics: List[Dict],
    uids: np.ndarray,
    top_miner_cap: int,
    identity_threshold: float,
    decay_rate: float,
    blend_factor: float,
    burn_uid: int,
    skip_burn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], bool]:
    """
    Applies the same blended exponential-decay ranking used by the old
    name-variation system, adapted for image variation scoring.

    Pipeline:
    1. Rank all miners by their composite reward (avg_validation_score +
       tiny identity tiebreak).  Rank 0 = best.
    2. Keep only the top ``top_miner_cap`` miners.
    3. Within that group filter by ``identity_threshold``
       (avg_identity_preservation >= threshold).
    4. Re-rank the qualified subset and compute:
           ranked_component  = exp(-decay_rate * rank)
           blended_reward    = blend_factor * ranked_component
                               + (1 - blend_factor) * original_score
    5. Miners who do not qualify receive a reward of 0.
    6. If nobody qualifies trigger a burn event.

    Returns:
        (final_rewards, uids, detailed_metrics, is_100_percent_burn)
    """
    uids = np.array(uids)

    # Identity preservation scores for the quality gate
    identity_scores = np.array([
        m.get("avg_identity_preservation", 0.0) for m in detailed_metrics
    ])

    # Global ranks — 0 = highest reward
    global_ranks = (-rewards).argsort().argsort()

    within_cap_mask = global_ranks < top_miner_cap
    qualified_mask = within_cap_mask & (identity_scores >= identity_threshold)
    qualified_indices = np.where(qualified_mask)[0]

    final_rewards = np.zeros_like(rewards)

    # Annotate every miner's metrics with ranking diagnostics
    for i, metrics in enumerate(detailed_metrics):
        metrics["ranking_info"] = {
            "initial_reward": float(rewards[i]),
            "global_rank": int(global_ranks[i]),
            "within_top_cap": bool(within_cap_mask[i]),
            "avg_identity_preservation": float(identity_scores[i]),
            "meets_identity_threshold": bool(identity_scores[i] >= identity_threshold),
            "is_qualified_for_ranking": bool(qualified_mask[i]),
            "final_blended_reward": 0.0,
        }

    if len(qualified_indices) > 0:
        qualified_original_rewards = rewards[qualified_indices]

        # Re-rank only among qualified miners
        qualified_ranks = (-qualified_original_rewards).argsort().argsort()

        # Exponential decay: rank 0 → 1.0, rank N → exp(-decay_rate * N)
        ranked_component = np.exp(-decay_rate * qualified_ranks)

        # Blend rank-based reward with original score
        blended = (
            blend_factor * ranked_component
            + (1 - blend_factor) * qualified_original_rewards
        )

        final_rewards[qualified_indices] = blended

        for idx, qi in enumerate(qualified_indices):
            detailed_metrics[qi]["ranking_info"]["final_blended_reward"] = float(blended[idx])

        bt.logging.info(
            f"Applied blended ranking: {qualified_mask.sum()} / {len(rewards)} miners qualified "
            f"(top_cap={top_miner_cap}, identity_threshold={identity_threshold})."
        )
        return final_rewards, uids, detailed_metrics, False

    # ── No miners qualified → burn event ─────────────────────────────────────
    bt.logging.warning(
        "BURN EVENT: No miners passed top-cap + identity threshold filters. "
        "All emissions will be burned."
    )

    if skip_burn:
        # Defer burn to apply_reputation_rewards() when UAV grading is enabled
        bt.logging.warning(
            f"skip_burn=True — deferring 100% burn to reputation-weighted stage."
        )
        return final_rewards, uids, detailed_metrics, True

    extended_uids = np.append(uids, burn_uid)
    extended_rewards = np.append(final_rewards, 1.0)
    detailed_metrics.append({
        "uid": burn_uid,
        "is_burn": True,
        "burn_type": "100_percent",
        "ranking_info": {
            "initial_reward": 0.0,
            "global_rank": -1,
            "within_top_cap": False,
            "avg_identity_preservation": 0.0,
            "meets_identity_threshold": False,
            "is_qualified_for_ranking": False,
            "final_blended_reward": 1.0,
        },
    })
    return extended_rewards, extended_uids, detailed_metrics, True


def get_image_variation_rewards(
    self,
    challenge_id: str,
    s3_submissions_by_miner: Dict[str, Any],
    miner_uids: List[int],
    selected_variations: List[Dict],
    skip_burn: bool = False,
    phase4_image_data: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Compute KAV rewards for each miner based on their image variation submissions.

    Grading pipeline:
    1. Sign and POST the challenge data to the external grading API
       (GRADING_API_URL).  Payload format mirrors validator_api_test.py:
           { "signature": <signed_message>, "phase4_image_data": { ... } }
       The API returns per-miner, per-image scores inside "results_by_miner".
    2. For each miner:
       a. Determine how many variation types were requested.
       b. Match each requested type to the miner's submitted images.
          Missing submissions are treated as score=0 / identity_preservation=0.
       c. Compute:
            avg_validation_score      = mean(validation_score / 5.0)  [0 – 1]
            avg_identity_preservation = mean(identity_preservation)   [0 – 1]
       d. Composite ranking score = avg_validation_score
          + avg_identity_preservation × _IDENTITY_TIEBREAK_EPS
          (primary: validation score; tiebreaker: identity preservation)
    3. Apply blended exponential-decay ranking (same curve as old name-variation
       system):
       • Only top ``top_miner_cap`` (default 50) miners are considered.
       • Among those, only miners with avg_identity_preservation ≥ 0.6
         (``quality_threshold``) are eligible for a reward.
       • Eligible miners are re-ranked; reward = blend of exp(-decay*rank)
         and original score.
    4. Apply burn:
       • skip_burn=True  → no burn appended (burn applied later in
         apply_reputation_rewards when UAV grading is enabled).
       • skip_burn=False → miner rewards are rescaled to keep_fraction and
         burn UID 59 is appended with weight burn_fraction.

    Args:
        self:                    Validator instance (metagraph, config, wallet).
        challenge_id:            Unique ID for the current challenge round.
        s3_submissions_by_miner: Dict mapping str(uid) → submission metadata.
        miner_uids:              Ordered list of miner UIDs to score.
        selected_variations:     List of requested variation specs
                                 (each dict has at minimum a "type" key).
        skip_burn:               When True, burn UID is NOT appended.
        phase4_image_data:       Full phase4 challenge dict (built in forward.py)
                                 that is sent as the API payload body.  When
                                 omitted a minimal dict is constructed from the
                                 other arguments.

    Returns:
        Tuple of:
            rewards          – np.ndarray, one entry per miner (+ burn UID
                               if skip_burn=False).
            uids             – np.ndarray of corresponding UIDs.
            detailed_metrics – List of per-miner metric dicts for logging.
    """
    # ── 1. Call external grading API ─────────────────────────────────────────
    results_by_miner: Dict[str, Any] = {}
    api_succeeded = False

    if not s3_submissions_by_miner:
        bt.logging.info(
            f"No valid submissions for challenge {challenge_id}. "
            "Skipping grading API call — all miners will receive 0.0 KAV scores."
        )
    else:
        try:
            # Build phase4_image_data from individual args when not provided by caller
            if phase4_image_data is None:
                phase4_image_data = {
                    "challenge_id": challenge_id,
                    "requested_variations": selected_variations,
                    "variation_count": len(selected_variations),
                    "variation_types": [v.get("type", "") for v in selected_variations],
                    "variation_intensities": [v.get("intensity", "") for v in selected_variations],
                    "s3_submissions_by_miner": s3_submissions_by_miner,
                    "cycle": None,
                }
            elif "s3_submissions_by_miner" not in phase4_image_data:
                # Ensure submissions are included even if caller omitted them
                phase4_image_data = {**phase4_image_data, "s3_submissions_by_miner": s3_submissions_by_miner}

            # Sign the payload — matches the exact signing format in validator_api_test.py
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            requested_variations = phase4_image_data.get("requested_variations", selected_variations)
            image_requirements = format_variation_requirements(requested_variations)
            query_template = (
                f"[PHASE4] challenge_id: {challenge_id}"
                f"{image_requirements}"
            )
            query_labels = {
                "variation_count": phase4_image_data.get("variation_count", len(requested_variations)),
                "variation_types": phase4_image_data.get("variation_types", [v.get("type", "") for v in requested_variations]),
                "variation_intensities": phase4_image_data.get("variation_intensities", [v.get("intensity", "") for v in requested_variations]),
                "challenge_id": challenge_id,
                "cycle": phase4_image_data.get("cycle"),
            }
            hotkey = self.wallet.hotkey
            message_to_sign = (
                f"Hotkey: {hotkey} \n timestamp: {timestamp} \n "
                f"query_template: {query_template} \n query_labels: {query_labels}"
            )
            signed_contents = sign_message(self.wallet, message_to_sign, output_file=None)

            payload = {
                "signature": signed_contents,
                "phase4_image_data": phase4_image_data,
            }

            for attempt in range(GRADING_RETRY_ATTEMPTS):
                try:
                    bt.logging.info(
                        f"Calling grading API at {GRADING_API_URL} "
                        f"for challenge {challenge_id} with {len(s3_submissions_by_miner)} miners "
                        f"(attempt {attempt + 1}/{GRADING_RETRY_ATTEMPTS})."
                    )
                    resp = requests.post(GRADING_API_URL, json=payload, timeout=1200)
                    resp.raise_for_status()
                    api_json = resp.json()
                    results_by_miner = api_json.get("results_by_miner", {})
                    api_succeeded = True
                    bt.logging.info(
                        f"Grading API returned results for {len(results_by_miner)} miners."
                    )
                    break
                except Exception as exc:
                    if attempt < GRADING_RETRY_ATTEMPTS - 1:
                        bt.logging.warning(
                            f"Grading API attempt {attempt + 1}/{GRADING_RETRY_ATTEMPTS} failed: {exc}. "
                            f"Retrying in {GRADING_RETRY_DELAY_SECONDS}s..."
                        )
                        time.sleep(GRADING_RETRY_DELAY_SECONDS)
                    else:
                        raise
        except Exception as exc:
            bt.logging.error(
                f"Grading API call failed: {exc}. "
                f"All miners will receive 0.0 KAV scores for this round."
            )

    # ── 2. Grade each miner ───────────────────────────────────────────────────
    num_requested = max(len(selected_variations), 1)
    requested_types = [v.get("type", "") for v in selected_variations]

    rewards = np.zeros(len(miner_uids), dtype=float)
    detailed_metrics: List[Dict] = []

    for i, uid in enumerate(miner_uids):
        hotkey = (
            self.metagraph.hotkeys[uid]
            if uid < len(self.metagraph.hotkeys)
            else "unknown"
        )

        miner_result = results_by_miner.get(str(uid), {})
        raw_submissions = miner_result.get("submissions", [])

        # Build type → first-matching submission lookup
        submission_by_type: Dict[str, Dict] = {}
        for sub in raw_submissions:
            vtype = sub.get("variation_type", "")
            if vtype and vtype not in submission_by_type:
                submission_by_type[vtype] = sub

        # Score each requested slot; missing slots count as 0
        per_variation: List[Dict] = []
        for vtype in requested_types:
            sub = submission_by_type.get(vtype)
            if sub is None:
                per_variation.append({
                    "variation_type": vtype,
                    "submitted": False,
                    "validation_score_raw": 0,
                    "validation_score_norm": 0.0,
                    "identity_preservation": 0.0,
                })
            else:
                vs_raw = int(sub.get("validation_score", 0))
                ip = float(sub.get("identity_preservation", 0.0))
                per_variation.append({
                    "variation_type": vtype,
                    "submitted": True,
                    "validation_score_raw": vs_raw,
                    "validation_score_norm": vs_raw / 5.0,
                    "identity_preservation": ip,
                    "comment": sub.get("comment", ""),
                    "status": sub.get("status", ""),
                })

        # Averages over all requested slots (missing slots penalize average)
        avg_vs_norm = sum(v["validation_score_norm"] for v in per_variation) / num_requested
        avg_ip = sum(v["identity_preservation"] for v in per_variation) / num_requested

        # Composite score: validation score + tiny identity tiebreak
        rewards[i] = avg_vs_norm + avg_ip * _IDENTITY_TIEBREAK_EPS

        detailed_metrics.append({
            "uid": uid,
            "miner_hotkey": hotkey,
            "api_graded": api_succeeded and bool(miner_result),
            "num_requested": num_requested,
            "num_submitted": sum(1 for v in per_variation if v["submitted"]),
            "per_variation_scores": per_variation,
            "avg_validation_score": float(avg_vs_norm),
            "avg_identity_preservation": float(avg_ip),
            # final_reward kept as avg_vs_norm for downstream logging clarity
            "final_reward": float(avg_vs_norm),
        })

    # ── 3. Blended ranking with identity preservation threshold ───────────────
    burn_fraction = getattr(self.config.neuron, "burn_fraction", 0.65)
    top_miner_cap = getattr(self.config.neuron, "top_miner_cap", 50)
    # quality_threshold repurposed as identity_preservation minimum (default 0.6)
    identity_threshold = getattr(self.config.neuron, "quality_threshold", 0.6)
    decay_rate = getattr(self.config.neuron, "decay_rate", 0.05)
    blend_factor = getattr(self.config.neuron, "blend_factor", 0.7)

    rewards, uids_array, detailed_metrics, is_100_percent_burn = (
        _apply_blended_rank_cap_with_quality(
            rewards=rewards,
            detailed_metrics=detailed_metrics,
            uids=miner_uids,
            top_miner_cap=top_miner_cap,
            identity_threshold=identity_threshold,
            decay_rate=decay_rate,
            blend_factor=blend_factor,
            burn_uid=BURN_UID,
            skip_burn=skip_burn,
        )
    )

    # ── 4. Burn handling ──────────────────────────────────────────────────────
    if skip_burn:
        bt.logging.info(
            "skip_burn=True — burn will be applied in apply_reputation_rewards()."
        )
        return rewards, uids_array, detailed_metrics

    if not is_100_percent_burn:
        if BURN_UID not in uids_array:
            keep_fraction = 1.0 - burn_fraction
            total_reward_sum = float(np.sum(rewards))

            if total_reward_sum > 0:
                rewards = rewards * (keep_fraction / total_reward_sum)

            uids_array = np.append(uids_array, BURN_UID)
            rewards = np.append(rewards, burn_fraction)

            detailed_metrics.append({
                "uid": BURN_UID,
                "is_burn": True,
                "burn_type": f"{int(burn_fraction * 100)}_percent",
                "ranking_info": {
                    "initial_reward": 0.0,
                    "global_rank": -1,
                    "within_top_cap": False,
                    "avg_identity_preservation": 0.0,
                    "meets_identity_threshold": False,
                    "is_qualified_for_ranking": False,
                    "final_blended_reward": burn_fraction,
                },
            })

            bt.logging.info(
                f"Applied {burn_fraction * 100:.1f}% emission burn: "
                f"UID {BURN_UID} gets {burn_fraction:.2%}, "
                f"miners share {keep_fraction:.2%}."
            )
        else:
            bt.logging.warning(
                f"Burn UID {BURN_UID} already present — skipping burn application."
            )
    else:
        bt.logging.info("100% burn occurred (no miners qualified).")

    if len(rewards) != len(uids_array):
        raise ValueError(
            f"Length mismatch after burn: rewards={len(rewards)}, uids={len(uids_array)}"
        )

    bt.logging.info(
        f"get_image_variation_rewards complete: {len(rewards)} entries "
        f"(including burn UID). api_succeeded={api_succeeded}."
    )
    return rewards, uids_array, detailed_metrics


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
