# forward.py

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
Validator Forward Module

Implements the forward function that drives each validation round:
1. Select random miners to query.
2. Fetch a base face image from the MIID API.
3. Build an ImageRequest (6 variations: 2 background, lighting, expression, pose, screen-replay).
4. Send the request to miners in batches; collect S3 submission references.
5. Grade submissions via the external grading API (KAV) using
   get_image_variation_rewards().
6. Combine KAV scores with reputation (UAV) via apply_reputation_rewards().
7. Update miner weights and upload results to the MIID server.
"""

import time
import bittensor as bt
import json
import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

from MIID.protocol import IdentitySynapse, ImageRequest, VariationRequest
from MIID.validator.reward import get_image_variation_rewards, apply_reputation_rewards
from MIID.utils.uids import get_random_uids
from MIID.utils.sign_message import sign_message

from MIID.validator.base_images import fetch_image_from_api
from MIID.validator.drand_utils import (
    calculate_target_round,
    calculate_reveal_buffer,
    REVEAL_DELAY_SECONDS,
)
from MIID.validator.image_variations import (
    format_variation_requirements,
    get_random_indoor_background_variation,
    get_random_outdoor_background_variation,
    get_random_variation_by_type,
    select_screen_replay_variation,
)


# Import your new upload_data function here
from MIID.utils.misc import upload_data

# =============================================================================
# Reputation Cache (persists across forward passes)
# =============================================================================

# Module-level cache for reputation data (persists across forward passes)
_cached_rep_data: Dict[str, Dict] = {}
_cached_rep_version: Optional[str] = None

# Module-level pending queue for failed uploads (persists across forward passes)
_pending_allocations: List[Dict] = []
_pending_file_path: Optional[Path] = None

# =============================================================================
# Phase 4: Image Cycling State
# =============================================================================
# Tracks the current position in the image cycle only.
# Global index advances by 1 per forward pass and selects `image_index`.

_phase4_global_index: int = 0
_phase4_state_file_path: Optional[Path] = None


def _load_phase4_state(file_path: Path) -> int:
    """Load the Phase 4 global index from disk on startup."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get("global_index", 0)
        except Exception:
            return 0
    return 0


def _save_phase4_state(file_path: Path, global_index: int):
    """Save the Phase 4 global index to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump({"global_index": global_index}, f)


def reset_phase4_state(file_path: Path) -> None:
    """Reset Phase 4 cycle state to 0 so the next forward starts with image index 0."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump({"global_index": 0}, f)
    bt.logging.info("Phase 4: Reset state to global_index=0 (cycle starts with background variation)")


def _load_pending_allocations(file_path: Path) -> List[Dict]:
    """Load pending allocations from disk on startup."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_pending_allocations(file_path: Path, pending: List[Dict]):
    """Save pending allocations to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(pending, f, indent=2)


def _clear_pending_allocations(file_path: Path):
    """Clear pending after successful send (overwrite with empty array)."""
    _save_pending_allocations(file_path, [])

# =============================================================================

EPOCH_MIN_TIME = 360  # seconds
MIID_SERVER = "http://52.44.186.20:5000/upload_data"

PHASE4_ENABLED = True


async def dendrite_with_retries(
    dendrite: bt.Dendrite,
    axons: list,
    synapse: IdentitySynapse,
    deserialize: bool,
    timeout: float,
    cnt_attempts: int = 3,
):
    """
    Send requests to miners with automatic retry logic for failed connections.

    Args:
        dendrite:     The dendrite object to use for communication.
        axons:        List of axons to query.
        synapse:      The synapse object containing the request.
        deserialize:  Whether to deserialize the response.
        timeout:      Timeout per attempt in seconds.
        cnt_attempts: Number of retry attempts for failed connections.

    Returns:
        List of IdentitySynapse responses from miners.
    """
    res = [None] * len(axons)
    idx = list(range(len(axons)))
    axons_for_retry = axons.copy()
    
    def create_default_response():
        return IdentitySynapse(
            image_request=synapse.image_request,
            s3_submissions=[],
            process_time=None,
        )
    
    for attempt in range(cnt_attempts):
        responses = await dendrite(
            axons=axons_for_retry,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout * (1 + attempt * 0.1),
        )
        
        new_idx = []
        new_axons = []
        
        for i, response in enumerate(responses):
            process_time = None
            if hasattr(response, "dendrite") and hasattr(response.dendrite, "process_time"):
                try:
                    process_time = float(response.dendrite.process_time)
                except (ValueError, TypeError):
                    process_time = None

            if hasattr(response, 'dendrite'):
                if (response.dendrite.status_code is not None
                        and int(response.dendrite.status_code) == 422):
                    if attempt == cnt_attempts - 1:
                        res[idx[i]] = response
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_for_retry[i])
                else:
                    response.process_time = process_time  # <-- attach it
                    res[idx[i]] = response
            
            else:
                if hasattr(response, 's3_submissions'):
                    response.process_time = process_time
                    res[idx[i]] = response
                else:
                    # Retry or assign default
                    if attempt == cnt_attempts - 1:
                        res[idx[i]] = create_default_response()
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_for_retry[i])
        
        if len(new_idx) <= 50:  # Only retry if more than 50 miners failed
            bt.logging.info(f"Only {len(new_idx)} miners failed (≤50 threshold). Giving them default responses instead of retrying.")
            # Give default responses to failed miners
            for i in new_idx:
                res[i] = create_default_response()
            break
        
        idx = new_idx
        axons_for_retry = new_axons
        await asyncio.sleep(5 * (attempt + 1))
    
    # Fill any remaining None
    for i, r in enumerate(res):
        if r is None:
            res[i] = create_default_response()
    
    return res


async def forward(self):
    """
    Validator forward pass — one full validation round.

    Steps:
    1.  Select random miners.
    2.  Fetch base face image from the MIID API.
    3.  Build ImageRequest (6 variations: indoor bg, outdoor bg, lighting, expression, pose, screen-replay).
    4.  Query miners in batches; collect S3 submission references.
    5.  Compute KAV rewards via get_image_variation_rewards() (calls grading API).
    6.  Optionally combine with UAV via apply_reputation_rewards().
    7.  Update scores and set weights.
    8.  Upload results to the MIID server.
    """

    # --- Wandb run setup ---
    wandb_disabled = (
        hasattr(self.config, 'wandb')
        and hasattr(self.config.wandb, 'disable')
        and self.config.wandb.disable
    )
    if not wandb_disabled:
        bt.logging.info("Creating new wandb run for this validation round")
        self.new_wandb_run()
    # --- END WANDB SETUP ---

    request_start = time.time()
    
    bt.logging.info("Updating and querying available uids")

    # 1) Get random UIDs to query
    available_axon_size = len(self.metagraph.axons) - 1  # Exclude self
    miner_selection_size = min(available_axon_size, self.config.neuron.sample_size)
    miner_uids = get_random_uids(self, k=miner_selection_size)

    bt.logging.debug(f"⚙️ Miner selection size: {miner_selection_size}")
    bt.logging.debug(f"📋 Available axon size: {available_axon_size}")

    miner_uids = miner_uids.tolist()
    bt.logging.info(f"Selected {len(miner_uids)} miners to query: {miner_uids}")

    request_timeout = self.config.neuron.timeout
    bt.logging.info(f"Using request timeout of {request_timeout} seconds")

    # 3) Build ImageRequest
    image_request = None
    challenge_id = None
    selected_variations = None  # Track what variations were requested

    if PHASE4_ENABLED:
        try:
            # API is expected to return a single image.
            image_result = fetch_image_from_api(self.wallet)
            if image_result is None:
                bt.logging.warning("Phase 4 disabled: Could not fetch base image from API")
            else:
                image_filename, base64_image = image_result

                # Always request (6 variations):
                # 1–2) background_edit: indoor + outdoor
                # 3–5) lighting, expression, pose (one each)
                # 6) screen_replay (independent random devices + cues)
                indoor_background_var = get_random_indoor_background_variation()
                outdoor_background_var = get_random_outdoor_background_variation()
                lighting_var = get_random_variation_by_type("lighting_edit")
                expression_var = get_random_variation_by_type("expression_edit")
                pose_var = get_random_variation_by_type("pose_edit")
                screen_replay_var = select_screen_replay_variation()

                selected_variations = [
                    indoor_background_var,
                    outdoor_background_var,
                    lighting_var,
                    expression_var,
                    pose_var,
                    screen_replay_var,
                ]

                # Drand unlock at T+40 min (batch1 20m + batch2 20m); grading window T+40–60m
                reveal_delay = calculate_reveal_buffer(
                    getattr(self.config.neuron, "reveal_delay_seconds", REVEAL_DELAY_SECONDS)
                )
                target_round, reveal_timestamp = calculate_target_round(reveal_delay)
                bt.logging.info(
                    f"Phase 4: drand reveal in {reveal_delay}s ({reveal_delay / 60:.0f} min)"
                )

                # Generate unique challenge ID
                challenge_id = f"challenge_{int(time.time())}_{self.wallet.hotkey.ss58_address[:8]}"

                # Convert to VariationRequest objects
                variation_requests = [
                    VariationRequest(
                        type=v["type"],
                        intensity=v["intensity"],
                        description=v["description"],
                        detail=v["detail"],
                    )
                    for v in selected_variations
                ]

                # Create image request
                image_request = ImageRequest(
                    base_image=base64_image,
                    image_filename=image_filename,
                    variation_requests=variation_requests,
                    target_drand_round=target_round,
                    reveal_timestamp=reveal_timestamp,
                    challenge_id=challenge_id,
                )

                # Log what was selected
                screen_replay_devices = screen_replay_var.get("device_type", "unknown")
                screen_replay_cues = screen_replay_var.get("visual_cue_keys", [])
                bt.logging.info(
                    f"Phase 4: API image + random variation selection - "
                    f"Image '{image_filename}', "
                    f"background_edit indoor={indoor_background_var['intensity']}, "
                    f"outdoor={outdoor_background_var['intensity']}, "
                    f"lighting={lighting_var['intensity']}, "
                    f"expression={expression_var['intensity']}, "
                    f"pose={pose_var['intensity']}, "
                    f"screen_replay: devices={screen_replay_devices}, cues={screen_replay_cues}, "
                    f"Total requested: {len(selected_variations)}, "
                    f"drand round {target_round}"
                )

        except Exception as e:
            bt.logging.warning(f"Phase 4: Could not create image request: {e}")
            import traceback
            bt.logging.debug(f"Phase 4 error traceback: {traceback.format_exc()}")
            image_request = None
            selected_variations = None

    # 4) Prepare synapse
    request_synapse = IdentitySynapse(
        timeout=request_timeout,
        image_request=image_request,
    )
    bt.logging.info(f"Querying {len(miner_uids)} miners with image variation request")
    await asyncio.sleep(3)

    # 5) Query miners in batches
    start_time = time.time()
    uid_response_map: Dict[int, IdentitySynapse] = {}
    batch_size = self.config.neuron.batch_size
    total_batches = (len(miner_uids) + batch_size - 1) // batch_size
    
    for i in range(0, len(miner_uids), batch_size):
        batch_uids = miner_uids[i:i+batch_size]
        batch_axons = [self.metagraph.axons[uid] for uid in batch_uids]
        
        bt.logging.debug(f"🔄 Batch uids: {batch_uids}")
        await asyncio.sleep(3)  # Large sleep; adjust as desired

        bt.logging.info(f"Processing batch {i//batch_size + 1}/{total_batches} with {len(batch_uids)} miners")
        batch_start_time = time.time()
        
        batch_responses = await dendrite_with_retries(
            dendrite=self.dendrite,
            axons=batch_axons,
            synapse=request_synapse,
            deserialize=False,
            timeout=request_timeout,
            cnt_attempts=3,
        )
        
        batch_duration = time.time() - batch_start_time
        bt.logging.info(f"Batch {i//batch_size + 1} completed in {batch_duration:.1f}s")

        # Map each response to its corresponding UID
        for idx_resp, response in enumerate(batch_responses):
            uid = batch_uids[idx_resp]
            uid_response_map[uid] = response

            if not hasattr(response, 's3_submissions'):
                bt.logging.warning(f"Miner {uid}: response missing 's3_submissions'.")
            elif response.s3_submissions is None:
                bt.logging.warning(f"Miner {uid}: s3_submissions is None.")
            elif not response.s3_submissions:
                bt.logging.warning(f"Miner {uid}: returned empty s3_submissions.")
            else:
                bt.logging.info(f"Miner {uid}: returned {len(response.s3_submissions)} S3 submissions.")

        if i + batch_size < len(miner_uids):
            sleep_time = 2
            bt.logging.info(f"Sleeping for {sleep_time}s before next batch")
            await asyncio.sleep(sleep_time)
    
    end_time = time.time()
    bt.logging.info(f"Query completed in {end_time - start_time:.2f} seconds")

    all_responses = [uid_response_map[uid] for uid in miner_uids]

    # Log valid vs invalid responses
    valid_count = sum(
        1 for r in all_responses
        if hasattr(r, 's3_submissions') and r.s3_submissions
    )
    bt.logging.info(f"Received {valid_count} valid responses out of {len(all_responses)}")

    # Build s3_submissions_by_miner for grading
    s3_submissions_by_miner: Dict[str, Any] = {}
    for uid, response in uid_response_map.items():
        if PHASE4_ENABLED and hasattr(response, 's3_submissions') and response.s3_submissions:
            miner_hotkey = str(self.metagraph.axons[uid].hotkey)
            s3_data = []
            for sub in response.s3_submissions:
                s3_data.append({
                    "s3_key":       sub.s3_key,
                    "image_hash":   sub.image_hash,
                    "signature":    sub.signature,
                    "variation_type": sub.variation_type,
                    "path_signature": sub.path_signature,
                })
            s3_submissions_by_miner[str(uid)] = {
                "hotkey":           miner_hotkey,
                "submissions":      s3_data,
                "submission_count": len(s3_data),
            }

    # Build phase4_image_data for the grading API — mirrors the structure used
    # in validator_api_test.py so the signing and payload format match exactly.
    phase4_image_data: Optional[Dict] = None
    if PHASE4_ENABLED and image_request is not None and selected_variations:
        phase4_image_data = {
            "cycle": "Phase4-C4-Sandbox",
            "challenge_id": challenge_id,
            "base_image_filename": image_request.image_filename,
            "target_drand_round": image_request.target_drand_round,
            "reveal_timestamp": image_request.reveal_timestamp,
            "requested_variations": selected_variations,
            "variation_count": len(selected_variations),
            "variation_types": [v["type"] for v in selected_variations],
            "variation_intensities": [v["intensity"] for v in selected_variations],
            "s3_submissions_by_miner": s3_submissions_by_miner,
        }

    # 6) Compute rewards (KAV + optional UAV)
    uav_grading_enabled = getattr(self.config.neuron, 'UAV_grading', False)

    if uav_grading_enabled:
        # Get KAV rewards WITHOUT burn — burn applied after KAV+UAV in apply_reputation_rewards()
        kav_rewards, kav_uids, detailed_metrics = get_image_variation_rewards(
            self,
            challenge_id=challenge_id or "",
            s3_submissions_by_miner=s3_submissions_by_miner,
            miner_uids=miner_uids,
            selected_variations=selected_variations or [],
            skip_burn=True,
            phase4_image_data=phase4_image_data,
        )

        global _cached_rep_data, _cached_rep_version, _pending_allocations, _pending_file_path

        # Initialize pending file path (once per validator session)
        if _pending_file_path is None:
            _pending_file_path = (
                Path(self.config.logging.logging_dir)
                / "validator_results"
                / "pending_allocations.json"
            )
            _pending_allocations = _load_pending_allocations(_pending_file_path)
            if _pending_allocations:
                bt.logging.info(f"Loaded {len(_pending_allocations)} pending allocations from disk")

        # With skip_burn=True, kav_rewards/kav_uids contain only miners (no burn UID)
        # Convert to list for apply_reputation_rewards
        miner_uids_list = kav_uids.tolist() if hasattr(kav_uids, 'tolist') else list(kav_uids)

        # Get config values (Phase 4 - Cycle 1)
        burn_fraction = getattr(self.config.neuron, 'burn_fraction', 0.65)
        kav_weight = getattr(self.config.neuron, 'kav_weight', 0.10)
        uav_weight = getattr(self.config.neuron, 'uav_weight', 0.90)

        # Apply reputation weighting (UAV + combine + burn in one call)
        # Burn is applied ONCE here after KAV + UAV are combined
        rewards, updated_uids, combined_metrics = apply_reputation_rewards(
            kav_rewards=kav_rewards,  # Raw KAV quality scores (no burn applied yet)
            uids=miner_uids_list,
            rep_data=_cached_rep_data,  # From previous forward pass (may be empty on first run)
            metagraph=self.metagraph,
            burn_fraction=burn_fraction,
            kav_weight=kav_weight,
            uav_weight=uav_weight,
            kav_metrics=detailed_metrics,
        )

        bt.logging.info(
            f"Applied reputation rewards: {len(miner_uids_list)} miners, "
            f"using rep_snapshot_version={_cached_rep_version or 'None (first run)'}"
        )
    else:
        # UAV grading disabled: Use KAV-only scoring with burn applied directly
        bt.logging.info("UAV_grading disabled. Using KAV-only scoring with burn applied.")
        rewards, updated_uids, detailed_metrics = get_image_variation_rewards(
            self,
            challenge_id=challenge_id or "",
            s3_submissions_by_miner=s3_submissions_by_miner,
            miner_uids=miner_uids,
            selected_variations=selected_variations or [],
            skip_burn=False,
            phase4_image_data=phase4_image_data,
        )
        combined_metrics = detailed_metrics

    # Verify UID-reward mapping before updating scores
    bt.logging.info("=== UID-REWARD MAPPING VERIFICATION ===")
    for i, uid in enumerate(updated_uids):
        reward = rewards[i] if i < len(rewards) else 0.0
        if uid == 59:
            bt.logging.info(f"UID {uid}: IS BURNED EVENT. NO REWARD OR RESPONSE.")
        else:
            has_response = uid_response_map.get(uid) is not None
            bt.logging.info(f"UID {uid}: Reward={reward:.4f}, HasResponse={has_response}")
    bt.logging.info("=== END UID-REWARD MAPPING VERIFICATION ===")

    self.update_scores(rewards, updated_uids)
    bt.logging.info(f"REWARDS: {rewards}  for UIDs: {updated_uids}")

    # 7) Save results locally
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(self.config.logging.logging_dir, "validator_results")
    os.makedirs(results_dir, exist_ok=True)
    
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "phase4_image_data": {
            **(phase4_image_data or {}),
            "note": "Sandbox image variations with S3 uploads for YANEZ Sandbox testing",
            "enabled": PHASE4_ENABLED and image_request is not None,
            "s3_bucket": "yanez-miid-sn54",
            # Fallbacks for when image_request was unavailable (phase4_image_data is None)
            "challenge_id": challenge_id,
            "base_image_filename": image_request.image_filename if image_request else None,
            "target_drand_round":  image_request.target_drand_round if image_request else None,
            "reveal_timestamp":    image_request.reveal_timestamp if image_request else None,
            "requested_variations": selected_variations or [],
            "variation_count":  len(selected_variations) if selected_variations else 0,
            "variation_types":  [v["type"] for v in selected_variations] if selected_variations else [],
            "variation_intensities": [v["intensity"] for v in selected_variations] if selected_variations else [],
            "s3_submissions_by_miner": s3_submissions_by_miner,
        },
        "responses": {},
        "rewards": {},
    }

    for i, uid in enumerate(miner_uids):
        # Get the response for this specific UID from our mapping
        response = uid_response_map.get(uid)
        axon = self.metagraph.axons[uid]
        axon_data = {
            "ip":         str(axon.ip)       if hasattr(axon, 'ip')         else None,
            "port":       int(axon.port)     if hasattr(axon, 'port')       else None,
            "hotkey":     str(axon.hotkey),
            "coldkey":    str(axon.coldkey)  if hasattr(axon, 'coldkey')    else None,
            "version":    int(axon.version)  if hasattr(axon, 'version')    else None,
            "protocol":   int(axon.protocol) if hasattr(axon, 'protocol')   else None,
            "is_serving": bool(axon.is_serving) if hasattr(axon, 'is_serving') else None,
        }
        response_data = {
            "uid":            int(uid),
            "hotkey":         str(self.metagraph.axons[uid].hotkey),
            "axon":           axon_data,
            "response_time":  response.process_time if response else None,
            "s3_submissions": s3_submissions_by_miner.get(str(uid), {}),
            "scoring_details": detailed_metrics[i] if i < len(detailed_metrics) else {},
        }
        if response is None:
            response_data["error"] = {"message": "No response received"}
        elif not (hasattr(response, 's3_submissions') and response.s3_submissions):
            if hasattr(response, 'dendrite') and hasattr(response.dendrite, 'status_code'):
                response_data["error"] = {
                    "status_code":    response.dendrite.status_code,
                    "status_message": getattr(response.dendrite, 'status_message', 'Unknown error'),
                }
            else:
                response_data["error"] = {"message": "Empty or invalid response"}

        results["responses"][str(uid)] = response_data
        results["rewards"][str(uid)] = float(rewards[i]) if i < len(rewards) else 0.0
    
    # logging the spec_version before setting weights
    bt.logging.info(f"Spec version for setting weights: {self.spec_version}")
    (success, uint_uids, uint_weights) = self.set_weights()
    bt.logging.info(f"Weights set successfully: {success}")

    results["Weights"] = {
        "spec_version":   self.spec_version,
        "hotkey":         str(self.wallet.hotkey.ss58_address),
        "timestamp":      timestamp,
        "dendrite_timeout": request_timeout,
        "Did_it_set_weights": success,
        "uids":    [int(uid)    for uid    in uint_uids]    if uint_uids    else [],
        "weights": [int(weight) for weight in uint_weights] if uint_weights else [],
    }

    results["metagraph_scores"] = {
        "timestamp":     timestamp,
        "total_miners":  len(self.scores),
        "scores_by_uid": {
            str(uid): {
                "uid":        int(uid),
                "hotkey":     str(self.metagraph.axons[uid].hotkey) if uid < len(self.metagraph.axons) else "unknown",
                "score":      float(self.scores[uid]),
                "was_queried": uid in miner_uids,
            }
            for uid in range(len(self.scores))
        },
    }

    # Reward allocation tracking
    if uav_grading_enabled:
        # Create new allocation for this forward pass
        new_allocation = {
            "timestamp":            timestamp,
            "rep_snapshot_version": _cached_rep_version,
            "miners":               combined_metrics,
        }

        # Add to pending allocations and save to disk
        _pending_allocations.append(new_allocation)
        _save_pending_allocations(_pending_file_path, _pending_allocations)

        # Build reward_allocation with ALL pending allocations (for retry on previous failures)
        results["reward_allocation"] = {
            "rep_snapshot_version": _cached_rep_version,
            "cycle_id":             f"cycle_{timestamp}",
            "pending_count":        len(_pending_allocations),
            "allocations":          _pending_allocations,
        }
        bt.logging.info(f"Added reward_allocation: {len(_pending_allocations)} pending allocation(s)")
    else:
        # UAV grading disabled: No reward allocation tracking
        results["reward_allocation"] = {
            "enabled": False,
            "note":    "UAV_grading disabled — using KAV-only scoring",
        }

    # Save the query and responses to a JSON file (now including weights and reward_allocation)
    json_path = os.path.join(run_dir, f"results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    bt.logging.info(f"Saved validator results to: {json_path}")
    
    # Prepare extra data for wandb logging
    wandb_extra_data = {
        "variation_count":  len(selected_variations) if selected_variations else 0,
        "dendrite_timeout": request_timeout,
    }

    # Upload to MIID server
    hotkey             = self.wallet.hotkey
    message_to_sign    = f"Hotkey: {hotkey} \n timestamp: {timestamp}"
    signed_contents    = sign_message(self.wallet, message_to_sign, output_file=None)
    results["signature"] = signed_contents

    is_testnet = (
        self.config.netuid == 322
        and hasattr(self.config, 'subtensor')
        and getattr(self.config.subtensor, 'network', None) == "test"
        and "test.finney.opentensor.ai" in getattr(self.config.subtensor, 'chain_endpoint', '')
    )

    upload_success = False
    upload_response = None
    # If for some reason uploading the data fails, we should just log it and continue.
    # Server might go down but should not be a unique point of failure for the subnet
    try:
        if is_testnet:
            bt.logging.info("Testnet detected — skipping upload to MIID server")
            upload_success = True
        else:
            bt.logging.info(f"Uploading data to: {MIID_SERVER}")
            upload_response = upload_data(MIID_SERVER, hotkey, results)
            upload_success = upload_response is not None

        if upload_success:
            bt.logging.info(
                "Testnet: upload skipped (treated as successful)"
                if is_testnet else "Data uploaded successfully to external server"
            )

            # ==========================================================================
            # Cache rep_data from response for NEXT forward pass (Phase 4 - Cycle 3 Sandbox)
            # ==========================================================================
            if uav_grading_enabled:
                if upload_response and upload_response.get("rep_cache"):
                    _cached_rep_version = upload_response.get("rep_snapshot_version")
                    _cached_rep_data    = upload_response.get("rep_cache", {})
                    bt.logging.info(
                        f"Updated rep cache: version={_cached_rep_version}, "
                        f"miners={len(_cached_rep_data)}"
                    )

                # Clear pending allocations after successful upload
                _pending_allocations.clear()
                _clear_pending_allocations(_pending_file_path)
                bt.logging.info("Cleared pending allocations after successful upload")
            # ==========================================================================
        else:
            bt.logging.error("Failed to upload data to external server")
            if uav_grading_enabled:
                bt.logging.warning(
                    f"Upload failed. {len(_pending_allocations)} allocation(s) pending for retry"
                )
    except Exception as e:
        bt.logging.error(f"Uploading data failed: {str(e)}")
        upload_success = False
    
    wandb_extra_data["upload_success"] = upload_success

    # Call log_step from the Validator instance AFTER the upload attempt
    self.log_step(
        uids=miner_uids, # Pass the list of uids
        metrics=detailed_metrics, # Pass the detailed metrics list
        rewards=rewards, # Pass the numpy array of rewards
        extra_data=wandb_extra_data # Pass additional context
    )
    
    # Delete JSON file and directories ONLY after successful upload
    if upload_success:
        bt.logging.info(f"Upload successful. Cleaning up local files...")
        bt.logging.info(f"Deleting json file: {json_path}")
        bt.logging.info(f"Deleting rundir: {run_dir}")
        bt.logging.info(f"Deleting validator_results dir: {results_dir}")
        try:
            os.remove(json_path)
            os.rmdir(run_dir)
            os.rmdir(results_dir)
            bt.logging.info("Successfully cleaned up all local files")
        except Exception as e:
            bt.logging.error(f"Error deleting files: {e}")
            bt.logging.warning(f"You might want to delete these files manually: {json_path}, {run_dir}, {results_dir}")
    else:
        bt.logging.warning("Upload failed. Keeping local files for debugging.")
        bt.logging.warning("You might want to reach out to the MIID team to add your hotkey to the allowlist.")
        bt.logging.info(f"JSON file preserved at: {json_path}")
        bt.logging.info(f"Run directory preserved at: {run_dir}")
    
    # --- FINISH WANDB RUN AFTER EACH FORWARD PASS ---
    # Finish the wandb run after weights are set and logged (unless wandb is disabled)
    if self.wandb_run and not wandb_disabled:
        bt.logging.info("Finishing wandb run after completing validation cycle")
        try:
            self.wandb_run.finish()
            # Clean up all wandb run folders after finishing
            self.cleanup_all_wandb_runs()
        except Exception as e:
            bt.logging.error(f"Error finishing wandb run: {e}")
        finally:
            self.wandb_run = None
    # --- END WANDB FINISH ---
    
    # 10) Set weights and enforce min epoch time
    
    request_end = time.time()
    if request_end - request_start < EPOCH_MIN_TIME:
        sleep_secs = EPOCH_MIN_TIME - (request_end - request_start)
        bt.logging.info(f"Finished quickly; sleeping for {sleep_secs:.1f}s")
        await asyncio.sleep(sleep_secs)

    bt.logging.info("Forward pass complete.")
    await asyncio.sleep(5)

    return True