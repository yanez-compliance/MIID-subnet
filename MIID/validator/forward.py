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

This module implements the forward function for generating Threat Scenarios.
The forward function is responsible for:
1. Selecting random miners to query
2. Generating threat scenarios and identity test cases
3. Requesting execution vectors (name variations) that could be used to bypass detection systems
4. Evaluating the effectiveness of the execution vectors returned by miners
5. Rewarding miners based on the quality of their execution vectors
6. Saving the results
7. Uploading the same results to the external endpoint (Flask-based) via HTTP POST

The module simulates identity screening bypass attempts by generating and evaluating 
name variations that could potentially evade detection systems.
"""

import time
import bittensor as bt
import json
import os
import random
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

from MIID.protocol import IdentitySynapse, ImageRequest, VariationRequest
from MIID.validator.reward import get_name_variation_rewards, apply_reputation_rewards
from MIID.utils.uids import get_random_uids
from MIID.utils.sign_message import sign_message
from MIID.validator.query_generator import QueryGenerator, add_uav_requirements

# Phase 4 imports
from MIID.validator.base_images import load_random_base_image, validate_base_images_folder
from MIID.validator.drand_utils import calculate_target_round, calculate_reveal_buffer
from MIID.validator.image_variations import (
    select_random_variations,
    format_variation_requirements
)


# Import your new upload_data function here
from MIID.utils.misc import upload_data

# =============================================================================
# Reputation Cache (Phase 4 - Cycle 1)
# =============================================================================

# Module-level cache for reputation data (persists across forward passes)
_cached_rep_data: Dict[str, Dict] = {}
_cached_rep_version: Optional[str] = None

# Module-level pending queue for failed uploads (persists across forward passes)
_pending_allocations: List[Dict] = []
_pending_file_path: Optional[Path] = None


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
MIID_SERVER = "http://52.44.186.20:5000/upload_data" ## MIID server

# =============================================================================
# Phase 4: Image Variation Configuration
# =============================================================================
PHASE4_ENABLED = True  # Set to False to disable Phase 4 features
# Variation types and counts are now dynamically selected per challenge
# See: MIID/validator/image_variations.py for definitions
# =============================================================================

# =============================================================================
# UAV Grading Configuration
# =============================================================================
# UAV_grading is controlled via config.neuron.UAV_grading (default: False)
# When False: Uses KAV-only scoring with burn applied directly
# When True: Uses reputation-weighted rewards (KAV + UAV) with burn applied after combination
# =============================================================================

async def dendrite_with_retries(dendrite: bt.Dendrite, axons: list, synapse: IdentitySynapse,
                                deserialize: bool, timeout: float, cnt_attempts=3):
    """
    Send requests to miners with automatic retry logic for failed connections.
    
    Args:
        dendrite: The dendrite object to use for communication
        axons: List of axons to query
        synapse: The synapse object containing the request
        deserialize: Whether to deserialize the response
        timeout: Timeout for each request in seconds
        cnt_attempts: Number of retry attempts for failed connections
        
    Returns:
        List of responses from miners
    """
    res = [None] * len(axons)
    idx = list(range(len(axons)))
    axons_for_retry = axons.copy()
    
    def create_default_response():
        return IdentitySynapse(
            identity=synapse.identity,
            query_template=synapse.query_template,
            variations={},
            process_time=process_time
        )
    
    for attempt in range(cnt_attempts):
        responses = await dendrite(
            axons=axons_for_retry,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout * (1 + attempt * 0.1)
        )
        
        new_idx = []
        new_axons = []
        
        for i, response in enumerate(responses):
            #bt.logging.info(f"#########################################Response {i}: {response}#########################################")
            #bt.logging.info(f"#########################################Response type: {type(response)}#########################################")
            
            process_time = None
            if hasattr(response, "dendrite") and hasattr(response.dendrite, "process_time"):
                try:
                    process_time = float(response.dendrite.process_time)
                except (ValueError, TypeError):
                    process_time = None
                        
            if isinstance(response, dict):
                # Got the variations dictionary directly
                complete_response = IdentitySynapse(
                    identity=synapse.identity,
                    query_template=synapse.query_template,
                    variations=response,
                    process_time=process_time
                )
                res[idx[i]] = complete_response
            
            elif hasattr(response, 'dendrite'):
                # Check status code
                if (response.dendrite.status_code is not None and 
                    int(response.dendrite.status_code) == 422):
                    if attempt == cnt_attempts - 1:
                        res[idx[i]] = response
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_for_retry[i])
                else:
                    response.process_time = process_time  # <-- attach it
                    res[idx[i]] = response
            
            else:
                # If the response has variations attribute, treat it as a valid response
                if hasattr(response, 'variations'):
                    response.process_time = process_time  # <-- attach it
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


def process_new_variations_structure(uid_response_map, miner_uids, self, expected_uav_seed_name: Optional[str] = None):
    """Process the new variations structure and extract UAV data.

    This function updates each response's variations to the legacy list-of-lists
    while collecting UAV information separately for logging and result export.
    
    Args:
        uid_response_map: Dictionary mapping UIDs to responses
        miner_uids: List of miner UIDs
        self: Validator instance
        expected_uav_seed_name: Optional. If provided, only accept UAVs for this specific seed name.
                               UAVs for other seeds will be ignored and logged as warnings.
    """
    uav_summary = {
        "total_miners_with_uav": 0,
        "total_uavs_collected": 0,
        "miners_with_coordinates": 0,
        "rejected_uavs": 0,  # Track UAVs from unexpected seeds
    }
    uav_by_miner = {}

    for uid in miner_uids:
        response = uid_response_map.get(uid)
        if not response or not hasattr(response, "variations") or response.variations is None:
            continue

        miner_variations = {}
        miner_uav_data = {"uavs": {}, "valid_count": 0, "has_coordinates": False}

        try:
            items = response.variations.items() if isinstance(response.variations, dict) else []
            for seed_name, seed_data in items:
                if isinstance(seed_data, list):
                    # Old format: variations only
                    miner_variations[seed_name] = seed_data
                elif isinstance(seed_data, dict):
                    # New format: { "variations": [...], "uav": {...} }
                    if "variations" in seed_data:
                        miner_variations[seed_name] = seed_data.get("variations") or []

                    if seed_data.get("uav"):
                        # Only accept UAV for the expected seed name
                        if expected_uav_seed_name and seed_name != expected_uav_seed_name:
                            bt.logging.warning(
                                f"Miner {uid}: Rejected UAV for unexpected seed '{seed_name}'. "
                                f"Expected UAV only for '{expected_uav_seed_name}'"
                            )
                            uav_summary["rejected_uavs"] += 1
                            continue
                        
                        uav = seed_data["uav"]
                        if (
                            isinstance(uav, dict)
                            and uav.get("address")
                            and uav.get("label")
                        ):
                            miner_uav_data["uavs"][seed_name] = uav
                            miner_uav_data["valid_count"] += 1
                            uav_summary["total_uavs_collected"] += 1

                            if (
                                uav.get("latitude") is not None
                                and uav.get("longitude") is not None
                            ):
                                miner_uav_data["has_coordinates"] = True
                        else:
                            bt.logging.warning(
                                f"Miner {uid}: Invalid UAV for '{seed_name}'"
                            )
        except Exception as e:
            bt.logging.warning(f"Miner {uid}: Error processing variations: {e}")

        # Normalize response variations for reward calculation
        response.variations = miner_variations

        # Store UAV data if any valid UAVs found
        if miner_uav_data["valid_count"] > 0:
            uav_by_miner[str(uid)] = {
                "hotkey": str(self.metagraph.axons[uid].hotkey),
                **miner_uav_data,
            }
            uav_summary["total_miners_with_uav"] += 1
            if miner_uav_data["has_coordinates"]:
                uav_summary["miners_with_coordinates"] += 1

    return uav_summary, uav_by_miner


async def forward(self):
    """
    The forward function is called by the validator every time step.

    This function implements a threat detection simulation that:
    1. Selects a random set of miners to query
    2. Generates a complex threat scenario using an LLM
    3. Creates a list of identity names as test cases
    4. Sends the threat scenario to miners, requesting execution vectors (name variations)
    5. Evaluates the effectiveness of execution vectors at bypassing identity detection
    6. Updates miner scores based on the quality and diversity of their execution vectors
    7. Saves the results 
    
    Returns:
        The result of the forward function from the MIID.validator module
    """

    # --- CREATE NEW WANDB RUN FOR EACH FORWARD PASS ---
    # Ensure we have a wandb run for this forward pass (unless wandb is disabled)
    wandb_disabled = hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'disable') and self.config.wandb.disable
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
    axons = [self.metagraph.axons[uid] for uid in miner_uids]

    bt.logging.debug(f"🔧 Miner axons: {axons}")
    bt.logging.debug(f"⚙️ Miner selection size: {miner_selection_size}")
    bt.logging.debug(f"📋 Available axon size: {available_axon_size}")

    miner_uids = miner_uids.tolist()
    bt.logging.info(f"Selected {len(miner_uids)} miners to query: {miner_uids}")

    # 2) Use the existing query generator instance
    query_generator = self.query_generator
    
    # Use the query generator
    challenge_start_time = time.time()
    seed_names_with_labels, query_template, query_labels, successful_model, successful_timeout, successful_judge_model, successful_judge_timeout, generation_log = await query_generator.build_queries()
    
    # Identify high-risk identities and randomly select one for UAV request
    high_risk_identities = [item for item in seed_names_with_labels if item.get('label') == 'High Risk']
    uav_seed_name = None
    selected_identity = None
    if high_risk_identities:
        selected_identity = random.choice(high_risk_identities)
        uav_seed_name = selected_identity['name']
        bt.logging.info(f"Selected high-risk identity '{uav_seed_name}' with address '{selected_identity['address']}' for UAV request")
    else:
        bt.logging.warning("No high-risk identities found. Skipping UAV request.")
    
    # Add UAV field to all identities (true for selected one, false for others)
    for identity in seed_names_with_labels:
        identity['UAV'] = (identity['name'] == uav_seed_name) if uav_seed_name else False
    
    # Store the base query template before adding UAV requirements (for formatted queries)
    base_query_template = query_template
    
    # Append Phase 3 UAV requirements to the query template sent to miners (only for selected identity)
    query_template = add_uav_requirements(query_template, uav_seed_name=uav_seed_name)
    challenge_end_time = time.time()
    bt.logging.info(f"Time to generate challenges: {int(challenge_end_time - challenge_start_time)}s")

    # Extract the names, addresses, and DOBs for use in existing logic
    seed_names = [item['name'] for item in seed_names_with_labels]
    seed_addresses = [item['address'] for item in seed_names_with_labels]
    seed_dob = [item['dob'] for item in seed_names_with_labels]
    
    # Create identity list with [name, dob, address] arrays
    identity_list = [[item['name'], item['dob'], item['address']] for item in seed_names_with_labels]

    # Calculate timeout based on the number of identities and complexity
    base_timeout = self.config.neuron.timeout  # Double from 60 to 120 seconds
    # More generous allocation - especially for LLM operations
    adaptive_timeout = base_timeout + (len(seed_names) * 20) + (query_labels['variation_count'] * 10)
    adaptive_timeout = min(self.config.neuron.max_request_timeout, max(120, adaptive_timeout))  # clamp [120, max_request_timeout]
    bt.logging.info(f"Using adaptive timeout of {adaptive_timeout} seconds for {len(seed_names)} identities")
    
    # ==========================================================================
    # Phase 4: Create Image Request
    # ==========================================================================
    image_request = None
    challenge_id = None
    selected_variations = None  # Track what variations were requested

    if PHASE4_ENABLED:
        try:
            # Validate base images folder
            is_valid, validation_msg = validate_base_images_folder()
            if not is_valid:
                bt.logging.warning(f"Phase 4 disabled: {validation_msg}")
            else:
                # Load a random base image
                image_filename, base64_image = load_random_base_image()

                # Calculate drand round for reveal (after all miners respond)
                reveal_delay = calculate_reveal_buffer(adaptive_timeout)
                target_round, reveal_timestamp = calculate_target_round(reveal_delay)

                # Generate unique challenge ID
                challenge_id = f"challenge_{int(time.time())}_{self.wallet.hotkey.ss58_address[:8]}"

                # Randomly select variation types with intensities (2-4 variations)
                selected_variations = select_random_variations(min_variations=2, max_variations=4)

                # Convert to VariationRequest objects
                variation_requests = [
                    VariationRequest(
                        type=v["type"],
                        intensity=v["intensity"],
                        description=v["description"],
                        detail=v["detail"]
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
                    challenge_id=challenge_id
                )

                # Log what was selected
                variation_summary = ", ".join(
                    f"{v['type']}({v['intensity']})" for v in selected_variations
                )
                bt.logging.info(
                    f"Phase 4: Created image request for '{image_filename}', "
                    f"variations: [{variation_summary}], "
                    f"drand round {target_round}, reveal at {reveal_timestamp}"
                )
        except Exception as e:
            bt.logging.warning(f"Phase 4: Could not create image request: {e}")
            image_request = None
            selected_variations = None

    # Add image variation requirements to query template (if Phase 4 enabled)
    if selected_variations:
        image_variation_text = format_variation_requirements(selected_variations)
        query_template = query_template + image_variation_text
        bt.logging.debug(f"Phase 4: Added image variation requirements to query template")
    # ==========================================================================

    # 5) Prepare the synapse
    request_synapse = IdentitySynapse(
        identity=identity_list,
        query_template=query_template,
        variations={},
        timeout=adaptive_timeout,
        image_request=image_request  # Phase 4: Add image request
    )

    if query_generator.use_default_query:  
        bt.logging.info(f"Querying {len(miner_uids)} miners with default query template")
    else:
        bt.logging.info(f"Querying {len(miner_uids)} miners with complex query")

    bt.logging.debug(f"📄 Request synapse: {request_synapse}")
    await asyncio.sleep(3)

    # 6) Query the network in batches
    start_time = time.time()
    # Use a dictionary to maintain UID-response mapping instead of separate lists
    uid_response_map = {}
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
            timeout=adaptive_timeout,
            cnt_attempts=3
        )
        
        batch_duration = time.time() - batch_start_time
        bt.logging.info(f"Batch {i//batch_size + 1} completed in {batch_duration:.1f}s")

        # Map each response to its corresponding UID
        for idx_resp, response in enumerate(batch_responses):
            uid = batch_uids[idx_resp]
            uid_response_map[uid] = response
            
            if not hasattr(response, 'variations'):
                bt.logging.warning(f"Miner {uid} returned response without 'variations' attribute.")
            elif response.variations is None:
                bt.logging.warning(f"Miner {uid} returned None in 'variations'.")
            elif not response.variations:
                bt.logging.warning(f"Miner {uid} returned empty variations dictionary.")
            else:
                total_variations = sum(len(v) for v in response.variations.values())
                # Enhanced logging for name structure validation
                # for name, variations in response.variations.items():
                #     name_parts = name.split()
                #     if len(name_parts) > 1:  # Multi-part name
                #         #bt.logging.info(f"Validating variations for multi-part name '{name}' (first: '{name_parts[0]}', last: '{name_parts[-1]}')")
                #         # Validate variation structure
                #         for var in variations:
                #             var_parts = var.split()
                #     #         if len(var_parts) < 2:
                #     #             bt.logging.warning(f"Miner {uid} returned single-part variation '{var}' for multi-part name '{name}'")
                #     # #else:  # Single-part name
                #     #     #bt.logging.info(f"Validating variations for single-part name '{name}'")
                        
                bt.logging.info(f"Miner {uid} returned {len(response.variations)} identity variations with {total_variations} total variations.")
        
        if i + batch_size < len(miner_uids):
            sleep_time = 2
            bt.logging.info(f"Sleeping for {sleep_time}s before next batch")
            await asyncio.sleep(sleep_time)
    
    end_time = time.time()
    bt.logging.info(f"Query completed in {end_time - start_time:.2f} seconds")

    # Normalize new structure to legacy for rewards and collect UAV data
    # Only accept UAVs for the expected high-risk identity
    uav_summary, uav_by_miner = process_new_variations_structure(uid_response_map, miner_uids, self, expected_uav_seed_name=uav_seed_name)
    bt.logging.info(
        f"UAV Collection: {uav_summary.get('total_miners_with_uav', 0)} miners provided UAVs, "
        f"{uav_summary.get('total_uavs_collected', 0)} total collected"
    )
    if uav_summary.get('rejected_uavs', 0) > 0:
        bt.logging.warning(
            f"Rejected {uav_summary.get('rejected_uavs', 0)} UAV(s) from unexpected seed names. "
            f"Expected UAV only for '{uav_seed_name}'"
        )

    # Create ordered lists from the mapping to maintain consistency (after normalization)
    all_responses = [uid_response_map[uid] for uid in miner_uids]

    # 7) Compute rewards
    bt.logging.info(f"Received identity variation responses for {len(all_responses)} miners")
    valid_responses = 0
    for i, response in enumerate(all_responses):
        if hasattr(response, 'variations') and response.variations:
            valid_responses += 1
        else:
            bt.logging.warning(f"Miner {miner_uids[i]} returned invalid or empty response")
    
    bt.logging.info(f"Received {valid_responses} valid responses out of {len(all_responses)}")

    seed_script = [item['script'] for item in seed_names_with_labels]
    
    # Check if UAV grading is enabled (default: False)
    uav_grading_enabled = getattr(self.config.neuron, 'UAV_grading', False)
    
    if uav_grading_enabled:
        # Phase 3: Get KAV rewards WITHOUT burn - burn will be applied after reputation weighting
        kav_rewards, kav_uids, detailed_metrics = get_name_variation_rewards(
            self,
            seed_names,
            seed_dob,
            seed_addresses,
            seed_script,
            all_responses,
            miner_uids,
            variation_count=query_labels['variation_count'],
            phonetic_similarity=query_labels['phonetic_similarity'],
            orthographic_similarity=query_labels['orthographic_similarity'],
            rule_based=query_labels.get('rule_based'),  # Pass rule-based metadata
            skip_burn=True  # Phase 3: Skip burn here, apply after KAV+UAV in apply_reputation_rewards()
        )

        # ==========================================================================
        # REPUTATION-WEIGHTED REWARDS (Phase 4 - Cycle 1)
        # ==========================================================================
        global _cached_rep_data, _cached_rep_version, _pending_allocations, _pending_file_path

        # Initialize pending file path (once per validator session)
        if _pending_file_path is None:
            _pending_file_path = Path(self.config.logging.logging_dir) / "validator_results" / "pending_allocations.json"
            _pending_allocations = _load_pending_allocations(_pending_file_path)
            if _pending_allocations:
                bt.logging.info(f"Loaded {len(_pending_allocations)} pending allocations from previous session")

        # With skip_burn=True, kav_rewards/kav_uids contain only miners (no burn UID)
        # Convert to list for apply_reputation_rewards
        miner_uids_list = kav_uids.tolist() if hasattr(kav_uids, 'tolist') else list(kav_uids)

        # Get config values (Phase 4 - Cycle 1)
        burn_fraction = getattr(self.config.neuron, 'burn_fraction', 0.75)
        kav_weight = getattr(self.config.neuron, 'kav_weight', 0.20)
        uav_weight = getattr(self.config.neuron, 'uav_weight', 0.80)

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
            kav_metrics=detailed_metrics
        )

        bt.logging.info(
            f"Applied reputation rewards: {len(miner_uids_list)} miners, "
            f"using rep_snapshot_version={_cached_rep_version or 'None (first run)'}"
        )
        # ==========================================================================
        # END REPUTATION-WEIGHTED REWARDS
        # ==========================================================================
    else:
        # UAV grading disabled: Use KAV-only scoring with burn applied directly
        bt.logging.info("UAV_grading disabled. Using KAV-only scoring with burn applied.")
        rewards, updated_uids, detailed_metrics = get_name_variation_rewards(
            self,
            seed_names,
            seed_dob,
            seed_addresses,
            seed_script,
            all_responses,
            miner_uids,
            variation_count=query_labels['variation_count'],
            phonetic_similarity=query_labels['phonetic_similarity'],
            orthographic_similarity=query_labels['orthographic_similarity'],
            rule_based=query_labels.get('rule_based'),  # Pass rule-based metadata
            skip_burn=False  # Apply burn directly in get_name_variation_rewards
        )
        # Use detailed_metrics as combined_metrics for consistency
        combined_metrics = detailed_metrics

    # Verify UID-reward mapping before updating scores
    bt.logging.info("=== UID-REWARD MAPPING VERIFICATION ===")
    for i, uid in enumerate(updated_uids):
        reward = rewards[i] if i < len(rewards) else 0.0
        response = uid_response_map.get(uid) if uid != 59 else None  # Burn UID won't have a response
        has_response = response is not None
        if uid == 59:
            bt.logging.info(f"UID {uid}: IS BURNED EVENT. NO REWARD OR RESPONSE.")
        else:
            bt.logging.info(f"UID {uid}: Reward={reward:.4f}, HasResponse={has_response}")
    bt.logging.info("=== END UID-REWARD MAPPING VERIFICATION ===")

    self.update_scores(rewards, updated_uids)
    bt.logging.info(f"REWARDS: {rewards}  for UIDs: {updated_uids}")

    # 8) Save results locally
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(self.config.logging.logging_dir, "validator_results")
    os.makedirs(results_dir, exist_ok=True)
    
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Format example queries with actual identities to show what was sent to miners
    formatted_queries = {}
    for identity in seed_names_with_labels:
        try:
            # Use base query template (without UAV) for all identities except the UAV-selected one
            # For the UAV-selected identity, use the full query template (with UAV requirements)
            if uav_seed_name and identity['name'] == uav_seed_name:
                template_to_use = query_template
            else:
                template_to_use = base_query_template
            
            # Format the query template with the actual identity name
            formatted_query = template_to_use.replace("{name}", identity['name'])
            formatted_queries[identity['name']] = {
                "query": formatted_query,
                "identity": {
                    "name": identity['name'],
                    "dob": identity['dob'],
                    "address": identity['address']
                }
            }
        except Exception as e:
            bt.logging.error(f"Error formatting query for identity '{identity.get('name', 'unknown')}': {str(e)}")
            formatted_queries[identity.get('name', 'unknown')] = {
                "query": f"Error formatting query: {str(e)}",
                "identity": identity
            }
    
    # Save the query and responses to a JSON file
    results = {
        "timestamp": timestamp,
        "seed_names_with_labels": seed_names_with_labels,
        "seed_names": seed_names,
        "seed_addresses": seed_addresses,
        "seed_dob": seed_dob,
        "query_template": query_template,
        "query_labels": query_labels,
        "formatted_queries": formatted_queries,  # Add the formatted queries
        "request_synapse": {
            "identity": identity_list,
            "query_template": query_template,
            "dendrite_timeout": adaptive_timeout
        },
        # Phase 3 UAV data (collected and scored in Cycle 1 execution)
        "uav_data": {
            "cycle": "Phase4-C1-Sandbox",
            "note": "UAVs are collected and scored in Cycle 1 execution",
            "summary": uav_summary,
            "by_miner": uav_by_miner,
        },
        # Phase 4: Image variation data for YANEZ post-validation
        "phase4_image_data": {
            "cycle": "Phase4-C1-Sandbox",
            "note": "Image variations with S3 uploads, post-validation scoring by YANEZ",
            "enabled": PHASE4_ENABLED and image_request is not None,
            "s3_bucket": "yanez-miid-sn54",
            "challenge_id": challenge_id,
            "base_image_filename": image_request.image_filename if image_request else None,
            "target_drand_round": image_request.target_drand_round if image_request else None,
            "reveal_timestamp": image_request.reveal_timestamp if image_request else None,
            # Detailed variation requests with type + intensity for scoring
            "requested_variations": selected_variations if selected_variations else [],
            # Summary for quick reference
            "variation_count": len(selected_variations) if selected_variations else 0,
            "variation_types": [v["type"] for v in selected_variations] if selected_variations else [],
            "variation_intensities": [v["intensity"] for v in selected_variations] if selected_variations else [],
            "s3_submissions_by_miner": {},  # Populated below
        },
        "query_generation": {
            "use_default_query": self.query_generator.use_default_query,
            "configured_model": getattr(self.config.neuron, 'ollama_model_name', "llama3.1:latest"),
            "model_used": successful_model,  # Actual model that succeeded
            "timeout_used": successful_timeout,  # Actual timeout that succeeded
            "generation_time": challenge_end_time - challenge_start_time,
            "generation_log": generation_log,
            # Enhanced generation details
            "generation_attempts": generation_log.get("attempts", []),
            "generation_decision": generation_log.get("decision", "unknown"),
            "final_template": generation_log.get("final_template", query_template),
            "total_attempts": len(generation_log.get("attempts", [])),
            "successful_attempt_index": next((i for i, attempt in enumerate(generation_log.get("attempts", [])) 
                                           if attempt.get("status") in ["success", "success_after_repair", "proceeded_with_invalid_template"]), None)
        },
        "query_validation": {
            "judge_enabled": self.query_generator.use_judge_model,
            "judge_model_used": successful_judge_model,  # Judge model that succeeded (if any)
            "judge_timeout_used": successful_judge_timeout,  # Judge timeout that succeeded (if any)
            "judge_strict_mode": self.query_generator.judge_strict_mode,
            "judge_on_static_pass": self.query_generator.judge_on_static_pass,
            # Enhanced validation details from generation_log
            "validation_details": generation_log.get("validation", {}),
            "static_issues": generation_log.get("validation", {}).get("static_issues", []),
            "judge_issues": generation_log.get("validation", {}).get("judge_issues", []),
            "final_issues": generation_log.get("validation", {}).get("final_issues", []),
            "validation_decision": generation_log.get("validation", {}).get("decision", "unknown"),
            # Validation summary for quick insights
            "validation_summary": {
                "static_checks_passed": len(generation_log.get("validation", {}).get("static_issues", [])) == 0,
                "judge_was_used": successful_judge_model is not None,
                "judge_found_issues": len(generation_log.get("validation", {}).get("judge_issues", [])) > 0,
                "final_issues_count": len(generation_log.get("validation", {}).get("final_issues", [])),
                "template_has_hints": "[VALIDATION HINTS]" in query_template if query_template else False
            }
        },
        "responses": {},
        "rewards": {}
    }

    for i, uid in enumerate(miner_uids):
        # Get the response for this specific UID from our mapping
        response = uid_response_map.get(uid)
        
        if response is not None:
            bt.logging.info(f"#########################################Response Time miner {uid}: {response.process_time}#########################################")

            # Get the axon for this UID
            axon = self.metagraph.axons[uid]
            
            # Convert the response to a serializable format
            response_data = {
                "uid": int(uid),
                "hotkey": str(self.metagraph.axons[uid].hotkey),
                "axon": {
                    "ip": str(axon.ip) if hasattr(axon, 'ip') else None,
                    "port": int(axon.port) if hasattr(axon, 'port') else None,
                    "hotkey": str(axon.hotkey),
                    "coldkey": str(axon.coldkey) if hasattr(axon, 'coldkey') else None,
                    "version": int(axon.version) if hasattr(axon, 'version') else None,
                    "protocol": int(axon.protocol) if hasattr(axon, 'protocol') else None,
                    "is_serving": bool(axon.is_serving) if hasattr(axon, 'is_serving') else None
                },
                "response_time": response.process_time,  # When we processed this response
                "variations": {},
                "error": None,
                "scoring_details": detailed_metrics[i] if i < len(detailed_metrics) else {}
            }
            
            # Add variations if available
            if hasattr(response, 'variations') and response.variations is not None:
                response_data["variations"] = response.variations
            else:
                # Log error information if available
                if hasattr(response, 'dendrite') and hasattr(response.dendrite, 'status_code'):
                    response_data["error"] = {
                        "status_code": response.dendrite.status_code,
                        "status_message": getattr(response.dendrite, 'status_message', 'Unknown error')
                    }
                else:
                    response_data["error"] = {
                        "message": "Invalid response format",
                        "response_type": str(type(response))
                    }

            # Phase 4: Collect S3 submissions
            if PHASE4_ENABLED and hasattr(response, 's3_submissions') and response.s3_submissions:
                miner_hotkey = str(self.metagraph.axons[uid].hotkey)
                s3_data = []
                for submission in response.s3_submissions:
                    s3_data.append({
                        "s3_key": submission.s3_key,
                        "image_hash": submission.image_hash,
                        "signature": submission.signature,
                        "variation_type": submission.variation_type
                    })
                results["phase4_image_data"]["s3_submissions_by_miner"][str(uid)] = {
                    "hotkey": miner_hotkey,
                    "submissions": s3_data,
                    "submission_count": len(s3_data)
                }
                bt.logging.info(f"Phase 4: Collected {len(s3_data)} S3 submissions from miner {uid}")
        else:
            # Handle case where no response was received for this UID
            bt.logging.warning(f"No response received for miner {uid}")
            
            # Get the axon for this UID
            axon = self.metagraph.axons[uid]
            
            response_data = {
                "uid": int(uid),
                "hotkey": str(self.metagraph.axons[uid].hotkey),
                "axon": {
                    "ip": str(axon.ip) if hasattr(axon, 'ip') else None,
                    "port": int(axon.port) if hasattr(axon, 'port') else None,
                    "hotkey": str(axon.hotkey),
                    "coldkey": str(axon.coldkey) if hasattr(axon, 'coldkey') else None,
                    "version": int(axon.version) if hasattr(axon, 'version') else None,
                    "protocol": int(axon.protocol) if hasattr(axon, 'protocol') else None,
                    "is_serving": bool(axon.is_serving) if hasattr(axon, 'is_serving') else None
                },
                "response_time": None,
                "variations": {},
                "error": {
                    "message": "No response received",
                    "response_type": "None"
                },
                "scoring_details": detailed_metrics[i] if i < len(detailed_metrics) else {}
            }
        
        # Add to the results
        results["responses"][str(uid)] = response_data
        results["rewards"][str(uid)] = float(rewards[i]) if i < len(rewards) else 0.0
    
    # logging the spec_version before setting weights
    bt.logging.info(f"Spec version for setting weights: {self.spec_version}")
    (success, uint_uids, uint_weights) = self.set_weights()
    bt.logging.info(f"Weights set successfully: {success}")
    bt.logging.debug(f"📊 Uids: {uint_uids}")
    bt.logging.debug(f"⚖️ Weights: {uint_weights}")
    
    # Always add weights info to results, regardless of success
    results["Weights"] = {
        "spec_version": self.spec_version,
        "hotkey": str(self.wallet.hotkey.ss58_address),
        "timestamp": timestamp,
        "model_name": successful_model or getattr(self.config.neuron, 'ollama_model_name', "llama3.1:latest"),
        "query_generator_timeout": successful_timeout,
        "judge_model": successful_judge_model,
        "judge_timeout": successful_judge_timeout,
        "dendrite_timeout": adaptive_timeout,
        "Did_it_set_weights": success,
        "uids": [int(uid) for uid in uint_uids] if uint_uids else [],
        "weights": [int(weight) for weight in uint_weights] if uint_weights else []
    }
    bt.logging.debug(f"📈 Results: {results['Weights']}")
    
    # Add metagraph scores for all miners
    results["metagraph_scores"] = {
        "timestamp": timestamp,
        "total_miners": len(self.scores),
        "scores_by_uid": {}
    }
    bt.logging.debug(f"📊 Metagraph scores: {results['metagraph_scores']}")
    # Add scores for each UID in the metagraph
    for uid in range(len(self.scores)):
        results["metagraph_scores"]["scores_by_uid"][str(uid)] = {
            "uid": int(uid),
            "hotkey": str(self.metagraph.axons[uid].hotkey) if uid < len(self.metagraph.axons) else "unknown",
            "score": float(self.scores[uid]),
            "was_queried": uid in miner_uids
        }
    
    bt.logging.debug(f"📋 Metagraph scores added for {len(self.scores)} miners")
    
    if not success:
        bt.logging.error("Failed to set weights. Exiting.")
    else:
        bt.logging.info("Weights set successfully.")

    # Save the query and responses to a JSON file (now including weights)
    json_path = os.path.join(run_dir, f"results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    bt.logging.info(f"Saved validator results to: {json_path}")
    
    # Prepare extra data for wandb logging
    wandb_extra_data = {
        "query_template": query_template,
        "variation_count": query_labels.get('variation_count'),
        "seed_names_count": len(seed_names_with_labels),
        "query_generation_model": successful_model,
        "query_generator_timeout": successful_timeout,
        "judge_model": successful_judge_model,
        "judge_timeout": successful_judge_timeout,
        "judge_enabled": self.query_generator.use_judge_model,
        "dendrite_timeout": adaptive_timeout,
        #"valid_responses": valid_responses,
        #"total_responses": len(all_responses),
        # Include query labels directly
        "query_labels": query_labels,
        # Add the path to the saved JSON results
        #"json_results_path": json_path
    }

    # ==========================================================================
    # 9) Add reward_allocation to results (Phase 4 - Cycle 1)
    # ==========================================================================
    if uav_grading_enabled:
        # Create new allocation for this forward pass
        new_allocation = {
            "timestamp": timestamp,
            "rep_snapshot_version": _cached_rep_version,
            "miners": combined_metrics  # Full breakdown per miner from apply_reputation_rewards()
        }

        # Add to pending allocations and save to disk
        _pending_allocations.append(new_allocation)
        _save_pending_allocations(_pending_file_path, _pending_allocations)

        # Build reward_allocation with ALL pending allocations (for retry on previous failures)
        results["reward_allocation"] = {
            "rep_snapshot_version": _cached_rep_version,
            "cycle_id": f"cycle_{timestamp}",
            "pending_count": len(_pending_allocations),
            "allocations": _pending_allocations  # ALL pending allocations
        }

        bt.logging.info(
            f"Added reward_allocation to results: {len(_pending_allocations)} pending allocation(s)"
        )
    else:
        # UAV grading disabled: No reward allocation tracking
        results["reward_allocation"] = {
            "enabled": False,
            "note": "UAV_grading disabled - using KAV-only scoring"
        }
    # ==========================================================================

    # 10) Upload to external endpoint (moved to a separate utils function)
    # Adjust endpoint URL/hotkey if needed
    results_json_string = json.dumps(results, sort_keys=True)

    hotkey = self.wallet.hotkey
    bt.logging.debug(f"🔑 Hotkey: {hotkey}")
    message_to_sign = f"Hotkey: {hotkey} \n timestamp: {timestamp} \n query_template: {query_template} \n query_labels: {query_labels}"
    signed_contents = sign_message(self.wallet, message_to_sign, output_file=None)
    results["signature"] = signed_contents

    upload_response = None
    upload_success = False
    # If for some reason uploading the data fails, we should just log it and continue.
    # Server might go down but should not be a unique point of failure for the subnet
    try:
        bt.logging.info(f"Uploading data to: {MIID_SERVER}")
        upload_response = upload_data(MIID_SERVER, hotkey, results)
        upload_success = upload_response is not None

        if upload_success:
            bt.logging.info("Data uploaded successfully to external server")

            # ==========================================================================
            # Cache rep_data from response for NEXT forward pass (Phase 3 - Cycle 2)
            # ==========================================================================
            if uav_grading_enabled:
                if upload_response.get("rep_cache"):
                    _cached_rep_version = upload_response.get("rep_snapshot_version")
                    _cached_rep_data = upload_response.get("rep_cache", {})
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
        if uav_grading_enabled:
            bt.logging.warning(
                f"Upload failed. {len(_pending_allocations)} allocation(s) pending for retry"
            )
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
        bt.logging.info(f"Finished quickly; sleeping for {EPOCH_MIN_TIME - (request_end - request_start)}s")
        await asyncio.sleep(EPOCH_MIN_TIME - (request_end - request_start))

    bt.logging.info("All batches processed, waiting 30 more seconds...")
    await asyncio.sleep(5)

    return True
