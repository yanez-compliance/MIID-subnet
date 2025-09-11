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
from typing import List, Dict, Any, Tuple
from datetime import datetime

from MIID.protocol import IdentitySynapse
from MIID.validator.reward import get_name_variation_rewards
from MIID.utils.uids import get_random_uids
from MIID.utils.sign_message import sign_message
from MIID.validator.query_generator import QueryGenerator


# Import your new upload_data function here
from MIID.utils.misc import upload_data

EPOCH_MIN_TIME = 360  # seconds
MIID_SERVER = "http://52.44.186.20:5000/upload_data" ## MIID server

async def dendrite_with_retries(dendrite: bt.dendrite, axons: list, synapse: IdentitySynapse,
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
            names=synapse.names,
            query_template=synapse.query_template,
            variations={}
        )
    
    for attempt in range(cnt_attempts):
        responses = await dendrite(
            axons=axons_for_retry,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout * (1 + attempt * 1.0)
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
                    names=synapse.names,
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
                        response.process_time = process_time
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons_for_retry[i])
        
        if not new_idx:
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
    challenge_end_time = time.time()
    bt.logging.info(f"Time to generate challenges: {int(challenge_end_time - challenge_start_time)}s")

    # Extract just the names for use in existing logic
    seed_names = [item['name'] for item in seed_names_with_labels]

    # Calculate timeout based on the number of names and complexity
    base_timeout = self.config.neuron.timeout  # Double from 60 to 120 seconds
    # More generous allocation - especially for LLM operations
    adaptive_timeout = base_timeout + (len(seed_names) * 20) + (query_labels['variation_count'] * 10)
    adaptive_timeout = min(self.config.neuron.max_request_timeout, max(120, adaptive_timeout))  # clamp [120, max_request_timeout]
    bt.logging.info(f"Using adaptive timeout of {adaptive_timeout} seconds for {len(seed_names)} names")

    # 5) Prepare the synapse
    request_synapse = IdentitySynapse(
        names=seed_names,
        query_template=query_template,
        variations={},
        timeout=adaptive_timeout
    )

    if query_generator.use_default_query:  
        bt.logging.info(f"Querying {len(miner_uids)} miners with default query template")
    else:
        bt.logging.info(f"Querying {len(miner_uids)} miners with complex query")

    bt.logging.debug(f"📄 Request synapse: {request_synapse}")
    await asyncio.sleep(3)

    # 6) Query the network in batches
    start_time = time.time()
    all_responses = []
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
            cnt_attempts=7
        )
        
        batch_duration = time.time() - batch_start_time
        bt.logging.info(f"Batch {i//batch_size + 1} completed in {batch_duration:.1f}s")

        for idx_resp, response in enumerate(batch_responses):
            uid = batch_uids[idx_resp]
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
                        
                bt.logging.info(f"Miner {uid} returned {len(response.variations)} names with {total_variations} total variations.")
        
        all_responses.extend(batch_responses)
        
        if i + batch_size < len(miner_uids):
            sleep_time = 2
            bt.logging.info(f"Sleeping for {sleep_time}s before next batch")
            await asyncio.sleep(sleep_time)
    
    end_time = time.time()
    bt.logging.info(f"Query completed in {end_time - start_time:.2f} seconds")

    # 7) Compute rewards
    bt.logging.info(f"Received name variation responses for {len(all_responses)} miners")
    valid_responses = 0
    for i, response in enumerate(all_responses):
        if hasattr(response, 'variations') and response.variations:
            valid_responses += 1
        else:
            bt.logging.warning(f"Miner {miner_uids[i]} returned invalid or empty response")
    
    bt.logging.info(f"Received {valid_responses} valid responses out of {len(all_responses)}")

    rewards, detailed_metrics = get_name_variation_rewards(
        self, 
        seed_names,
        all_responses, 
        miner_uids,
        variation_count=query_labels['variation_count'],
        phonetic_similarity=query_labels['phonetic_similarity'],
        orthographic_similarity=query_labels['orthographic_similarity'],
        rule_based=query_labels.get('rule_based')  # Pass rule-based metadata
    )

    self.update_scores(rewards, miner_uids)
    bt.logging.info(f"REWARDS: {rewards}  for MINER UIDs: {miner_uids}")

    # 8) Save results locally
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(self.config.logging.logging_dir, "validator_results")
    os.makedirs(results_dir, exist_ok=True)
    
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Format example queries with actual names to show what was sent to miners
    formatted_queries = {}
    for name in seed_names:
        try:
            # Format the query template with the actual name
            formatted_query = query_template.replace("{name}", name)
            formatted_queries[name] = formatted_query
        except Exception as e:
            bt.logging.error(f"Error formatting query for name '{name}': {str(e)}")
            formatted_queries[name] = f"Error formatting query: {str(e)}"
    
    # Save the query and responses to a JSON file
    results = {
        "timestamp": timestamp,
        "seed_names_with_labels": seed_names_with_labels,
        "seed_names": seed_names,
        "query_template": query_template,
        "query_labels": query_labels,
        "formatted_queries": formatted_queries,  # Add the formatted queries
        "request_synapse": {
            "names": seed_names,
            "query_template": query_template,
            "dendrite_timeout": adaptive_timeout
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
        if i < len(all_responses):
            bt.logging.info(f"#########################################Response Time miner {uid}: {all_responses[i].process_time}#########################################")

            # Convert the response to a serializable format
            response_data = {
                "uid": int(uid),
                "hotkey": str(self.metagraph.axons[uid].hotkey),
                "response_time": all_responses[i].process_time,  # When we processed this response
                "variations": {},
                "error": None,
                "scoring_details": detailed_metrics[i] if i < len(detailed_metrics) else {}
            }
            
            # Add variations if available
            if hasattr(all_responses[i], 'variations') and all_responses[i].variations is not None:
                response_data["variations"] = all_responses[i].variations
            else:
                # Log error information if available
                if hasattr(all_responses[i], 'dendrite') and hasattr(all_responses[i].dendrite, 'status_code'):
                    response_data["error"] = {
                        "status_code": all_responses[i].dendrite.status_code,
                        "status_message": getattr(all_responses[i].dendrite, 'status_message', 'Unknown error')
                    }
                else:
                    response_data["error"] = {
                        "message": "Invalid response format",
                        "response_type": str(type(all_responses[i]))
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

    # 9) Upload to external endpoint (moved to a separate utils function)
    # Adjust endpoint URL/hotkey if needed
    results_json_string = json.dumps(results, sort_keys=True)
    
    hotkey = self.wallet.hotkey
    bt.logging.debug(f"🔑 Hotkey: {hotkey}")
    message_to_sign = f"Hotkey: {hotkey} \n timestamp: {timestamp} \n query_template: {query_template} \n query_labels: {query_labels}"
    signed_contents = sign_message(self.wallet, message_to_sign, output_file=None)
    results["signature"] = signed_contents

    upload_success = False
    #If for some reason uploading the data fails, we should just log it and continue. Server might go down but should not be a unique point of failure for the subnet
    try:
        bt.logging.info(f"Uploading data to: {MIID_SERVER}")
        upload_success = upload_data(MIID_SERVER, hotkey, results) 
        if upload_success:
            bt.logging.info("Data uploaded successfully to external server")
        else:
            bt.logging.error("Failed to upload data to external server")
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
        bt.logging.info(f"Finished quickly; sleeping for {EPOCH_MIN_TIME - (request_end - request_start)}s")
        await asyncio.sleep(EPOCH_MIN_TIME - (request_end - request_start))

    bt.logging.info("All batches processed, waiting 30 more seconds...")
    await asyncio.sleep(5)

    return True
