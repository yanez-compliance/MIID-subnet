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
Name Variation Validator Module

This module implements a Bittensor validator that queries miners for alternative
spellings of names. The validator sends requests to miners with a list of names
and a query template, then evaluates the quality of the variations returned by
each miner.

The validator follows these steps:
1. Select a random set of miners to query
2. Generate a list of random names to request variations for
3. Send the request to the selected miners
4. Evaluate the quality of the variations returned by each miner
5. Update the scores of the miners based on their performance

The evaluation considers factors such as:
- The number of variations provided
- The uniqueness of the variations
- The similarity of the variations to the original name

This validator helps incentivize miners to provide high-quality name variations
that can be used for various applications such as identity verification, name
matching, and data augmentation.
"""

import time
import traceback
import datetime as dt
import json
import base64
import wandb
import os
import shutil
import copy
from dotenv import load_dotenv
from pathlib import Path
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Load environment variables from .env file (e.g., vali.env)
# This will load WANDB_API_KEY if set in the file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), 'vali.env')) # it has the WANDB_API_KEY
# You might need to adjust the path if your .env file is elsewhere

# Remove offline mode
# os.environ["WANDB_MODE"] = "offline"

# Set wandb to not prompt, but still upload
#os.environ["WANDB_SILENT"] = "true"

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from MIID.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from MIID.validator import forward
from MIID.validator.forward import reset_phase4_state
# Import only what's needed from the validator module
# Import reward function if needed for metrics (or maybe just pass rewards to log_step)
from MIID.validator.reward import get_name_variation_rewards
import ollama
from MIID.validator.query_generator import QueryGenerator
from MIID.utils.sign_message import sign_message


class Validator(BaseValidatorNeuron):
    """
    Validator neuron for the name variation protocol.
    
    This validator sends requests to miners to generate alternative spellings for names,
    and rewards miners based on the quality of their responses. It evaluates miners on:
    
    1. Responsiveness - whether they respond to requests at all
    2. Completeness - whether they provide variations for all requested names
    3. Quantity - the number of variations provided (up to a reasonable limit)
    4. Quality - the uniqueness and similarity of variations to the original names
    
    The validator maintains a scoring system that rewards miners who consistently
    provide high-quality name variations, which helps to improve the overall
    quality of the network.
    """
    # llama3.1:latest is the default model for the validator
    # this is 8B model and it is faster than llama3.1: 70B or 405B models
    # is a more general-purpose model with enhanced reasoning and general knowledge capabilities, suitable for a broader range of applications.
    # it is a good balance between speed and quality
    # llama3.2:latest is 1B model and it is efficient for on device inference
    # llama3.3:latest is 70B model and it is the latest and most powerful model but require more memory and time to run
    # it is recommended to use llama3.1:latest for the validator
    # but you can try other models and see which one performs better for your use case
    
    DEFAULT_LLM_MODEL = "llama3.1:latest" # llama3.1:latest is the default model for the validator

    # Base URL for Flask app (same host as upload_data in forward.py)
    MIID_IMAGES_SERVER = os.environ.get("MIID_IMAGES_SERVER", "http://52.44.186.20:5000")

    def __init__(self, config=None):
        """
        Initialize the Name Variation Validator.
        
        This sets up the validator with configuration for the name variation protocol,
        including how many names to sample in each query.
        
        Args:
            config: Configuration object for the validator
        """
        bt.logging.info("Initializing Validator")

        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        
        # # Make sure self.config exists
        # if not hasattr(self, 'config') or self.config is None:
        #     bt.logging.warning("self.config is None, creating a new config object")
        #     self.config = bt.config()
        
        # # Configuration for the name variation protocol
        # # Create the name_variation config object if it doesn't exist
        # if not hasattr(self.config, 'name_variation') or self.config.name_variation is None:
        #     bt.logging.warning("name_variation config is None, creating a new config object")
        #     self.config.name_variation = bt.config()
        
        # # Explicitly create the sample_size attribute with a default value
        # # This is a more direct approach that should work regardless of the config object's implementation
        # self.config.name_variation.sample_size = 5
        
        # # Log the configuration to verify it's set correctly
        # bt.logging.info(f"Name variation sample size: {self.config.name_variation.sample_size}")

        # Initialize wandb run
        self.step = 0
        self.wandb_run = None # Initialize wandb_run as None
        
        # Auto-detect testnet and set wandb project name accordingly
        if (self.config.netuid == 322 and 
            hasattr(self.config, 'subtensor') and 
            hasattr(self.config.subtensor, 'network') and 
            self.config.subtensor.network == "test" and
            hasattr(self.config, 'subtensor') and 
            hasattr(self.config.subtensor, 'chain_endpoint') and 
            "test.finney.opentensor.ai" in self.config.subtensor.chain_endpoint):
            
            # Override wandb project name for testnet
            if hasattr(self.config, 'wandb'):
                original_project_name = getattr(self.config.wandb, 'project_name', 'MIID')
                self.config.wandb.project_name = "subnet322-test"
                bt.logging.info(f"Detected testnet configuration. Changing wandb project from '{original_project_name}' to 'subnet322-test'")
            else:
                bt.logging.warning("Wandb config not found, cannot set testnet project name")
        
        # Check if wandb is disabled via config
        if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'disable') and self.config.wandb.disable:
            bt.logging.info("Wandb is disabled via config. Skipping wandb initialization.")
        else:
            # Log the final wandb project name that will be used
            if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'project_name'):
                bt.logging.info(f"Wandb project name set to: {self.config.wandb.project_name}")
            else:
                bt.logging.warning("Wandb project name not found in config")
        # else:
        #     self.new_wandb_run() # Start the first wandb run - REMOVED: Each forward pass will create its own run

        # Initialize Ollama with the same approach as in miner.py
        if hasattr(self.config, 'neuron') and hasattr(self.config.neuron, 'ollama_model_name'):
            self.model_name = self.config.neuron.ollama_model_name
        else:
            self.model_name = self.DEFAULT_LLM_MODEL
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
        
        self._ensure_models_are_pulled()
        
        bt.logging.info(f"Using LLM model: {self.model_name}")
        
        # Initialize the query generator
        self.query_generator = QueryGenerator(self.config)
        
        # Download base images from Flask API using validator's hotkey (signed request)
        self._download_base_images_from_api()

        # Reset Phase 4 state on startup so each run starts from the beginning of the image/variation cycle
        phase4_state_path = Path(self.config.logging.logging_dir) / "validator_results" / "phase4_state.json"
        # reset_phase4_state(phase4_state_path)

        bt.logging.info("Ollama initialized")
        bt.logging.info(f"Using LLM model: {self.model_name}")
        bt.logging.info("Finished initializing Validator")
        bt.logging.info("----------------------------------")
        time.sleep(1)

    def _download_base_images_from_api(self):
        """Download base images from Flask API using validator's hotkey (signed request).
        
        Calls POST /images/<hotkey> with a signed message (same pattern as forward.py upload).
        Images are saved to MIID/validator/base_images directory.
        This only runs during initialization.
        """
        if not REQUESTS_AVAILABLE:
            bt.logging.warning("requests library not available. Skipping base images download.")
            return

        try:
            # Get the base images directory path (relative to repo root)
            repo_root = Path(__file__).parent.parent  # Go up from neurons/ to repo root
            base_images_dir = repo_root / "MIID" / "validator" / "base_images"
            base_images_dir.mkdir(parents=True, exist_ok=True)
            bt.logging.info(f"Base images directory: {base_images_dir}")

            # Ensure the folder is empty before requesting fresh images from API.
            for path in base_images_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            bt.logging.info("Cleared existing base images before API download")

            hotkey = self.wallet.hotkey
            hotkey_address = hotkey.ss58_address
            bt.logging.info(f"Requesting base images for hotkey: {hotkey_address}")

            # Sign message the same way as in forward.py for upload_data
            message_to_sign = (
                f"Hotkey: {hotkey} \n timestamp: {time.time()} \n request: base_images"
            )
            signed_contents = sign_message(self.wallet, message_to_sign, output_file=None)

            base_url = self.MIID_IMAGES_SERVER.rstrip("/")
            url = f"{base_url}/images/{hotkey_address}"
            payload = {"signature": signed_contents}

            response = requests.post(url, json=payload, timeout=30)
            if response.status_code != 200:
                bt.logging.warning(
                    f"Base images API returned {response.status_code}: {response.text[:200]}"
                )
                bt.logging.warning("Validator will continue with existing base images in the directory")
                return

            data = response.json()
            images = data.get("images") or []
            if not images:
                bt.logging.warning("No base images returned for this hotkey")
                bt.logging.warning("Validator will continue with existing base images in the directory")
                return

            downloaded_count = 0
            for item in images:
                filename = item.get("filename")
                b64 = item.get("data_base64")
                if not filename or not b64:
                    continue
                try:
                    raw = base64.standard_b64decode(b64)
                    image_path = base_images_dir / filename
                    image_path.write_bytes(raw)
                    bt.logging.info(f"Downloaded base image: {filename} ({len(raw)} bytes)")
                    downloaded_count += 1
                except Exception as e:
                    bt.logging.debug(f"Failed to save image {filename}: {e}")

            if downloaded_count > 0:
                bt.logging.info(f"Successfully downloaded {downloaded_count} base image(s) from API")
            else:
                bt.logging.warning("No images could be saved from API response")
                bt.logging.warning("Validator will continue with existing base images in the directory")

        except Exception as e:
            bt.logging.error(f"Error downloading base images from API: {e}")
            bt.logging.error(traceback.format_exc())
            bt.logging.warning("Validator will continue with existing base images in the directory")

    def _get_model_name_from_response(self, model_data: any) -> str:
        """Safely extract model name from ollama list response item."""
        if isinstance(model_data, dict):
            return model_data.get('name') or model_data.get('model')
        # For pydantic-like objects
        return getattr(model_data, 'name', getattr(model_data, 'model', None))

    def _ensure_models_are_pulled(self):
        """
        Ensures that the primary and all fallback models are available locally.
        """
        bt.logging.info("Ensuring all required Ollama models are available locally.")
        
        # Get primary model
        primary_model = getattr(self.config.neuron, 'ollama_model_name', "llama3.1:latest")
        
        # Get fallback models
        fallback_models = getattr(self.config.neuron, 'ollama_fallback_models', ['llama3.2:latest', 'tinyllama:latest'])
        
        # Also include judge model(s)
        primary_judge_model = getattr(self.config.neuron, 'ollama_judge_model', 'gemma3:latest')
        judge_fallback_models = getattr(self.config.neuron, 'ollama_judge_fallback_models', ['llama3.2:latest', 'tinyllama:latest'])

        # Build unique ordered list
        all_models = []
        for m in [primary_model, *fallback_models, primary_judge_model, *judge_fallback_models]:
            if m and m not in all_models:
                all_models.append(m)
        
        for model_name in all_models:
            try:
                bt.logging.info(f"Checking if model '{model_name}' is available locally.")
                response = ollama.list()
                
                model_is_pulled = False
                for model_data in response['models']:
                    if self._get_model_name_from_response(model_data) == model_name:
                        model_is_pulled = True
                        break

                if not model_is_pulled:
                    bt.logging.info(f"Model '{model_name}' not found locally. Pulling now...")
                    ollama.pull(model_name)
                    bt.logging.info(f"Successfully pulled model '{model_name}'.")
                else:
                    bt.logging.info(f"Model '{model_name}' is already available.")
            except Exception as e:
                bt.logging.error(f"Failed to check or pull model '{model_name}': {e}")
                # We log the error and continue. The validator might still be able to run with the models it has.
                # Consider whether to raise the exception if a model is critical.
    
    def manual_cleanup_wandb_runs(self):
        """Manually clean up all wandb run folders. Can be called anytime for maintenance."""
        bt.logging.info("Starting manual cleanup of all wandb run folders")
        self.cleanup_all_wandb_runs()
        bt.logging.info("Manual cleanup completed")

    def cleanup_all_wandb_runs(self):
        """Clean up all wandb run folders in the wandb directory."""
        # Check if cleanup is enabled via config
        cleanup_enabled = getattr(self.config.wandb, 'cleanup_runs', True)
        if not cleanup_enabled:
            bt.logging.debug("Wandb run cleanup is disabled via config. Skipping cleanup.")
            return
            
        try:
            wandb_dir = "wandb"
            if not os.path.exists(wandb_dir):
                bt.logging.debug("Wandb directory not found")
                return
            
            # Find all run directories
            run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
            
            if not run_dirs:
                bt.logging.debug("No wandb run directories found to clean up")
                return
            
            bt.logging.info(f"Found {len(run_dirs)} wandb run directories to clean up")
            
            # Delete all run directories
            for run_dir_name in run_dirs:
                run_dir_path = os.path.join(wandb_dir, run_dir_name)
                if os.path.exists(run_dir_path) and os.path.isdir(run_dir_path):
                    bt.logging.info(f"Cleaning up wandb run folder: {run_dir_path}")
                    shutil.rmtree(run_dir_path)
                    bt.logging.info(f"Successfully deleted wandb run folder: {run_dir_path}")
            
            # Also clean up the latest-run symlink if it exists
            latest_run_link = os.path.join(wandb_dir, "latest-run")
            if os.path.islink(latest_run_link):
                try:
                    os.unlink(latest_run_link)
                    bt.logging.debug("Removed latest-run symlink")
                except Exception as e:
                    bt.logging.debug(f"Could not remove latest-run symlink: {e}")
            
            bt.logging.info(f"Successfully cleaned up all {len(run_dirs)} wandb run folders")
                
        except Exception as e:
            bt.logging.error(f"Error cleaning up wandb run folders: {e}")
            bt.logging.debug(traceback.format_exc())

    def cleanup_wandb_run_folder(self, run_id=None):
        """Clean up the wandb run folder after the run is finished."""
        # Check if cleanup is enabled via config
        cleanup_enabled = getattr(self.config.wandb, 'cleanup_runs', True)
        if not cleanup_enabled:
            bt.logging.debug("Wandb run cleanup is disabled via config. Skipping cleanup.")
            return
            
        try:
            # Get the current wandb run directory
            if wandb.run and hasattr(wandb.run, 'dir'):
                run_dir = wandb.run.dir
            elif run_id:
                # If we have a run_id, construct the path
                run_dir = os.path.join("wandb", f"run-{run_id}")
            else:
                # Try to find the most recent run directory
                wandb_dir = "wandb"
                if os.path.exists(wandb_dir):
                    run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
                    if run_dirs:
                        # Sort by creation time and get the most recent
                        run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(wandb_dir, x)), reverse=True)
                        run_dir = os.path.join(wandb_dir, run_dirs[0])
                    else:
                        bt.logging.debug("No wandb run directories found to clean up")
                        return
                else:
                    bt.logging.debug("Wandb directory not found")
                    return
            
            # Check if the directory exists and is a wandb run directory
            if os.path.exists(run_dir) and os.path.isdir(run_dir):
                bt.logging.info(f"Cleaning up wandb run folder: {run_dir}")
                shutil.rmtree(run_dir)
                bt.logging.info(f"Successfully deleted wandb run folder: {run_dir}")
                
                # Also clean up the latest-run symlink if it exists
                latest_run_link = os.path.join("wandb", "latest-run")
                if os.path.islink(latest_run_link):
                    try:
                        os.unlink(latest_run_link)
                        bt.logging.debug("Removed latest-run symlink")
                    except Exception as e:
                        bt.logging.debug(f"Could not remove latest-run symlink: {e}")
            else:
                bt.logging.debug(f"Wandb run directory not found or not a directory: {run_dir}")
                
        except Exception as e:
            bt.logging.error(f"Error cleaning up wandb run folder: {e}")
            bt.logging.debug(traceback.format_exc())

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Check if wandb is disabled
        if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'disable') and self.config.wandb.disable:
            bt.logging.debug("Wandb is disabled. Skipping run creation.")
            self.wandb_run = None
            return
        
        # Create a unique run id for this run.
        run_id = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        wandb_name = "validator-" + str(self.uid) + "-" + run_id
        # Make sure to finish the previous run if it exists
        if self.wandb_run:
            try:
                bt.logging.info("Finishing previous wandb run before creating new one")
                self.wandb_run.finish()
                # Clean up all previous run folders after finishing
                self.cleanup_all_wandb_runs()
            except Exception as e:
                bt.logging.error(f"Error finishing previous wandb run: {e}")
            finally:
                self.wandb_run = None

        try:
            # Create the wandb run with connection to servers
            settings = wandb.Settings(
                # Prevent capturing the overall summary on run end
                x_server_side_derived_summary=True,
                # Suppress console logs if you'd like less noise
                silent=False,
                show_info=True,
                show_warnings=True,
                show_errors=True,
            )
            self.wandb_run = wandb.init(
                name=wandb_name,
                project=self.config.wandb.project_name,
                entity=self.config.wandb.entity,
                tags=["validation", "subnet54", "automated", "per-forward-pass"],
                group="neuron-validation-batch",
                job_type="validation",
                # Use anonymous="allow" instead of "must" to prefer API key auth when available
                anonymous="allow",
                config={
                    "uid": self.uid,
                    "hotkey": self.wallet.hotkey.ss58_address,
                    "run_name": run_id,
                    "version": self.spec_version,
                    # Add other relevant config from self.config
                    "sample_size": getattr(self.config.neuron, 'sample_size', None),
                    "batch_size": getattr(self.config.neuron, 'batch_size', None),
                    "timeout": getattr(self.config.neuron, 'timeout', None),
                    #"logging_dir": getattr(self.config.logging, 'logging_dir', None),
                },
                allow_val_change=True,
                reinit=True, # Allows reinitializing runs, useful with max_run_steps config
                settings=settings
            )

            bt.logging.info(f"Started new wandb run for forward pass: {wandb_name}")
            
            # Check if we're connected to the wandb servers
            if wandb.run and hasattr(wandb.run, 'mode') and wandb.run.mode == "online":
                bt.logging.info("Connected to wandb servers! Data will be uploaded.")
            elif wandb.run and hasattr(wandb.run, 'settings') and wandb.run.settings.mode == "online":
                bt.logging.info("Connected to wandb servers! Data will be uploaded.")
            elif wandb.run:
                bt.logging.info("wandb run initialized successfully.")
            else:
                bt.logging.warning("Not connected to wandb servers. Run may be in offline mode.")
                
        except Exception as e:
            bt.logging.error(f"Error initializing wandb: {str(e)}")
            bt.logging.error(traceback.format_exc())
            self.wandb_run = None  # Make sure it's set to None on error

    def log_step(
            self,
            uids,
            metrics, # Pass detailed metrics from forward
            rewards,
            extra_data=None # Optional dict for additional data from forward
    ):
        """Logs data for the current step to wandb, creating a new run if needed."""
        # Check if wandb is disabled
        if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'disable') and self.config.wandb.disable:
            bt.logging.debug("Wandb is disabled. Skipping log_step.")
            return
        
        # Check if wandb run is initialized
        if not self.wandb_run:
            bt.logging.warning("wandb_run not initialized. Skipping log_step.")
            # REMOVED: No longer create new runs here - each forward pass manages its own run
            return

        # Increment step count
        self.step += 1

        # Create a deep copy of metrics to avoid modifying the original object, which is used elsewhere
        metrics_for_wandb = copy.deepcopy(metrics)

        # Remove the verbose 'variations' list from the metrics copy to reduce log size
        for miner_metrics in metrics_for_wandb:
            if isinstance(miner_metrics, dict) and 'name_metrics' in miner_metrics:
                for name, name_data in miner_metrics['name_metrics'].items():
                    if isinstance(name_data, dict) and 'first_name' in name_data and isinstance(name_data.get('first_name'), dict) and 'metrics' in name_data['first_name']:
                        name_data['first_name']['metrics'].pop('variations', None)
                    if isinstance(name_data, dict) and 'last_name' in name_data and isinstance(name_data.get('last_name'), dict) and 'metrics' in name_data['last_name']:
                        name_data['last_name']['metrics'].pop('variations', None)

        # NOTE: Commented out automatic wandb run creation based on max_run_steps
        # since we now explicitly manage wandb runs to end after weights are set
        # # If we have already completed max_run_steps then we will complete the current wandb run and make a new one.
        # max_run_steps = self.config.wandb.max_run_steps
        # if self.step % max_run_steps == 0 and max_run_steps > 0:
        #     bt.logging.info(
        #         f"Validator has completed {self.step} run steps. Creating a new wandb run."
        #     )
        #     self.new_wandb_run()

        # Prepare logging data
        step_log = {
            "timestamp": time.time(),
            "uids": uids, # Assuming uids is already a list of ints
            "uid_metrics": {},
            **(extra_data or {}) # Include extra data passed from forward
        }

        # Populate metrics per UID
        for i, uid in enumerate(uids):
            uid_str = str(uid)
            step_log["uid_metrics"][uid_str] = {
                "uid": uid,
                "weight": float(self.scores[uid]) if uid < len(self.scores) else 0.0, # Ensure score exists
                "reward": float(rewards[i]) if i < len(rewards) else 0.0
            }
            # Add detailed metrics if available and correctly structured
            if i < len(metrics_for_wandb) and isinstance(metrics_for_wandb[i], dict):
                 step_log["uid_metrics"][uid_str].update(metrics_for_wandb[i])
            else:
                 # Log placeholder if metrics structure is unexpected
                 step_log["uid_metrics"][uid_str]["detailed_metrics_error"] = "Metrics structure invalid or missing"


        # Data specifically for graphing
        graphed_data = {
            #"block": self.metagraph.block.item(), # Ensure block is item()
            "average_reward": float(rewards.mean()) if hasattr(rewards, 'mean') else 0.0,
            "uid_rewards": {
                str(uids[i]): float(rewards[i]) for i in range(len(uids)) if i < len(rewards)
            },
            # "uid_weights": {
            #      str(uid): float(self.scores[uid]) for uid in uids if uid < len(self.scores)
            #  },
        }

        bt.logging.debug(f"Logging step_log keys: {list(step_log.keys())}")
        bt.logging.debug(f"Logging graphed_data keys: {list(graphed_data.keys())}")

        # Log data to wandb
        try:
            log_payload = {**graphed_data, "step_details": step_log}
            self.wandb_run.log(log_payload, step=self.step)
            bt.logging.info(f"Logged step {self.step} to Wandb")

            # # Log JSON results as artifact if path is provided
            # json_results_path = step_log.get("json_results_path")
            # if json_results_path and os.path.isfile(json_results_path):
            #     bt.logging.info(f"Logging results JSON as artifact: {json_results_path}")
            #     artifact_name = f"validator_results_step_{self.step}"
            #     artifact = wandb.Artifact(artifact_name, type="validation_results")
            #     artifact.add_file(json_results_path)
            #     self.wandb_run.log_artifact(artifact)
            #     bt.logging.info(f"Logged artifact {artifact_name}")
            # elif json_results_path:
            #     bt.logging.warning(f"Could not find results JSON file for artifact logging: {json_results_path}")

        except Exception as e:
             bt.logging.error(f"Error logging step {self.step} to Wandb: {e}")
             bt.logging.error(traceback.format_exc())

    async def forward(self):
        """
        Validator forward pass.
        
        This method is called periodically to query miners and evaluate their responses.
        The forward pass consists of:
        1. Generating a list of random names
        2. Querying the miners for name variations
        3. Rewarding the miners based on the quality of their responses
        
        The results of each query are saved to a file for analysis, and the
        scores of the miners are updated based on their performance.
        
        Returns:
            The result of the forward function from the MIID.validator module
        """
        try:
            # The forward function now handles the core logic
            # It will call self.log_step internally when needed
            res = await forward(self)
            return res
        except Exception as e:
            bt.logging.error("Got error in forward function")
            bt.logging.info(traceback.format_exc())
            return None

    async def build_queries(self):
        """Create test queries for miners using the QueryGenerator class"""
        return await self.query_generator.build_queries()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Validator running... {time.time()}")
            time.sleep(5)
