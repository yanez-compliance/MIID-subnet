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
Face Variation Validator Module

This validator sends base face images to miners and asks them to generate
face variations (pose, lighting, expression, background, screen_replay). The
validator evaluates submitted image variations via an external grading API
and rewards miners based on image quality (KAV) and reputation (UAV).
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

load_dotenv(dotenv_path=os.path.join(os.getcwd(), 'vali.env'))

import bittensor as bt

from MIID.base.validator import BaseValidatorNeuron
from MIID.validator import forward
from MIID.validator.forward import reset_phase4_state
from MIID.utils.sign_message import sign_message


class Validator(BaseValidatorNeuron):
    """
    Face Variation Validator Neuron.

    Sends base face images to miners, receives S3 submission references,
    and rewards miners based on image variation quality (KAV, 10%) and
    reputation (UAV, 90%).
    """

    MIID_IMAGES_SERVER = os.environ.get("MIID_IMAGES_SERVER", "http://52.44.186.20:5000")

    def __init__(self, config=None):
        bt.logging.info("Initializing Validator")

        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.step = 0
        self.wandb_run = None

        # Auto-detect testnet and override wandb project name accordingly
        if (self.config.netuid == 322
                and hasattr(self.config, 'subtensor')
                and hasattr(self.config.subtensor, 'network')
                and self.config.subtensor.network == "test"
                and hasattr(self.config.subtensor, 'chain_endpoint')
                and "test.finney.opentensor.ai" in self.config.subtensor.chain_endpoint):
            if hasattr(self.config, 'wandb'):
                original_project_name = getattr(self.config.wandb, 'project_name', 'MIID')
                self.config.wandb.project_name = "subnet322-test"
                bt.logging.info(
                    f"Detected testnet. Changing wandb project from '{original_project_name}' to 'subnet322-test'"
                )

        if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'disable') and self.config.wandb.disable:
            bt.logging.info("Wandb is disabled via config.")
        else:
            if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'project_name'):
                bt.logging.info(f"Wandb project name: {self.config.wandb.project_name}")

        # Download base images from Flask API (signed request)
        self._download_base_images_from_api()

        # Reset Phase 4 cycle state on startup
        phase4_state_path = Path(self.config.logging.logging_dir) / "validator_results" / "phase4_state.json"
        reset_phase4_state(phase4_state_path)

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

            # Clear existing images before downloading fresh ones
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
            bt.logging.warning("Continuing with existing base images in directory")

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

    def log_step(self, uids, metrics, rewards, extra_data=None):
        """Logs data for the current step to wandb."""
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
                "weight": float(self.scores[uid]) if uid < len(self.scores) else 0.0,
                "reward": float(rewards[i]) if i < len(rewards) else 0.0,
            }
            # Add detailed metrics if available and correctly structured
            if i < len(metrics_for_wandb) and isinstance(metrics_for_wandb[i], dict):
                step_log["uid_metrics"][uid_str].update(metrics_for_wandb[i])


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

        try:
            log_payload = {**graphed_data, "step_details": step_log}
            self.wandb_run.log(log_payload, step=self.step)
            bt.logging.info(f"Logged step {self.step} to Wandb")
        except Exception as e:
            bt.logging.error(f"Error logging step {self.step} to Wandb: {e}")
            bt.logging.error(traceback.format_exc())

    async def forward(self):
        """
        Validator forward pass.

        Creates an image variation challenge, queries miners for their S3
        submissions, grades submissions via the external API (KAV), and
        combines with reputation scores (UAV) to update miner weights.
        """
        try:
            res = await forward(self)
            return res
        except Exception as e:
            bt.logging.error("Got error in forward function")
            bt.logging.info(traceback.format_exc())
            return None


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Validator running... {time.time()}")
            time.sleep(5)
