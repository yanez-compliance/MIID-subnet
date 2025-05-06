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
import wandb
import os
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., vali.env)
# This will load WANDB_API_KEY if set in the file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), 'vali.env')) 
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
# Import wandb constants from the validator module
from MIID.validator import WANDB_PROJECT, WANDB_ENTITY, MAX_RUN_STEPS_PER_WANDB_RUN
# Import reward function if needed for metrics (or maybe just pass rewards to log_step)
from MIID.validator.reward import get_name_variation_rewards
import ollama
from MIID.validator.query_generator import QueryGenerator

# Define version (replace with actual version logic if available)
__version__ = "0.0.1"

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
    
    DEFAULT_LLM_MODEL = "llama3.1:latest"

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
        self.new_wandb_run() # Start the first wandb run

        # Initialize Ollama with the same approach as in miner.py
        if hasattr(self.config, 'neuron') and hasattr(self.config.neuron, 'ollama_model_name'):
            self.model_name = self.config.neuron.ollama_model_name
        else:
            self.model_name = self.DEFAULT_LLM_MODEL
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
        
        bt.logging.info(f"Using LLM model: {self.model_name}")
        
        # Check if Ollama is available
        try:
            # Check if model exists locally first
            models_response = ollama.list()
            models = models_response.get('models', [])
            bt.logging.info(f"Ollama models response: {models_response}") # Log the raw response
            
            # Robust check for model name
            model_exists = False
            if isinstance(models, list):
                for model_info in models:
                    # Check if model_info is a dict and has 'name'
                    if isinstance(model_info, dict) and model_info.get('name') == self.model_name:
                        model_exists = True
                        break 
            else:
                bt.logging.warning(f"Unexpected format for ollama models list: {type(models)}")

            if model_exists:
                bt.logging.info(f"Model {self.model_name} already pulled")
            else:
                # Model not found locally, pull it
                bt.logging.info(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
        except Exception as e:
            bt.logging.error(f"Error initializing Ollama: {e}")
            raise e
        
        bt.logging.info("Ollama initialized")
        bt.logging.info(f"Using LLM model: {self.model_name}")
        bt.logging.info("Finished initializing Validator")
        bt.logging.info("----------------------------------")
        time.sleep(1)

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        run_id = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        wandb_name = "validator-" + str(self.uid) + "-" + run_id
        # Make sure to finish the previous run if it exists
        if self.wandb_run:
            self.wandb_run.finish()

        try:
            # Create the wandb run with connection to servers
            self.wandb_run = wandb.init(
                name=wandb_name,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                tags=["validation", "subnet322", "automated"],
                group="neuron-validation-batch",
                job_type="validation",
                # Use anonymous="allow" instead of "must" to prefer API key auth when available
                anonymous="allow",
                config={
                    "uid": self.uid,
                    "hotkey": self.wallet.hotkey.ss58_address,
                    "run_name": run_id,
                    "version": __version__,
                    # Add other relevant config from self.config
                    "sample_size": getattr(self.config.neuron, 'sample_size', None),
                    "batch_size": getattr(self.config.neuron, 'batch_size', None),
                    "timeout": getattr(self.config.neuron, 'timeout', None),
                    "logging_dir": getattr(self.config.logging, 'logging_dir', None),
                },
                allow_val_change=True,
                reinit=True # Allows reinitializing runs, useful with MAX_RUN_STEPS_PER_WANDB_RUN
            )

            bt.logging.info(f"Started new wandb run: {name}")
            
            # Check if we're connected to the wandb servers
            if wandb.run and wandb.run.mode == "online":
                bt.logging.info("Connected to wandb servers! Data will be uploaded.")
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
        # Check if wandb run is initialized
        if not self.wandb_run:
            bt.logging.warning("wandb_run not initialized. Skipping log_step.")
            self.new_wandb_run() # Attempt to start a new run
            if not self.wandb_run: # If still not initialized, return
                 bt.logging.error("Failed to initialize wandb run in log_step.")
                 return

        # Increment step count
        self.step += 1

        # If we have already completed MAX_RUN_STEPS_PER_WANDB_RUN steps then we will complete the current wandb run and make a new one.
        if self.step % MAX_RUN_STEPS_PER_WANDB_RUN == 0 and MAX_RUN_STEPS_PER_WANDB_RUN > 0:
            bt.logging.info(
                f"Validator has completed {self.step} run steps. Creating a new wandb run."
            )
            self.new_wandb_run()

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
            if i < len(metrics) and isinstance(metrics[i], dict):
                 step_log["uid_metrics"][uid_str].update(metrics[i])
            else:
                 # Log placeholder if metrics structure is unexpected
                 step_log["uid_metrics"][uid_str]["detailed_metrics_error"] = "Metrics structure invalid or missing"


        # Data specifically for graphing
        graphed_data = {
            "block": self.metagraph.block.item(), # Ensure block is item()
            "average_reward": float(rewards.mean()) if hasattr(rewards, 'mean') else 0.0,
            "uid_rewards": {
                str(uids[i]): float(rewards[i]) for i in range(len(uids)) if i < len(rewards)
            },
            "uid_weights": {
                 str(uid): float(self.scores[uid]) for uid in uids if uid < len(self.scores)
             },
        }

        bt.logging.debug(f"Logging step_log keys: {list(step_log.keys())}")
        bt.logging.debug(f"Logging graphed_data keys: {list(graphed_data.keys())}")

        # Log data to wandb
        try:
            log_payload = {**graphed_data, "step_details": step_log}
            self.wandb_run.log(log_payload, step=self.step)
            bt.logging.info(f"Logged step {self.step} to Wandb")

            # Log JSON results as artifact if path is provided
            json_results_path = step_log.get("json_results_path")
            if json_results_path and os.path.isfile(json_results_path):
                bt.logging.info(f"Logging results JSON as artifact: {json_results_path}")
                artifact_name = f"validator_results_step_{self.step}"
                artifact = wandb.Artifact(artifact_name, type="validation_results")
                artifact.add_file(json_results_path)
                self.wandb_run.log_artifact(artifact)
                bt.logging.info(f"Logged artifact {artifact_name}")
            elif json_results_path:
                bt.logging.warning(f"Could not find results JSON file for artifact logging: {json_results_path}")

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
            time.sleep(60)
