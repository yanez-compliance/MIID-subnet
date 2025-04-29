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
Name Variation Miner Module

This module implements a Bittensor miner that generates alternative spellings for names
using a local LLM (via Ollama). 
######### Ollama should be installed and running on the machine. ########
The miner receives requests from validators containing
a list of names and a query template, processes each name through the LLM, extracts
the variations from the LLM's response, and returns them to the validator.

The miner follows these steps:
1. Receive a request with names and a query template
2. For each name, query the LLM to generate variations
3. Process the LLM responses to extract clean variations
4. Return the variations to the validator

The processing logic handles different response formats from LLMs, including:
- Comma-separated lists
- Line-separated lists
- Space-separated lists with numbering

For debugging and analysis, the miner also saves:
- Raw LLM responses
- Processed variations in JSON format
- A pandas DataFrame with the variations

Each mining run is saved with a unique timestamp identifier to distinguish between
different runs and facilitate analysis of results over time.
"""

import time
import typing
import bittensor as bt
import ollama
import pandas as pd
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

# Bittensor Miner Template:
from MIID.protocol import IdentitySynapse

# import base miner class which takes care of most of the boilerplate
from MIID.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Name Variation Miner Neuron
    
    This miner receives requests from validators to generate alternative spellings for names,
    and responds with variations generated using a local LLM (via Ollama).
    
    The miner handles the following tasks:
    - Processing incoming requests for name variations
    - Querying a local LLM to generate variations
    - Extracting and cleaning variations from LLM responses
    - Returning the processed variations to the validator
    - Saving intermediate results for debugging and analysis
    
    Each mining run is saved with a unique timestamp identifier to distinguish between
    different runs and facilitate analysis of results over time.
    
    Configuration:
    - model_name: The Ollama model to use (default: 'tinyllama:latest')
    - output_path: Directory for saving mining results (default: logging_dir/mining_results)
    """

    def __init__(self, config=None):
        """
        Initialize the Name Variation Miner.
        
        Sets up the LLM client and creates directories for storing mining results.
        Each run will be saved in a separate directory with a unique timestamp.
        
        Args:
            config: Configuration object for the miner
        """
        super(Miner, self).__init__(config=config)
        
        # Initialize the LLM client
        # You can override this in your config by setting model_name
        # Ensure we have a valid model name, defaulting to llama3.2:1b if not specified
        self.model_name = getattr(self.config, 'model_name', None)
        if self.model_name is None:
            self.model_name = 'llama3.2:1b'
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
        
        bt.logging.info(f"Using LLM model: {self.model_name}")
        
        # Check if Ollama is available
        try:
            # Check if model exists locally first
            models = ollama.list().get('models', [])
            model_exists = any(model.get('name') == self.model_name for model in models)
            
            if model_exists:
                bt.logging.info(f"Model {self.model_name} already pulled")
            else:
                # Model not found locally, pull it
                bt.logging.info(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
        except Exception as e:
            bt.logging.error(f"Error with Ollama: {str(e)}")
            bt.logging.error("Make sure Ollama is installed and running on this machine")
            bt.logging.error("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            bt.logging.error("Start Ollama: ollama serve")
            raise RuntimeError("Ollama is required for this miner. Please install and start Ollama.")
        
        # Create a directory for storing mining results
        # This helps with debugging and analysis
        self.output_path = os.path.join(self.config.logging.logging_dir, "mining_results")
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")

    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        """
        Process a name variation request by generating variations for each name.
        
        This is the main entry point for the miner's functionality. It:
        1. Receives a request with names and a query template
        2. Processes each name through the LLM
        3. Extracts variations from the LLM responses
        4. Returns the variations to the validator
        
        Each run is assigned a unique timestamp ID and results are saved in a
        dedicated directory for that run.
        
        Args:
            synapse: The IdentitySynapse containing names and query template
            
        Returns:
            The synapse with variations field populated with name variations
        """
        # Generate a unique run ID using timestamp
        run_id = int(time.time())
        bt.logging.info(f"Starting run {run_id} for {len(synapse.names)} names")
        
        # Get timeout from synapse (default to 120s if not specified)
        timeout = getattr(synapse, 'timeout', 120.0)
        bt.logging.info(f"Request timeout: {timeout:.1f}s for {len(synapse.names)} names")
        start_time = time.time()
        
        # Create a run-specific directory
        run_dir = os.path.join(self.output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # This will store all responses from the LLM in a format that can be processed later
        # Format: ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
        Response_list = []
        
        # Track which names we've processed
        processed_names = []
        
        # Process each name in the request, respecting the timeout
        for name in tqdm(synapse.names, desc="Processing names"):
            # Check if we're approaching the timeout (reserve 15% for processing)
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            time_buffer = timeout * 0.15  # Reserve 15% of total time for final processing
            
            # If time is running out, skip remaining names
            if remaining < time_buffer:
                bt.logging.warning(
                    f"Time limit approaching ({elapsed:.1f}/{timeout:.1f}s), "
                    f"processed {len(processed_names)}/{len(synapse.names)} names. "
                    f"Skipping remaining names to ensure timely response."
                )
                break
            
            # Format the response list for later processing
            Response_list.append("Respond")
            Response_list.append("---")
            Response_list.append("Query-" + name)
            Response_list.append("---")
            
            # Format the query with the current name
            formatted_query = synapse.query_template.replace("{name}", name)
            
            # Query the LLM with timeout awareness
            try:
                bt.logging.info(f"Generating variations for name: {name}, remaining time: {remaining:.1f}s")
                # Pass a more limited timeout to the LLM call to ensure we stay within bounds
                name_respond = self.Get_Respond_LLM(formatted_query)
                Response_list.append(name_respond)
                processed_names.append(name)
            except Exception as e:
                bt.logging.error(f"Error querying LLM for name {name}: {str(e)}")
                Response_list.append("Error: " + str(e))
        
        # Check if we've managed to process at least some names
        if not processed_names:
            bt.logging.error("Could not process any names within the timeout period")
            synapse.variations = {}
            return synapse
        
        # Process the responses to extract variations, but be aware of remaining time
        remaining = timeout - (time.time() - start_time)
        bt.logging.info(f"Processing responses with {remaining:.1f}s remaining of {timeout:.1f}s timeout")
        
        # Only proceed with processing if we have enough time
        if remaining > 1.0:  # Ensure at least 1 second for processing
            variations = self.process_variations(Response_list, run_id, run_dir)
            bt.logging.info(f"======== FINAL VARIATIONS===============================================: {variations}")
            # Set the variations in the synapse for return to the validator
            synapse.variations = variations
        else:
            bt.logging.warning(f"Insufficient time for processing responses, returning empty result")
            synapse.variations = {}
        
        # Log final timing information
        total_time = time.time() - start_time
        bt.logging.info(
            f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed. "
            f"Processed {len(processed_names)}/{len(synapse.names)} names."
        )
        
        bt.logging.info(f"======== SYNAPSE VARIATIONS===============================================: {synapse.variations}")
        bt.logging.info(f"==========================Processed variations for {len(synapse.variations)} names in run {run_id}")
        bt.logging.info(f"==========================Synapse: {synapse}")
        bt.logging.info("========================================================================================")
        return synapse
    
    def Get_Respond_LLM(self, prompt: str) -> str:
        """
        Query the LLM using Ollama.
        
        This function sends a prompt to the LLM and returns its response.
        It uses the Ollama client to communicate with a locally running LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If there's an error communicating with the LLM
        """
        # Use Ollama to query the LLM
        try:
            response = ollama.chat(
                self.model_name, 
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
                options={
                    # Add a reasonable timeout to ensure we don't get stuck
                    "num_predict": 1024
                }
            )
            
            # Extract and return the content of the response
            return response['message']['content']
        except Exception as e:
            bt.logging.error(f"LLM query failed: {str(e)}")
            raise
    
    def process_variations(self, Response_list: List[str], run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """
        Process LLM responses to extract name variations.
        
        This function takes the raw LLM responses and extracts the name variations
        using the Process_function. It handles the parsing and cleaning of the
        LLM outputs, ensuring that all variations are properly cleaned before
        being returned or saved.
        
        Args:
            Response_list: List of LLM responses in the format:
                          ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
            
        Returns:
            Dictionary mapping each name to its list of variations
        """
        bt.logging.info(f"Processing {len(Response_list)} responses")
        # Split the responses by "Respond" to get individual responses
        Responds = "".join(Response_list).split("Respond")
        
        # Create a dictionary to store each name and its variations
        name_variations = {}
        
        # Process each response to extract variations
        for i in range(1, len(Responds)):
            try:
                # Process the response to extract the name and variations
                # Returns: (seed_name, processing_method, variations_list)
                llm_respond = self.Process_function(Responds[i], False)
                
                # Extract the seed name and variations
                name = llm_respond[0]
                
                # Filter out empty or NaN variations
                variations = [var for var in llm_respond[2] if not pd.isna(var) and var != ""]
                
                # Clean each variation before storing
                cleaned_variations = []
                for var in variations:
                    # Remove unwanted characters
                    cleaned_var = var.replace(")", "").replace("(", "").replace("]", "").replace("[", "").replace(",", "")
                    # Remove leading/trailing whitespace
                    cleaned_var = cleaned_var.strip()
                    # Only add non-empty variations
                    if cleaned_var:
                        cleaned_variations.append(cleaned_var)
                
                # Store the cleaned variations for this name
                name_variations[name] = cleaned_variations
                bt.logging.info(f"=================== Name variations: {name_variations}")
                
                bt.logging.info(f"Processed {len(cleaned_variations)} variations for {name}")
            except Exception as e:
                bt.logging.error(f"Error processing response {i}: {e}")
        
        # # Save processed variations to JSON for debugging and analysis
        # self.save_variations_to_json(name_variations, run_id, run_dir)
        
        return name_variations
    
    def save_variations_to_json(self, name_variations: Dict[str, List[str]], run_id: int, run_dir: str) -> None:
        """
        Save processed variations to JSON and DataFrame for debugging and analysis.
        
        This function saves the processed variations in multiple formats:
        1. A pandas DataFrame saved as a pickle file in the run-specific directory
        2. A JSON file with the name variations in the run-specific directory
        3. A JSON file with the model name and run ID in the main output directory
        
        Each file is named with the run ID to distinguish between different runs.
        
        Args:
            name_variations: Dictionary mapping names to variations
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
        """
        bt.logging.info(f"=================== Name variations: {name_variations}")
        bt.logging.info(f"=================== Run ID: {run_id}")
        bt.logging.info(f"=================== Run directory: {run_dir}")
        bt.logging.info("Saving variations to JSON and DataFrame")

        # Find the maximum number of variations for any name
        max_variations = max([len(vars) for vars in name_variations.values()]) if name_variations else 0
        bt.logging.info(f"Maximum number of variations found: {max_variations}")
        
        # Create a DataFrame with columns for the name and each variation
        columns = ['Name'] + [f'Var_{i+1}' for i in range(max_variations)]
        result_df = pd.DataFrame(columns=columns)
        
        # Fill the DataFrame with names and their variations, padding with empty strings if needed
        for i, (name, variations) in enumerate(name_variations.items()):
            row_data = [name] + variations + [''] * (max_variations - len(variations))
            result_df.loc[i] = row_data
        
        # Note: We no longer need to clean the data here since it's already cleaned
        # in the process_variations function
        
        # Save DataFrame to pickle for backup and analysis
        # Include run_id in the filename
        #df_path = os.path.join(run_dir, f"variations_df_{run_id}.pkl")
        #result_df.to_pickle(df_path)
        
        # Convert DataFrame to JSON format
        json_data = {}
        for i, row in result_df.iterrows():
            name = row['Name']
            # Extract non-empty variations
            variations = [var for var in row[1:] if var != ""]
            json_data[name] = variations
        
        # Save to JSON file
        # Include run_id in the filename
        # json_path = os.path.join(run_dir, f"variations_{run_id}.json")
        # import json
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=4)
        # bt.logging.info(f"Saved variations to: {json_path}")
        # bt.logging.info(f"DataFrame shape: {result_df.shape} with {max_variations} variation columns")
    
    def Clean_extra(self, payload: str, comma: bool, line: bool, space: bool) -> str:
        """
        Clean the LLM output by removing unwanted characters.
        
        This function removes various characters from the LLM output to make it
        easier to parse and extract the name variations. It can selectively
        remove commas, newlines, and spaces based on the parameters.
        
        Args:
            payload: The text to clean
            comma: Whether to remove commas
            line: Whether to remove newlines
            space: Whether to remove spaces
            
        Returns:
            The cleaned text
        """
        # Remove punctuation and quotes
        payload = payload.replace(".", "")
        payload = payload.replace('"', "")
        payload = payload.replace("'", "")
        payload = payload.replace("-", "")
        payload = payload.replace("and ", "")
        
        # Optionally remove spaces, commas, and newlines
        if space:
            payload = payload.replace(" ", "")
        if comma:
            payload = payload.replace(",", "")
        if line:
            payload = payload.replace("\\n", "")
            
        return payload
    
    def Process_function(self, string: str, debug: bool) -> Tuple[str, str, List[str], Optional[str]]:
        """
        Process the LLM response to extract the seed name and variations.
        
        This function parses the LLM response to extract:
        1. The original seed name
        2. The list of name variations
        
        It handles different response formats from LLMs:
        - Comma-separated lists (preferred format)
        - Line-separated lists
        - Space-separated lists with numbering
        
        The function is flexible and can handle any number of variations, not just 10.
        
        Args:
            string: The LLM response in the format:
                   "---\nQuery-{name}\n---\n{response}"
            debug: Whether to return debug information
            
        Returns:
            Tuple containing:
            - seed_name: The original name
            - processing_method: The method used to process the response (r1, r2, or r3)
            - variations_list: The list of extracted variations
            - payload: (if debug=True) The processed payload
        """
        # Split the response by "---" to extract the query and response parts
        splits = string.split('---')
        
        # Extract the seed name from the query part
        seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
        seed = self.Clean_extra(seed, True, True, True)  # Clean the original seed name
        
        # Extract the response payload
        payload = splits[-1]
        
        # Case 1: Comma-separated list (preferred format)
        if len(payload.split(",")) > 3:  # Check if we have at least 3 commas
            # Clean the payload but keep commas
            payload = self.Clean_extra(payload, False, True, True)
            
            # Remove numbering
            for num in range(10):
                payload = payload.replace(str(num), "")
                
            # Split by comma and take all variations
            name_list = list(payload.split(",")[1:])  # Skip the first element (often empty or contains intro text)
            
            # Clean each variation
            Cleaned_name_list = []
            for name in name_list:
                # Handle case where LLM includes a prefix like "Here are 10 alternative spellings for the name Rowena: Rowenna"
                if ":" in name:
                    c_name = name.split(":")[-1]
                    Cleaned_name_list.append(c_name)
                # Skip if the variation is too long compared to the original (likely an error)
                elif len(name) > 2*len(seed):
                    Cleaned_name_list.append(np.nan)
                else:
                    Cleaned_name_list.append(name)
                    
            # Return the results - we accept any number of variations
            if debug:
                return seed, "r1", Cleaned_name_list, payload
            else:
                return seed, "r1", Cleaned_name_list
        
        # Case 2 & 3: Non-comma separated formats
        else:
            # Case 2: Line-separated list
            len_ans = len(payload.split("\\n"))
            if len_ans > 2:  # Multiple lines, use this to separate the names
                # Clean the payload but keep newlines
                payload = self.Clean_extra(payload, True, False, True)
                
                # Remove numbering
                for num in range(10):
                    payload = payload.replace(str(num), "")
                    
                # Split by newline and take all variations
                name_list = list(payload.split("\\n"))
                
                # Clean each variation
                Cleaned_name_list = []
                for name in name_list:
                    if ":" in name:
                        c_name = name.split(":")[-1]
                        Cleaned_name_list.append(c_name)
                    elif len(name) > 2*len(seed):
                        Cleaned_name_list.append(np.nan)
                    else:
                        Cleaned_name_list.append(name)
                        
                # Return the results
                if debug:
                    return seed, "r2", Cleaned_name_list, payload
                else:
                    return seed, "r2", Cleaned_name_list
            
            # Case 3: Space-separated list with numbering
            else: 
                # Clean the payload but keep spaces
                payload = self.Clean_extra(payload, True, True, False)
                
                # Remove numbering
                for num in range(10):
                    payload = payload.replace(str(num), "")
                    
                # Split by space
                name_list = list(payload.split(" "))
                
                # Clean each variation
                Cleaned_name_list = []
                for name in name_list:
                    if ":" in name:
                        c_name = name.split(":")[-1]
                        Cleaned_name_list.append(c_name)
                    elif len(name) > 2*len(seed):
                        Cleaned_name_list.append(np.nan)
                    elif len(name) != 0:  # Skip empty strings
                        Cleaned_name_list.append(name)
                        
                # Return the results
                if debug:
                    return seed, "r3", Cleaned_name_list, payload
                else:
                    return seed, "r3", Cleaned_name_list

    async def blacklist(
        self, synapse: IdentitySynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        
        This function implements security checks to ensure that only authorized
        validators can query this miner. It verifies:
        1. Whether the request has a valid dendrite and hotkey
        2. Whether the hotkey is registered in the metagraph
        3. Whether the hotkey has validator permissions (if required)
        
        Args:
            synapse: A IdentitySynapse object constructed from the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether the request should be blacklisted
                - str: The reason for the decision
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # Get the UID of the sender
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        # Check if the hotkey is registered in the metagraph
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Check if the hotkey has validator permissions (if required)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        # If all checks pass, allow the request
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: IdentitySynapse) -> float:
        """
        The priority function determines the order in which requests are handled.
        
        This function assigns a priority to each request based on the stake of the
        calling entity. Requests with higher priority are processed first, which
        ensures that validators with more stake get faster responses.
        
        Args:
            synapse: The IdentitySynapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.
                  Higher values indicate higher priority.
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # Get the UID of the caller
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )
        
        # Use the stake as the priority
        # Higher stake = higher priority
        priority = float(
            self.metagraph.S[caller_uid]
        )
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Miner running... {time.time()}")
            time.sleep(30)
