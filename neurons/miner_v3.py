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
from MIID.miner.dob_variations import generate_dobes_variations
import httpx
import asyncio
import json
from datetime import datetime

# Bittensor Miner Template:
from MIID.protocol import IdentitySynapse

# import base miner class which takes care of most of the boilerplate
from MIID.base.miner import BaseMinerNeuron

from bittensor.core.errors import NotVerifiedException

from datetime import datetime
import json
from MIID.miner.nvgen_service import is_latin_name, make_key
from MIID.validator.reward import get_name_variation_rewards


def _group_indices_by_country(addresses):
    """Returns (country->list_of_indices) and (index->country)."""
    idxs_by_cty = {}
    idx_to_cty = {}
    for i, addr in enumerate(addresses):
        idx_to_cty[i] = addr
        if addr not in idxs_by_cty:
            idxs_by_cty[addr] = []
        idxs_by_cty[addr].append(i)
    return idxs_by_cty, idx_to_cty

def _build_batch_map(countries_counts):
    """
    Build the /generate/batch payload in 'map' form:
    { "map": { "Country|": N, ... } }
    We leave city empty (country-only), using the strict formatter server-side.
    """
    mp = {}
    for country, n in countries_counts.items():
        if not country:
            continue
        key = f"{country}|"
        # FastAPI batch endpoint expects integer counts >=1
        if n > 0:
            mp[key] = int(n)
    return {"map": mp}

async def _post_json(client: httpx.AsyncClient, url: str, payload: dict, timeout_s: float):
    return await client.post(url, json=payload, timeout=timeout_s)

async def _write_json_async(path: str, obj):
    def _write():
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)
    await asyncio.to_thread(_write)

def _distribute_addresses_to_identities(
    addresses_by_country: dict,
    idxs_by_cty: dict,
    var_count: int,
) -> Dict[int, List[str]]:
    """
    Returns mapping: index -> list[str] of length var_count,
    pulling from the pool for that index's country.
    """
    per_idx_addrs = {}
    for country, idxs in idxs_by_cty.items():
        pool = addresses_by_country.get(country, [])
        # cycle per index so everyone gets var_count items
        for i, idx in enumerate(idxs):
            per_idx_addrs[idx] = pool[i*var_count :(i + 1)* var_count]
    return per_idx_addrs

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
    WHITELISTED_VALIDATORS = {
        "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54": "RoundTable21",
        "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr": "Yanez", 
        "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs": "Yuma",
        "5HbUFHW4XVhbQvMbSy7WDjvhHb62nuYgP1XBsmmz9E2E2K6p": "OpenTensor",
        "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54": "Rizzo",
        "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN": "TAO.com",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u": "tao.bot",
        "5GuPvuyKBJAWQbEGAkMbfRpG5qDqqhML8uDVSWoFjqcKKvDU": "Testnet_omar",
        "5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt": "Testnet_asem",
        "5Hmypa1isVpEwrQTYkGGKga54C13XnUj3fBJPHxH2etZkCF7": "Local Test Validator",
        "5DLLwfW9vw3c5Yw6V7FTPojMrojPsZyrHqrf9keusGMXcBUF": "Local Test Validator2",
        "5DZXBWkPedMDYoUrUNLA79naxrjZDCHseT5kfz9et3eeBMZU": "Local Test Validator3",
        "5CZi3t7LBUEoUWTqtwQnbjTT2LYXtNpEmCerXS5ZagzMEj9d": "Local Test Validator4",
        "5Ejk5HeFxruA61fSYE1pzupPf8893Fjq9EUDZxRKjSYG9oD2": "Testnet Validator"
    }

    def __init__(self, config=None):
        """
        Initialize the Name Variation Miner.
        
        Sets up the LLM client and creates directories for storing mining results.
        Each run will be saved in a separate directory with a unique timestamp.
        
        Args:
            config: Configuration object for the miner
        """
        super(Miner, self).__init__(config=config)
        
        # Create a directory for storing mining results
        # This helps with debugging and analysis
        self.output_path = os.path.join(self.config.logging.logging_dir, "mining_results")
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")
        # self.axon.verify_fns[IdentitySynapse.__name__] = self._verify_validator_request
        self.output_path = self.config.neuron.full_path
        bt.logging.info(f"Full path: {self.output_path}")
        bt.logging.info(f"NVGen url: {self.config.neuron.nvgen_url}")

    async def _verify_validator_request(self, synapse: IdentitySynapse) -> None:
        """
        Rejects any RPC that is not cryptographically proven to come from
        one of the whitelisted validator hotkeys.

        Signature *must* be present and valid.  If anything is missing or
        incorrect we raise `NotVerifiedException`, which the Axon middleware
        converts into a 401 reply.
        """
        # ----------  basic sanity checks  ----------
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey    = synapse.dendrite.hotkey
        # signature = synapse.dendrite.signature
        nonce     = synapse.dendrite.nonce
        uuid      = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        # 1 — is the sender even on our allow‑list?
        if hotkey not in self.WHITELISTED_VALIDATORS:
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        # 3 — run all the standard Bittensor checks (nonce window, replay,
        #     timeout, signature, …).  This *does not* insist on a signature,
        #     so we still do step 4 afterwards.
        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        bt.logging.info(
            f"Verifying message: {message}"
        )

        await self.axon.default_verify(synapse)

        # 5 — all good ➜ let the middleware continue
        bt.logging.info(
            f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})"
        )


    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        # ----- timing & bookkeeping -----
        raw_timeout = float(getattr(synapse, "timeout", 120.0))
        bt.logging.info(
            f"Request timeout: {raw_timeout:.1f}s for {len(synapse.identity)} names. "
            f"Validator: {synapse.dendrite.hotkey}"
        )
        overall_budget = max(10.0, raw_timeout - 10.0)
        started = time.time()

        try:
            validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            validator_uid = -1

        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        names = [iden[0] for iden in synapse.identity]
        dobes = [iden[1] for iden in synapse.identity]
        addrs = [iden[2] for iden in synapse.identity]

        # ----- run dir -----
        output_path = self.output_path
        output_path = os.path.expanduser(output_path) if output_path.startswith("~") else os.path.abspath(output_path)
        run_dir = os.path.join(output_path, f"validator_{validator_uid}", f"run_{current_datetime}")
        os.makedirs(run_dir, exist_ok=True)

        # ----- retry policy -----
        max_retries = 3
        base_delay = 1.0

        base_url = f'http://{getattr(self.config.neuron, "nvgen_url", "localhost:8000")}'
        addr_base_url = f'http://{getattr(self.config.neuron, "adgen_url", "144.76.38.143:8888")}'
        task_url = f'{base_url}/task'                    # name variations
        addr_batch_url = f'{addr_base_url}/generate/batch'    # address variations (batch)
        bt.logging.info(f"Task url: {task_url}")
        bt.logging.info(f"Address batch url: {addr_batch_url}")

        response_data = {}  # will fill on success

        # Precompute country groupings
        idxs_by_cty, idx_to_cty = _group_indices_by_country(addrs)
        bt.logging.info(f"idxs_by_cty: {idxs_by_cty}")

        for attempt in range(1, max_retries + 1):
            elapsed = time.time() - started
            remaining = overall_budget - elapsed
            if remaining <= 0:
                raise asyncio.TimeoutError("Overall timeout budget exhausted")

            per_attempt_budget = max(5.0, remaining)

            try:
                # One shared client per attempt
                async with httpx.AsyncClient(timeout=httpx.Timeout(per_attempt_budget - 1.0)) as client:
                    # 1) Fire name-task request
                    name_payload = {
                        "names": names,
                        "query_template": synapse.query_template,
                        "timeout": max(1.0, per_attempt_budget - 2.0),
                        "miner_uid": self.uid,
                        "validator_uid": validator_uid,
                    }
                    bt.logging.info(f"[attempt={attempt}] POST /task payload keys={list(name_payload.keys())}")

                    name_req = asyncio.create_task(_post_json(client, task_url, name_payload, per_attempt_budget - 0.5))

                    # 2) While waiting, we can prepare DOBs (CPU) and compute country counts
                    #    (variation_count needed later, but dobes can be ready)
                    dobes_task = asyncio.create_task(asyncio.to_thread(generate_dobes_variations, dobes, 11))

                    # 3) Await name response to learn variation_count
                    resp = await asyncio.wait_for(name_req, timeout=per_attempt_budget)
                    bt.logging.info(f"[attempt={attempt}] /task HTTP {resp.status_code}")
                    resp.raise_for_status()
                    data = resp.json()
                    try:
                        name_variations, metric, query_params = data
                    except Exception:
                        raise ValueError("Unexpected /task response shape; expected [name_variations, metric, query_params]")

                    variation_count = int(query_params.get("variation_count", 10))
                    bt.logging.info(f"variation_count={variation_count}")

                    # 4) Build batch counts per country and fire address-batch request
                    countries_counts = {}
                    for cty, idxs in idxs_by_cty.items():
                        if not cty:
                            continue
                        countries_counts[cty] = variation_count * len(idxs)
                    bt.logging.info(f"countries_counts: {countries_counts}")
                    addr_payload = _build_batch_map(countries_counts)

                    bt.logging.info(f"[attempt={attempt}] POST /generate/batch map_keys={list(addr_payload.get('map', {}).keys())[:6]}... total={len(addr_payload.get('map', {}))}")
                    addr_req = asyncio.create_task(_post_json(client, addr_batch_url, addr_payload, per_attempt_budget - 0.5))

                    # 5) Meanwhile finish DOB variations
                    dob_variations = await dobes_task

                    # 6) Await address response
                    addr_resp = await asyncio.wait_for(addr_req, timeout=per_attempt_budget)
                    bt.logging.info(f"[attempt={attempt}] /generate/batch HTTP {addr_resp.status_code}")
                    addr_resp.raise_for_status()
                    addr_data = addr_resp.json()
                    # addr_data: {"results": {"Country||": {"addresses": [...]}, ...}}
                    results_block = addr_data.get("results", {}) if isinstance(addr_data, dict) else {}
                    # Normalize: country key format in cache is "Country||"
                    pool_by_country = {}
                    for key, entry in results_block.items():
                        # key is "Country||" or "Country|City"
                        ctry = key.split("||", 1)[0] if "||" in key else key.split("|", 1)[0]
                        pool_by_country[ctry] = entry.get("addresses", []) or []

                    # 7) Distribute addresses per identity (by country)
                    per_idx_addrs = _distribute_addresses_to_identities(pool_by_country, idxs_by_cty, variation_count)

                    bt.logging.info(f"per_idx_addrs: {per_idx_addrs}")

                    # 8) Build final response
                    response_data = {}
                    for idx, (name, dob, original_addr) in enumerate(synapse.identity):
                        key = f"{name}"
                        response_data[key] = []

                        variations_for_name = name_variations.get(name, [])
                        dobs_for_dob = dob_variations.get(dob, [])

                        if not variations_for_name:
                            bt.logging.warning(f"No name variations for '{name}'")
                        if not dobs_for_dob:
                            bt.logging.warning(f"No DOB variations for '{dob}'")

                        # fallback: if we didn’t get a pool, use original address
                        addr_list_for_idx = per_idx_addrs.get(idx, [original_addr] * max(1, variation_count))

                        for j in range(min(len(variations_for_name), variation_count)):
                            name_var = variations_for_name[j]
                            dob_var = dobs_for_dob[j] if j < len(dobs_for_dob) else ""
                            if dob_var is None:
                                dob_var = ""
                            final_addr = addr_list_for_idx[j] if j < len(addr_list_for_idx) else original_addr
                            response_data[key].append([name_var, dob_var, final_addr])

                    synapse.variations = response_data

                    if True:
                        try:
                            seed_names = names
                            seed_dob = dobes
                            seed_addresses = addrs
                            # Avoid heavy transliteration path during miner-side evaluation
                            seed_script = ["latin" if is_latin_name(name) else "non-latin" for name in seed_names]
                            rewards_arr, detailed_metrics = get_name_variation_rewards(
                                None,
                                seed_names=seed_names,
                                seed_dob=seed_dob,
                                seed_addresses=seed_addresses,
                                seed_script=seed_script,
                                responses=[synapse],
                                uids=[self.uid],
                                variation_count=variation_count,
                                phonetic_similarity=query_params.get("phonetic_similarity"),
                                orthographic_similarity=query_params.get("orthographic_similarity"),
                                rule_based=query_params,
                            )
                            # Save evaluation results
                            try:
                                eval_path = os.path.join(run_dir, "reward_eval.json")
                                with open(eval_path, "w") as f:
                                    json.dump(
                                        {
                                            "rewards": [float(x) for x in rewards_arr.tolist()],
                                            "metrics": detailed_metrics,
                                        },
                                        f,
                                        indent=4,
                                    )
                            except Exception as e:
                                bt.logging.error(f"Error saving reward_eval.json: {e}")
                        except Exception as e:
                            import traceback
                            bt.logging.error(traceback.format_exc())
                            bt.logging.error(f"Reward evaluation failed: {e}")
                    # ----- optional: persist metrics & task payloads in background -----
                    try:
                        async def _write_files():
                            task_path = os.path.join(run_dir, "task.json")
                            metric_path = os.path.join(run_dir, "metric.json")
                            task_blob = {
                                "identity": synapse.identity,
                                "query_template": synapse.query_template,
                                "query_template_hash": make_key(names, synapse.query_template),
                                "query_params": query_params,
                                "timeout_budget": overall_budget,
                                "countries_counts": countries_counts,
                            }
                            await _write_json_async(task_path, task_blob)
                            await _write_json_async(metric_path, metric)

                        await _write_files()
                    except Exception as e:
                        bt.logging.error(f"Error saving artifacts: {e}")

                    # success -> break out
                    break

            except Exception as e:
                bt.logging.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    bt.logging.error("All attempts failed")
                    raise
                # jittered backoff
                delay = base_delay * (2 ** (attempt - 1)) + float(np.random.uniform(0, 1))
                delay = min(delay, max(0.0, overall_budget - (time.time() - started)))
                if delay > 0:
                    bt.logging.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)

        total_time = time.time() - started
        processed = len(getattr(synapse, "variations", {}))
        bt.logging.info(
            f"Request completed in {total_time:.2f}s of {overall_budget:.1f}s allowed. "
            f"Processed {processed}/{len(synapse.identity)} names. "
            f"Validator: {synapse.dendrite.hotkey}"
        )
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
        # Add ethical context and purpose explanation
        context_prompt = f"""IMPORTANT CONTEXT: This is for generating synthetic test data only.
Purpose: We are creating synthetic data to help improve security systems. This data will be used to:
1. Test system robustness
2. Identify potential vulnerabilities
3. Improve detection mechanisms
4. Generate training data for security systems

This is purely for defensive testing and system improvement. The data generated will not be used for any malicious purposes.

TASK: Based on this ethical context, please respond to the following query:

{prompt}

Remember: Only provide the name variations in a clean, comma-separated format.
"""

        # Use Ollama to query the LLM
        try:
            # Create Ollama client with configured URL
            client = ollama.Client(host=getattr(self.config.neuron, 'ollama_url', 'http://127.0.0.1:11434'))
            response = client.chat(
                self.model_name, 
                messages=[{
                    'role': 'user',
                    'content': context_prompt,
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
    
    def Clean_extra(self, payload: str, comma: bool, line: bool, space: bool, preserve_name_spaces: bool = False) -> str:
        """
        Clean the LLM output by removing unwanted characters.
        
        Args:
            payload: The text to clean
            comma: Whether to remove commas
            line: Whether to remove newlines
            space: Whether to remove spaces
            preserve_name_spaces: Whether to preserve spaces between names (for multi-part names)
        """
        # Remove punctuation and quotes
        payload = payload.replace(".", "")
        payload = payload.replace('"', "")
        payload = payload.replace("'", "")
        payload = payload.replace("-", "")
        payload = payload.replace("and ", "")
        
        # Handle spaces based on preservation flag
        if space:
            if preserve_name_spaces:
                # Replace multiple spaces with single space
                while "  " in payload:
                    payload = payload.replace("  ", " ")
            else:
                # Original behavior - remove all spaces
                payload = payload.replace(" ", "")
        
        if comma:
            payload = payload.replace(",", "")
        if line:
            payload = payload.replace("\\n", "")
        
        return payload.strip()

    def validate_variation(self, name: str, seed: str, is_multipart_name: bool) -> str:
        """
        Helper function to validate if a variation matches the seed name structure.
        
        Args:
            name: The variation to validate
            seed: The original seed name
            is_multipart_name: Whether the seed is a multi-part name
            
        Returns:
            str: The validated and cleaned variation, or np.nan if invalid
        """
        name = name.strip()
        if not name or name.isspace():
            return np.nan
        
        # Handle cases with colons (e.g., "Here are variations: Name")
        if ":" in name:
            name = name.split(":")[-1].strip()
        
        # Check length reasonability (variation shouldn't be more than 2x the seed length)
        if len(name) > 2 * len(seed):
            return np.nan
        
        # Check structure consistency with seed name
        name_parts = name.split()
        if is_multipart_name:
            # For multi-part seed names (e.g., "John Smith"), variations must also have multiple parts
            if len(name_parts) < 2:
                bt.logging.warning(f"Skipping single-part variation '{name}' for multi-part seed '{seed}'")
                return np.nan
        else:
            # For single-part seed names (e.g., "John"), variations must be single part
            if len(name_parts) > 1:
                bt.logging.warning(f"Skipping multi-part variation '{name}' for single-part seed '{seed}'")
                return np.nan
            
        return name

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
        
        The function ensures variations match the structure of the seed name:
        - Single-part seed names (e.g., "John") only get single-part variations
        - Multi-part seed names (e.g., "John Smith") only get multi-part variations
        
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
        
        # Extract and analyze the seed name structure
        seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
        seed_parts = seed.split()
        is_multipart_name = len(seed_parts) > 1
        seed = self.Clean_extra(seed, True, True, True, preserve_name_spaces=is_multipart_name)
        
        bt.logging.info(f"Processing seed name: '{seed}' (multipart: {is_multipart_name})")
        
        # Extract the response payload
        payload = splits[-1]
        
        # Case 1: Comma-separated list (preferred format)
        if len(payload.split(",")) > 3:  # Check if we have at least 3 commas
            # Clean the payload but keep commas for splitting
            payload = self.Clean_extra(payload, False, True, True, preserve_name_spaces=is_multipart_name)
            
            # Remove numbering prefixes
            for num in range(10):
                payload = payload.replace(str(num), "")
            
            # Split by comma and process each variation
            variations = []
            for name in payload.split(","):
                cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                if not pd.isna(cleaned_var):
                    variations.append(cleaned_var)
            
            if debug:
                return seed, "r1", variations, payload
            return seed, "r1", variations
        
        # Case 2 & 3: Non-comma separated formats
        else:
            # Case 2: Line-separated list
            len_ans = len(payload.split("\\n"))
            if len_ans > 2:  # Multiple lines indicate line-separated format
                # Clean the payload but preserve newlines for splitting
                payload = self.Clean_extra(payload, True, False, True, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                # Process line-separated variations
                variations = []
                for name in payload.split("\\n"):
                    cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                    if not pd.isna(cleaned_var):
                        variations.append(cleaned_var)
            
                if debug:
                    return seed, "r2", variations, payload
                return seed, "r2", variations
            
            # Case 3: Space-separated list
            else:
                # Clean the payload but preserve spaces for multi-part names
                payload = self.Clean_extra(payload, True, True, False, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                variations = []
                if is_multipart_name:
                    # For multi-part names, we need to carefully group the parts
                    current_variation = []
                    parts = payload.split()
                    
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        if ":" in part:  # New variation starts after colon
                            if current_variation:
                                # Process completed variation
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                            current_variation = [part.split(":")[-1].strip()]
                        else:
                            current_variation.append(part)
                            # Check if we have collected enough parts for a complete name
                            if len(current_variation) == len(seed_parts):
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                                current_variation = []
                
                    # Handle any remaining parts
                    if current_variation:
                        cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                else:
                    # For single-part names, simple space splitting is sufficient
                    for name in payload.split():
                        cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                
                if debug:
                    return seed, "r3", variations, payload
                return seed, "r3", variations

    async def blacklist(
        self, synapse: IdentitySynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        
        This function implements security checks to ensure that only authorized
        validators can query this miner. It verifies:
        1. Whether the request has a valid dendrite and hotkey
        2. Whether the hotkey is one of the ones on the white list
        
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

        if synapse.dendrite.hotkey not in self.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

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
