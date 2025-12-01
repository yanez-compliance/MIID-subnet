# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
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

import os
import subprocess
import argparse
import bittensor as bt
from .logging import setup_events_logger


def is_cuda_available():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT
        )
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=360,## MIID: 360 blocks = 4320 seconds = 72 minutes
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--neuron.model_name",
        type=str,
        help="The Ollama model to use (default: tinyllama:latest)",
        default="tinyllama:latest",
    )

    parser.add_argument(
        "--neuron.ollama_url",
        type=str,
        help="Url to ollama",
        default="http://127.0.0.1:11434",
    )

    parser.add_argument(
        "--neuron.ollama_request_timeout",
        type=int,
        help="Timeout for the Ollama request in seconds.",
        default=60,
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=360,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.use_default_query",
        action="store_true",
        help="If set, use the default query template.",
        default=False,
    )
    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=250,# MIID: 50 miners we want to query in a single step change to 100
    )

    parser.add_argument(
        "--neuron.batch_size",
        type=int,
        help="The number of miners to query in a single batch.",
        default=150, # MIID: 5 miners we want to query in a single batch change to 10
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.15,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=40960,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run. Auto-changes to 'subnet322-test' when running on testnet (netuid 322).",
        default="MIID"  # for project_name MIID for mainnet and subnet322-test for testnet
    )
    parser.add_argument(
        "--seed_names.sample_size",
        type=int,
        help="The number of seed names to generate for each validation round.",
        default=15,
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="MIID-dev-test" 
    )
    parser.add_argument(
        "--wandb.max_run_steps",
        type=int,
        help="The maximum number of steps per wandb run before creating a new run.",
        default=1
    )
    parser.add_argument(
        "--wandb.disable",
        action="store_true",
        help="Disable wandb logging entirely. Useful for debugging or when wandb is unavailable.",
        default=True,
    )
    parser.add_argument(
        "--wandb.cleanup_runs",
        action="store_true",
        help="Automatically delete wandb run folders after each run is finished. Useful for saving disk space.",
        default=True,
    )
    parser.add_argument(
        '--neuron.ollama_fallback_models',
        type=str,
        nargs='+',
        help="A list of fallback Ollama models to try if the primary model fails.",
        default=['llama3.2:latest', 'tinyllama:latest']
    )
    parser.add_argument(
        '--neuron.ollama_fallback_timeouts',
        type=int,
        nargs='+',
        help="A list of fallback timeouts (in seconds) to try for Ollama requests.",
        default=[100, 120]
    )
    parser.add_argument(
            "--neuron.ollama_url",
            type=str,
            help="Url to ollama",
            default="http://127.0.0.1:11434",
        )
    parser.add_argument(
            "--neuron.ollama_model_name",
            type=str,
            help="Model name to use with ollama",
            default="llama3.1:latest",
        )
    parser.add_argument(
        "--neuron.ollama_request_timeout",
        type=int,
        help="Timeout for the Ollama request in seconds.",
        default=90, # MIID: 60 seconds is the default timeout to wait for a response from the Ollama server change to 90 seconds
    )

    parser.add_argument(
        "--neuron.max_request_timeout",
        type=int,
        help="Maximum timeout for miner requests in seconds.",
        default=900, # MIID: Maximum timeout limit for adaptive timeout calculation
    )

    parser.add_argument(
        '--neuron.ollama_judge_model',
        type=str,
        help="The Ollama model to use for judging query templates (default: llama3.2:latest)",
        default="mistral:latest"
    )

    parser.add_argument(
        '--neuron.ollama_judge_timeout',
        type=int,
        help="Timeout for the Ollama judge request in seconds.",
        default=60
    )

    parser.add_argument(
        '--neuron.ollama_judge_fallback_models',
        type=str,
        nargs='+',
        help="A list of fallback Ollama models to try for judging if the primary fails.",
        default=['llama3.2:latest','tinyllama:latest']
    )

    parser.add_argument(
        '--neuron.ollama_judge_fallback_timeouts',
        type=int,
        nargs='+',
        help="A list of fallback timeouts (in seconds) to try for Ollama judge requests.",
        default=[90,100, 120]
    )

    parser.add_argument(
        '--neuron.use_judge_model',
        action='store_true',
        help="Enable LLM judge for query validation. Auto-enables if complex query generation is used.",
        default=True
    )

    parser.add_argument(
        '--neuron.judge_strict_mode',
        action='store_true',
        help="Enable strict mode for LLM judge (fails on JSON parsing errors). Default is lenient mode.",
        default=False
    )

    parser.add_argument(
        '--neuron.judge_on_static_pass',
        action='store_true',
        help="Run LLM judge even when static checks pass (default: disabled)",
        default=False
    )

    parser.add_argument(
        '--neuron.judge_failure_threshold',
        type=int,
        help="Number of consecutive judge failures before suggesting to disable judge (default: 10).",
        default=10
    )

    parser.add_argument(
        '--neuron.regenerate_on_invalid',
        action='store_true',
        help="When a generated query is structurally invalid (e.g., missing {name}), try next model/timeout. Default: False (append hints to the current query and proceed).",
        default=False
    )

    parser.add_argument(
        '--neuron.enable_repair_prompt',
        action='store_true',
        help="Attempt to repair an invalid query template by prompting the LLM with the issues and labels (default: False).",
        default=False
    )

    # --- Blended Ranking Reward System Arguments ---
    # Note: apply_ranking is always enabled (removed as configurable option)
    parser.add_argument(
        '--neuron.top_miner_cap',
        type=int,
        help="The maximum number of top miners to consider for ranking rewards.",
        default=50
    )
    parser.add_argument(
        '--neuron.quality_threshold',
        type=float,
        help="The minimum quality score a miner must achieve to be eligible for ranking rewards.",
        default=0.6
    )
    parser.add_argument(
        '--neuron.decay_rate',
        type=float,
        help="The decay rate for the exponential ranking reward curve.",
        default=0.05
    )
    parser.add_argument(
        '--neuron.blend_factor',
        type=float,
        help="The blend factor between rank-based reward and original score (e.g., 0.7 means 70% rank, 30% original score).",
        default=0.7
    )
    
    # --- Emission Burn Configuration ---
    parser.add_argument(
        '--neuron.burn_fraction',
        type=float,
        help="Fraction of emissions to burn to the burn UID when miners qualify.",
        default=0.75
    )
    # Note: burn_uid is hardcoded to 59 and not configurable
    
    # --- Nominatim Cache Configuration ---
    parser.add_argument(
        '--neuron.nominatim_cache_enabled',
        action='store_true',
        help="Enable caching of Nominatim API results to reduce API calls.",
        default=True
    )
    parser.add_argument(
        '--neuron.nominatim_cache_max_size',
        type=int,
        help="Maximum number of entries in the Nominatim cache. Lower values reduce memory usage.",
        default=10000
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
