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
            config.logging.logging_dir,
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
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    """Adds relevant arguments to the parser for operation."""

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
        default=360,
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
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name.",
        default="miner",
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
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name.",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=1200,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=250,
    )

    parser.add_argument(
        "--neuron.batch_size",
        type=int,
        help="The number of miners to query in a single batch.",
        default=150,
    )

    parser.add_argument(
        "--neuron.reveal_delay_seconds",
        type=int,
        help=(
            "Seconds from challenge start until drand timelock unlock (default: 2400 = 40 min). "
            "Aligned to a 1-hour session: 20 min batch 1, 20 min batch 2, unlock at 40 min, "
            "20 min API grading."
        ),
        default=2400,
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
        help="The wandb project name. Auto-changes to 'subnet322-test' when running on testnet.",
        default="MIID",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The wandb entity to log to.",
        default="MIID-dev-test",
    )

    parser.add_argument(
        "--wandb.max_run_steps",
        type=int,
        help="The maximum number of steps per wandb run before creating a new run.",
        default=1,
    )

    parser.add_argument(
        "--wandb.disable",
        action="store_true",
        help="Disable wandb logging entirely.",
        default=True,
    )

    parser.add_argument(
        "--wandb.cleanup_runs",
        action="store_true",
        help="Automatically delete wandb run folders after each run is finished.",
        default=True,
    )

    # --- Blended Ranking Reward System ---
    parser.add_argument(
        '--neuron.top_miner_cap',
        type=int,
        help="The maximum number of top miners to consider for ranking rewards.",
        default=50,
    )
    parser.add_argument(
        '--neuron.quality_threshold',
        type=float,
        help="The minimum quality score a miner must achieve to be eligible for ranking rewards.",
        default=0.6,
    )
    parser.add_argument(
        '--neuron.decay_rate',
        type=float,
        help="The decay rate for the exponential ranking reward curve.",
        default=0.05,
    )
    parser.add_argument(
        '--neuron.blend_factor',
        type=float,
        help="Blend factor between rank-based reward and original score.",
        default=0.7,
    )

    # --- Emission Burn Configuration ---
    parser.add_argument(
        '--neuron.burn_fraction',
        type=float,
        help="Fraction of emissions to burn to the burn UID when miners qualify. "
             "The remaining PARTNER_FRACTION (35%) routes to the commercial partner "
             "hotkey if registered on mainnet, otherwise is also burned.",
        default=0.30,
    )

    # --- UAV Grading Configuration ---
    parser.add_argument(
        '--neuron.UAV_grading',
        action='store_true',
        help="Enable UAV grading system with reputation-weighted rewards (KAV + UAV).",
        default=True,
    )
    parser.add_argument(
        '--neuron.kav_weight',
        type=float,
        help="Weight for KAV (image quality) scores in reputation-weighted rewards.",
        default=0.10,
    )
    parser.add_argument(
        '--neuron.uav_weight',
        type=float,
        help="Weight for UAV (reputation) scores in reputation-weighted rewards.",
        default=0.90,
    )

    # --- Nominatim Cache Configuration ---
    parser.add_argument(
        '--neuron.nominatim_cache_enabled',
        action='store_true',
        help="Enable caching of Nominatim API results.",
        default=True,
    )
    parser.add_argument(
        '--neuron.nominatim_cache_max_size',
        type=int,
        help="Maximum number of entries in the Nominatim cache.",
        default=10000,
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator
    after adding relevant arguments.
    """
    # bittensor >= 10.5.0 introduced BT_NO_PARSE_CLI_ARGS which defaults to
    # "true", causing bt.Config(parser) to skip all CLI arg parsing and return
    # only DEFAULTS (which has no `neuron` key). We must set it to "false" so
    # that wallet/subtensor/neuron args passed on the command line are parsed.
    os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")

    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    cls.add_args(parser)
    return bt.Config(parser)
