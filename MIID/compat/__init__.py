"""bittensor v11 compatibility shim.

bittensor v11 deleted a handful of names this codebase was built around:
``bt.logging``, ``bt.Config``, ``bt.Synapse``, ``bt.Axon``, ``bt.Dendrite``,
``bt.MockSubtensor``, ``bt.MockMetagraph``. None of these exist anymore, so
rather than rewriting every one of the hundreds of call sites across the
codebase that use them, this module installs drop-in replacements as
attributes of the real ``bittensor`` module at import time. This is safe
specifically *because* v11 no longer defines these names at all -- we are
filling a gap, not overriding real functionality.

This module MUST be imported before anything else in the MIID package
touches ``bittensor.logging`` / ``bittensor.Synapse`` / etc. -- see the top
of MIID/__init__.py.

Names that still exist for real in v11 (``bt.Wallet``, ``bt.Subtensor``,
``bt.Metagraph``) are intentionally left untouched; the handful of call
sites constructing/using them with the old (now-removed) method signatures
are fixed directly at their call sites instead.
"""

import bittensor as bt

from MIID.compat.logging_shim import bt_logging
from MIID.compat.config import Config
from MIID.compat.synapse import Synapse, TerminalInfo
from MIID.compat.transport import CompatAxon, CompatDendrite, NotVerifiedException
from MIID.compat.mock import CompatMockSubtensor

if not hasattr(bt, "logging"):
    bt.logging = bt_logging

if not hasattr(bt, "Config"):
    bt.Config = Config

if not hasattr(bt, "Synapse"):
    bt.Synapse = Synapse

if not hasattr(bt, "Axon"):
    bt.Axon = CompatAxon

if not hasattr(bt, "Dendrite"):
    bt.Dendrite = CompatDendrite

if not hasattr(bt, "MockSubtensor"):
    bt.MockSubtensor = CompatMockSubtensor

__all__ = [
    "bt_logging",
    "Config",
    "Synapse",
    "TerminalInfo",
    "CompatAxon",
    "CompatDendrite",
    "NotVerifiedException",
    "CompatMockSubtensor",
]
