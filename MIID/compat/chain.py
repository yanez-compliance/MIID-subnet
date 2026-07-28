"""Weight-setting and axon-serving helpers using v11's intent/execute model.

Replaces the old ``subtensor.set_weights(...)`` and ``subtensor.serve_axon(...)``
methods, which are gone in v11 (see the migration guide's "Transactions"
table: they map to the ``bt.SetWeights`` and ``bt.ServeAxon`` intents,
submitted through ``subtensor.execute(intent, wallet)``).
"""

from typing import Sequence

import bittensor as bt


def set_weights(
    subtensor,
    wallet,
    netuid: int,
    uids: Sequence[int],
    weights: Sequence[float],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
):
    """Set validator weights on chain via the bt.SetWeights intent.

    Unlike the old u16-quantized `(uids, weights)` pairs, v11's SetWeights
    intent accepts plain floats at any scale and handles clipping/
    normalization/quantization internally.
    """
    intent = bt.SetWeights(
        netuid=netuid,
        uids=list(int(u) for u in uids),
        weights=list(float(w) for w in weights),
        mechid=0,
        version_key=version_key,
    )
    return subtensor.execute(
        intent,
        wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )


def serve_axon(subtensor, wallet, netuid: int, ip: str, port: int):
    """Publish this neuron's IP:port on chain via the bt.ServeAxon intent."""
    intent = bt.ServeAxon(netuid=netuid, ip=ip, port=port)
    return subtensor.execute(intent, wallet)


def is_hotkey_registered(subtensor, netuid: int, hotkey_ss58: str) -> bool:
    """Replacement for the removed Subtensor.is_hotkey_registered(...)."""
    uid = subtensor.read("uid", hotkey_ss58=hotkey_ss58, netuid=netuid)
    return uid is not None


def get_current_block(subtensor) -> int:
    """Replacement for the removed Subtensor.get_current_block()."""
    return subtensor.block
