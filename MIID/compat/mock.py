"""Best-effort in-memory replacement for the old bt.MockSubtensor/MockMetagraph.

v11 removed mock chain support entirely -- the migration guide's advice is
"test against a local node" instead. `--mock` mode in this codebase is a
secondary/offline development path (the real validator/miner talk to a real
chain), so this is a lightweight, self-contained stand-in good enough to let
`--mock` keep working structurally; it is not a faithful chain simulator.
"""

import os
import tempfile
from typing import Optional

import bittensor as bt


def get_mock_wallet():
    """Replacement for the old `bittensor_wallet.mock.get_mock_wallet()`.

    Deliberately does NOT fall back to the standalone `bittensor_wallet`
    package's own mock helper even when that package happens to be installed
    (e.g. as a transitive dep of `bittensor-cli`): its `Keypair` is a
    different, older binding whose `.sign()`/`.verify()` use the legacy
    `data=` kwarg, whereas the `Keypair` bundled into `bittensor>=11` (used
    everywhere else in this codebase, e.g. MIID/utils/sign_message.py) uses
    `message=`. Mixing the two silently breaks signing. Always build the
    mock wallet the same way a real one is built, just on a throwaway path.
    """
    mock_dir = os.path.join(tempfile.gettempdir(), "miid_mock_wallet")
    os.makedirs(mock_dir, exist_ok=True)
    wallet = bt.Wallet(name="mock", hotkey="mock", path=mock_dir)
    # Reuse the same throwaway keys across runs instead of erroring out on the
    # second+ call (bt.Wallet.create_new_*key refuses to overwrite by default).
    if not os.path.exists(os.path.join(mock_dir, "mock", "coldkey")):
        wallet.create_new_coldkey(use_password=False, overwrite=False)
    if not os.path.exists(os.path.join(mock_dir, "mock", "hotkeys", "mock")):
        wallet.create_new_hotkey(use_password=False, overwrite=False)
    return wallet


class _MockResult:
    """Duck-types the .success/.message surface of bt.ExtrinsicResult."""

    def __init__(self, success: bool = True, message: str = "mock"):
        self.success = success
        self.message = message


class CompatMockSubtensor:
    def __init__(self, netuid: int, n: int = 16, wallet=None, network: str = "mock"):
        self.network = network
        self.netuid = netuid
        self.chain_endpoint = "mock://localhost"
        # Mirrors the real bt.Subtensor's `.endpoint` attribute, used for logging
        # (e.g. BaseNeuron.__init__'s "using network: {self.subtensor.endpoint}").
        self.endpoint = self.chain_endpoint
        self._block_counter = 1
        self._neurons = {}

        if wallet is not None:
            self._register_neuron(
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                stake=100000,
            )
        for i in range(1, n + 1):
            self._register_neuron(hotkey=f"miner-hotkey-{i}", coldkey="mock-coldkey", stake=100000)

    def _register_neuron(self, hotkey: str, coldkey: str, stake: float) -> int:
        uid = len(self._neurons)
        self._neurons[uid] = {"hotkey": hotkey, "coldkey": coldkey, "stake": stake}
        return uid

    @property
    def block(self) -> int:
        self._block_counter += 1
        return self._block_counter

    def read(self, name: str, **kwargs):
        if name == "metagraph":
            return self._metagraph_dict()
        if name == "uid":
            hotkey_ss58 = kwargs.get("hotkey_ss58")
            for uid, info in self._neurons.items():
                if info["hotkey"] == hotkey_ss58:
                    return uid
            return None
        raise NotImplementedError(f"CompatMockSubtensor.read({name!r}) is not implemented in mock mode.")

    def execute(self, intent, wallet, **kwargs):
        return _MockResult(success=True, message=f"mock-executed {type(intent).__name__}")

    def _metagraph_dict(self) -> dict:
        n = len(self._neurons)
        uids = list(range(n))
        return {
            "netuid": self.netuid,
            "block": self._block_counter,
            "num_uids": n,
            "hotkeys": [self._neurons[i]["hotkey"] for i in uids],
            "coldkeys": [self._neurons[i]["coldkey"] for i in uids],
            "axons": [
                {"ip": 2130706433, "port": 8091, "version": 1, "protocol": 4, "ip_type": 4}
                for _ in uids
            ],
            "active": [True] * n,
            "validator_permit": [i == 0 for i in uids],
            "last_update": [self._block_counter] * n,
            "emission": [0.0] * n,
            "dividends": [0.0] * n,
            "incentives": [0.0] * n,
            "consensus": [0.0] * n,
            "trust": [0.0] * n,
            "rank": [0.0] * n,
            "alpha_stake": [self._neurons[i]["stake"] for i in uids],
            "tao_stake": [0] * n,
            "total_stake": [self._neurons[i]["stake"] for i in uids],
            "min_allowed_weights": 0,
            "max_weights_limit": 1.0,
            "weights_rate_limit": 0,
            "tempo": 100,
        }
