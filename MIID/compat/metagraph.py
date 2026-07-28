"""Drop-in replacement for the old ``subtensor.metagraph(netuid)`` object.

bittensor v11 removed ``Subtensor.metagraph()`` and the old parallel-numpy-array
``bt.Metagraph``. The closest v11 equivalent is the raw ``metagraph`` read
(``subtensor.read("metagraph", netuid=...)``), which still returns
per-neuron parallel arrays (hotkeys, coldkeys, axons, stake, ...) — just as a
plain dict instead of an object. ``CompatMetagraph`` wraps that dict back
into the old attribute-style interface (``.hotkeys``, ``.axons``, ``.S``,
``.uids``, ``.n``, ``.validator_permit``, ``.last_update``, ...) so the rest
of the codebase (uids.py, forward.py, weight_utils.py, ...) needs no changes.
"""

import ipaddress

import numpy as np


class CompatAxonInfo:
    """Equivalent of the old bt.AxonInfo entry inside metagraph.axons[uid]."""

    __slots__ = ("uid", "hotkey", "coldkey", "ip", "port", "version", "protocol", "ip_type")

    def __init__(self, uid, hotkey, coldkey, raw_axon: dict):
        self.uid = uid
        self.hotkey = hotkey
        self.coldkey = coldkey
        raw_ip = raw_axon.get("ip") or 0
        try:
            self.ip = str(ipaddress.ip_address(raw_ip)) if raw_ip else "0.0.0.0"
        except ValueError:
            self.ip = "0.0.0.0"
        self.port = raw_axon.get("port") or 0
        self.version = raw_axon.get("version") or 0
        self.protocol = raw_axon.get("protocol") or 0
        self.ip_type = raw_axon.get("ip_type") or 0

    @property
    def is_serving(self) -> bool:
        return self.ip not in (None, "0.0.0.0") and bool(self.port)

    def ip_str(self) -> str:
        return f"{self.ip}:{self.port}"

    def __eq__(self, other):
        if not isinstance(other, CompatAxonInfo):
            return NotImplemented
        return (
            self.hotkey == other.hotkey
            and self.coldkey == other.coldkey
            and self.ip == other.ip
            and self.port == other.port
        )

    def __repr__(self):
        return f"CompatAxonInfo(uid={self.uid}, hotkey={self.hotkey!r}, ip={self.ip}:{self.port})"


class CompatMetagraph:
    """Old-style attribute-access metagraph, backed by the v11 'metagraph' read."""

    def __init__(self, netuid: int, raw: dict):
        self.netuid = netuid
        self._load(raw)

    def _load(self, raw: dict):
        self._raw = raw
        n = int(raw.get("num_uids", len(raw.get("hotkeys", []))))
        self.n = n
        self.uids = np.arange(n)
        self.block = raw.get("block", 0)
        self.hotkeys = list(raw.get("hotkeys", []))
        self.coldkeys = list(raw.get("coldkeys", []))
        self.axons = [
            CompatAxonInfo(uid, self.hotkeys[uid], self.coldkeys[uid], raw["axons"][uid])
            for uid in range(n)
        ]
        self.active = np.array(raw.get("active", [False] * n))
        self.validator_permit = np.array(raw.get("validator_permit", [False] * n))
        self.last_update = np.array(raw.get("last_update", [0] * n))
        self.emission = np.array(raw.get("emission", [0.0] * n), dtype=np.float32)
        self.dividends = np.array(raw.get("dividends", [0.0] * n), dtype=np.float32)
        self.incentive = np.array(raw.get("incentives", [0.0] * n), dtype=np.float32)
        self.consensus = np.array(raw.get("consensus", [0.0] * n), dtype=np.float32)
        self.trust = np.array(raw.get("trust", [0.0] * n), dtype=np.float32)
        self.rank = np.array(raw.get("rank", [0.0] * n), dtype=np.float32)
        self.total_stake = np.array(raw.get("total_stake", [0] * n), dtype=np.float64)
        self.alpha_stake = np.array(raw.get("alpha_stake", [0] * n), dtype=np.float64)
        self.tao_stake = np.array(raw.get("tao_stake", [0] * n), dtype=np.float64)
        # S: alias used throughout MIID for total stake, matching the old metagraph.
        self.S = self.total_stake
        self.min_allowed_weights = raw.get("min_allowed_weights", 0)
        self.max_weight_limit = raw.get("max_weights_limit", 1.0)
        self.weights_rate_limit = raw.get("weights_rate_limit", 0)
        self.tempo = raw.get("tempo", 0)

    def sync(self, subtensor=None, netuid: int = None):
        """Refetch and rebind in place, mirroring old Metagraph.sync()."""
        sub = subtensor
        target_netuid = netuid if netuid is not None else self.netuid
        if sub is None:
            raise ValueError("CompatMetagraph.sync() requires a subtensor instance.")
        raw = sub.read("metagraph", netuid=target_netuid)
        self._load(raw)
        return self


def get_metagraph(subtensor, netuid: int) -> CompatMetagraph:
    """Fetch a fresh CompatMetagraph for netuid via the v11 'metagraph' read."""
    raw = subtensor.read("metagraph", netuid=netuid)
    return CompatMetagraph(netuid, raw)
