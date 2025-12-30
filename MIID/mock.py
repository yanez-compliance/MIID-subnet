import time

import asyncio
import random
import numpy as np
import bittensor as bt

from typing import List, Optional


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey=f"miner-hotkey-{i}",
                coldkey="mock-coldkey",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.Dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.Axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # Update the status code and status message of the dendrite to match the axon
                    # TODO (developer): replace with your own expected synapse data
                    s.dummy_output = s.dummy_input * 2
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(process_time)
                else:
                    s.dummy_output = 0
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = str(timeout)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)


class LocalTestMetagraph:
    """
    Lightweight metagraph for local testing that uses direct substrate queries
    instead of the incompatible SubnetInfoRuntimeApi.

    This allows testing on local subtensor networks that don't have the
    required runtime APIs for bittensor 10.0.0.
    """

    def __init__(
        self,
        netuid: int,
        subtensor: "bt.Subtensor",
        wallet: Optional["bt.Wallet"] = None
    ):
        self.netuid = netuid
        self.subtensor = subtensor
        self.network = subtensor.network
        self.block = 0

        # Query neurons directly from substrate
        self._load_neurons_from_chain()

        bt.logging.info(f"LocalTestMetagraph: netuid={netuid}, n={self.n}")

    def _get_value(self, result):
        """Extract value from substrate query result (handles different return types)."""
        if result is None:
            return None
        if isinstance(result, str):
            return result
        if hasattr(result, 'value'):
            inner = result.value
            # Handle nested value (e.g., BittensorScaleType)
            if hasattr(inner, 'value'):
                return inner.value
            return inner
        return result

    def _load_neurons_from_chain(self):
        """Load neuron data using direct substrate queries."""
        substrate = self.subtensor.substrate

        # Get neuron count for this subnet
        try:
            n_result = substrate.query('SubtensorModule', 'SubnetworkN', [self.netuid])
            self.n = self._get_value(n_result) or 0
        except Exception as e:
            bt.logging.warning(f"Could not get subnet size: {e}")
            self.n = 0

        # Initialize arrays
        self.uids = np.array(list(range(self.n)))
        self.hotkeys = []
        self.coldkeys = []
        self.axons = []
        self.S = np.zeros(self.n)  # Stake
        self.R = np.zeros(self.n)  # Rank
        self.I = np.zeros(self.n)  # Incentive
        self.E = np.zeros(self.n)  # Emission
        self.C = np.zeros(self.n)  # Consensus
        self.T = np.zeros(self.n)  # Trust
        self.D = np.zeros(self.n)  # Dividends
        self.B = np.zeros(self.n)  # Bonds
        self.W = np.zeros((self.n, self.n))  # Weights
        self.last_update = np.zeros(self.n)  # Last update block
        self.active = np.ones(self.n, dtype=bool)  # Active neurons
        self.validator_permit = np.zeros(self.n, dtype=bool)  # Validator permits
        self.validator_trust = np.zeros(self.n)  # Validator trust

        # Query each neuron's data
        for uid in range(self.n):
            try:
                # Get hotkey for this UID
                hotkey_result = substrate.query(
                    'SubtensorModule', 'Keys', [self.netuid, uid]
                )
                hotkey = self._get_value(hotkey_result) or f"unknown-{uid}"
                self.hotkeys.append(hotkey)

                # Get coldkey (owner)
                try:
                    owner_result = substrate.query(
                        'SubtensorModule', 'Owner', [hotkey]
                    )
                    coldkey = self._get_value(owner_result) or "unknown"
                except:
                    coldkey = "unknown"
                self.coldkeys.append(coldkey)

                # Get stake
                try:
                    stake_result = substrate.query(
                        'SubtensorModule', 'TotalHotkeyAlpha', [hotkey, self.netuid]
                    )
                    stake_val = self._get_value(stake_result)
                    self.S[uid] = stake_val / 1e9 if stake_val else 0
                except:
                    self.S[uid] = 0

                # Get axon info
                try:
                    axon_result = substrate.query(
                        'SubtensorModule', 'Axons', [self.netuid, hotkey]
                    )
                    axon_data = self._get_value(axon_result)
                    if axon_data and isinstance(axon_data, dict):
                        axon = bt.AxonInfo(
                            version=axon_data.get('version', 0),
                            ip=self._int_to_ip(axon_data.get('ip', 0)),
                            port=axon_data.get('port', 0),
                            ip_type=axon_data.get('ip_type', 4),
                            hotkey=hotkey,
                            coldkey=coldkey,
                            protocol=axon_data.get('protocol', 4),
                            placeholder1=0,
                            placeholder2=0
                        )
                    else:
                        # No axon info - neuron not serving
                        axon = bt.AxonInfo(
                            version=0, ip="0.0.0.0", port=0, ip_type=4,
                            hotkey=hotkey, coldkey=coldkey, protocol=4,
                            placeholder1=0, placeholder2=0
                        )
                except Exception as e:
                    bt.logging.debug(f"Could not get axon for UID {uid}: {e}")
                    axon = bt.AxonInfo(
                        version=0, ip="0.0.0.0", port=0, ip_type=4,
                        hotkey=hotkey, coldkey=coldkey, protocol=4,
                        placeholder1=0, placeholder2=0
                    )
                self.axons.append(axon)

                bt.logging.debug(f"Loaded UID {uid}: hotkey={hotkey[:16]}..., stake={self.S[uid]:.2f}")

            except Exception as e:
                bt.logging.warning(f"Error loading neuron {uid}: {e}")
                self.hotkeys.append(f"error-{uid}")
                self.coldkeys.append("unknown")
                self.axons.append(bt.AxonInfo(
                    version=0, ip="0.0.0.0", port=0, ip_type=4,
                    hotkey=f"error-{uid}", coldkey="unknown", protocol=4,
                    placeholder1=0, placeholder2=0
                ))

        # Get current block
        try:
            self.block = substrate.get_block_number(None)
        except:
            self.block = 0

    def _int_to_ip(self, ip_int: int) -> str:
        """Convert integer IP to dotted notation."""
        if ip_int == 0:
            return "0.0.0.0"
        return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"

    def sync(self, block: Optional[int] = None, lite: bool = True, subtensor: Optional["bt.Subtensor"] = None):
        """Sync metagraph with chain."""
        if subtensor:
            self.subtensor = subtensor
        self._load_neurons_from_chain()

    def __deepcopy__(self, memo):
        """Custom deepcopy that avoids copying the subtensor (contains unpicklable objects)."""
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'subtensor':
                # Don't deepcopy subtensor - just reference it
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __repr__(self):
        return f"LocalTestMetagraph(netuid={self.netuid}, n={self.n}, network={self.network})"

    def __str__(self):
        return self.__repr__()
