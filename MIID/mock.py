import time

import asyncio
import random
import bittensor as bt

from typing import List, Optional

from MIID.compat.mock import CompatMockSubtensor
from MIID.compat.metagraph import get_metagraph
from MIID.compat.synapse import Synapse


# bittensor v11 has no chain/network mocks (bt.MockSubtensor / bt.MockMetagraph
# are gone -- the migration guide's advice is to test against a local node
# instead). MockSubtensor/MockMetagraph here are thin aliases over our own
# lightweight, in-memory stand-ins (see MIID/compat/mock.py and
# MIID/compat/metagraph.py) kept only so `--mock` mode keeps working
# structurally for local/offline development.
MockSubtensor = CompatMockSubtensor


def MockMetagraph(netuid: int = 1, network: str = "mock", subtensor=None):
    if subtensor is None:
        raise ValueError("MockMetagraph requires a subtensor instance.")
    metagraph = get_metagraph(subtensor, netuid)
    bt.logging.info(f"Metagraph: {metagraph}")
    bt.logging.info(f"Axons: {metagraph.axons}")
    return metagraph


class MockDendrite(bt.Dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List,
        synapse: Optional[Synapse] = None,
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")
        if synapse is None:
            synapse = Synapse()

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = self.preprocess_synapse_for_request(axon, synapse, timeout)
                # We just want to mock the response; real Synapse subclasses
                # (e.g. IdentitySynapse) don't have generic dummy_input/output
                # fields, so we just echo the request back with mock timing.
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = process_time
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                else:
                    s.dendrite.process_time = timeout
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"

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
