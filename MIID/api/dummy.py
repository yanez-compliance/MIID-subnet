# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from typing import List, Optional, Union, Any, Dict

# NOTE: bittensor v11 removed bt.subnets.SubnetsAPI entirely (see the
# migration guide's "Axon, Dendrite, and Synapse are gone" section) and this
# class is unused leftover template code (nothing else in MIID imports it).
# Kept only as a stub so importing this module doesn't crash; querying via
# the subnet's own protocol should go through MIID.compat.transport.CompatDendrite
# directly instead.


class DummyAPI:
    def __init__(self, wallet: "bt.Wallet"):
        raise NotImplementedError(
            "DummyAPI relied on bittensor.subnets.SubnetsAPI, which was removed in "
            "bittensor v11. Use MIID.compat.transport.CompatDendrite directly instead."
        )
