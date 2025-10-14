import os
import json
import asyncio
from typing import Any, Dict, List

import bittensor as bt

from MIID.protocol import IdentitySynapse
from neurons.miner_v2 import Miner


class _DummyWallet:
    def __init__(self, hotkey: str):
        class _HotKey:
            def __init__(self, addr: str):
                self.ss58_address = addr

        self.hotkey = _HotKey(hotkey)


class _DummyMetagraph:
    def __init__(self, hotkeys: List[str]):
        self.hotkeys = hotkeys
        # minimal stake vector for priority
        class _S(list):
            pass
        self.S = _S([1.0 for _ in hotkeys])


async def _mock_http_task_response(variation_count: int, names: List[str]) -> Any:
    # name_variations: dict[name] -> list[str]
    name_vars: Dict[str, List[str]] = {
        n: [f"{n}_v{i+1}" for i in range(variation_count)] for n in names
    }
    metric: Dict[str, Any] = {"final_reward": 0.0}
    query_params: Dict[str, Any] = {"variation_count": variation_count}
    return [name_vars, metric, query_params]


async def run_test(task_json_path: str, validator_hotkey: str = "5Hmypa1isVpEwrQTYkGGKga54C13XnUj3fBJPHxH2etZkCF7"):
    # Load task data
    with open(task_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = data.get("names", [])
    query_template = data.get("query_template", "Generate {name}")
    query_params = data.get("query_params", {"variation_count": 10})
    variation_count = int(query_params.get("variation_count", 10))

    # Build identity as [name, dob, address]; use placeholders for dob/address if not present
    # The provided _task.json contains only names; synthesize dob/address for testing
    identity: List[List[str]] = []
    for i, n in enumerate(names):
        dob = f"1980-01-{(i % 28) + 1:02d}"
        addr = f"Test City {i}, Testland"
        identity.append([n, dob, addr])

    # Create synapse and attach a dummy dendrite with hotkey
    syn = IdentitySynapse(
        identity=identity,
        query_template=query_template,
        timeout=120.0,
    )
    # Pydantic will coerce this dict into the expected TerminalInfo model
    syn.dendrite = {
        "hotkey": validator_hotkey,
        "nonce": 0,
        "uuid": "test-uuid",
        "signature": "",
    }

    # Build a minimal Miner instance without calling __init__
    # so we test only the forward method logic
    fake_miner: Miner = object.__new__(Miner)  # bypass __init__
    fake_miner.uid = 0
    fake_miner.metagraph = _DummyMetagraph([validator_hotkey])
    fake_miner.wallet = _DummyWallet("5MinerTestAddressXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # output dir for artifacts
    out_dir = os.path.abspath(os.path.join(os.getcwd(), "tmp_forward_test"))
    os.makedirs(out_dir, exist_ok=True)
    fake_miner.output_path = out_dir
    # minimal config with neuron.nvgen_url
    from types import SimpleNamespace
    fake_miner.config = SimpleNamespace(neuron=SimpleNamespace(nvgen_url="localhost:8001"))

    # Patch httpx.AsyncClient.post used inside forward by monkeypatching httpx at runtime
    import types
    import httpx

    orig_client = httpx.AsyncClient

    class _MockResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _MockClient(orig_client):
        async def post(self, url, json=None, **kwargs):  # type: ignore[override]
            payload = await _mock_http_task_response(variation_count, names)
            return _MockResponse(payload)

    httpx.AsyncClient = _MockClient  # type: ignore[assignment]

    try:
        out = await Miner.forward(fake_miner, syn)
        print(out)
    finally:
        httpx.AsyncClient = orig_client  # restore

    # Print results
    print("Variations keys:", list((out.variations or {}).keys())[:3], "...", len(out.variations or {}))
    # Dump a sample of one entry
    # for k, v in (out.variations or {}).items():
    #     print(k, "=>", v[:3])
    #     break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Path to _task.json")
    parser.add_argument("--hotkey", type=str, default="5Hmypa1isVpEwrQTYkGGKga54C13XnUj3fBJPHxH2etZkCF7")
    parser.add_argument("--logging.debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Ensure bittensor logging configured
    # bt.logging.enable_default()

    asyncio.run(run_test(args.task, args.hotkey))


