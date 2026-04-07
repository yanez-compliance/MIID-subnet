r"""
On-chain miner check (no local wallet): pass your miner's SS58 hotkey. Designed to run on
any machine with Python + bittensor — e.g. not the miner host.

MIID: mainnet netuid 54 (finney), testnet netuid 322 (test). See docs/miner.md.

================================================================================
COPY/PASTE — run on a different machine than the miner (use IP/port from script output)
================================================================================

  nc -zv <AXON_IP> <AXON_PORT>
  curl -v --max-time 10 "http://<AXON_IP>:<AXON_PORT>/IdentitySynapse"

Example with real IP/port (swap in your on-chain axon values):

  nc -zv 1.208.108.242 53124
  curl -v --max-time 10 "http://1.208.108.242:53124/IdentitySynapse"

Example output (TCP connect succeeds; curl often returns HTTP 500 for /IdentitySynapse
when the caller is not a validator — that still proves the axon is listening. Check
miner logs for non-validator / unauthorized messages.)

  $ nc -zv 1.208.108.242 53124
  Connection to 1.208.108.242 port 53124 [tcp/*] succeeded!

  $ curl -v --max-time 10 "http://1.208.108.242:53124/IdentitySynapse"
  *   Trying 1.208.108.242:53124...
  * Connected to 1.208.108.242 (1.208.108.242) port 53124
  > GET /IdentitySynapse HTTP/1.1
  > Host: 1.208.108.242:53124
  > User-Agent: curl/8.7.1
  > Accept: */*
  >
  * Request completely sent off
  < HTTP/1.1 500 Internal Server Error
  < date: Fri, 03 Apr 2026 16:45:21 GMT
  < server: uvicorn
  < name: Synapse
  < timeout: 12.0
  < bt_header_axon_status_code: 500
  < bt_header_axon_status_message: Internal Server Error #64a35c28-fa82-414a-bf13-499c76780a6a
  < bt_header_axon_process_time: 0.0032787322998046875
  < header_size: 184
  < total_size: 2174
  < computed_body_hash: a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
  < content-length: 73
  < content-type: application/json
  <
  * Connection #0 to host 1.208.108.242 left intact
  {"message":"Internal Server Error #64a35c28-fa82-414a-bf13-499c76780a6a"}

A second curl may show a different Internal Server Error id in bt_header_axon_status_message
and in the JSON body; that is normal.

================================================================================
CLI examples (this script)
================================================================================

  # Testnet — hotkey SS58 from `btcli wallet overview` on the miner machine (or cold/hotkey files)
  python MIID/miner/active_miner_check/is_my_miner_alive.py --hotkey 5YourHotkeySs58Here... --preset testnet

  # Mainnet
  python MIID/miner/active_miner_check/is_my_miner_alive.py --hotkey 5YourHotkeySs58Here... --preset mainnet

  python MIID/miner/active_miner_check/is_my_miner_alive.py --hotkey 5YourHotkeySs58Here... --preset mainnet --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443
"""

import argparse
import sys

import bittensor as bt

# MIID subnet IDs (see docs/miner.md)
MIID_MAINNET_NETUID = 54
MIID_TESTNET_NETUID = 322

_PRESET_NETUID = {"mainnet": MIID_MAINNET_NETUID, "testnet": MIID_TESTNET_NETUID}
_PRESET_NETWORK = {"mainnet": "finney", "testnet": "test"}


def _hotkey_from_argv() -> str | None:
    """Fallback if bt.Config does not surface custom --hotkey (depends on bittensor version)."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--hotkey" and i + 1 < len(argv):
            return argv[i + 1].strip()
        if a.startswith("--hotkey="):
            return a.split("=", 1)[1].strip()
    return None


def _apply_preset_defaults(config: bt.Config) -> None:
    """Apply finney/54 vs test/322 unless the user passed --netuid or --subtensor.network."""
    preset = getattr(config, "preset", "testnet")
    if preset not in _PRESET_NETUID:
        preset = "testnet"
    if "--netuid" not in sys.argv:
        config.netuid = _PRESET_NETUID[preset]
    if "--subtensor.network" not in sys.argv:
        config.subtensor.network = _PRESET_NETWORK[preset]


def _print_external_checks(ip: str, port: int) -> None:
    print("\n--- Same as module docstring: run on another host; substitute IP/port below ---")
    print(f"  nc -zv {ip} {port}")
    print(f'  curl -v --max-time 10 "http://{ip}:{port}/IdentitySynapse"')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check miner hotkey (SS58) on metagraph — no wallet file; use --hotkey. "
            "See file docstring for copy-paste nc/curl checks."
        ),
    )
    bt.Subtensor.add_args(parser)
    parser.add_argument(
        "--hotkey",
        type=str,
        required=True,
        metavar="SS58",
        help="Miner hotkey SS58 address (no wallet needed on this machine).",
    )
    parser.add_argument(
        "--preset",
        choices=["mainnet", "testnet"],
        default="testnet",
        help=(
            f"mainnet: subtensor.network finney, netuid {MIID_MAINNET_NETUID}; "
            f"testnet: network test, netuid {MIID_TESTNET_NETUID}. "
            "Overridden if you pass --netuid or --subtensor.network."
        ),
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=MIID_TESTNET_NETUID,
        help=(
            f"Subnet netuid (default from --preset: {MIID_MAINNET_NETUID} mainnet, "
            f"{MIID_TESTNET_NETUID} testnet)"
        ),
    )
    parser.add_argument(
        "--print-all-axons",
        action="store_true",
        help="List all axons with non-zero IP/port (like miner_check.py)",
    )
    config = bt.Config(parser)
    _apply_preset_defaults(config)
    hotkey_ss58 = str(getattr(config, "hotkey", "") or "").strip() or (_hotkey_from_argv() or "")
    if not hotkey_ss58:
        raise SystemExit(
            "Missing hotkey: pass --hotkey <SS58> (bt.Config did not expose it; file a bittensor issue if needed)."
        )

    preset = getattr(config, "preset", "testnet")
    print(f"Hotkey (ss58): {hotkey_ss58}")
    print(
        f"Preset: {preset}  |  subtensor.network: {getattr(config.subtensor, 'network', 'default')}  "
        f"netuid: {config.netuid}"
    )

    subtensor = bt.Subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    if getattr(config, "print_all_axons", False):
        print(f"\nTotal UIDs: {len(metagraph.uids)}")
        print(f"\n{'UID':<6} {'Hotkey':<50} {'IP':<20} {'Port':<8}")
        print("-" * 100)
        for uid in metagraph.uids:
            axon = metagraph.axons[uid]
            if axon.ip != "0.0.0.0" and axon.port != 0:
                print(f"{uid:<6} {axon.hotkey:<50} {axon.ip:<20} {axon.port:<8}")

    print("\n--- Your miner (from --hotkey) ---")
    if hotkey_ss58 not in metagraph.hotkeys:
        print(
            "Hotkey is NOT in this metagraph — not registered on this netuid/network, "
            "or metagraph not synced yet."
        )
        return

    uid = metagraph.hotkeys.index(hotkey_ss58)
    axon = metagraph.axons[uid]
    on_chain_ok = axon.ip != "0.0.0.0" and axon.port != 0

    print(f"UID: {uid}")
    print(f"IP on-chain:   {axon.ip}")
    print(f"Port on-chain: {axon.port}")
    print(f"Is serving:    {getattr(axon, 'is_serving', 'n/a')}")
    if on_chain_ok:
        print(
            "\nOn-chain axon looks advertised (non-zero IP and port). "
            "Confirm reachability with nc/curl (see top of this file)."
        )
        _print_external_checks(axon.ip, axon.port)
    else:
        print(
            "\nOn-chain IP/port are missing or zero — fix axon registration / "
            "`btcli subnet register` / firewall before expecting validators to connect."
        )


if __name__ == "__main__":
    main()
