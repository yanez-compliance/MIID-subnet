"""
Check that your miner hotkey (from your local wallet) is registered on-chain and that
advertised axon IP/port look valid. For reachability from the internet, run `nc` and
`curl` from a *different* machine than the miner (see printed instructions).

MIID subnets: mainnet netuid 54 (finney), testnet netuid 322 (test).

Examples:
  # Testnet (default preset: test + netuid 322)
  python MIID/miner/active_miner_check/is_my_miner_alive.py \\
    --wallet.name miner_wallet --wallet.hotkey miner_hotkey

  # Mainnet (finney + netuid 54)
  python MIID/miner/active_miner_check/is_my_miner_alive.py \\
    --preset mainnet --wallet.name miner_wallet --wallet.hotkey miner_hotkey

  # Mainnet with explicit endpoint (same as docs/miner.md)
  python MIID/miner/active_miner_check/is_my_miner_alive.py \\
    --preset mainnet --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \\
    --wallet.name miner_wallet --wallet.hotkey miner_hotkey
"""

import argparse
import sys

import bittensor as bt

# MIID subnet IDs (see docs/miner.md)
MIID_MAINNET_NETUID = 54
MIID_TESTNET_NETUID = 322

_PRESET_NETUID = {"mainnet": MIID_MAINNET_NETUID, "testnet": MIID_TESTNET_NETUID}
_PRESET_NETWORK = {"mainnet": "finney", "testnet": "test"}


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
    print("\n--- Manual checks (run on a different machine than the miner) ---")
    print(
        "If TCP connects and curl reaches the axon, validators can reach your port.\n"
        "A GET on /IdentitySynapse without validator credentials often returns HTTP 500; "
        "that still shows the process is listening. Check miner logs for messages about "
        "non-validator or unauthorized requests — that confirms traffic hit your axon.\n"
    )
    print(f"  nc -zv {ip} {port}")
    print(
        f'  curl -v --max-time 10 "http://{ip}:{port}/IdentitySynapse"'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check miner registration and on-chain axon (uses local wallet hotkey)."
    )
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    parser.add_argument(
        "--preset",
        choices=["mainnet", "testnet"],
        default="testnet",
        help=(
            f"mainnet: subtensor.network finney, netuid {MIID_MAINNET_NETUID}; "
            f"testnet: network test, netuid {MIID_TESTNET_NETUID}. "
            "Ignored for netuid/network if you pass --netuid or --subtensor.network."
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
    wallet = bt.Wallet(config=config)
    hotkey_ss58 = wallet.hotkey.ss58_address

    preset = getattr(config, "preset", "testnet")
    print(f"Wallet: {config.wallet.name} / {config.wallet.hotkey}")
    print(f"Hotkey (ss58): {hotkey_ss58}")
    print(
        f"Preset: {preset}  |  subtensor.network: {getattr(config.subtensor, 'network', 'default')}  "
        f"netuid: {config.netuid}"
    )

    subtensor = bt.Subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    if config.print_all_axons:
        print(f"\nTotal UIDs: {len(metagraph.uids)}")
        print(f"\n{'UID':<6} {'Hotkey':<50} {'IP':<20} {'Port':<8}")
        print("-" * 100)
        for uid in metagraph.uids:
            axon = metagraph.axons[uid]
            if axon.ip != "0.0.0.0" and axon.port != 0:
                print(f"{uid:<6} {axon.hotkey:<50} {axon.ip:<20} {axon.port:<8}")

    print("\n--- Your miner (from wallet hotkey) ---")
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
            "Confirm reachability with the commands below."
        )
        _print_external_checks(axon.ip, axon.port)
    else:
        print(
            "\nOn-chain IP/port are missing or zero — fix axon registration / "
            "`btcli subnet register` / firewall before expecting validators to connect."
        )


if __name__ == "__main__":
    main()
