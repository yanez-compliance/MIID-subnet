r"""
On-chain miner check (no local wallet): pass your miner's SS58 hotkey. Designed to run on
any machine with Python + bittensor — e.g. not the miner host.

MIID: mainnet netuid 54 (finney), testnet netuid 322 (test). See docs/miner.md.

================================================================================
COPY/PASTE — run on a different machine than the miner (use IP/port from script output)
================================================================================

  nc -zv <AXON_IP> <AXON_PORT>

  expected output:  
    Connection to <AXON_IP> port <AXON_PORT> [tcp/*] succeeded!

  curl -v --max-time 10 "http://<AXON_IP>:<AXON_PORT>/IdentitySynapse"

  expected output:  
    *   Trying <AXON_IP>:<AXON_PORT>...
    * Connected to <AXON_IP> (<AXON_IP>) port <AXON_PORT>
    ...
    > GET /IdentitySynapse HTTP/1.1
    ...
    < bt_header_axon_status_code: 500
    < bt_header_axon_status_message: Internal Server Error #<UUID>
    < bt_header_axon_process_time: 0.001527547836303711
    ...
    * Connection #0 to host <AXON_IP> left intact
    {"message":"Internal Server Error #<UUID>"}%

    Note: you should also see an error in the miner logs:
    verify: default_verify failed for <HOTKEY>: <HOTKEY> is not a whitelisted validator
"""

import bittensor as bt


def select_network_and_netuid():
    """Prompt for network and return (network_name, netuid)."""
    choice = input("Select network [test/main] (default: test): ").strip().lower()
    if choice in {"main", "mainnet"}:
        return "finney", 52
    return "test", 322


def get_local_hotkey_ss58():
    """
    Load a local wallet and return its hotkey SS58 address.
    Uses bittensor defaults unless user provides custom wallet/hotkey names.
    """
    wallet_name = input("Wallet name (default: default): ").strip() or "default"
    hotkey_name = input("Hotkey name (default: default): ").strip() or "default"
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    return wallet.hotkey.ss58_address


def main():
    network_name, netuid = select_network_and_netuid()
    subtensor = bt.subtensor(network=network_name)
    metagraph = subtensor.metagraph(netuid=netuid)
    my_hotkey = get_local_hotkey_ss58()

    print(f"\nUsing network: {network_name}")
    print(f"Using netuid: {netuid}")
    print(f"Local hotkey: {my_hotkey}")

    # Check all UIDs - find yours
    print(f"Total UIDs: {len(metagraph.uids)}")
    print(f"\n{'UID':<6} {'Hotkey':<50} {'IP':<20} {'Port':<8} {'Active':<8}")
    print("-" * 100)

    for uid in metagraph.uids:
        axon = metagraph.axons[uid]
        # Print all axons that have a non-zero IP
        if axon.ip != "0.0.0.0" and axon.port != 0:
            print(f"{uid:<6} {axon.hotkey:<50} {axon.ip:<20} {axon.port:<8}")

    print("\n--- Your specific miner ---")
    found = False
    for uid in metagraph.uids:
        if metagraph.axons[uid].hotkey == my_hotkey:
            axon = metagraph.axons[uid]
            print(f"UID: {uid}")
            print(f"IP on-chain:   {axon.ip}")
            print(f"Port on-chain: {axon.port}")
            print(f"Is serving:    {axon.is_serving}")
            found = True
            break

    if not found:
        print("Local hotkey not found in this metagraph.")


if __name__ == "__main__":
    main()