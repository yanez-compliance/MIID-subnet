#!/usr/bin/env python3
"""
Interactive message signer for miners (hotkey) or coldkey owners.

Usage:
    python sign_massge/sign_massage.py
"""

from datetime import datetime
from pathlib import Path

import bittensor


WALLETS_DIR = Path.home() / ".bittensor" / "wallets"


def list_wallet_names() -> list[str]:
    if not WALLETS_DIR.exists():
        return []
    return sorted(
        path.name
        for path in WALLETS_DIR.iterdir()
        if path.is_dir() and (path / "coldkeypub.txt").exists()
    )


def list_hotkeys(wallet_name: str) -> list[str]:
    hotkeys_dir = WALLETS_DIR / wallet_name / "hotkeys"
    if not hotkeys_dir.exists():
        return []
    return sorted(
        path.name
        for path in hotkeys_dir.iterdir()
        if path.is_file() and not path.name.endswith("pub.txt")
    )


def prompt_choice(prompt: str, options: list[str]) -> str:
    if not options:
        raise SystemExit(f"No options found for: {prompt}")

    if len(options) == 1:
        print(f"Using: {options[0]}")
        return options[0]

    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")

    while True:
        raw = input(f"Enter number (1-{len(options)}): ").strip()
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(options):
                return options[index - 1]
        print("Invalid choice, try again.")


def ask_role() -> str:
    print("Are you a miner or a coldkey owner?")
    print("  1. Miner (sign with hotkey)")
    print("  2. Coldkey owner (sign with coldkey)")
    while True:
        choice = input("Enter 1 or 2: ").strip().lower()
        if choice in {"1", "miner", "m"}:
            return "miner"
        if choice in {"2", "coldkey", "c", "owner"}:
            return "coldkey"
        print("Invalid choice, try again.")


def ask_message() -> str:
    print("\nPaste the message that needs to be signed, then press Enter:")
    message = input().strip()
    if not message:
        raise SystemExit("No message provided.")
    return message


def sign(message_text: str, keypair) -> str:
    timestamp = datetime.now()
    timezone_name = timestamp.astimezone().tzname() or "UTC"
    signed_message = f"<Bytes>On {timestamp} {timezone_name} {message_text}</Bytes>"
    signature = keypair.sign(data=signed_message)
    return (
        f"{signed_message}\n"
        f"\tSigned by: {keypair.ss58_address}\n"
        f"\tSignature: {signature.hex()}"
    )


def main() -> None:
    role = ask_role()
    message = ask_message()

    wallet_names = list_wallet_names()
    if not wallet_names:
        raise SystemExit(f"No wallets found in {WALLETS_DIR}")

    wallet_name = prompt_choice("\nSelect a wallet:", wallet_names)

    if role == "miner":
        hotkeys = list_hotkeys(wallet_name)
        if not hotkeys:
            raise SystemExit(f"No hotkeys found for wallet '{wallet_name}'")
        hotkey_name = prompt_choice("\nSelect a hotkey:", hotkeys)
        wallet = bittensor.Wallet(name=wallet_name, hotkey=hotkey_name)
        keypair = wallet.hotkey
        print(f"\nSigning as miner with {wallet_name}/{hotkey_name}...")
    else:
        wallet = bittensor.Wallet(name=wallet_name)
        keypair = wallet.coldkey
        print(f"\nSigning as coldkey owner with wallet '{wallet_name}'...")

    result = sign(message, keypair)
    print("\n" + "=" * 60)
    print("Signed message:")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
