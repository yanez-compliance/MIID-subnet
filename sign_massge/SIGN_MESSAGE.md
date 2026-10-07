# Sign a Bittensor Message

Use this to sign a short message with your **miner hotkey** or **coldkey**.  
Paste either block below into your terminal — no need for this repo.

## Requirements

```bash
pip install bittensor
```

You need a local wallet under `~/.bittensor/wallets/`.

---

## Copy & paste into terminal (hotkey)

Paste this **entire block** into your terminal. It will ask for wallet name, hotkey name, and the message.

```bash
cat > /tmp/sign_hotkey.py <<'EOF'
from datetime import datetime
import bittensor

wallet_name = input("Wallet name: ").strip()
hotkey_name = input("Hotkey name: ").strip()
message = input("Message to sign: ").strip()

timestamp = datetime.now()
timezone_name = timestamp.astimezone().tzname() or "UTC"
signed_message = f"<Bytes>On {timestamp} {timezone_name} {message}</Bytes>"

wallet = bittensor.Wallet(name=wallet_name, hotkey=hotkey_name)
signature = wallet.hotkey.sign(data=signed_message)

print(
    f"{signed_message}\n"
    f"\tSigned by: {wallet.hotkey.ss58_address}\n"
    f"\tSignature: {signature.hex()}"
)
EOF
python /tmp/sign_hotkey.py
```

---

## Copy & paste into terminal (coldkey)

Paste this **entire block** into your terminal. It will ask for wallet name and the message. You may be prompted for the coldkey password.

```bash
cat > /tmp/sign_coldkey.py <<'EOF'
from datetime import datetime
import bittensor

wallet_name = input("Wallet name: ").strip()
message = input("Message to sign: ").strip()

timestamp = datetime.now()
timezone_name = timestamp.astimezone().tzname() or "UTC"
signed_message = f"<Bytes>On {timestamp} {timezone_name} {message}</Bytes>"

wallet = bittensor.Wallet(name=wallet_name)
signature = wallet.coldkey.sign(data=signed_message)

print(
    f"{signed_message}\n"
    f"\tSigned by: {wallet.coldkey.ss58_address}\n"
    f"\tSignature: {signature.hex()}"
)
EOF
python /tmp/sign_coldkey.py
```

---

## Output format

You should get something like:

```text
<Bytes>On 2026-09-28 22:10:00.123456 EDT YOUR_MESSAGE</Bytes>
	Signed by: 5F...your_ss58_address...
	Signature: a1b2c3...
```

Share that full block (message + signed by + signature) as your signed message.
