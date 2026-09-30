# Sign a Bittensor Message

Use this to sign a short message with your **miner hotkey** or **coldkey**.  
Copy any of the scripts below into a `.py` file and run it — no need for this repo.

## Requirements

```bash
pip install bittensor
```

You need a local wallet under `~/.bittensor/wallets/`.

---

## Sign with a miner hotkey

Replace `WALLET_NAME`, `HOTKEY_NAME`, and `YOUR_MESSAGE`.

```python
from datetime import datetime
import bittensor

WALLET_NAME = "my_wallet"
HOTKEY_NAME = "my_hotkey"
MESSAGE = "YOUR_MESSAGE"


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


wallet = bittensor.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
print(sign(MESSAGE, wallet.hotkey))
```

---

## Sign with a coldkey

Replace `WALLET_NAME` and `YOUR_MESSAGE`. You may be prompted for the coldkey password.

```python
from datetime import datetime
import bittensor

WALLET_NAME = "my_wallet"
MESSAGE = "YOUR_MESSAGE"


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


wallet = bittensor.Wallet(name=WALLET_NAME)
print(sign(MESSAGE, wallet.coldkey))
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
