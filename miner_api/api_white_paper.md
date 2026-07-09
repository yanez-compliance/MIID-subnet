# Passive Liveness (PL) API Documentation

## Overview

The Passive Liveness (PL) API provides **screen replay detection** for identity
and liveness workflows. The system analyzes a submitted image and determines
whether it is likely a photo of a screen (for example, a face shown on a phone,
tablet, laptop, monitor, or TV) rather than a direct capture of a real person.

The PL API validates images through an internal detection pipeline and returns a
pass/fail decision plus an overall confidence score. Per-cue metrics, fusion
logic, and device classification are **not** exposed in API responses.

There is a rate limit because we do not want miners spamming the PL API. We want
carefully thought-out Unknown Attack Vectors (UAVs) that can attack our system.

> **NOTE:** Only miners with registered hotkeys on the subnet can call the PL
> API. Static validator hotkeys are always allowed; registered miner hotkeys are
> fetched from the chain and refreshed every 24 hours.

## Authentication

All API requests require authentication using **hotkey-based signature
verification**. You must include a signed message in your request that verifies
your hotkey identity.

**Rate Limiting:**

- One request per hotkey every **5 minutes (300 seconds)**
- Rate limits are enforced per hotkey address
- Use the `/rate-limit/status` endpoint to check your current status

Rate limiting ensures fair usage and system stability. Each authenticated hotkey
has its own independent rate limit counter. If a hotkey exceeds the rate limit,
the request is rejected with a `429` error.

## API Endpoints

| Method | Endpoint              | Auth | Rate limited | Purpose                                         |
| ------ | --------------------- | ---- | ------------ | ----------------------------------------------- |
| POST   | `/is_live`            | Yes  | Yes          | Screen replay / liveness check                  |
| POST   | `/is_ai`              | Yes  | Yes          | Face variations / AI-generated face detection   |
| POST   | `/rate-limit/status`  | Yes  | No           | Check your rate limit status (no quota used)    |
| GET    | `/health`             | No   | No           | Health check                                    |

> **Migration note:** The old `/validate` endpoint has been replaced by
> **`/is_live`**. It takes the same request payload and returns the same
> response format.

### 1. `POST /is_live`

Validate a single image for screen replay (liveness check).

```python
import base64
import requests
from datetime import datetime
import bittensor

# Miner information
wallet_name = "miner"
wallet_hotkey = "m"

# Path to the image to validate
image_path = "capture.png"
api_url = "http://98.90.28.118:5001/is_live"


def sign_message(wallet):
    """Generate a signed message for API authentication."""
    t = datetime.now()
    msg = f"<Bytes>On {t} {t.astimezone().tzname()} API Request</Bytes>"
    sig = wallet.hotkey.sign(data=msg)
    return (
        f"{msg}\n"
        f"\tSigned by: {wallet.hotkey.ss58_address}\n"
        f"\tSignature: {sig.hex()}"
    )


def encode_image(path):
    """Read an image file and return base64-encoded bytes."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Load wallet
wallet = bittensor.Wallet(name=wallet_name, hotkey=wallet_hotkey)

# Build payload
payload = {
    "image": encode_image(image_path),
    "signature": sign_message(wallet),
}

# Send request
response = requests.post(api_url, json=payload)
print(response.status_code)
print(response.json())
```

### 2. `POST /is_ai`

The **BitMind** face-variations AI checker. Detects whether a submitted face
image is AI-generated (synthetic) rather than a real capture. When a miner sends
an image, they get back a **`true`/`false`** decision plus a **confidence score**
indicating how likely the image is to be AI-generated.

Takes the same request payload as `/is_live` (see below).

### 3. `POST /rate-limit/status`

Check your current rate limit status **without** consuming your quota.

```json
{
  "hotkey": "5DUB7k...",
  "can_request": true,
  "last_request": 1735689600.0,
  "seconds_until_allowed": 0,
  "rate_limit_seconds": 300
}
```

## Request Format

**Required Fields:**

- `signature`: A signed message block from your hotkey.
- `image`: Base64-encoded image bytes (PNG, JPEG, etc.). A
  `data:image/png;base64,` prefix is optional and will be stripped
  automatically.

**Example Request:**

```json
{
  "signature": "<Bytes>On 2025-01-01 CST My test message</Bytes>\n\tSigned by: YOUR_HOTKEY\n\tSignature: YOUR_SIGNATURE",
  "image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Response Format

### `/is_live`

| Field                | Type    | Meaning                                                                                            |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------- |
| `is_screen_replay`   | boolean | `true` if the image is classified as a likely screen replay; `false` otherwise. Use for pass/fail. |
| `overall_confidence` | number  | Confidence score in `[0, 1]`. `0.0` when no screen replay is detected.                             |

Example (screen replay detected):

```json
{
  "is_screen_replay": true,
  "overall_confidence": 0.785
}
```

Example (not a screen replay):

```json
{
  "is_screen_replay": false,
  "overall_confidence": 0.0
}
```

```json
{
  "is_screen_replay": true,
  "overall_confidence": 0.0
}
```


### `/is_ai`

| Field                | Type    | Meaning                                                                             |
| -------------------- | ------- | ----------------------------------------------------------------------------------- |
| `is_ai`              | boolean | `true` if the image is detected as AI-generated (synthetic); `false` otherwise.      |
| `overall_confidence` | number  | Confidence score in `[0, 1]` for how likely the image is to be AI-generated.         |

Example (AI-generated detected):

```json
{
  "is_ai": true,
  "overall_confidence": 0.83
}
```

Example (not AI-generated):

```json
{
  "is_ai": false,
  "overall_confidence": 0.12
}
```

## Process Overview

When you submit an image to the PL system, the following high-level process
occurs:

1. **Authentication** — Your request is authenticated using hotkey signature
   verification.
2. **Rate limiting** — Quota is checked (one call per hotkey every 5 minutes).
3. **Image ingestion** — Base64 is decoded; the image is validated and converted
   to RGB PNG.
4. **Detection** — The image is analyzed by the internal detection pipeline
   (screen-replay for `/is_live`, AI/face-variation for `/is_ai`).
5. **Response** — You receive the decision flag and `overall_confidence` only.

## Error Responses

- **400 Bad Request:** Missing required fields or invalid request format.
- **401 Unauthorized:** Signature verification failed.
- **403 Forbidden:** Hotkey not authorized to use the API.
- **429 Too Many Requests:** Rate limit exceeded (wait 5 minutes between requests).

---

## Client Script: `miner_call_PL.py`

This repo includes a ready-to-use command-line client that handles signing,
image loading, and endpoint selection for you.

### What it does

1. Prompts you to choose a check: **`is_live`** or **`is_ai`**.
2. Loads the first image found in the local `image/` folder.
3. Signs a request message with your Bittensor wallet hotkey.
4. Sends the base64-encoded image plus the signature to the selected endpoint.
5. Prints the JSON response.

### Requirements

- Python 3.8+
- Packages: `bittensor`, `requests`
- A Bittensor wallet + hotkey that is authorized (allow-listed) by the API.

```bash
pip install bittensor requests
```

### Configuration

Edit these variables at the top of `miner_call_PL.py`:

| Variable        | Description                             | Default                    |
| --------------- | --------------------------------------- | -------------------------- |
| `wallet_name`   | Your Bittensor wallet (coldkey) name    | `"miner"`                  |
| `wallet_hotkey` | Your hotkey name                        | `"m"`                      |
| `api_base_url`  | Base URL of the API (no endpoint path)  | `http://98.90.28.118:5001` |
| `image_dir`     | Folder scanned for the image to send    | `./image`                  |

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`. The **first**
image (alphabetically) in `image_dir` is used.

### Usage

1. Put one image in the `image/` folder next to the script:

```
passive_liveness/
├── miner_call_PL.py
└── image/
    └── my_photo.png
```

2. Run the script:

```bash
python miner_call_PL.py
```

3. When prompted, pick the check to run:

```
Which check do you want to run?
  1) is_live  - Screen replay / liveness check
  2) is_ai    - Face variations / AI-generated face detection
Enter 1 or 2 (or 'is_live'/'is_ai'):
```

You can type `1` / `2`, or the endpoint name (`is_live` / `is_ai`) directly.

### Troubleshooting

| Symptom                             | Likely cause / fix                                            |
| ----------------------------------- | ------------------------------------------------------------ |
| `Image directory not found`         | Create the `image/` folder next to the script.               |
| `No image found in ...`             | Add a supported image file to `image/`.                      |
| `401 Signature verification failed` | Wrong wallet/hotkey, or signature format mismatch.           |
| `403 Unauthorized hotkey`           | Your hotkey is not on the API allow-list.                    |
| `429 Rate limit exceeded`           | Wait for the indicated number of seconds before retrying.    |
