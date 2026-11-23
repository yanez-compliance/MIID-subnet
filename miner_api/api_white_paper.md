# Miner grade API

## Overview

The goal is to **break the detectors**.

- **Face grade** — break **deepfake detection** (Deepfake V1).
- **Screen replay grade** — break **passive liveness** (Passive Liveness v3).

Miners can do either of two things. Both use the same two calls, and both
return whether it **broke**, a **score from -5 to 5**, and a short **comment**.

1. **Break check** — `python miner_api.py`. Send the picture only. Pick face
   to see if deepfake detection breaks, or screen to see if passive liveness
   breaks. No seed. No prompt.
2. **Exact grade** — `python exact_grade.py`. Say whether it is a face
   variation or a screen replay, and send the seed plus the prompt. The face
   prompt is the variation type and the intensity. The screen prompt is the
   cues the challenge asked for. Fill in `case.txt`, or leave it blank and
   answer the questions.

Per-cue metrics, fusion logic, and device classification are not exposed.

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

- One request per hotkey every **1 minute (60 seconds)**
- Rate limits are enforced per hotkey address
- Use the `/rate-limit/status` endpoint to check your current status

Rate limiting ensures fair usage and system stability. Each authenticated hotkey
has its own independent rate limit counter. If a hotkey exceeds the rate limit,
the request is rejected with a `429` error.

## API Endpoints

| Method | Endpoint                 | Auth | Rate limited | Purpose                                      |
| ------ | ------------------------ | ---- | ------------ | -------------------------------------------- |
| POST   | `/face_grade`            | Yes  | Yes          | Face grade. Goal: break deepfake detection.  |
| POST   | `/screen_replay_grade`   | Yes  | Yes          | Screen replay grade. Goal: break passive liveness. |
| POST   | `/rate-limit/status`     | Yes  | No           | Check your rate limit status (no quota used) |
| GET    | `/health`                | No   | No           | Health check                                 |

Base URL: `https://api.yanez.ai:5000`

### 1. `POST /face_grade`

**Goal: break deepfake detection.**

- Break check: send the picture.
- Exact grade: also send the seed, `variation_type`, and `intensity`
  (`light`, `medium`, or `far`).

```python
import base64
import requests
from datetime import datetime
import bittensor

# Miner information
wallet_name = "miner"
wallet_hotkey = "m"

# Break check: the picture is enough.
image_path = "image/my_variation.png"
api_url = "https://api.yanez.ai:5000/face_grade"


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

# Exact grade adds the seed and the face prompt:
# payload["seed_image"] = encode_image("image/seed.png")
# payload["variation_type"] = "background_out"
# payload["intensity"] = "medium"

# Send request
response = requests.post(api_url, json=payload)
print(response.status_code)
print(response.json())
```

### 2. `POST /screen_replay_grade`

**Goal: break passive liveness.** A break check sends the picture. An exact
grade also sends the seed and the cue prompt.

```json
{
  "image": "<base64 of the screen replay>",
  "seed_image": "<base64 of the seed face>",
  "cues": ["screen_glare_hotspots", "moire_pixel_grid"],
  "signature": "<signed message>"
}
```

Leave out `seed_image` and `cues` when you only want the break check.

Cue names: `moire_pixel_grid`, `screen_glare_hotspots`,
`perspective_keystone_distortion`, `gamma_contrast_shift`, `edge_crop_cues`.

### 3. `POST /rate-limit/status`

Check your current rate limit status **without** consuming your quota.

```json
{
  "hotkey": "5DUB7k...",
  "can_request": true,
  "last_request": 1735689600.0,
  "seconds_until_allowed": 0,
  "rate_limit_seconds": 60
}
```

## Request Format

**Break check** needs:

- `signature`: A signed message block from your hotkey.
- `image`: Base64-encoded picture (PNG, JPEG, and so on). A
  `data:image/png;base64,` prefix is optional and will be stripped.

**Exact grade** adds:

- `seed_image`: Base64-encoded seed face.
- Face prompt: `variation_type` (`background_in`, `background_out`,
  `lighting_edit+expression_edit`, `lighting_edit+pose_edit`, or
  `pose_edit+expression_edit`) and `intensity` (`light`, `medium`, or `far`).
- Screen prompt: `cues`, the cue names the challenge asked for.

**Break check:**

```json
{
  "signature": "<Bytes>On 2025-01-01 CST My test message</Bytes>\n\tSigned by: YOUR_HOTKEY\n\tSignature: YOUR_SIGNATURE",
  "image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Exact face grade:**

```json
{
  "signature": "<Bytes>On 2025-01-01 CST My test message</Bytes>\n\tSigned by: YOUR_HOTKEY\n\tSignature: YOUR_SIGNATURE",
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "seed_image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "variation_type": "background_out",
  "intensity": "medium"
}
```

## Response Format

Both `/face_grade` and `/screen_replay_grade` return the same fields.

| Field     | Type    | Meaning                                                                 |
| --------- | ------- | ----------------------------------------------------------------------- |
| `breaks`  | boolean | `true` when the detector rejects the image. Face: deepfake. Screen: passive liveness. |
| `score`   | number  | From **-5** to **5**.                                                   |
| `comment` | string  | Why the image got that score.                                           |

Example (face breaks deepfake detection):

```json
{
  "breaks": true,
  "score": 5,
  "comment": "Deepfake detection rejects this image. Identity held and the requested edit is present."
}
```

Example (screen replay does not break passive liveness):

```json
{
  "breaks": false,
  "score": 3,
  "comment": "Passive liveness still accepts this image. The asked-for cues are present and identity held."
}
```

## Process Overview

1. **Authentication** — Your request is authenticated using hotkey signature verification.
2. **Rate limiting** — Quota is checked (one call per hotkey every 1 minute).
3. **Image ingestion** — Base64 is decoded; the image is validated and converted to RGB PNG.
4. **Grade** — Face grade checks deepfake detection. Screen replay grade checks passive liveness. A break check uses the picture. An exact grade also uses the seed and the prompt.
5. **Response** — You receive `breaks`, `score` (from -5 to 5), and `comment`.

## Error Responses

- **400 Bad Request:** Missing required fields or invalid request format.
- **401 Unauthorized:** Signature verification failed.
- **403 Forbidden:** Hotkey not authorized to use the API.
- **429 Too Many Requests:** Rate limit exceeded (wait 1 minute between requests).

---

## Two client scripts

| You want | Run | You provide |
| -------- | --- | ----------- |
| Did this picture break deepfake, or passive liveness? | `python miner_api.py` | The picture |
| The exact grade for a face variation or a screen replay | `python exact_grade.py` | The picture, the seed, and the prompt (`case.txt`) |

Put the picture you made in `variation/`. Put the original face in `seed/`.
`image/` has been renamed to `variation/`.

`miner_api.py` asks which detector and which picture. If `variation/` has one
picture, it uses that file. Skip the questions:

```bash
python miner_api.py --face gamma
python miner_api.py --screen gamma
```

`exact_grade.py` reads `case.txt`. Fill it in and the script runs. Leave a
line blank and it asks for that part, then saves the answers. One picture in
`variation/` and one picture in `seed/` are picked automatically. Skip the
questions:

```bash
python exact_grade.py --face gamma --seed original --type 2 --intensity medium
python exact_grade.py --screen replay --seed original --cue1 --cue2
```

`--type 1` through `5` is the face variation. `--cue1` through `--cue5` are
moiré, glare, perspective, gamma/contrast, and edge/crop.

### What you fill in for the exact grade

| Blank | Face variation | Screen replay |
| ----- | -------------- | ------------- |
| `grade` | `face` | `screen` |
| `image` | The variation you made | The screen replay you made |
| `seed` | The original seed face | The original seed face |
| `variation_type` | The face prompt | leave blank |
| `intensity` | `light`, `medium`, or `far` | leave blank |
| `cues` | leave blank | The screen prompt |

You can type the number from the comment in `case.txt` instead of the long name.
`1` for a face is an indoor background. `1, 2` for cues is moiré and screen glare.

Both scripts print whether it broke, the score from -5 to 5, and the comment.

### Requirements

- Python 3.8+
- Packages: `bittensor`, `requests`
- A Bittensor wallet + hotkey that is authorized (allow-listed) by the API.

```bash
pip install bittensor requests
```

### Configuration

Edit these variables at the top of `miner_api.py` and again at the top of
`exact_grade.py`. Each script has its own copy.

| Variable        | Description                             | Default                      |
| --------------- | --------------------------------------- | ---------------------------- |
| `wallet_name`   | Your Bittensor wallet (coldkey) name    | `"miner"`                    |
| `wallet_hotkey` | Your hotkey name                        | `"m"`                        |
| `api_base_url`  | Base URL of the API (no endpoint path)  | `https://api.yanez.ai:5000`  |
| `variation_dir` | Folder for the picture you made         | `./variation`                |
| `seed_dir`      | Folder for the original seed face. `exact_grade.py` only. | `./seed`     |
| `case_path`     | Exact-grade fill-in file. `exact_grade.py` only. | `./case.txt`          |

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.

### Usage

1. Put the picture you made in `variation/`. For an exact grade, put the seed in `seed/`.

```
miner_api/
├── miner_api.py
├── exact_grade.py
├── case.txt
├── variation/
│   └── gamma.jpg
└── seed/
    └── original.png
```

2. Break check. One picture is enough. The name skips the question:

```bash
python miner_api.py --face gamma
```

3. Exact grade. A face in `case.txt` looks like this:

```
grade = face
image = gamma
seed = original
variation_type = 2
intensity = medium
cues =
```

A screen replay looks like this:

```
grade = screen
image = replay
seed = original
variation_type =
intensity =
cues = 1, 2
```

The same screen replay on the command line:

```bash
python exact_grade.py --screen replay --seed original --cue1 --cue2
```

If `case.txt` is blank and you pass no flags, `exact_grade.py` asks which kind
of picture, which file is the variation, which file is the seed, and then the
prompt. Your answers are written into `case.txt`.

### Troubleshooting

| Symptom                             | Likely cause / fix                                            |
| ----------------------------------- | ------------------------------------------------------------ |
| `Could not find ... in variation/`  | Put the picture you made in `variation/`, or pass its name.  |
| `Could not find ... in seed/`       | Put the original face in `seed/`.                            |
| `401 Signature verification failed` | Wrong wallet/hotkey, or signature format mismatch.           |
| `403 Unauthorized hotkey`           | Your hotkey is not on the API allow-list.                    |
| `429 Rate limit exceeded`           | Wait for the indicated number of seconds before retrying.    |
