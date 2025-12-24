# Phase 4 Sandbox Plan

## Overview

This sandbox tests the Phase 4 image variation pipeline with minimal dependencies. The goal is to validate the end-to-end flow before integrating production components.

---

## Sandbox Scope

| Component | Sandbox Implementation | Production (Future) |
|-----------|----------------------|---------------------|
| **Base Images** | Pre-loaded folder on validator | FLUXSynID generation |
| **Miner Model** | Placeholder function (returns copies) | Actual image generation model |
| **S3 Storage** | Placeholder URL (TBD) | YANEZ-owned S3 bucket |
| **Drand Encryption** | Real tlock encryption | Same |
| **Post-Validation** | YANEZ handles separately | Same |

---

## Sandbox Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SANDBOX DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. VALIDATOR: Load Base Image                                               │
│     ├── Read image from local folder: ./base_images/                         │
│     ├── Encode to Base64                                                     │
│     └── Create IdentitySynapse with target_drand_round                       │
│                                                                              │
│  2. VALIDATOR → MINER: Send Synapse                                          │
│     ├── base_image (Base64)                                                  │
│     ├── variation_types[] (pose, expression, lighting, background)           │
│     └── target_drand_round (calculated from current time + delay)            │
│                                                                              │
│  3. MINER: Process Request                                                   │
│     ├── Decode Base64 image                                                  │
│     ├── Call generate_variations() [PLACEHOLDER - returns copies for now]    │
│     ├── Sign each variation with wallet.hotkey                               │
│     ├── Encrypt with drand timelock (tlock.encrypt)                          │
│     └── Upload to S3 (placeholder URL)                                       │
│                                                                              │
│  4. MINER → VALIDATOR: Return S3Submissions                                  │
│     ├── s3_key (path to encrypted file)                                      │
│     ├── image_hash (SHA256 of original image)                                │
│     └── signature (wallet signature)                                         │
│                                                                              │
│  5. VALIDATOR: Online Validation                                             │
│     ├── Validate KAV (Name/DOB/Address) - existing logic                     │
│     ├── Apply previous cycle reputation (if any)                             │
│     ├── SET WEIGHTS on-chain                                                 │
│     └── Upload results to YANEZ server (existing flow)                       │
│                                                                              │
│  6. YANEZ: Post-Validation (Separate System)                                 │
│     ├── Download encrypted images from S3                                    │
│     ├── Wait for drand beacon, decrypt                                       │
│     ├── Run image validation (face match, quality, etc.)                     │
│     └── Update reputation for next cycle                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### Phase 1: Validator Setup

- [ ] Create `base_images/` folder in validator directory
- [ ] Add 5-10 sample face images for testing
- [ ] Implement `load_base_image()` function to read from folder
- [ ] Extend `IdentitySynapse` with `ImageRequest` field
- [ ] Calculate `target_drand_round` based on response timeout

### Phase 2: Miner Implementation

- [ ] Implement `decode_base_image()` function
- [ ] Create `generate_variations()` placeholder function
  - For now: return 3-5 copies of the original image
  - TODO: integrate actual model later
- [ ] Implement `sign_image()` using wallet.hotkey
- [ ] Implement `encrypt_with_drand()` using tlock
- [ ] Implement `upload_to_s3()` with placeholder URL
- [ ] Return `S3Submission` objects in synapse response

### Phase 3: Validator Response Handling

- [ ] Parse `S3Submission[]` from miner responses
- [ ] Store S3 references in results JSON
- [ ] Existing: KAV validation continues as-is
- [ ] Existing: Upload to YANEZ server continues as-is

### Phase 4: Testing

- [ ] Test end-to-end flow with single miner
- [ ] Test with multiple miners
- [ ] Verify drand encryption/decryption works
- [ ] Verify S3 upload works (when URL provided)

---

## Validator Code Changes

### 1. Load Base Images

```python
# MIID/validator/base_images.py

import os
import base64
from pathlib import Path
from typing import List, Tuple
import random

# Base images folder (relative to validator)
BASE_IMAGES_DIR = Path(__file__).parent / "base_images"

def load_random_base_image() -> Tuple[str, bytes]:
    """Load a random base image from the local folder.

    Returns:
        Tuple of (filename, base64_encoded_image)
    """
    if not BASE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Base images folder not found: {BASE_IMAGES_DIR}")

    # Get all image files
    image_files = list(BASE_IMAGES_DIR.glob("*.png")) + \
                  list(BASE_IMAGES_DIR.glob("*.jpg")) + \
                  list(BASE_IMAGES_DIR.glob("*.jpeg"))

    if not image_files:
        raise FileNotFoundError(f"No images found in {BASE_IMAGES_DIR}")

    # Select random image
    selected_file = random.choice(image_files)

    # Read and encode
    with open(selected_file, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    return selected_file.name, base64_image


def load_all_base_images() -> List[Tuple[str, str]]:
    """Load all base images from the local folder.

    Returns:
        List of (filename, base64_encoded_image) tuples
    """
    if not BASE_IMAGES_DIR.exists():
        return []

    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for img_path in BASE_IMAGES_DIR.glob(ext):
            with open(img_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            images.append((img_path.name, base64_image))

    return images
```

### 2. Create Image Request in Forward

```python
# In MIID/validator/forward.py - Add to forward() function

from MIID.validator.base_images import load_random_base_image
from MIID.validator.drand_utils import calculate_target_round

# After creating the synapse, add image request
async def forward(self):
    # ... existing code ...

    # === PHASE 4: Add image request ===
    try:
        image_filename, base64_image = load_random_base_image()

        # Calculate drand round for reveal (after all miners respond)
        # Use adaptive_timeout + buffer to ensure all responses are in
        reveal_delay = adaptive_timeout + 60  # 60 second buffer
        target_round, reveal_timestamp = calculate_target_round(reveal_delay)

        image_request = {
            "base_image": base64_image,
            "image_filename": image_filename,
            "variation_types": ["pose", "expression", "lighting", "background"],
            "target_drand_round": target_round,
            "reveal_timestamp": reveal_timestamp,
            "requested_variations": 3  # Request 3-5 variations
        }

        bt.logging.info(f"Phase 4: Added image request for '{image_filename}', drand round {target_round}")
    except Exception as e:
        bt.logging.warning(f"Phase 4: Could not load base image: {e}")
        image_request = None

    # Add to synapse
    request_synapse = IdentitySynapse(
        identity=identity_list,
        query_template=query_template,
        variations={},
        timeout=adaptive_timeout,
        image_request=image_request  # NEW: Phase 4
    )

    # ... rest of forward() ...
```

### 3. Drand Utilities

```python
# MIID/validator/drand_utils.py

import time
import requests
from typing import Tuple

DRAND_QUICKNET_URL = "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"

def calculate_target_round(delay_seconds: int) -> Tuple[int, int]:
    """Calculate the drand round for reveal time.

    Args:
        delay_seconds: Seconds from now until reveal

    Returns:
        Tuple of (target_round, reveal_timestamp)
    """
    try:
        info = requests.get(f"{DRAND_QUICKNET_URL}/info", timeout=5).json()
        genesis = info["genesis_time"]
        period = info["period"]  # 3 seconds for quicknet

        target_time = int(time.time()) + delay_seconds
        target_round = (target_time - genesis) // period + 1
        reveal_timestamp = genesis + (target_round - 1) * period

        return target_round, reveal_timestamp
    except Exception as e:
        # Fallback: estimate based on known parameters
        # Quicknet genesis: 1692803367, period: 3s
        genesis = 1692803367
        period = 3
        target_time = int(time.time()) + delay_seconds
        target_round = (target_time - genesis) // period + 1
        reveal_timestamp = genesis + (target_round - 1) * period

        return target_round, reveal_timestamp
```

---

## Miner Code Changes

### 1. Image Variation Generator (Placeholder)

```python
# MIID/miner/image_generator.py

import base64
import hashlib
from typing import List, Dict
from PIL import Image
import io

def decode_base_image(base64_image: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    image_bytes = base64.b64decode(base64_image)
    return Image.open(io.BytesIO(image_bytes))


def generate_variations(
    base_image: Image.Image,
    variation_types: List[str],
    num_variations: int = 3
) -> List[Dict]:
    """Generate image variations from base image.

    SANDBOX: Returns copies of the original image.
    TODO: Replace with actual model call (FLUX, SD, etc.)

    Args:
        base_image: PIL Image of the base face
        variation_types: List of variation types requested
        num_variations: Number of variations to generate

    Returns:
        List of dicts with:
            - image: PIL Image
            - variation_type: str
            - image_bytes: bytes
            - image_hash: str (SHA256)
    """
    variations = []

    for i in range(num_variations):
        # Select variation type (cycle through requested types)
        var_type = variation_types[i % len(variation_types)]

        # ============================================
        # SANDBOX: Just return copies of original
        # TODO: Replace with actual model call:
        #   variation_image = model.generate(base_image, var_type)
        # ============================================
        variation_image = base_image.copy()

        # Convert to bytes
        buffer = io.BytesIO()
        variation_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Calculate hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        variations.append({
            "image": variation_image,
            "variation_type": var_type,
            "image_bytes": image_bytes,
            "image_hash": image_hash
        })

    return variations
```

### 2. Drand Encryption

```python
# MIID/miner/drand_encrypt.py

import secrets
from typing import Optional

# Try to import timelock, fall back gracefully
try:
    from timelock import Timelock
    TLOCK_AVAILABLE = True
except ImportError:
    TLOCK_AVAILABLE = False

DRAND_QUICKNET_PK = "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"


def encrypt_for_drand(data: bytes, target_round: int) -> bytes:
    """Encrypt data with drand timelock.

    Args:
        data: Raw bytes to encrypt
        target_round: Drand round when decryption becomes possible

    Returns:
        Encrypted bytes
    """
    if not TLOCK_AVAILABLE:
        raise ImportError("timelock package not installed. Run: pip install timelock")

    tlock = Timelock(DRAND_QUICKNET_PK)
    ephemeral_sk = bytearray(secrets.token_bytes(32))

    return tlock.tle(target_round, data, ephemeral_sk)


def encrypt_image_for_drand(image_bytes: bytes, target_round: int) -> Optional[bytes]:
    """Encrypt image bytes with drand timelock.

    Returns None if encryption fails.
    """
    try:
        return encrypt_for_drand(image_bytes, target_round)
    except Exception as e:
        print(f"Drand encryption failed: {e}")
        return None
```

### 3. S3 Upload (Placeholder)

```python
# MIID/miner/s3_upload.py

import hashlib
import time
from typing import Dict, Optional

# Placeholder S3 URL - will be provided later
S3_UPLOAD_URL = "https://placeholder-s3-url.example.com"  # TBD by YANEZ


def upload_to_s3(
    encrypted_data: bytes,
    miner_hotkey: str,
    signature: str,
    image_hash: str,
    target_round: int,
    challenge_id: str,
    variation_type: str
) -> Optional[str]:
    """Upload encrypted image to S3.

    SANDBOX: Returns a mock S3 key. Actual upload TBD.

    Args:
        encrypted_data: Tlock-encrypted image bytes
        miner_hotkey: Miner's hotkey for path
        signature: Wallet signature
        image_hash: SHA256 of original image
        target_round: Drand round for reveal
        challenge_id: Challenge identifier
        variation_type: Type of variation

    Returns:
        S3 key (path) or None if upload fails
    """
    # Generate S3 key path
    timestamp = int(time.time())
    s3_key = f"submissions/{challenge_id}/{miner_hotkey}/{variation_type}_{timestamp}.png.tlock"

    # ============================================
    # SANDBOX: Mock upload - just return the key
    # TODO: Implement actual S3 upload when URL is provided
    #
    # Example future implementation:
    # response = requests.put(
    #     f"{S3_UPLOAD_URL}/{s3_key}",
    #     data=encrypted_data,
    #     headers={
    #         "x-miner-hotkey": miner_hotkey,
    #         "x-signature": signature,
    #         "x-image-hash": image_hash,
    #         "x-target-round": str(target_round)
    #     }
    # )
    # if response.status_code != 200:
    #     return None
    # ============================================

    print(f"[SANDBOX] Mock S3 upload: {s3_key} ({len(encrypted_data)} bytes)")

    return s3_key
```

### 4. Miner Forward Integration

```python
# In MIID/miner/forward.py - Add Phase 4 handling

from MIID.miner.image_generator import decode_base_image, generate_variations
from MIID.miner.drand_encrypt import encrypt_image_for_drand
from MIID.miner.s3_upload import upload_to_s3
from MIID.utils.sign_message import sign_message

def process_image_request(synapse, wallet) -> List[Dict]:
    """Process Phase 4 image variation request.

    Args:
        synapse: IdentitySynapse with image_request
        wallet: Miner wallet for signing

    Returns:
        List of S3Submission dicts
    """
    image_request = synapse.image_request
    if not image_request:
        return []

    try:
        # 1. Decode base image
        base_image = decode_base_image(image_request["base_image"])

        # 2. Generate variations (SANDBOX: returns copies)
        variations = generate_variations(
            base_image,
            image_request["variation_types"],
            image_request.get("requested_variations", 3)
        )

        # 3. Process each variation
        s3_submissions = []
        target_round = image_request["target_drand_round"]
        challenge_id = image_request.get("challenge_id", "sandbox_test")

        for var in variations:
            # Sign the image hash
            message = f"challenge:{challenge_id}:hash:{var['image_hash']}"
            signature = sign_message(wallet, message)

            # Encrypt with drand timelock
            encrypted_data = encrypt_image_for_drand(
                var["image_bytes"],
                target_round
            )

            if encrypted_data is None:
                continue

            # Upload to S3
            s3_key = upload_to_s3(
                encrypted_data=encrypted_data,
                miner_hotkey=wallet.hotkey.ss58_address,
                signature=signature,
                image_hash=var["image_hash"],
                target_round=target_round,
                challenge_id=challenge_id,
                variation_type=var["variation_type"]
            )

            if s3_key:
                s3_submissions.append({
                    "s3_key": s3_key,
                    "image_hash": var["image_hash"],
                    "signature": signature,
                    "variation_type": var["variation_type"]
                })

        return s3_submissions

    except Exception as e:
        print(f"Error processing image request: {e}")
        return []
```

---

## Protocol Extension

### Updated IdentitySynapse

```python
# In MIID/protocol.py - Extend IdentitySynapse

from typing import List, Dict, Optional
from pydantic import Field

class ImageRequest(BaseModel):
    """Phase 4: Image variation request from validator."""
    base_image: str  # Base64 encoded image
    image_filename: str
    variation_types: List[str]  # ["pose", "expression", "lighting", "background"]
    target_drand_round: int
    reveal_timestamp: int
    requested_variations: int = 3
    challenge_id: Optional[str] = None


class S3Submission(BaseModel):
    """Phase 4: Miner's S3 submission response."""
    s3_key: str  # Path to encrypted file in S3
    image_hash: str  # SHA256 of original image
    signature: str  # Wallet signature
    variation_type: str  # Which variation type this is


class IdentitySynapse(bt.Synapse):
    """Extended synapse for Phase 4 image variations."""

    # Existing fields
    identity: List[List[str]] = Field(default_factory=list)
    query_template: str = ""
    variations: Optional[Dict[str, Any]] = None
    process_time: Optional[float] = None

    # Phase 4 additions
    image_request: Optional[ImageRequest] = None  # Validator → Miner
    s3_submissions: Optional[List[S3Submission]] = None  # Miner → Validator
```

---

## Dependencies

Add to `requirements.txt`:

```
# Phase 4 Sandbox
timelock>=0.1.0  # Drand timelock encryption
Pillow>=9.0.0    # Image processing
```

---

## Testing Steps

### 1. Unit Tests

```bash
# Test base image loading
python -c "from MIID.validator.base_images import load_random_base_image; print(load_random_base_image())"

# Test drand round calculation
python -c "from MIID.validator.drand_utils import calculate_target_round; print(calculate_target_round(300))"

# Test image generation (sandbox)
python -c "
from PIL import Image
from MIID.miner.image_generator import generate_variations
img = Image.new('RGB', (256, 256), 'red')
print(generate_variations(img, ['pose', 'expression'], 3))
"
```

### 2. Integration Test

```bash
# Run validator with Phase 4 enabled
python -m MIID.validator --netuid <netuid> --wallet.name <wallet> --logging.debug

# Run miner with Phase 4 handling
python -m MIID.miner --netuid <netuid> --wallet.name <wallet> --logging.debug
```

### 3. End-to-End Verification

1. Validator loads base image from folder
2. Validator sends synapse with `image_request`
3. Miner receives and generates variations (copies for sandbox)
4. Miner encrypts with drand and "uploads" to S3 (mock)
5. Miner returns `s3_submissions` in response
6. Validator includes submissions in results JSON
7. Results uploaded to YANEZ server

---

## Folder Structure

```
MIID/
├── validator/
│   ├── base_images/           # NEW: Local folder for base images
│   │   ├── face_001.png
│   │   ├── face_002.png
│   │   └── ...
│   ├── base_images.py         # NEW: Load base images
│   ├── drand_utils.py         # NEW: Drand utilities
│   └── forward.py             # MODIFIED: Add image request
│
├── miner/
│   ├── image_generator.py     # NEW: Generate variations
│   ├── drand_encrypt.py       # NEW: Tlock encryption
│   ├── s3_upload.py           # NEW: S3 upload (mock)
│   └── forward.py             # MODIFIED: Process image request
│
└── protocol.py                # MODIFIED: Add ImageRequest, S3Submission
```

---

## Next Steps After Sandbox

1. **Integrate real model**: Replace `generate_variations()` placeholder with actual FLUX/SD model
2. **S3 configuration**: YANEZ provides actual S3 bucket URL and credentials
3. **Post-validation**: YANEZ implements image validation in separate system
4. **Reputation integration**: Connect post-validation scores to reputation system

---

## References

- [PHASE4_PLAN.md](./PHASE4_PLAN.md) - Full Phase 4 specification
- [DRAND_TLOCK_BITTENSOR_GUIDE.md](./S3_drand/DRAND_TLOCK_BITTENSOR_GUIDE.md) - Drand encryption details
- [forward.py](../../MIID/validator/forward.py) - Current validator forward implementation
