# MIID Subnet Phase 4: Synthetic Image Variations

## Overview

Phase 4 extends the MIID (Multimodal Inorganic Identities Dataset) Subnet to include **synthetic image variations** of people for KYC/identity verification. This phase builds on the existing identity variation system (names, addresses, DOBs) by adding a visual component.

**Subnet**: 54 (Bittensor)
**Phase**: 4 Cycle 1
**Focus**: Image variation generation with identity preservation

---

## Objectives

1. Generate realistic variations of synthetic person images while preserving identity
2. Create a dataset of image variations for KYC verification training
3. Extend the existing identity system with visual biometric data
4. Validate face consistency across variations using embedding similarity
5. Integrate image rewards with existing KAV/UAV scoring system

---

## Cycle-Based Reward Structure

Phase 4 follows the same **cycle-based validation pattern** as Phase 3. Understanding this is critical:

### Key Concept: Reputation Evolves Over Cycles

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CYCLE-BASED REWARD FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CYCLE N (Current):                                                              │
│  ├── ONLINE VALIDATION (Immediate Weight Setting)                                │
│  │   ├── KAV Scores: Name/DOB/Address validation → immediate                     │
│  │   └── Reputation: From PREVIOUS cycle (N-1) → applied immediately             │
│  │                                                                               │
│  └── Formula: weight = f(KAV_score, reputation_from_cycle_N-1)                   │
│                                                                                  │
│  POST-VALIDATION (After Cycle N Ends):                                           │
│  ├── UAV Address Validation: novelty × impact × quality × reputation × penalty   │
│  ├── Image Post-Validation: face matching, quality, cheat detection              │
│  │                                                                               │
│  └── Results → Update REPUTATION for CYCLE N+1 (NOT current weights!)            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**IMPORTANT**: Post-validation does NOT affect current cycle weights. It updates **reputation** for the **NEXT cycle**.

### Online Validation (Immediate - Sets Current Weights)

| Component | Source | Description |
|-----------|--------|-------------|
| **KAV Scores** | Current cycle | Name/DOB/Address validation (immediate) |
| **Previous Reputation** | Cycle N-1 | Earned from previous post-validation (UAV + Image) |

```python
# Cycle N: Online Weight Setting (IMMEDIATE)
def calculate_online_weight(miner_uid: int, cycle: int) -> float:
    # Current cycle KAV validation
    kav_score = validate_kav_online(miner_uid)  # Name/DOB/Address

    # Previous cycle reputation (includes UAV + Image post-validation)
    previous_reputation = get_reputation_from_cycle(miner_uid, cycle - 1)

    # Rank-Quality Fused Formula (from Phase 3)
    # r(fused) = 0.7 × e^(-0.05 × rank) + 0.3 × Quality
    rank_component = 0.7 * math.exp(-0.05 * miner_rank)
    quality_component = 0.3 * (kav_score * (1 + previous_reputation))

    return rank_component + quality_component
```

### Post-Validation (After Cycle Ends - Updates Future Reputation)

Post-validation runs after the cycle, classifying submissions and updating reputation for NEXT cycle:

| Validation Type | Timing | Effect |
|-----------------|--------|--------|
| **UAV Address** | Post-cycle | Updates reputation for Cycle N+1 |
| **Image Variations** | Post-cycle (after drand reveal) | Updates reputation for Cycle N+1 |

```python
# Post-Cycle N: Reputation Update for Cycle N+1
def post_validate_and_update_reputation(miner_uid: int, cycle: int):
    # UAV Address Post-Validation (same as Phase 3)
    uav_score = validate_uav_addresses(miner_uid, cycle)
    # Formula: novelty × impact × quality × reputation × penalty

    # Image Post-Validation (NEW in Phase 4)
    image_score = validate_images_post_drand(miner_uid, cycle)
    # Includes: face matching, quality, watermark, cheat detection

    # Combine for reputation update
    post_validation_score = combine_post_scores(uav_score, image_score)

    # Update reputation for NEXT cycle (N+1), NOT current
    update_reputation_for_cycle(miner_uid, cycle + 1, post_validation_score)
```

### Cycle Timeline

```
Cycle N-1 (Past)          Cycle N (Current)           Cycle N+1 (Future)
     │                          │                          │
     │  ┌──────────────────────┐│  ┌──────────────────────┐│
     │  │ Post-Validation      ││  │ Post-Validation      ││
     │  │ (UAV + Image)        ││  │ (UAV + Image)        ││
     │  │        │             ││  │        │             ││
     │  └────────┼─────────────┘│  └────────┼─────────────┘│
     │           │              │           │              │
     │           ▼              │           ▼              │
     │    ┌─────────────┐       │    ┌─────────────┐       │
     │    │ Reputation  │───────┼───►│ Reputation  │───────┼───► (affects future)
     │    │ for N       │       │    │ for N+1     │       │
     │    └─────────────┘       │    └─────────────┘       │
     │           │              │                          │
     │           ▼              │                          │
     │    ┌─────────────────────┤                          │
     │    │ Online Validation   │                          │
     │    │ KAV + Reputation    │                          │
     │    │        │            │                          │
     │    │        ▼            │                          │
     │    │ SET WEIGHTS NOW     │                          │
     │    └─────────────────────┘                          │
```

### Image Post-Validation Scoring (Like UAV)

Similar to UAV addresses, image variations are assessed in batch **after drand reveal**:

| Score | Meaning | Criteria |
|-------|---------|----------|
| **+5** | Excellent | Perfect face match (ArcFace >= 0.9), high quality, correct variation type |
| **+2** | Good | Face match passes (>= 0.8), acceptable quality |
| **0** | Neutral | Minor issues but acceptable |
| **-3** | Poor | Quality issues, wrong variation type, face borderline |
| **-5** | Reject | Wrong person (face < 0.8), spam, duplicate, collusion |

**Note**: These scores affect **reputation for the next cycle**, not current weights.

---

## Architecture

### Data Flow (Within a Single Challenge)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 4 CHALLENGE FLOW                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. CHALLENGE PHASE (T=0)                                                       │
│     Validator selects base image from database (FLUXSynID placeholder for later)│
│     Sends to miners with target_drand_round (e.g., T+5m)                        │
│                                                                                 │
│  2. SUBMISSION PHASE (T=0 to T+4m)                                              │
│     Miners generate variations locally                                          │
│     Miners encrypt variations with drand timelock                               │
│     Miners sign with wallet (sign_message)                                      │
│     Miners upload ENCRYPTED files to S3                                         │
│     Miners return S3 paths + hashes + signatures to validator (via synapse)     │
│                                                                                 │
│  3. ONLINE VALIDATION & WEIGHT SETTING (Immediate)                              │
│     Validator receives S3 references from miners                                │
│     Validator validates KAV (Name/DOB/Address) - IMMEDIATE                      │
│     Validator applies PREVIOUS CYCLE reputation                                 │
│     Validator SETS WEIGHTS based on: KAV + previous_reputation                  │
│     Validator stores S3 refs for later post-validation                          │
│     ⚠️ Validator does NOT download or process images here                       │
│                                                                                 │
│  4. POST-VALIDATION (After Cycle Ends + Drand Reveal)                           │
│     Post-validator downloads encrypted files from S3                            │
│     Post-validator waits for drand round, decrypts ALL simultaneously           │
│     Post-validator verifies signatures (verify_message)                         │
│     Post-validator runs ALL image validation:                                   │
│       - Face matching (ArcFace similarity >= 0.8)                               │
│       - Quality assessment                                                      │
│       - Watermark detection                                                     │
│       - Duplicate/collusion detection                                           │
│                                                                                 │
│  5. REPUTATION UPDATE (For NEXT Cycle)                                          │
│     Post-validation scores (UAV + Image) → update REPUTATION                    │
│     ⚠️ This reputation is used in NEXT cycle, NOT current                       │
│     Reputation persists and affects future online weight calculations           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Cycle vs Challenge Distinction

| Concept | Scope | Duration |
|---------|-------|----------|
| **Challenge** | Single image variation request | ~72 minutes (1 Bittensor epoch for drand reveal) |
| **Cycle** | Full validation period | Multiple challenges |
| **Online Validation** | Per challenge | Immediate |
| **Post-Validation** | Per cycle | After cycle ends |
| **Weight Setting** | Per challenge | Uses previous cycle reputation |
| **Reputation Update** | Per cycle | Affects next cycle |

### Base Image Source Strategy

| Phase | Source | Description |
|-------|--------|-------------|
| **Sandbox (Current)** | Base Image Database | Pre-loaded images shared with validators via Flask app (TBD) |
| **Execution (Future)** | FLUXSynID (Placeholder) | Dynamic image generation - implementation details TBD |

**Current Implementation (Sandbox)**:
- Validators receive base images from a shared database
- Database distribution method TBD (likely hourly sync via Flask app)
- Same images available to all validators for consistency

**Future Implementation (FLUXSynID)**:
- Placeholder for dynamic synthetic image generation
- FLUXSynID integration details to be defined later
- Will enable unique base images per challenge

### Key Components

| Component | Description |
|-----------|-------------|
| **Base Image Database** | Pre-loaded images for sandbox (shared via Flask app - TBD) |
| **FLUXSynID (Future)** | Placeholder - dynamic synthetic ID photo generation |
| **Miner Image Generator** | Miners use any model to create variations |
| **S3 Storage** | Encrypted image storage with wallet signature verification |
| **Drand Timelock** | Time-locked encryption for fair simultaneous reveal |
| **Post-Validator** | ALL image validation: face matching, quality, watermarks, cheat detection |

**Note**: The validator does NOT process images in real-time. It only collects S3 references and stores them. All image processing happens in post-validation.

---

## S3 Storage with Wallet Verification

### Overview

Miners upload image variations to S3 encrypted with their wallet signature. This ensures:
- **Identity verification**: Only the miner who claims the upload can prove ownership
- **Tamper prevention**: Images cannot be modified after submission
- **Path protection**: Miners cannot write to other miners' S3 paths (via path_signature)
- **Auditable**: All uploads are traceable to specific miner hotkeys

### Path Signature Security

To prevent malicious miners from overwriting other miners' submissions, each miner derives a unique `path_signature` that becomes part of the S3 path:

```python
# Path signature derivation (miner-side)
path_message = f"{challenge_id}:{wallet.hotkey.ss58_address}"
path_signature = wallet.hotkey.sign(path_message.encode()).hex()[:16]

# S3 path structure (attacker cannot forge path_signature without private key)
s3_key = f"submissions/{challenge_id}/{miner_hotkey}/{path_signature}/{variation_type}.png.tlock"
```

**Why it works**: An attacker who knows the victim's hotkey still cannot compute the victim's `path_signature` because they don't have the victim's private key.

### Miner Upload Flow

```python
# Miner signs and uploads images to S3
def upload_variation_to_s3(wallet, challenge_id: str, image_data: bytes, variation_type: str) -> dict:
    # 1. Generate path_signature (unique per miner, unpredictable by others)
    path_message = f"{challenge_id}:{wallet.hotkey.ss58_address}"
    path_signature = wallet.hotkey.sign(path_message.encode()).hex()[:16]

    # 2. Create message with image hash
    image_hash = hashlib.sha256(image_data).hexdigest()
    message = f"challenge:{challenge_id}:hash:{image_hash}"

    # 3. Sign with wallet hotkey
    signature = wallet.hotkey.sign(message.encode()).hex()

    # 4. Encrypt image with drand timelock (see Drand section)
    encrypted_data = tlock_encrypt(image_data, target_drand_round)

    # 5. Upload to S3 with path_signature in path (prevents path hijacking)
    s3_key = f"submissions/{challenge_id}/{wallet.hotkey.ss58_address}/{path_signature}/{variation_type}.png.tlock"
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=encrypted_data,
        Metadata={
            "hotkey": wallet.hotkey.ss58_address,
            "signature": signature,
            "image_hash": image_hash,
            "path_signature": path_signature,
            "timestamp": str(int(time.time()))
        }
    )

    return {
        "s3_key": s3_key,
        "signature": signature,
        "image_hash": image_hash,
        "path_signature": path_signature
    }
```

### YANEZ Post-Validation Verification Flow

**Note**: This verification is performed by YANEZ during post-validation, NOT by the validator in real-time. The validator only collects S3 references during online validation.

```python
# YANEZ post-validator verifies miner identity and downloads images
from MIID.utils.verify_message import verify_message
from substrateinterface import Keypair

def verify_and_download(miner_hotkey: str, submission: dict) -> bytes:
    # 1. Download encrypted file from S3
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=submission["s3_key"])
    encrypted_data = response["Body"].read()
    metadata = response["Metadata"]

    # 2. Verify miner hotkey matches
    if metadata["hotkey"] != miner_hotkey:
        raise ValueError("Hotkey mismatch - possible tampering")

    # 3. Verify signature
    keypair = Keypair(ss58_address=miner_hotkey, ss58_format=42)
    message = f"challenge:{challenge_id}:hash:{metadata['image_hash']}"
    signature = bytes.fromhex(metadata["signature"])

    if not keypair.verify(data=message, signature=signature):
        raise ValueError(f"Invalid signature for miner {miner_hotkey}")

    # 4. Wait for drand round and decrypt (see Drand section)
    decrypted_data = tlock_decrypt(encrypted_data, drand_signature)

    # 5. Verify hash matches
    actual_hash = hashlib.sha256(decrypted_data).hexdigest()
    if actual_hash != metadata["image_hash"]:
        raise ValueError("Hash mismatch - image was modified")

    return decrypted_data
```

### S3 Bucket Configuration

**Bucket**: `s3://yanez-miid-sn54/`

Miners upload encrypted image variations to this YANEZ-managed S3 bucket.

### S3 Bucket Structure

```
yanez-miid-sn54/
├── challenges/
│   └── <challenge_id>/
│       ├── seed_image.png              # Base image from validator
│       ├── target_round.txt            # Drand round for reveal
│       └── metadata.json               # Challenge parameters
│
└── submissions/
    └── <challenge_id>/
        └── <miner_hotkey>/
            └── <path_signature>/       # Unique per miner (prevents path hijacking)
                ├── pose_edit_123456.png.tlock    # Encrypted with tlock
                ├── expression_edit_123457.png.tlock
                ├── lighting_edit_123458.png.tlock
                └── submission_metadata.json      # Hashes, signatures, path_signature
```

**Security Note**: The `<path_signature>` folder is derived from `sign(challenge_id:miner_hotkey)[:16]`. Attackers cannot write to other miners' paths without their private key.

---

## Drand Timelock Encryption

### Why Timelock?

Timelock encryption serves several critical purposes:

1. **Protect Miner Work**: Miners' generated images are encrypted until reveal, preventing others from seeing/copying their work
2. **Prevent Relay Attacks**: Without timelock, an attacker could intercept a miner's submission and re-submit it as their own. Timelock ensures all submissions are committed before any can be read
3. **Anti-Cheating in Post-Validation**: When YANEZ performs post-validation, all images are decrypted simultaneously, making cross-miner comparison fair and preventing miners from adjusting based on others' work
4. **Future Cycle Protection**: As the system evolves, validators may perform real-time validation - timelock ensures fair competition

With drand timelock:
- All submissions are encrypted until reveal time
- Decryption only possible after drand releases signature
- All miners' work revealed simultaneously for fair post-validation by YANEZ

### Drand Quicknet Configuration

```python
# Drand quicknet (3-second periods)
DRAND_QUICKNET_URL = "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
DRAND_QUICKNET_PK = "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"

def calculate_target_round(delay_seconds: int) -> tuple:
    """Calculate drand round for reveal time."""
    info = requests.get(f"{DRAND_QUICKNET_URL}/info").json()
    genesis = info["genesis_time"]
    period = info["period"]  # 3 seconds

    target_time = int(time.time()) + delay_seconds
    target_round = (target_time - genesis) // period + 1
    reveal_timestamp = genesis + (target_round - 1) * period

    return target_round, reveal_timestamp
```

### Miner: Encrypt with Timelock

```python
from timelock import Timelock
import secrets

def tlock_encrypt(data: bytes, target_round: int) -> bytes:
    """Encrypt data for future drand round."""
    tlock = Timelock(DRAND_QUICKNET_PK)
    ephemeral_sk = bytearray(secrets.token_bytes(32))
    return tlock.tle(target_round, data, ephemeral_sk)
```

### YANEZ Post-Validator: Decrypt After Reveal

**Note**: Decryption is performed by YANEZ during post-validation, NOT by validators in real-time. In future cycles, validators may perform this step.

```python
def wait_for_round(target_round: int) -> bytes:
    """Wait for drand round and fetch signature."""
    info = requests.get(f"{DRAND_QUICKNET_URL}/info").json()
    target_time = info["genesis_time"] + (target_round - 1) * info["period"]

    while time.time() < target_time:
        time.sleep(1)

    # Fetch signature
    response = requests.get(f"{DRAND_QUICKNET_URL}/public/{target_round}")
    return bytes.fromhex(response.json()["signature"])


def tlock_decrypt(encrypted_data: bytes, drand_signature: bytes) -> bytes:
    """Decrypt timelock-encrypted data."""
    tlock = Timelock(DRAND_QUICKNET_PK)
    return tlock.tld(encrypted_data, drand_signature)
```

### Timeline

```
Time ────────────────────────────────────────────────────────────────►

T+0         T+30s       T+60s       T+90s       T+120s      T+72min (REVEAL)
 │           │           │           │           │           │
 │ Challenge │ Miner A   │ Miner B   │ Miner C   │ Deadline  │ Drand
 │ Created   │ Encrypts  │ Encrypts  │ Encrypts  │           │ Releases
 │           │ & Uploads │ & Uploads │ & Uploads │           │ Signature
 │           │           │           │           │           │
 │◄──────────│───────────│───────────│───────────│───────────│────────►
 │           SUBMISSION WINDOW                   │    ALL REVEALED
 │           (All encrypted, validator           │    SIMULTANEOUSLY
 │            CANNOT see any submission)         │
 │                                               │
 │  Note: Reveal happens after 1 Bittensor epoch │
 │  (~72 min = 360 blocks × 12s) to allow        │
 │  post-validation at cycle end                 │
```

---

## Protocol Extension

### IdentitySynapse Additions

The existing `IdentitySynapse` will be extended with image-related fields. **Note**: Miners do NOT return images in the synapse - they return S3 references.

```python
# Phase 4: Single variation request with type and intensity
class VariationRequest(BaseModel):
    type: str        # pose_edit, lighting_edit, expression_edit, background_edit
    intensity: str   # light, medium, far
    description: str # Human-readable description of the type
    detail: str      # Intensity-specific guideline (e.g., "±30° rotation")

# Phase 4: Image variation request from validator to miner
class ImageRequest(BaseModel):
    base_image: str                              # Base64 encoded image
    image_filename: str                          # Original filename for reference
    variation_requests: List[VariationRequest]   # Dynamic selection with type + intensity
    target_drand_round: int                      # Drand round for reveal (~72 min)
    reveal_timestamp: int                        # Unix timestamp when reveal occurs
    challenge_id: str                            # Unique ID: challenge_{timestamp}_{validator_hotkey[:8]}

    @property
    def requested_variations(self) -> int:
        return len(self.variation_requests)

    @property
    def variation_types(self) -> List[str]:
        return [v.type for v in self.variation_requests]

# Phase 4: Miner's S3 submission response
class S3Submission(BaseModel):
    s3_key: str         # Path to encrypted image in S3
    image_hash: str     # SHA256 hash of original (unencrypted) image
    signature: str      # Wallet signature proving ownership
    variation_type: str # Which type this variation addresses
    path_signature: str # Unique path component: sign(challenge_id:miner_hotkey)[:16]

class IdentitySynapse(bt.Synapse):
    # Existing fields
    identity: List[List[str]]  # [name, dob, address]
    query_template: str
    variations: Optional[Dict[str, Union[List[List[str]], SeedData]]] = None

    # Phase 4 additions
    image_request: Optional[ImageRequest] = None      # Validator → Miner
    s3_submissions: Optional[List[S3Submission]] = None  # Miner → Validator
```

**Key changes**:
1. `VariationRequest` includes type + intensity (dynamically selected per challenge)
2. `challenge_id` includes validator hotkey for uniqueness across validators
3. `target_drand_round` set to ~72 minutes (1 Bittensor epoch) for post-validation
4. Miners return S3 references, NOT actual images

---

## Variation Types and Intensity Bins

Validators randomly select **2-4 variation types** per challenge, each with a randomly assigned **intensity level** (Light/Medium/Far). This prevents miners from gaming the system with fixed responses.

### Variation Type Definitions (YEVS-style)

| Family | Description | Light | Medium | Far |
|--------|-------------|-------|--------|-----|
| **pose_edit** | Change head pose (yaw/pitch/roll) while keeping identity | ±15° rotation (slight head tilt) | ±30° rotation (clear head turn) | >±45° rotation (near-profile view) |
| **lighting_edit** | Modify illumination direction, intensity, or color temperature | Subtle brightness/contrast, soft shadows | Directional light, noticeable shadows | Strong shadows, dramatic contrast, unusual color temp |
| **expression_edit** | Change facial expression while preserving identity | Subtle (neutral ↔ slight smile) | Clear change (smile, serious) | Strong (laughing, surprised) |
| **background_edit** | Change background environment | Color/blur adjustment, similar background | Different setting (office ↔ outdoor) | Dramatic/unusual environment |

### Dynamic Selection Per Challenge

Each challenge randomly selects:
1. **Which types**: 2-4 from the 4 available types
2. **Which intensity**: Light, Medium, or Far (can be different per type)

Example challenge request:
```python
requested_variations = [
    {"type": "pose_edit", "intensity": "medium", "detail": "±30° rotation..."},
    {"type": "expression_edit", "intensity": "light", "detail": "Subtle change..."},
    {"type": "background_edit", "intensity": "far", "detail": "Dramatic environment..."}
]
```

### Query Template Integration

Variation requirements are appended to the query template sent to miners:

```
[IMAGE VARIATION REQUIREMENTS]
For the face image provided, generate the following variations while preserving identity:

1. pose_edit (medium): ±30° rotation (clear head turn, profile partially visible)
2. expression_edit (light): Neutral to slight smile, minor brow movement
3. background_edit (far): Unusual or contrasting environment, complex scene

IMPORTANT: The subject's face must remain recognizable across all variations.
```

### Scoring (Post-Validation by YANEZ)

Post-validation will judge:
- **Did the miner follow the requested type?** (pose vs expression vs lighting vs background)
- **Did the miner match the intensity level?** (Light changes should be subtle, Far should be dramatic)
- **Is identity preserved?** (ArcFace similarity >= 0.8)

Intensity compliance is a **guideline** - exact measurements are not required, but the variation should clearly match the requested intensity level.

---

## Validation System

**IMPORTANT**: ALL image validation happens in post-validation. The validator does NOT process images in real-time.

### Validator Role (Real-time)

The validator only:
1. Sends challenge with base image + target drand round
2. Receives S3 references (paths, hashes, signatures) from miners
3. Stores S3 references in database
4. Does NOT download, decrypt, or validate images

### Post-Validation (Batch Processing - After Drand Reveal)

ALL image processing happens here, after the drand round releases:

1. **Download & Decrypt**: Fetch encrypted images from S3, decrypt with drand beacon
2. **Signature Verification**: Verify miner signatures using `verify_message`
3. **Face Matching**: ArcFace embedding comparison (threshold >= 0.8)
4. **Prompt Adherence**: Verify variations match requested types
5. **Quality Assessment**: Image quality, resolution compliance
6. **Watermark Detection**: Check for unwanted watermarks
7. **Cheat Detection**: Duplicate detection, collusion analysis

```python
# Post-validation flow (runs after drand reveal)
def post_validate_image(challenge_id: str, miner_hotkey: str, s3_submission: dict) -> float:
    # 1. Download encrypted image from S3
    encrypted_data = s3_client.get_object(Bucket=BUCKET, Key=s3_submission["s3_key"])

    # 2. Wait for drand round and decrypt
    drand_sig = wait_for_drand_round(target_round)
    image_data = tlock_decrypt(encrypted_data, drand_sig)

    # 3. Verify signature
    if not verify_miner_signature(miner_hotkey, s3_submission):
        return -5  # Reject

    # 4. Verify hash matches
    if sha256(image_data) != s3_submission["image_hash"]:
        return -5  # Tampered

    # 5. Face matching
    base_embedding = get_base_embedding(challenge_id)
    var_embedding = arcface.get_embedding(image_data)
    face_similarity = cosine_similarity(base_embedding, var_embedding)

    if face_similarity < 0.8:
        return -5  # Wrong person

    # 6. Quality, watermark, cheat detection
    quality_score = assess_quality(image_data)
    has_watermark = detect_watermark(image_data)
    is_duplicate = check_duplicate(image_data, seen_hashes)

    # 7. Calculate final score
    return calculate_image_score(face_similarity, quality_score, has_watermark, is_duplicate)
```

---

## Cheat Detection (Post-Validation Only)

All cheat detection happens during post-validation, after drand reveal when all images are decrypted simultaneously.

### Detection Types

| Cheat Type | Detection Method | Penalty |
|------------|------------------|---------|
| **Same Image Resubmission** | Perceptual hash comparison | -5 score |
| **Collusion/Sharing** | Cross-miner image similarity | -3 score |
| **Wrong Person** | Face embedding < 0.8 threshold | -5 score (rejection) |
| **Low Quality/Spam** | Quality metrics below threshold | -5 score |
| **Signature Failure** | Invalid wallet signature | -5 score (rejection) |
| **Hash Mismatch** | Image hash doesn't match declared hash | -5 score (tampering) |

### Implementation

```python
# Duplicate detection using perceptual hashing (post-validation)
def detect_duplicate(image_data: bytes, seen_hashes: Set[str]) -> bool:
    img_hash = imagehash.phash(Image.open(BytesIO(image_data)))
    for seen in seen_hashes:
        if img_hash - imagehash.hex_to_hash(seen) < 5:  # Hamming distance threshold
            return True
    return False

# Collusion detection across miners (post-validation)
# Run after ALL images are decrypted simultaneously
def detect_collusion(challenge_id: str, decrypted_images: Dict[str, List[bytes]]) -> List[Tuple[str, str]]:
    colluding_pairs = []
    miner_hashes = {
        hotkey: [imagehash.phash(Image.open(BytesIO(img))) for img in images]
        for hotkey, images in decrypted_images.items()
    }

    for m1, hashes1 in miner_hashes.items():
        for m2, hashes2 in miner_hashes.items():
            if m1 >= m2:
                continue
            for h1 in hashes1:
                for h2 in hashes2:
                    if h1 - h2 < 5:  # Too similar
                        colluding_pairs.append((m1, m2))
                        break
    return colluding_pairs
```

---

## Cycle Structure (Aligned with Phase 3)

Phase 4 follows the same cycle structure as Phase 3:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4 CYCLE TIMELINE                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  t0: BASELINE                                                                 │
│  └── Initial state, no reputation yet                                         │
│                                                                               │
│  t1: SANDBOX (Cycle 1 Initialization)                                         │
│  ├── Fixed base images (from database, same for all miners)                   │
│  ├── Miners submit image variations                                           │
│  ├── Online validation: KAV only (no reputation yet)                          │
│  └── Post-validation starts: Building initial reputation                      │
│                                                                               │
│  t2: EXECUTION (Cycle 1 Live)                                                 │
│  ├── Base images from database (FLUXSynID placeholder for future)             │
│  ├── Online validation: KAV + reputation from t1 sandbox                      │
│  ├── Validators set weights immediately                                       │
│  └── Post-validation runs: Updates reputation for next cycle                  │
│                                                                               │
│  t3: NEXT CYCLE (Cycle 2 Sandbox)                                             │
│  ├── Miners receive post-validation rewards (UAV + Image)                     │
│  ├── Reputation from t2 now affects online weight calculation                 │
│  └── Cycle continues...                                                       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Phase 4 Cycle 1: Sandbox (t1)

**Purpose**: Controlled testing, building initial reputation

| Aspect | Configuration |
|--------|---------------|
| **Base Images** | Fixed images from database (same for all miners) |
| **Online Validation** | KAV (Name/DOB/Address) only |
| **Reputation Used** | None (no previous cycle) |
| **Post-Validation** | Image validation runs, builds initial reputation |
| **Weight Effect** | Reduced weight, learning phase |

### Phase 4 Cycle 1: Execution (t2)

**Purpose**: Full production with reputation-based weights

| Aspect | Configuration |
|--------|---------------|
| **Base Images** | From database (FLUXSynID placeholder for future) |
| **Online Validation** | KAV + reputation from sandbox (t1) |
| **Reputation Used** | From sandbox phase post-validation |
| **Post-Validation** | Full image validation, updates reputation for t3 |
| **Weight Effect** | Full weight based on KAV + previous reputation |

### What Miners Earn When

| Phase | What's Evaluated | When Weight Set | When Reputation Updated |
|-------|------------------|-----------------|-------------------------|
| **Sandbox (t1)** | KAV only | Immediate (KAV-based) | After sandbox ends → affects t2 |
| **Execution (t2)** | KAV + Image variations | Immediate (KAV + t1 reputation) | After execution → affects t3 |
| **Next Cycle (t3)** | KAV + Image variations | Immediate (KAV + t2 reputation) | Continues... |

---

## Reward Calculation

### Two-Stage Reward System

**Stage 1: Online Weight Setting (Immediate)**
```python
# Validator sets weights IMMEDIATELY after receiving submissions
def set_online_weight(miner_uid: int, cycle: int) -> float:
    # KAV: Immediate validation of identity variations
    kav_score = validate_kav(miner_uid)  # Name/DOB/Address

    # Reputation: From PREVIOUS cycle (includes UAV + Image post-validation)
    prev_reputation = get_reputation(miner_uid, cycle - 1)

    # Rank-Quality Fused (from Phase 3)
    # r(fused) = 0.7 × e^(-0.05 × rank) + 0.3 × Quality
    weight = 0.7 * exp(-0.05 * rank) + 0.3 * (kav_score * (1 + prev_reputation))
    return weight
```

**Stage 2: Post-Validation (Reputation Update for Next Cycle)**
```python
# Post-validator runs AFTER cycle, updates reputation for NEXT cycle
def calculate_image_reputation_score(miner_uid: int) -> float:
    # Image quality assessment (runs during post-validation)
    face_similarity = get_arcface_score(miner_uid)      # 0.40 weight
    variation_count = get_variation_count_score(miner_uid)  # 0.15 weight
    prompt_adherence = get_prompt_adherence(miner_uid)  # 0.25 weight
    quality = get_quality_score(miner_uid)              # 0.10 weight
    uniqueness = get_uniqueness_score(miner_uid)        # 0.10 weight

    image_score = (
        0.40 * face_similarity +
        0.15 * variation_count +
        0.25 * prompt_adherence +
        0.10 * quality +
        0.10 * uniqueness
    )

    # Combined with UAV post-validation for reputation update
    uav_score = get_uav_post_score(miner_uid)
    combined_post_score = combine_uav_and_image(uav_score, image_score)

    # This updates reputation for NEXT cycle, NOT current
    return combined_post_score
```

### Image Post-Validation Score Components

| Component | Weight | Description | When Calculated |
|-----------|--------|-------------|-----------------|
| **Face Similarity** | 40% | ArcFace embedding similarity (>= 0.8) | Post-validation |
| **Variation Count** | 15% | Valid variations / requested variations | Post-validation |
| **Prompt Adherence** | 25% | Variations matching requested types | Post-validation |
| **Quality** | 10% | Image quality (resolution, artifacts) | Post-validation |
| **Uniqueness** | 10% | Unique vs other miners' submissions | Post-validation |

### Formula

```
r(image_post) = 0.40 × face_score + 0.15 × count_score + 0.25 × prompt_score + 0.10 × quality_score + 0.10 × unique_score
```

**Important**: This score feeds into **reputation update**, which affects **NEXT cycle** weights, not current.

---

## Technical Specifications

### Synapse Configuration

| Parameter | Value |
|-----------|-------|
| **Timeout** | 120 seconds |
| **Variations per Request** | 3-5 |
| **Image Delivery** | S3 (encrypted with drand timelock) |
| **Synapse Returns** | S3 paths, hashes, signatures (NOT images) |
| **Resolution** | Configurable (512x512 to 1024x1024) |
| **Face Match Threshold** | 0.8 (checked in post-validation) |

### Base Image Generation (Validator)

**Tool**: FLUXSynID (https://github.com/Raul2718/FLUXSynID)

FLUXSynID features:
- Controllable identity attributes (age, ethnicity, facial features)
- Document-style and live-capture photo generation
- FLUX.1 diffusion model for synthesis
- Quality filtering with face recognition models

### Miner Requirements

- Any image generation model allowed
- Must preserve face identity across variations
- Must address requested variation types
- Response within 120 seconds

---

## Database Schema Extensions

### New Tables

```sql
-- Seed images for Phase 4
CREATE TABLE seed_images (
    id BIGSERIAL PRIMARY KEY,
    identity_id BIGINT REFERENCES identity(id),
    image_hash TEXT NOT NULL UNIQUE,
    image_data TEXT NOT NULL,  -- Base64
    resolution_width INT NOT NULL,
    resolution_height INT NOT NULL,
    source TEXT NOT NULL,  -- 'fluxsynid', 'database'
    face_embedding VECTOR(512),  -- ArcFace embedding
    created_at TIMESTAMP DEFAULT NOW()
);

-- Image variations from miners
CREATE TABLE image_variations (
    id BIGSERIAL PRIMARY KEY,
    seed_image_id BIGINT REFERENCES seed_images(id),
    miner_id BIGINT REFERENCES miner(id),
    variation_type TEXT NOT NULL,  -- 'pose', 'expression', 'lighting', 'background'
    image_data TEXT NOT NULL,  -- Base64
    image_hash TEXT NOT NULL,
    face_embedding VECTOR(512),
    face_similarity FLOAT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Image validation results
CREATE TABLE image_validation (
    id BIGSERIAL PRIMARY KEY,
    variation_id BIGINT REFERENCES image_variations(id),
    face_match_passed BOOLEAN NOT NULL,
    face_similarity_score FLOAT,
    prompt_adherence_score FLOAT,
    quality_score FLOAT,
    watermark_detected BOOLEAN DEFAULT FALSE,
    is_duplicate BOOLEAN DEFAULT FALSE,
    is_collusion BOOLEAN DEFAULT FALSE,
    validation_status TEXT,  -- 'pending', 'passed', 'failed'
    validated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Image observations (linking variations to queries)
CREATE TABLE image_observations (
    id BIGSERIAL PRIMARY KEY,
    seed_image_id BIGINT REFERENCES seed_images(id),
    variation_id BIGINT REFERENCES image_variations(id),
    miner_id BIGINT REFERENCES miner(id),
    validator_id BIGINT REFERENCES validator(id),
    query_id BIGINT REFERENCES query(id),
    source_file_id BIGINT REFERENCES source_file(id),
    observed_at TIMESTAMP DEFAULT NOW()
);
```

---

## Integration with Phase 3

Phase 4 images are linked to Phase 3 identity data:

```
Identity (name, DOB, address) ←→ Seed Image ←→ Image Variations
```

This enables:
- Complete synthetic identity profiles with visual component
- Cross-validation between text and image data
- Enhanced KYC training datasets

---

## Implementation Checklist

### Protocol Changes
- [ ] Extend IdentitySynapse with ImageRequest type (validator → miner)
- [ ] Add S3Submission type for miner response (S3 paths, hashes, signatures)
- [ ] Add challenge_id and target_drand_round fields
- [ ] Update Synapse timeout to 120 seconds

### Validator Changes (Minimal - just collects S3 refs)
- [ ] Integrate FLUXSynID for base image generation
- [ ] Send challenges with base image + target drand round
- [ ] Receive and store S3 submission references
- [ ] Update reward calculation to use post-validation scores

### Post-Validator Changes (NEW - all image processing)
- [ ] Download encrypted images from S3
- [ ] Wait for drand round and decrypt
- [ ] Verify miner signatures (verify_message)
- [ ] Implement ArcFace face embedding extraction
- [ ] Face matching validation (threshold 0.8)
- [ ] Quality assessment
- [ ] Watermark detection
- [ ] Duplicate/collusion detection
- [ ] Calculate image scores (like UAV scoring)

### Miner Changes
- [ ] Handle image requests in forward pass
- [ ] Implement image variation generation
- [ ] Encrypt images with drand timelock
- [ ] Sign with wallet (sign_message)
- [ ] Upload encrypted variations to S3
- [ ] Return S3 paths, hashes, signatures (NOT images)

### Database
- [ ] Create seed_images table
- [ ] Create image_submissions table (S3 refs, not image data)
- [ ] Create image_validation table (post-validation scores)
- [ ] Create image_observations table

### Testing
- [ ] Sandbox testing with fixed images
- [ ] Drand timelock encrypt/decrypt testing
- [ ] S3 upload/download testing
- [ ] Wallet signature verification testing
- [ ] Face matching accuracy validation (post-validation)
- [ ] Cheat detection testing (post-validation)

---

## Dependencies

### Python Packages

```
# Face Recognition & Image Processing
insightface>=0.7.3    # ArcFace face recognition
imagehash>=4.3.1      # Perceptual hashing
Pillow>=10.0.0        # Image processing
numpy>=1.24.0         # Array operations
torch>=2.0.0          # Model inference

# Drand Timelock
timelock>=0.1.0       # Drand timelock encryption
bittensor-drand>=0.1.0  # Bittensor drand integration

# S3 Storage
boto3>=1.26.0         # AWS S3 client
httpx>=0.24.0         # Async HTTP client

# Cryptography
substrateinterface>=1.7.0  # Keypair verification
```

### External Services

- FLUXSynID (self-hosted or API) - Base image generation
- AWS S3 or compatible storage - Image storage
- Drand quicknet - Timelock encryption
- Optional: GPU for face embedding extraction

---

## S3 Infrastructure

- [ ] Create S3 bucket with appropriate permissions
- [ ] Set up miner upload credentials flow
- [ ] Implement S3 auth server (reference: S3_drand/s3_auth_server.py)

---

## Resolved Questions

| Question | Answer |
|----------|--------|
| **Watermark handling specifics** | Details handled in post-validation by YANEZ - skip for now |
| **Exact quality metrics** | Post-validation details handled by YANEZ team in separate repo |
| **Post-validation timing** | Weeks (TBD - exact timing to be determined) |
| **Reputation integration** | Will be decided later when coding - for now just conceptual |
| **S3 bucket ownership** | YANEZ owns the S3 bucket |
| **Drand reveal delay** | After validator finishes receiving ALL miner responses (see forward.py) |

**Note**: Cheat detection, quality metrics, and reputation integration details will be implemented by YANEZ in the post-validation system (separate repo/machine). What we get back is the reputation update for the next cycle.

---

## Key Learnings from S3_drand Folder

The S3_drand folder contains reference implementations for:

1. **DRAND_TLOCK_BITTENSOR_GUIDE.md**: Comprehensive guide on drand timelock encryption
   - How drand provides verifiable randomness
   - Timelock encryption using Identity-Based Encryption (IBE)
   - MANTIS subnet (SN123) as reference implementation
   - Complete miner/validator code examples

2. **drnad1.md**: Design document for commit-reveal with drand
   - StorageLog structure for tracking encrypted commits
   - Random audit sampling using drand as RNG
   - S3 verification hooks

3. **neurons/miner.py**: Reference miner implementation
   - Image variation generation
   - S3 upload with signature
   - Hotkey-based authentication

4. **MIID/utils/sign_message.py** & **verify_message.py**: Wallet signature utilities
   - Sign messages with wallet hotkey
   - Verify signatures using Keypair

---

## References

- [FLUXSynID Repository](https://github.com/Raul2718/FLUXSynID)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [MIID Phase 3 Documentation](../Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf)
- [UAV Reward System](../Unknown%20Attack%20Vectors%20(UAV)%20Reward%20System-updates.pdf)
- [Drand Documentation](https://docs.drand.love/)
- [Timelock PyPI Package](https://pypi.org/project/timelock/)
- [Bittensor Commit-Reveal](https://github.com/opentensor/bittensor-drand)
- [MANTIS Subnet (SN123)](https://github.com/Barbariandev/MANTIS) - Reference for timelock implementation
