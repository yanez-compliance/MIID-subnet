# MIID Validator

This document explains how to run a MIID validator on the Bittensor network. The MIID subnet (Subnet 54) focuses on **identity-preserving face image variations** for KYC and fraud-detection research.

## Overview

MIID validators:
1. Download base face images from the MIID images API (signed request)
2. Build an `ImageRequest` with **6 standard variations** (indoor/outdoor background, screen-replay, and combined edits)
3. Query a random sample of miners in batches; miners return **S3 submission references** (encrypted until drand reveal)
4. Wait for the **drand timelock** (~40 minutes) so submissions can be decrypted
5. Grade submissions via the **external grading API** (KAV — image quality / identity preservation)
6. Optionally combine KAV with **UAV** (reputation) and apply emission burn / partner allocation
7. Update scores, set weights, and upload results to the MIID server

Validators do **not** run local LLMs or generate images. Scoring is handled by the external grading API; miners do the GPU-heavy image generation.

## Requirements

- Python 3.10 or higher
- Git
- A Bittensor wallet with TAO for staking
- Reliable internet connection (MIID API, grading API, drand, Bittensor network)
- Weights & Biases account and API key (optional; disabled by default — see [Weights & Biases Guide](weights_and_biases.md))
- Sufficient disk for temporary results and wandb folders (minimum **20GB** free recommended)

### Compute / hardware (validator)

Validators are **CPU + network** workloads. There is **no GPU or Ollama requirement**.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2–4 cores | 4–8 cores |
| RAM | 8GB | 16GB |
| Disk | 20GB free | 40GB+ free |
| GPU | Not required | — |
| Network | Stable broadband | Low-latency, always-on |

Each forward pass is a long session (~**1 hour** wall clock): miner queries in batches, then ~40 minutes waiting for drand unlock, then grading API calls and weight setting. Plan for continuous uptime and outbound HTTPS access.

## Installation

### Option 1: Automated Setup (Recommended)

1. Clone the MIID repository:
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. Run the automated setup script:
```bash
bash scripts/validator/setup.sh
```

This script will:
- Install system dependencies
- Create a Python virtual environment (`validator_env`)
- Install Python requirements, the MIID package, and Bittensor
- **Not** install Ollama (validators no longer use a local LLM)

3. Activate the virtual environment:
```bash
source validator_env/bin/activate
```

If you encounter issues with `python-venv` during installation:
```bash
sudo apt-get update
sudo apt-get install python3-venv
# Or for specific Python versions:
sudo apt-get install python3.10-venv  # for Python 3.10
sudo apt-get install python3.11-venv  # for Python 3.11
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. Create and activate a virtual environment:
```bash
python3 -m venv validator_env
source validator_env/bin/activate
```

3. Install dependencies and the MIID package:
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install bittensor
```

## Installation Recommendations

- **For beginners**: Use the automated setup script (Option 1).
- **For production**:
  - Run under systemd, supervisor, or pm2 for continuous operation
  - Monitor uptime and outbound connectivity to the MIID / grading APIs
  - Keep wallet keys backed up securely
  - Ensure outbound access to the images server, grading API, and drand endpoints

## Running a Validator

1. Stake to the subnet:
```bash
btcli stake add --netuid 54 --amount 100 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

2. Start your validator:
```bash
python neurons/validator.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

On startup the validator:
- Loads state and (unless disabled) prepares wandb
- Downloads fresh **base images** via a signed POST to the MIID images API into `MIID/validator/base_images/`
- Resets Phase 4 cycle state (`validator_results/phase4_state.json`)

For logging details see the [Logging Guide](logging.md). For wandb see the [Weights & Biases Integration Guide](weights_and_biases.md).

## Configuration Options

Common command-line arguments (defaults from `MIID/utils/config.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--neuron.timeout` | `1200` | Dendrite timeout per miner request (seconds) |
| `--neuron.sample_size` | `250` | Max miners to query per forward pass |
| `--neuron.batch_size` | `150` | Miners queried per batch |
| `--neuron.reveal_delay_seconds` | `2400` | Drand unlock delay (40 min; aligns to 1-hour session) |
| `--neuron.UAV_grading` | `True` | Combine KAV with UAV reputation scoring |
| `--neuron.kav_weight` | `0.10` | Weight of image-quality (KAV) in miner share |
| `--neuron.uav_weight` | `0.90` | Weight of reputation (UAV) in miner share |
| `--neuron.burn_fraction` | `0.30` | Emission burn fraction when miners qualify |
| `--neuron.top_miner_cap` | `50` | Cap on top miners for blended ranking rewards |
| `--neuron.quality_threshold` | `0.6` | Min avg identity preservation to qualify |
| `--wandb.disable` | `True` | Disable wandb (pass flag to change behavior per argparse) |
| `--wandb.cleanup_runs` | `True` | Delete wandb run folders after each run |

Example with custom settings:
```bash
python neurons/validator.py \
  --netuid 54 \
  --wallet.name your_wallet_name \
  --wallet.hotkey your_hotkey \
  --subtensor.network finney \
  --neuron.timeout 1200 \
  --neuron.sample_size 250 \
  --neuron.batch_size 150 \
  --wandb.cleanup_runs
```

### Disk Space Management

The validator cleans up wandb run folders after each forward pass when cleanup is enabled:

```bash
# Remove all wandb run folders
rm -rf ./wandb/run-*

# Or remove wandb runs older than 7 days
find ./wandb -name "run-*" -type d -mtime +7 -exec rm -rf {} \;
```

On mainnet, local `validator_results` JSON is deleted after a successful upload to the MIID server. On testnet, results files are kept for review and upload is skipped.

## How It Works

### Startup

1. Connect to the subnet metagraph and load validator state
2. Fetch base face images from the MIID images server (`MIID_IMAGES_SERVER`, signed by the validator hotkey)
3. Reset Phase 4 image-cycle state for a clean start

### Challenge construction (each forward pass)

1. Select up to `sample_size` random miner UIDs
2. Fetch a base face image from the MIID API
3. Build a standard **6-variation** challenge via `build_standard_challenge_variations()`:
   1. `background_edit` (indoor)
   2. `background_edit` (outdoor)
   3. `screen_replay` (device + visual cues)
   4. Combined: lighting + expression
   5. Combined: lighting + pose
   6. Combined: pose + expression
4. Attach a **drand target round** / reveal timestamp (~T+40 minutes)
5. Wrap into an `IdentitySynapse` with an `ImageRequest` (base image + variation requests + challenge ID)

Images are expected as professional passport-style portraits (3:4, head-and-shoulders; recommended ~1015×1350).

### Miner querying

1. Query miners in batches of `batch_size`
2. Retry failed connections (up to 3 attempts); if ≤50 miners fail, assign empty default responses instead of retrying
3. Collect `s3_submissions` (S3 key, image hash, signatures, variation type)

Miners encrypt outputs until the drand round unlocks; the validator only stores submission metadata until grading time.

### Reveal and grading

Session schedule (≈1 hour):

| Window | What happens |
|--------|----------------|
| 0–20 min | Batch querying (part 1) |
| 20–40 min | Batch querying (part 2) / wait |
| ~40 min | Drand reveal — encrypted images unlock |
| 40–60 min | External grading API window |

After reveal, `get_image_variation_rewards()` POSTs signed challenge + S3 submission data to the grading API. The API returns per-miner validation and identity-preservation scores. Rewards use blended exponential-decay ranking (top `top_miner_cap`, identity threshold ≥ `quality_threshold`).

### Scoring and emissions

- **KAV (10% of miner share by default):** Online image quality / compliance from the grading API
- **UAV (90% of miner share by default):** Reputation from the MIID server snapshot returned on successful upload (when `--neuron.UAV_grading` is enabled)
- **Burn (~30%):** Routed to burn UID when miners qualify
- **Partner pool (~35%):** Commercial partner hotkey on mainnet when present; otherwise burned

If UAV grading is disabled, the validator uses KAV-only scoring with burn applied directly.

Scores are updated with a moving average, then weights are set on-chain.

### Results and upload

Each forward pass writes a results JSON under the logging directory (`validator_results/`), including:

- `phase4_image_data` — challenge ID, base image filename, drand round, requested variations, S3 submissions by miner
- `responses` — per-UID axon info, response time, submissions, scoring details
- `rewards` — per-UID reward floats
- `Weights` — spec version, success flag, uids/weights set
- `metagraph_scores` — current score vector
- `reward_allocation` — pending UAV allocation snapshots (when UAV grading is on)

The payload is signed and uploaded to the MIID server (`/upload_data`). Successful uploads refresh the reputation cache for the next pass and clear pending allocations. Failed uploads keep pending allocations for retry.

## Advanced Configuration

Relevant modules:

- `neurons/validator.py` — validator neuron lifecycle, base-image download, wandb
- `MIID/validator/forward.py` — forward pass orchestration
- `MIID/validator/image_variations.py` — variation challenge definitions
- `MIID/validator/reward.py` — KAV grading API + UAV reputation rewards
- `MIID/validator/drand_utils.py` — timelock reveal timing
- `MIID/validator/base_images.py` — local base-image helpers / API fetch
