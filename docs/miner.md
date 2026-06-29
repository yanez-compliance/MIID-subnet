# MIID Miner

This document explains how to run a MIID miner on the Bittensor network. The MIID subnet (Subnet 54) focuses on **identity-preserving face image variations** for KYC and fraud-detection research.

## Overview

MIID miners receive **face image variation requests** from validators:

- Receive a base face image and variation requirements
- **Variation types (Phase 4 Cycle 2):** pose_edit, lighting_edit, expression_edit, background_edit, and **screen_replay**
- Generate identity-preserving image variations using a diffusion model
- Encrypt and upload results to S3
- Return signed submission references to the validator

**Image generation is the only scored task.** Validators score miners exclusively on face image variation quality. There is no separate CPU-only or name-variation mining path — you need a GPU and the full image-generation stack to earn rewards.

## Subnet Phases & Timeline

| Date | Milestone |
|------|-----------|
| **Mar 16 - Apr 20, 2026** | **Phase 4 Cycle 2 Execution (current stage; reward allocation live)** |
| Mar 16, 2026 | Phase 4 Cycle 2 sandbox ends; execution phase begins |
| Jan 22 - Mar 2, 2026 | Phase 4 Cycle 1 Execution |
| Jan 8 - Jan 22, 2026 | Phase 4 Cycle 1 sandbox/calibration window |

For the full roadmap and detailed architecture, see [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf).

## Requirements

- Python 3.10 or higher
- A Bittensor wallet with TAO for registration
- A Hugging Face account with an API token (free)
- Access to the base image models used by the miner:
  - [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (used by `flux_klein` and `pulid_flux2`)
  - [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (used by `pulid` fallback and `flux_kontext` alternative)
  - Optional alternative: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- **GPU with at least 8GB VRAM** (NVIDIA CUDA or Apple Silicon MPS) — required for production mining
- CPU: 8 physical cores recommended (12+ preferred when handling image jobs continuously)
- At least 16GB RAM (32GB+ recommended)
- Storage: 80GB minimum free disk, 150GB+ recommended (OS + repo + venv + diffusion model weights + cache + outputs)
- Base Python packages from `requirements.txt` plus miner image packages from `requirements-miner.txt` (`torch`, `diffusers`, `transformers`, `Pillow`, `opencv-python`, etc.)
- Open port 8091 for validator communication ([Network Setup Guide](network_setup.md))

> **Machine sizing:** Target at least **8 cores / 16GB RAM / 80GB free disk + 8GB VRAM GPU** (better: 12+ cores / 32GB RAM / 150GB+ disk + higher-VRAM GPU).

---

## Installation

**If you are not used to the command line or Python**, use **Option 1**. It runs one script that installs dependencies for you.

**If you are comfortable with Linux/macOS, Python virtual environments, and package managers**, use **Option 2** (step-by-step guide below).

### Option 1: Automated setup (recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. **Run the setup script** (targets **Ubuntu/Debian** with `apt`; installs system packages, Python venv, image-generation dependencies, AdaFace, and more):
```bash
bash scripts/miner/setup.sh
```

On **macOS** (or if you prefer not to use `apt`), use **Option 2** instead.

3. **Activate the virtual environment** (do this every time you open a new terminal before running the miner):
```bash
source miner_env/bin/activate
```

4. **Set your Hugging Face token and GPU device** before the first miner run (create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept model licenses):
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"   # or mps (Apple) or cpu (slow, not recommended)
```
If the script could not pre-download FLUX weights, run `python -m MIID.miner.downloading_model` after `HF_TOKEN` is set.

5. **Create a wallet and register** (needs a small amount of TAO; skip if you already have a registered miner). For first-time setup, we recommend starting on testnet first (see [First-time setup: run on testnet first](#first-time-setup-run-on-testnet-first)):
```bash
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
```

6. **Start the miner**
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug
```

You should see `Phase 4 image generation: ENABLED` in the logs. If you see `DISABLED`, complete the [troubleshooting steps](#phase-4-image-generation-disabled-missing-packages) below — without image generation enabled you will not earn rewards.

### First-time setup: run on testnet first

For first-time setup, testnet is the safest place to validate your miner end-to-end before running on mainnet.  
Our MIID testnet netuid is **`322`**.

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 322 --subtensor.network test --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug --axon.port <YOUR_OPEN_PORT> --axon.ip 0.0.0.0 --axon.external_ip <YOUR_PUBLIC_IP> --axon.external_port <YOUR_PUBLIC_PORT>
```

View your testnet results on [Weights & Biases (MIID subnet 322)](https://wandb.ai/MIID-dev-test/subnet322-test/table?nw=nwuseraromanhigh).

Optional walkthrough video (different subnet, but setup flow is very similar): [YouTube guide](https://www.youtube.com/watch?v=UH_sOZSIk10&list=PLqsRtfujbWUklHVRqOAHQ7tzXYZ8OAZlR).

Flag breakdown for the command above:

- `--netuid 322`: selects the subnet to join; use `322` for MIID testnet.
- `--subtensor.network test`: points Bittensor to the test network instead of finney/mainnet.
- `--subtensor.chain_endpoint wss://test.finney.opentensor.ai:443`: sets the websocket RPC endpoint for chain communication.
- `--wallet.name miner_wallet`: chooses the coldkey wallet name that holds your miner identity.
- `--wallet.hotkey miner_hotkey`: chooses the hotkey used by the miner process to sign and serve requests.
- `--logging.debug`: enables verbose logs to make first-run troubleshooting easier.
- `--axon.port <YOUR_OPEN_PORT>`: local listening port where your miner axon serves validator requests.
- `--axon.ip 0.0.0.0`: binds the service to all local interfaces so it can accept incoming traffic.
- `--axon.external_ip <YOUR_PUBLIC_IP>`: advertises the public IP validators should use to reach your miner.
- `--axon.external_port <YOUR_PUBLIC_PORT>`: advertises the public port validators should use (often your forwarded/NAT port).

### Option 2: Manual installation

Follow the **[step-by-step guide](#step-by-step-guide-manual-installation)** from **Step 1** through **Step 12**, then start the miner.

### Installation recommendations

- **Non-technical users**: Prefer **Option 1**, then Steps 3–6 above (activate → env vars → wallet/register → run).
- **Technical users**: Use **Option 2** if you want full control, custom paths, or a non-Ubuntu setup.
- **Production**: After either path, consider a process manager (systemd, supervisor, tmux, pm2 — see [Background Mining Guide](background_mining.md)), monitoring, and logging.

---

## Step-by-step guide (manual installation)

Use this section when you chose **Option 2**, or when you need to **verify or redo** a single step.

### Step 1: Install System Dependencies

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake git curl wget jq nano
```

**macOS:**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python3 git curl wget jq cmake
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

From this point on, **all commands are run from inside the `MIID-subnet` folder**.

### Step 3: Create and Activate a Virtual Environment

```bash
python3 -m venv miner_env
source miner_env/bin/activate
```

**Every time you open a new terminal**, activate again:

```bash
cd MIID-subnet
source miner_env/bin/activate
```

### Step 4: Install Base Python Dependencies

```bash
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e .
```

### Step 5: Install Miner Image-Generation Packages

```bash
pip install -r requirements-miner.txt
```

Validators do **not** need `requirements-miner.txt` — only miners.

**Optional:** `pip install timelock` (drand encryption; otherwise sandbox raw-bytes fallback).

### Step 6: Create a Hugging Face Account and Token

1. [huggingface.co/join](https://huggingface.co/join)
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a **Read** token (`hf_...`)
3. Accept licenses: [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), and optionally [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
4. Set the token:

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

> To persist: add to `~/.bashrc` or `~/.zshrc`.

### Step 7: Set Up AdaFace (Face Identity Validation)

```bash
git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace
mkdir -p MIID/miner/AdaFace/pretrained
gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
```

Manual download: [Google Drive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) → place at `MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt`.

### Step 8: Set Your GPU Device

```bash
export FLUX_DEVICE="cuda"    # NVIDIA
# export FLUX_DEVICE="mps"   # Apple Silicon
# export FLUX_DEVICE="cpu"    # slow — not viable for production mining
```

### Step 9: Pre-Download the Diffusion Model (Recommended)

```bash
python -m MIID.miner.downloading_model
```

> **Note:** FLUX.2-klein is the default baseline in code. You can experiment with other diffusion setups in `MIID/miner/generate_variations.py` (model list and `MIID_MODEL`). Better models can give you an edge.

### Step 10: Create a Bittensor Wallet

```bash
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
```

**Write down and securely store your mnemonic seed phrases.**

### Step 11: Register Your Miner on the Subnet

```bash
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
```

### Step 12: Start the Miner

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py \
  --netuid 54 \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --wallet.name miner_wallet \
  --wallet.hotkey miner_hotkey \
  --logging.debug
```

The miner logs `Phase 4 image generation: ENABLED`, then **randomly picks one of three image models** per session (unless `MIID_MODEL` is set).

### Quick Start Cheat Sheet

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake git curl wget jq nano
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
python3 -m venv miner_env
source miner_env/bin/activate
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e .
pip install -r requirements-miner.txt
pip install timelock      # optional
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace
mkdir -p MIID/miner/AdaFace/pretrained
gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
export FLUX_DEVICE="cuda"
python -m MIID.miner.downloading_model
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug
```

---

## Running a Miner (register + start)

If you **already created a wallet and registered**, skip step 1 below.

1. Register your miner to the subnet (skip if already registered):
```bash
btcli subnet register --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

2. Start your miner:
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --logging.debug
```

For detailed information about logging setup and management, see the [Logging Guide](logging.md).

For running your miner in the background (recommended for production), see the [Background Mining Guide](background_mining.md).

---

## Configuration Options

Command-line arguments:

- `--neuron.logging.debug`: Enable debug logging
- `--neuron.log_responses`: Save miner responses for analysis
- `--neuron.response_cache_dir`: Directory to store response logs

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` or `HUGGINGFACE_TOKEN` | Your Hugging Face API token | none |
| `FLUX_DEVICE` | Device for image generation (`cuda`, `mps`, or `cpu`) | `cpu` |
| `MIID_MODEL` | Force `flux_klein`, `flux_kontext`, `pulid`, `pulid_flux2`, or `qwen` | random |
| `MIID_INFERENCE_STEPS`, `MIID_GUIDANCE_SCALE` | Tune generation (see `MIID/miner/generate_variations.py`) | see code |

Example with custom configuration:
```bash
export HF_TOKEN="hf_..."
export FLUX_DEVICE="cuda"
export MIID_MODEL="flux_klein"
python neurons/miner.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney --neuron.logging.debug
```

---

## How It Works

### Face Image Variation (Phase 4)

1. The miner receives an image request containing:
   - A base face image (base64-encoded)
   - A list of variation requests, each specifying a **type** and **intensity**
   - The schema supports multiple variation requests, but current validator runtime sends **one variation per request** and cycles sequentially

2. For each requested variation, the miner:
   - Decodes the base image
   - Generates a variation using the diffusion pipeline (`generate_variations.py`) with the specified parameters
   - Validates that face identity is preserved (AdaFace similarity check)
   - Encrypts the result using drand timelock encryption when available
   - Uploads the encrypted image to S3

3. The miner returns signed S3 submission references to the validator.

**Variation types and intensities (Cycle 2):**

| Type | Light | Medium | Far |
|------|-------|--------|-----|
| **Pose edit** | ±15° rotation (slight tilt/turn) | ±30° rotation (clear head turn) | >±45° rotation (near-profile) |
| **Expression edit** | Slight smile, minor brow movement | Smile, serious, mildly surprised | Laughing, surprised, concerned |
| **Lighting edit** | Subtle brightness/contrast change | Directional light, noticeable shadows | Strong shadows, dramatic contrast |
| **Background edit** | Color shift, blur adjustment | Different environment type | Dramatic/contrasting environment |

**Screen replay (Cycle 2 only):** Simulate the face as shown on a device screen (e.g. photo on phone/tablet/laptop/monitor/TV). At least **two** of these visual cues must be clearly visible: moiré/pixel grid, screen glare hotspots, perspective/keystone distortion, gamma or contrast shift typical of display capture, or edge/crop cues (screen borders, bezel reflections).

Validators may also request **accessories** on background edits (head coverings, hats, etc.).

---

## How the Three Image Models Work

The miner ships with **three** diffusion models. By default, the miner **randomly picks one at the start of each query** (unless `MIID_MODEL` is set).

| Model | Description | VRAM needed |
|-------|-------------|-------------|
| **FLUX.2-klein** (`flux_klein`) | Fast baseline model; lowest overhead and most stable default path | ~8 GB |
| **PuLID** (`pulid`) | Identity-focused path: tries Nunchaku PuLID on CUDA, otherwise falls back to FLUX.1-Kontext | ~12 GB+ (Nunchaku path); fallback depends on Kontext |
| **PuLID-FLUX2** (`pulid_flux2`) | FLUX.2-klein backbone for PuLID-FLUX2-style identity experiments | ~8 GB |

Default behavior is:
- If `MIID_MODEL` is set, the miner uses that exact model.
- If `MIID_MODEL` is not set and `MIID_MODEL_RANDOM` is unset (or set to `1`), it randomly picks from the **three base models** above at the start of each query.
- If `MIID_MODEL` is not set and `MIID_MODEL_RANDOM=0`, it uses `flux_klein`.

If a model fails to load, the miner falls back to `flux_klein`. Licenses, pipeline details, and model wiring are documented in `MIID/miner/generate_variations.py`.

### Recommended alternatives (easy to enable now)

These are already wired in code and can be enabled by setting `MIID_MODEL`:

1. **FLUX.1-Kontext (`flux_kontext`)**
   - Strong text-guided editing quality.
   - Best on higher-memory GPUs (commonly ~24 GB class).
   - Model: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

2. **Qwen Image Edit (`qwen`)**
   - Strong instruction-following image editing.
   - Requires newer diffusers build and `torchvision`.
   - Model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

---

## Reward System

**Validators score miners exclusively on face image variation quality.** There is no separate reward path for name, DOB, or address variations.

You can view your live scores and cycle rewards on the [Yanez Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/).

### How Rewards Are Calculated

Each validation round:
1. Validators send image variation challenges to miners
2. Miners generate and upload encrypted face variations to S3
3. Submissions go through automated pre-checks, identity preservation checks, and manual validation
4. Validators set weights based on image variation scores

**Important:** Phase 4 Cycle 2 execution runs **Mar 16 – Apr 20, 2026**. Face-variation data submitted in Cycle 2 is collected and validated for reward distribution in the next cycle.

### Face Variation Scoring

Face variation rewards are **reputation-based**. Your face variation scores feed into your overall miner reputation, which determines your share of the reward pool.

Face variations go through a multi-stage validation pipeline:

1. **Automatic pre-checks**: Corrupt images, duplicates of the seed, and duplicates of other miners' submissions are filtered out immediately
2. **Identity preservation**: AdaFace similarity must be above threshold (ada_sim >= 0.6 for basic pass)
3. **Variation detection**: Automated systems detect what type of variation was actually produced
4. **Manual validation**: Reviewers check whether the miner followed the query, preserved identity, and produced the exact requested transformation

**Score hints from automated checks:**
- **Score 0**: Failed pre-checks (corrupt, duplicate, identity not preserved)
- **Score -1**: Duplicate detected
- **Score 2**: Partial match (more than one variation detected, or intensity mismatch)
- **Score 3**: Acceptable (identity preserved, label close to claimed)
- **Score 5**: Exact transformation match with identity preserved

For detailed examples of accepted and rejected face variations, see [Face Variation Reward System (PDF)](Face%20Variation%20Reward%20System.pdf).

## How Does a Miner Compete?

Miners compete to generate the highest-quality face image variations:

- Run the image generation pipeline reliably while minimizing latency and compute cost
- Produce **exactly** the requested transformation — no more, no less — while preserving the subject's identity
- Experiment with better diffusion models, fine-tuned parameters, and generation settings
- Validate identity preservation locally before submitting (AdaFace similarity >= 0.7 is the miner-side threshold)
- Build and maintain reputation over time for higher reward tiers

## Common Face Variation Rejection Reasons

- **Identity not preserved**: The generated face looks like a different person (ada_sim below 0.4). Use models and settings that maintain facial features.
- **Copy-paste backgrounds**: Simply pasting the face onto a new background results in negative scores. The background edit must be a natural transformation.
- **Multiple unintended variations**: If asked for a pose edit but the expression also changes significantly, the submission gets a partial match penalty.
- **Corrupted/unreadable images**: Ensure your generation pipeline outputs valid, uncorrupted images.
- **Duplicates**: Submitting the seed image as-is or duplicating another miner's output.

## Performance Tips

1. **Use a GPU** — image generation on CPU is too slow for production and you will not keep up with validator requests.
2. **Tune your generation parameters**: Check `MIID/miner/generate_variations.py`.
   - `NUM_INFERENCE_STEPS` (default 20): Increase for higher quality (slower), decrease for speed.
   - `GUIDANCE_SCALE` (default 3.5): Adjust how closely the model follows the prompt.
3. **Timeout Awareness**: Phase 4 requests have an extended timeout (+15 minutes) to allow for generation, but speed still matters for throughput.
4. **Validate identity preservation locally** before submitting (AdaFace similarity >= 0.7 is the miner-side threshold).
5. **Produce only the requested variation type and intensity** — extra changes hurt your score.
6. **Experiment** with different diffusion models or fine-tuned checkpoints for better results.
7. **Ensure background edits are natural transformations**, not copy-paste composites.
8. Ensure reliable internet connectivity and monitor logs for errors.
9. Keep your miner code up to date with the latest repository changes.

---

## Troubleshooting

### "Phase 4 image generation: DISABLED (missing packages)"

```bash
pip install -r requirements-miner.txt
```

Then complete the AdaFace, Hugging Face token, GPU device, and model download steps above and restart. **Without this enabled, you cannot earn rewards.**

### "Missing Hugging Face token"

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### "Failed to load model — falling back to flux_klein"

Normal if Kontext cannot load.

### AdaFace / checkpoint issues

See [Step 7](#step-7-set-up-adaface-face-identity-validation).

### Port 8091

See [Network Setup Guide](network_setup.md).

### Out of memory

```bash
export MIID_MODEL="flux_klein"
```

---

## Running in the Background

See [Background Mining Guide](background_mining.md). Export `HF_TOKEN` and `FLUX_DEVICE` before starting.

---

## Reference Documents

- [Yanez Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/) — Live miner scores, cycle rewards, and reputation leaderboard
- [Background Mining Guide](background_mining.md) — Run in the background with tmux or pm2
- [Network Setup Guide](network_setup.md) — Port and firewall configuration
- [Logging Guide](logging.md) — Logging setup and management
- [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf) — Full subnet architecture, roadmap, and phase descriptions
- [Face Variation Reward System (PDF)](Face%20Variation%20Reward%20System.pdf) — Face variation validation pipeline, scoring, and examples of accepted/rejected submissions
