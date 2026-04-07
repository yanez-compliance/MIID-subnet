# MIID Miner

This document explains how to run a MIID miner on the Bittensor network. The MIID subnet (Subnet 54) focuses on generating identity data for compliance and security research, including name spelling variations and identity-preserving face image variations.

## Overview

MIID miners handle three types of tasks from validators:

**1. Identity Demographic Variation Generation (Active; lower weight in current cycle)**
- Receive identity data (name, DOB, address) and a query template
- Generate spelling/transliteration variations using a local LLM
- Return structured variations to the validator
- This task remains active in Phase 4, but with lower reward weight than UAV-focused components

**2. Address UAV — Unknown Attack Vectors (Phase 3; not in scope Phase 4 Cycle 2)**
- For a selected seed identity, generate an address that looks legitimate but might fail geocoding or validation
- Provide the address variant, an explanation label, and latitude/longitude coordinates
- *UAV address submissions are not accepted in Phase 4 Cycle 2.*

**3. Face Image Variation Generation (Phase 4 — Current)**
- Receive a base face image and variation requirements
- **Variation types (Phase 4 Cycle 2):** pose_edit, lighting_edit, expression_edit, background_edit, and **screen_replay**
- Generate identity-preserving image variations using a diffusion model
- Encrypt and upload results to S3
- Return signed submission references to the validator

All tasks can be sent in a single synapse. Validators decide which tasks to include per request.

## Subnet Phases & Timeline

The subnet has evolved in cycles. Use these dates from the latest docs:

| Date | Milestone |
|------|-----------|
| **Mar 16 - Apr 13, 2026** | **Phase 4 Cycle 2 Execution (current stage; reward allocation live)** |
| Mar 16, 2026 | Phase 4 Cycle 2 sandbox ends; execution phase begins |
| Jan 22 - Mar 2, 2026 | Phase 4 Cycle 1 Execution |
| Jan 8 - Jan 22, 2026 | Phase 4 Cycle 1 sandbox/calibration window |
| Dec 31, 2025 | Phase 3 Cycle 1 UAV intake stopped |

For the full roadmap and detailed architecture, see [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf).

## Requirements

### Base Requirements (Name Variations)
- Python 3.10 or higher
- A Bittensor wallet with TAO for registration
- A local LLM via Ollama (default: llama3.1:latest)
- Sufficient storage for LLM model weights (~10GB or more depending on model)
- At least 8GB RAM (16GB+ recommended)
- Open port 8091 for validator communication ([Network Setup Guide](network_setup.md))

### Additional Requirements for Phase 4 (Face Image Variations)
- A Hugging Face account with an API token (free)
- Access to the base image models used by the miner:
  - [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (used by `flux_klein` and `pulid_flux2`)
  - [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (used by `pulid` fallback and `flux_kontext` alternative)
  - Optional alternative: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- GPU with at least 8GB VRAM recommended (NVIDIA CUDA or Apple Silicon MPS)
- Additional storage for diffusion model weights (typically ~10GB for FLUX.2-klein; more if you also use Kontext/Qwen)
- Base Python packages from `requirements.txt` plus miner image packages from `requirements-miner.txt` (`torch`, `diffusers`, `transformers`, `Pillow`, `opencv-python`, etc.)

---

There are **two setup paths** for installation:

| Path | What it does | Extra packages | GPU needed? |
|------|-------------|----------------|-------------|
| **Basic** | Name/address/DOB variation generation only | None beyond base install | No |
| **Full (recommended)** | Name variations **+ face image variations** (Phase 4) | `requirements-miner.txt` (`diffusers`, `transformers`, `accelerate`, `opencv-python`, AdaFace setup) | Yes (8+ GB VRAM) |

> **Which should I pick?** If you have a GPU and want to earn the maximum possible rewards (including face-variation reputation), choose **Full**. If you just want to get started quickly or do not have a GPU, choose **Basic** -- you can upgrade to Full later without losing anything.

---

## Installation

**If you are not used to the command line or Python**, use **Option 1**. It runs one script that installs most dependencies for you. You clone the repo **once**, then follow the numbered steps under Option 1 through to “Start the miner.”

**If you are comfortable with Linux/macOS, Python virtual environments, and package managers**, you can use **Option 2** instead: follow the [step-by-step guide](#step-by-step-guide-manual-installation) below from Step 1. You still clone the repo **only once** (in Step 2 of that guide). Option 2 does not repeat the clone commands from Option 1—it is a separate path.

### Option 1: Automated setup (recommended)

1. **Clone the repository** (this is the only time you need to clone):
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. **Run the setup script** (targets **Ubuntu/Debian** with `apt`; installs system packages, Python venv, Ollama, `pip install -e .`, and more):
```bash
bash scripts/miner/setup.sh
```
When asked, choose **Basic** (name variations only) or **Full** (name + face images). Non-interactive: `bash scripts/miner/setup.sh --basic` or `bash scripts/miner/setup.sh --full`.

On **macOS** (or if you prefer not to use `apt`), use **Option 2** instead—the step-by-step guide works on macOS.

3. **Activate the virtual environment** (do this every time you open a new terminal before running the miner):
```bash
source miner_env/bin/activate
```

4. **If you chose Full**, set your Hugging Face token and GPU device **before** the first miner run (create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept model licenses—see [Path B](#path-b-full-setup-name--image-variations) for details):
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"   # or mps (Apple) or cpu (slow)
```
If the script could not pre-download FLUX weights, run `python -m MIID.miner.downloading_model` after `HF_TOKEN` is set.

5. **Create a wallet and register** (needs a small amount of TAO; skip if you already have a registered miner). If this is your first time setting up, we recommend starting on testnet first (see [First-time setup: run on testnet first](#first-time-setup-run-on-testnet-first)):
```bash
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
```

6. **Start the miner**
```bash
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug
```
Use the same wallet/hotkey names you created. If you chose **Basic**, you may see `Phase 4 image generation: DISABLED` in the logs—that is expected.

### First-time setup: run on testnet first

For first-time setup, testnet is the safest place to validate your miner end-to-end before running on mainnet.  
Our MIID testnet netuid is **`322`**.

```bash
python neurons/miner.py --netuid 322 --subtensor.network test --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug --axon.port <YOUR_OPEN_PORT> --axon.ip 0.0.0.0 --axon.external_ip <YOUR_PUBLIC_IP> --axon.external_port <YOUR_PUBLIC_PORT>
```

View your testnet results on [taostats.io](https://taostats.io).

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

Do **not** clone the repository here if you already cloned it for Option 1. Option 2 is for people who want to run each command themselves.

Follow the **[step-by-step guide](#step-by-step-guide-manual-installation)** from **Step 1** through **Step 7**, then either **[Path A](#path-a-basic-setup-name-variations-only)** (Basic) or **[Path B](#path-b-full-setup-name--image-variations)** (Full). The guide includes a single `git clone` in Step 2, venv and `pip install -e .`, Ollama, wallet, register, and optional Phase 4 steps.

### Installation recommendations

- **Non-technical users**: Prefer **Option 1**, then only Steps 3–6 above (activate → env vars if Full → wallet/register → run).
- **Technical users**: Use **Option 2** if you want full control, custom paths, or a non-Ubuntu setup; use the automated script if you want speed and consistency.
- **Production**: After either path, consider a process manager (systemd, supervisor, tmux, pm2—see [Background Mining Guide](background_mining.md)), monitoring, logging, and a GPU for LLM and image workloads.

---

## Step-by-step guide (manual installation)

Use this section when you chose **Option 2**, or when you need to **verify or redo** a single step (for example reinstalling packages). If you used **Option 1**, the script already performed many of these steps on Ubuntu/Debian—you can skip ahead to wallet creation and starting the miner unless something failed.

### Step 1: Install System Dependencies

Open a terminal and install the base packages.

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

This installs the shared packages (`bittensor`, `torch`, `Pillow`, `ollama`, etc.).

### Step 5: Install Ollama (for Name Variations)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:latest
ollama list
```

You should see `llama3.1:latest` in the list.

### Step 6: Create a Bittensor Wallet

```bash
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
```

**Write down and securely store your mnemonic seed phrases.**

### Step 7: Register Your Miner on the Subnet

```bash
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
```

---

## Path A: Basic Setup (Name Variations Only)

If you completed Steps 1-7 above, you are ready to run. No extra packages are needed.

### Run the Miner (Basic)

```bash
python neurons/miner.py \
  --netuid 54 \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --wallet.name miner_wallet \
  --wallet.hotkey miner_hotkey \
  --logging.debug
```

You will see:

```
Phase 4 image generation: DISABLED (missing packages).
```

This is expected. Name/address/DOB variation requests work normally; image requests are skipped.

**When you are ready for image-variation rewards**, continue to [Path B](#path-b-full-setup-name--image-variations).

### Basic Quick Start Cheat Sheet

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake git curl wget jq nano
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
python3 -m venv miner_env
source miner_env/bin/activate
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e .
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:latest
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug
```

---

## Path B: Full Setup (Name + Image Variations)

Complete Steps 1-7 first. Then enable Phase 4.

### Step 8: Install Miner Image-Generation Packages

```bash
pip install -r requirements-miner.txt
```

Validators do **not** need `requirements-miner.txt` -- only miners.

**Optional:** `pip install photomaker` (third image model; falls back to FLUX.2-klein if missing). **Optional:** `pip install timelock` (drand encryption; otherwise sandbox raw-bytes fallback).

### Step 9: Create a Hugging Face Account and Token

1. [huggingface.co/join](https://huggingface.co/join)
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) -- create a **Read** token (`hf_...`)
3. Accept licenses: [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), and optionally [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
4. Set the token:

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

> To persist: add to `~/.bashrc` or `~/.zshrc`.

### Step 10: Set Up AdaFace (Face Identity Validation)

```bash
git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace
mkdir -p MIID/miner/AdaFace/pretrained
gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
```

Manual download: [Google Drive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) → place at `MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt`.

### Step 11: Set Your GPU Device

```bash
export FLUX_DEVICE="cuda"    # NVIDIA
# export FLUX_DEVICE="mps"   # Apple Silicon
# export FLUX_DEVICE="cpu"    # slow
```

### Step 12: Pre-Download the Diffusion Model (Recommended)

```bash
python -m MIID.miner.downloading_model
```

> **Note:** FLUX.2-klein is the default baseline in code. You can experiment with other diffusion setups in `MIID/miner/generate_variations.py` (model list and `MIID_MODEL`). Better models can give you an edge.

### Run the Miner (Full)

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

### Full Quick Start Cheat Sheet

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
pip install photomaker    # optional
pip install timelock      # optional
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:latest
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

If you **already created a wallet and registered** in [Option 1](#option-1-automated-setup-recommended) (steps 5–6) or in [Steps 6–7](#step-6-create-a-bittensor-wallet) of the manual guide, **skip step 1 below** and go straight to starting the miner. If you **already started the miner** from those sections, you do not need to repeat this whole block—it is here as a quick reference.

1. Register your miner to the subnet (skip if already registered):
```bash
btcli subnet register --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

2. Start your miner:
```bash
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --logging.debug
```

For detailed information about logging setup and management, see the [Logging Guide](logging.md).

For running your miner in the background (recommended for production), see the [Background Mining Guide](background_mining.md).

---

## Configuration Options

You can configure your miner with the following command-line arguments:

- `--neuron.model_name`: The Ollama model to use (default: tinyllama:latest)
- `--neuron.logging.debug`: Enable debug logging
- `--neuron.log_responses`: Save miner responses for analysis
- `--neuron.response_cache_dir`: Directory to store response logs

Environment variables for Phase 4 (Full path):
- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: Your Hugging Face API token
- `FLUX_DEVICE`: Device for image generation (`cuda`, `mps`, or `cpu`)
- `MIID_MODEL`: Force `flux_klein`, `flux_kontext`, or `photomaker`
- `MIID_INFERENCE_STEPS`, `MIID_GUIDANCE_SCALE`: Tune generation (see `MIID/miner/generate_variations.py`)

Example with custom configuration:
```bash
export HF_TOKEN="hf_..."
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney --neuron.model_name mistral:7b --neuron.logging.debug
```

### Environment Variables (reference)

| Variable | Description | Default | Path |
|----------|-------------|---------|------|
| `HF_TOKEN` | Hugging Face token | none | Full |
| `FLUX_DEVICE` | `cuda`, `mps`, or `cpu` | `cpu` | Full |
| `MIID_MODEL` | Force a specific image model | random | Full |

---

## How It Works

### Task 1: Name Variation Generation

1. The miner receives a request from a validator containing:
   - A list of identities (name, DOB, address) to generate variations for
   - A query template with `{name}`, `{dob}`, `{address}` placeholders

2. For each identity, the miner:
   - Formats the query template with the identity data
   - Sends the formatted query to the local LLM (Ollama)
   - Extracts the generated variations from the LLM response

3. The miner processes all responses and returns a dictionary mapping each input name to a list of variations.

### Task 2: Address UAV (Unknown Attack Vectors)

*Not in scope for Phase 4 Cycle 2 — address UAV submissions are not accepted.*

When in scope, validators mark one high-risk seed identity for UAV in each challenge. For that selected seed, the miner should return:

1. An **address variant** that looks legitimate but might fail geocoding (e.g., common typos, local abbreviations, missing street directions)
2. A **label** explaining why the address could be valid
3. **Latitude/longitude** coordinates for the address

The UAV response is included alongside the normal name variations for the selected seed identity:
```json
{
  "seed_name": {
    "variations": [["name_var", "dob_var", "addr_var"], ...],
    "uav": {
      "address": "123 Main Str, City",
      "label": "Common typo",
      "latitude": 40.7128,
      "longitude": -74.0060
    }
  }
}
```

Notes:
- UAVs submitted under the wrong seed name can be rejected by validator-side parsing.
- `address` and `label` are mandatory for UAV payloads; coordinates should be provided.

### Task 3: Face Image Variation (Phase 4)

1. The miner receives an image request containing:
   - A base face image (base64-encoded)
   - A list of variation requests, each specifying a **type** and **intensity**
   - The schema supports multiple variation requests, but current validator runtime sends **one variation per request** and cycles sequentially

2. For each requested variation, the miner:
   - Decodes the base image
   - Generates a variation using the diffusion pipeline (FLUX / PhotoMaker per `generate_variations.py`) with the specified parameters
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

## How the Three Image Models Work (Full path)

The miner ships with **three** diffusion models; each session **randomly picks one** unless `MIID_MODEL` is set.

| Model | Description | VRAM needed |
|-------|-------------|-------------|
| **FLUX.2-klein** (`flux_klein`) | Fast baseline model; lowest overhead and most stable default path | ~8 GB |
| **PuLID** (`pulid`) | Identity-focused path: tries Nunchaku PuLID on CUDA, otherwise falls back to FLUX.1-Kontext | ~12 GB+ (Nunchaku path); fallback depends on Kontext |
| **PuLID-FLUX2** (`pulid_flux2`) | FLUX.2-klein backbone for PuLID-FLUX2-style identity experiments | ~8 GB |

Default behavior is:
- If `MIID_MODEL` is set, the miner uses that exact model.
- If `MIID_MODEL` is not set, it defaults to `flux_klein`.
- If `MIID_MODEL_RANDOM=1`, it randomly picks from the **three base models** above.

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

### Paid model recommendations (future integration)

These are not active in the miner by default, but are good candidates for advanced setups:

- **Soul**
- **Grok Imagination**
- **Seedream**
- **Nonobana**
- **Nonobana2**

---

## Reward System

### How Rewards Are Calculated

Rewards combine:
- **KAV** (online quality score)
- **UAV** (reputation-based component)

You can view your live scores, cycle rewards, and reputation on the [Yanez Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/).

**KAV quality score breakdown:**

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Address variations** | **70%** | Realistic, geocodable addresses within the seed country/city |
| **Name quality** | **20%** | Phonetic/orthographic similarity, count, uniqueness, rule compliance |
| **DOB variations** | **10%** | Coverage of date ranges (±1 day, ±3 days, ±30 days, ±90 days, ±365 days, year+month) |

> **Address is 70% of the KAV score.** This is the single most important factor in quality scoring. Addresses must pass heuristic checks (proper format, 30-300 chars, real structure), region validation (correct country/city), and API geocoding.

**KAV/UAV reward split (of the 30% kept after burn):**

| Pool | Share | Who Gets It |
|------|-------|-------------|
| **KAV (Known Attack Vectors)** | 15% | All miners, based on quality score |
| **UAV (Unknown Attack Vectors)** | 85% | Established miners only, based on reputation |

New miners receive **only KAV rewards** (15% of the kept fraction) until they build UAV reputation. This means building reputation early is critical to accessing the larger reward pool.

### KAV -- Online Quality Scoring

All miners are scored on:
- **Phonetic similarity** (30% of quality): How well variations sound like the original (Soundex, Metaphone, NYSIIS)
- **Orthographic similarity** (70% of quality): Edit distance from the original
- **Count**: Correct number of variations (penalty for too few or too many)
- **Uniqueness**: Diverse variations (duplicates penalized)
- **Rule compliance**: Whether variations follow specified transformation rules
- **Address validation**: Three-stage check -- heuristic format, region match, then API geocoding
- **DOB variation coverage**: At least one variation in each date range category

### UAV -- Reputation-Based Rewards

A reputation-weighted system that applies across both name variations and face variations:
- New miners start with zero UAV reputation and build it over time
- Higher reputation unlocks access to the 85% UAV reward pool
- Reputation decays if quality drops

**Reputation tiers and multipliers:**

| Tier | Score Range | Multiplier |
|------|-------------|------------|
| Diamond | 50+ | 1.15x |
| Gold | 30 - 50 | 1.10x |
| Silver | 15 - 30 | 1.05x |
| Bronze | 5 - 15 | 1.02x |
| Neutral | 0.1 - 5 | 1.00x |
| Watch | < 0.1 | 0.90x |

For detailed scoring rules, examples, and the latest updates, see [Unknown Attack Vectors (UAV) Reward System (PDF)](Unknown%20Attack%20Vectors%20(UAV)%20Reward%20System-updates.pdf).

### Face Variation Scoring (Phase 4)

Face variation rewards are **reputation-based**, following the same reputation framework as UAV. Your face variation scores feed into your overall miner reputation, which determines your share of the reward pool.

**Important:** Phase 4 Cycle 2 execution runs **Mar 16 – Apr 13, 2026**. Face-variation data submitted in Cycle 2 is collected and validated for reward distribution in the next cycle.

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

Miners in the Yanez subnet compete to generate the highest-quality identity data. Success depends not just on participation but on outperforming others across several dimensions:

- Run the core mining code to generate valid responses while minimizing latency and compute cost.
- Accurately decode structured and abstract constraints to deliver precise, tailored identity outputs.
- Experiment with better models, fine-tuned parameters, and advanced prompt engineering to stay ahead.
- Ensure outputs are formatted and error-free to avoid penalties and increase scores.
- For face variations: produce **exactly** the requested transformation -- no more, no less -- while preserving the subject's identity.
- Build and maintain your reputation over time to access higher UAV reward tiers.

## Common Face Variation Rejection Reasons

Understanding why submissions get rejected helps you improve quality:

- **Identity not preserved**: The generated face looks like a different person (ada_sim below 0.4). Use models and settings that maintain facial features.
- **Copy-paste backgrounds**: Simply pasting the face onto a new background results in negative scores. The background edit must be a natural transformation.
- **Multiple unintended variations**: If asked for a pose edit but the expression also changes significantly, the submission gets a partial match penalty.
- **Corrupted/unreadable images**: Ensure your generation pipeline outputs valid, uncorrupted images.
- **Duplicates**: Submitting the seed image as-is or duplicating another miner's output.

## Performance Tips

### Name and Address Variations
1. **Prioritize address quality** -- addresses are 70% of your score. Generate realistic, properly formatted addresses within the seed country/city that can pass geocoding validation
2. Addresses must be 30-300 characters, contain real street structure (numbers, letters, commas), and match the seed region
3. Cover all DOB variation ranges (±1 day, ±3 days, ±30 days, ±90 days, ±365 days, year+month only)
4. Use a powerful GPU if available to speed up LLM inference
5. Consider using a high-quality LLM model for better variations
6. Implement custom post-processing to filter and enhance LLM outputs

### Face Image Variations
1. **Use a GPU** -- image generation on CPU is too slow for production.
2. **Tune your generation parameters**: Check `MIID/miner/generate_variations.py`.
   - `NUM_INFERENCE_STEPS` (default 20): Increase for higher quality (slower), decrease for speed.
   - `GUIDANCE_SCALE` (default 3.5): Adjust how closely the model follows the prompt.
3. **Timeout Awareness**: Phase 4 requests have an extended timeout (+15 minutes) to allow for generation, but speed still matters for throughput.
4. **Validate identity preservation locally** before submitting (AdaFace similarity >= 0.7 is the miner-side threshold).
5. **Produce only the requested variation type and intensity** -- extra changes hurt your score.
6. **Experiment** with different diffusion models or fine-tuned checkpoints for better results.
7. **Ensure background edits are natural transformations**, not copy-paste composites.

### General
1. Ensure your miner has reliable internet connectivity
2. Monitor your miner's logs for errors or performance issues
3. Regularly update your techniques to remain competitive
4. Keep your miner code up to date with the latest repository changes

---

## Troubleshooting

### "Ollama is required for this miner"
```bash
ollama serve &
```

### "Phase 4 image generation: DISABLED (missing packages)"
```bash
pip install -r requirements-miner.txt
```
Then complete Path B steps (Hugging Face, AdaFace, device, model download) and restart.

### "Missing Hugging Face token"
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### "Failed to load model -- falling back to flux_klein"
Normal if Kontext or PhotoMaker cannot load.

### AdaFace / checkpoint issues
See [Step 10](#step-10-set-up-adaface-face-identity-validation).

### Port 8091
See [Network Setup Guide](network_setup.md).

### Out of memory
```bash
export MIID_MODEL="flux_klein"
```

---

## Running in the Background

See [Background Mining Guide](background_mining.md). **Basic:** no `HF_TOKEN`/`FLUX_DEVICE` needed. **Full:** export both before starting.

---

## Upgrading from Basic to Full

```bash
cd MIID-subnet
source miner_env/bin/activate
pip install -r requirements-miner.txt
pip install photomaker    # optional
pip install timelock      # optional
git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace
mkdir -p MIID/miner/AdaFace/pretrained
gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python -m MIID.miner.downloading_model
```

Restart the miner; you should see `Phase 4 image generation: ENABLED`.

---

## Reference Documents

These documents provide detailed information about the reward system, validation pipeline, and subnet architecture:

- [Yanez Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/) -- Live miner scores, cycle rewards, online KAV scores, and UAV reputation leaderboard
- [Background Mining Guide](background_mining.md) -- Run in the background with tmux or pm2
- [Network Setup Guide](network_setup.md) -- Port and firewall configuration
- [Logging Guide](logging.md) -- Logging setup and management
- [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf) -- Full subnet architecture, roadmap, and phase descriptions
- [Unknown Attack Vectors (UAV) Reward System (PDF)](Unknown%20Attack%20Vectors%20(UAV)%20Reward%20System-updates.pdf) -- UAV scoring rules, reputation mechanics, and worked examples
- [Face Variation Reward System (PDF)](Face%20Variation%20Reward%20System.pdf) -- Face variation validation pipeline, scoring, and examples of accepted/rejected submissions
