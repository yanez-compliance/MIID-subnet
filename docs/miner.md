# MIID Miner

This document explains how to run a MIID miner on the Bittensor network. The MIID subnet (Subnet 54) focuses on generating identity data for compliance and security research, including name spelling variations and identity-preserving face image variations.

## Overview

MIID miners handle three types of tasks from validators:

**1. Identity Demographic Variation Generation (Active; lower weight in current cycle)**
- Receive identity data (name, DOB, address) and a query template
- Generate spelling/transliteration variations using a local LLM
- Return structured variations to the validator
- This task remains active in Phase 4, but with lower reward weight than UAV-focused components

**2. Address UAV -- Unknown Attack Vectors (Phase 3)**
- For a selected seed identity, generate an address that looks legitimate but might fail geocoding or validation
- Provide the address variant, an explanation label, and latitude/longitude coordinates
- Examples: typos ("123 Main Str"), abbreviations ("456 Oak Av"), missing directions ("789 1st St")
- Validators select one high-risk seed per challenge; miners should return UAV output for that selected seed

**3. Face Image Variation Generation (Phase 4 -- Current)**
- Receive a base face image and variation requirements (pose, expression, lighting, background)
- Generate identity-preserving image variations using a diffusion model
- Encrypt and upload results to S3
- Return signed submission references to the validator

All tasks can be sent in a single synapse. Validators decide which tasks to include per request.

## Subnet Phases & Timeline

The subnet has evolved in cycles. Use these dates from the latest docs:

| Date | Milestone |
|------|-----------|
| **Jan 22 - Mar 2, 2026** | **Phase 4 Cycle 1 Execution (current stage, reward allocation live)** |
| Jan 12, 2026 | Manual UAV grading nearly complete, leaderboard published (<2% pending) |
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
- Access to the [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (accept the license on HuggingFace)
- GPU with at least 8GB VRAM recommended (NVIDIA CUDA or Apple Silicon MPS)
- Additional ~10GB storage for diffusion model weights
- `torch`, `diffusers`, `transformers`, and `Pillow` Python packages (included in `requirements.txt`)

## Installation

### Option 1: Automated Setup (Recommended)

1. First, clone the MIID repository:
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. Run the automated setup script:
```bash
bash scripts/miner/setup.sh
```

This script will:
- Install all system dependencies
- Install Python 3.10+ and required packages
- Install Ollama and pull the llama3.1 model
- Create a virtual environment (miner_env)
- Install the MIID package and Bittensor

After running the script:
```bash
source miner_env/bin/activate
```

### Option 2: Manual Installation

If you prefer to install components manually:

1. Clone the MIID repository:
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. Install dependencies:
```bash
python -m pip install -e .
```

3. Install Ollama:
Visit [ollama.ai](https://ollama.ai) for installation instructions.

4. Pull the default LLM model:
```bash
ollama pull llama3.1:latest
```

### Phase 4 Setup: Face Image Generation

To participate in Phase 4 face variation tasks, complete these additional steps:

1. **Create a Hugging Face account** at [huggingface.co/join](https://huggingface.co/join).

2. **Accept the FLUX.2-klein model license** at [huggingface.co/black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) -- click "Agree and access repository".

3. **Create a Hugging Face token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is sufficient).

4. **Set your token** in the environment before running the miner:
```bash
export HF_TOKEN="hf_..."
```

5. **Download the model** (recommended before first run so the miner doesn't download during operation):
```bash
python -m MIID.miner.downloading_model
```

6. **Set your device** (optional, defaults to CPU):
```bash
# NVIDIA GPU (recommended)
export FLUX_DEVICE="cuda"

# Apple Silicon
export FLUX_DEVICE="mps"

# CPU only (slow, not recommended for production)
export FLUX_DEVICE="cpu"
```

> **Note:** FLUX.2-klein is the default base model. You can swap in other diffusion models (FLUX.1-dev, FLUX.1-schnell, Stable Diffusion, SDXL, or custom checkpoints) by changing `MODEL_ID` in `MIID/miner/generate_variations.py`. Better models can give you an edge.

## Installation Recommendations

- **For beginners**: Use the automated setup script (Option 1) for the smoothest experience.
- **For experienced users**: Either option works well. The setup script ensures consistent environments, while manual installation offers more control.
- **For production**: Use the setup script to ensure all dependencies are properly installed, then consider additional hardening:
  - Use a service manager like systemd or supervisor
  - Set up monitoring and logging
  - Use a GPU for both LLM inference and image generation

## Running a Miner

1. Register your miner to the subnet:
```bash
btcli subnet register --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

2. Start your miner:
```bash
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --logging.debug
```

For detailed information about logging setup and management, see the [Logging Guide](logging.md).

For running your miner in the background (recommended for production), see the [Background Mining Guide](background_mining.md).

## Configuration Options

You can configure your miner with the following command-line arguments:

- `--neuron.model_name`: The Ollama model to use (default: tinyllama:latest)
- `--neuron.logging.debug`: Enable debug logging
- `--neuron.log_responses`: Save miner responses for analysis
- `--neuron.response_cache_dir`: Directory to store response logs

Environment variables for Phase 4:
- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: Your Hugging Face API token
- `FLUX_DEVICE`: Device for image generation (`cuda`, `mps`, or `cpu`)

Example with custom configuration:
```bash
export HF_TOKEN="hf_..."
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney --neuron.model_name mistral:7b --neuron.logging.debug
```

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

Validators mark one high-risk seed identity for UAV in each challenge. For that selected seed, the miner should return:

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
   - Generates a variation using the FLUX diffusion model with the specified parameters
   - Validates that face identity is preserved (AdaFace similarity check)
   - Encrypts the result using drand timelock encryption
   - Uploads the encrypted image to S3

3. The miner returns signed S3 submission references to the validator.

**Variation types and intensities:**

| Type | Light | Medium | Far |
|------|-------|--------|-----|
| **Pose edit** | ±15° rotation (slight tilt/turn) | ±30° rotation (clear head turn) | >±45° rotation (near-profile) |
| **Expression edit** | Slight smile, minor brow movement | Smile, serious, mildly surprised | Laughing, surprised, concerned |
| **Lighting edit** | Subtle brightness/contrast change | Directional light, noticeable shadows | Strong shadows, dramatic contrast |
| **Background edit** | Color shift, blur adjustment | Different environment type | Dramatic/contrasting environment |

Validators may also request **accessories** on background edits (head coverings, hats, etc.).

## Reward System

### How Rewards Are Calculated

Rewards combine:
- **KAV** (online quality score)
- **UAV** (reputation-based component)

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
| **KAV (Known Attack Vectors)** | 20% | All miners, based on quality score |
| **UAV (Unknown Attack Vectors)** | 80% | Established miners only, based on reputation |

New miners receive **only KAV rewards** (20% of the kept fraction) until they build UAV reputation. This means building reputation early is critical to accessing the larger reward pool.

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
- Higher reputation unlocks access to the 80% UAV reward pool
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

**Important:** In **Phase 4 Cycle 1 Execution**, miners submit face-variation data that is collected and validated. Reward distribution based on this Cycle 1 face data is applied in **Phase 4 Cycle 2 Execution**.

Face variations go through a multi-stage validation pipeline:

1. **Automatic pre-checks**: Corrupt images, duplicates of the seed, and duplicates of other miners' submissions are filtered out immediately
2. **Identity preservation**: AdaFace similarity must be above threshold (ada_sim >= 0.4 for basic pass)
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

## Reference Documents

These documents provide detailed information about the reward system, validation pipeline, and subnet architecture:

- [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf) -- Full subnet architecture, roadmap, and phase descriptions
- [Unknown Attack Vectors (UAV) Reward System (PDF)](Unknown%20Attack%20Vectors%20(UAV)%20Reward%20System-updates.pdf) -- UAV scoring rules, reputation mechanics, and worked examples
- [Face Variation Reward System (PDF)](Face%20Variation%20Reward%20System.pdf) -- Face variation validation pipeline, scoring, and examples of accepted/rejected submissions
