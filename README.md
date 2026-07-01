<div align="center">
<picture>
    <source srcset="YanezSubnetLogo.png" media="(prefers-color-scheme: dark)">
    <source srcset="YanezSubnetLogo.png" media="(prefers-color-scheme: light)">
    <img src="YanezSubnetLogo.png" width="300">
</picture>

# **MIID Subnet 54 - Identity Testing Network**
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1351934165964296232)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Helpful Hints](https://img.shields.io/badge/Helpful-Hints-blue)](docs/helpful_hints.md)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/yanez-compliance/MIID-subnet)

[⛏️ Mining Guide](docs/miner.md) • [🧑‍🏫 Validator Guide](docs/validator.md) • [📊 Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/)
</div>

---

## 🔍 What is MIID?

**MIID** (Multimodal Inorganic Identity Dataset) is a next-generation **identity testing** and **identity data generation** subnet designed to enhance fraud detection, KYC systems, and biometric verification. Our goal is to provide **financial institutions, security systems, and AI researchers** with a robust dataset of **identity-preserving face image variations** that help identify deepfake and presentation-attack evasion techniques.

By incentivizing **miners** to create high-quality face image variations, **MIID** serves as a critical tool in financial crime prevention, identity resolution, and security intelligence.

## 🎯 Why MIID Matters

Fraudsters use **identity manipulation techniques** to evade detection — including deepfakes, screen replays, and biometric spoofing. **Sanctioned individuals**, **high-risk entities**, and **money launderers** exploit weaknesses in KYC and IDV systems.

MIID **tests and enhances these systems** by:
- ✅ **Simulating Face-Based Adversarial Scenarios** for KYC and biometric screening
- ✅ **Evaluating Identity-Preserving Image Transformations**
- ✅ **Providing Adversarial Face Data for Model Training**

This network helps **governments, financial institutions, and researchers** improve their fraud detection models, making the financial ecosystem safer.

---

## ⚙️ How It Works

### 🛠️ **Miners: Generate Face Image Variations**
Miners process image variation requests from validators and return **identity-preserving face image variations**.

- Receive base face images and variation requirements from validators
- Generate variations using diffusion models: **pose_edit**, **lighting_edit**, **expression_edit**, **background_edit**, and **screen_replay** (Cycle 2)
- Encrypt and upload results to S3; return signed submission references
- **Image generation is the only scored task** — a GPU and the full image stack are required to earn rewards

### 🧑‍🏫 **Validators: Evaluate & Score Miners**
Validators ensure the dataset maintains **high-quality** and **real-world relevance**.

- Issue face image variation challenges
- Run automated pre-checks and identity preservation validation
- Perform manual validation to assess transformation accuracy
- Set miner weights based on image variation quality and reputation

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **GPU with 8GB+ VRAM** (NVIDIA CUDA or Apple Silicon MPS)
- **Hugging Face account and API token** (for diffusion model access)
- **Bittensor wallet with TAO**
- **16GB+ RAM** (32GB recommended)
- **80GB+ free disk** (for model weights and cache)
- **Open port 8091** for miner-to-validator communication ([Network Setup Guide](docs/network_setup.md))

### 1️⃣ **Setup for Miners**
```bash
# Install dependencies (includes image-generation stack)
bash scripts/miner/setup.sh

# Activate the miner environment
source miner_env/bin/activate

# Set required environment variables
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"   # or mps for Apple Silicon

# Start mining
pm2 start python --name neuron-miner -- neurons/miner.py --netuid 54 --wallet.name your-wallet --wallet.hotkey your-hotkey --subtensor.network finney
```

### 2️⃣ **Setup for Validators**
```bash
# Install dependencies
bash scripts/validator/setup.sh

# Activate the validator environment
source validator_env/bin/activate

# Start validating
pm2 start python --name neuron-validator -- neurons/validator.py --netuid 54 --wallet.name your_wallet --wallet.hotkey your_hotkey --subtensor.network finney
```

For detailed instructions, check our **[Mining Guide](docs/miner.md)** and **[Validator Guide](docs/validator.md)**.

---

## 🔥 Why Join MIID?

### 🔐 **Be Part of the Future of Digital Identity Security**
- Help **banks, fintech, and law enforcement agencies** strengthen KYC and biometric fraud detection.
- Contribute to **privacy-preserving AI research**.
- Earn rewards while **enhancing AI-driven face verification and presentation-attack detection**.

### 🏆 **Incentives for Participants**
- **Miners**: Earn rewards for producing high-quality, identity-preserving face image variations.
- **Validators**: Gain influence in network security and reward distribution.

### 🌎 **Real-World Impact**
MIID is not just another AI dataset—it's a **live, evolving system** that **challenges and improves** real-world fraud detection models. Every contribution makes financial systems **safer and more secure**.

---
## 🛣️ Roadmap

### Phase 1: Initial Launch & Name-Based Threat Scenarios (June 2025) [Read more details here](docs/Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf)
- Deploy MIID subnet on Bittensor mainnet. 
- Enable validators to test known threat scenarios against miner responses.
- Introduce name-based execution vectors: phonetic, orthographic, and rule-based variations.

### Phase 2: Miner-Contributed Threat Scenarios (Q4 2025)
- Expand Threat Scenario Query System to allow miners to propose unknown threat scenarios.
- Introduce a **Post-Evaluation System** to systematically validate and assess new miner-submitted threat scenarios.
- Support new evasion tactics, including nickname-based threats, transliteration-based alterations, and middle name manipulations.
- Improve validator scoring and introduce penalties for repetitive or low-value submissions.

### Phase 3: Location UAV + LDS V1 Post-Validation (Q4 2025)
- Add support for location-based unknown attack vectors (UAV) and obfuscation patterns.
- Establish post-validation workflows and LDS V1 (beta → full) to separate signal from noise.
- Use validated UAV quality to build a reputation signal that carries into future cycles.

### Phase 4: Deepfake / Face-Based Adversarial Testing for KYC (Q1 2026) — **Current**
- Validator-provided seed face images and deepfake-style transformation families.
- Cycle 1: pose_edit, lighting_edit, expression_edit, background_edit. Cycle 2 adds screen_replay.
- **Image generation is the sole scored miner task** in the current cycle.

### Phase 5–11 (2026–2027): Identity Realism & Simulation
- Expand biometric attack families beyond Cycle 1 (e.g., swap/recapture/morphing) (Q1 2026)
- Generate and validate synthetic documents (Q2 2026)
- Simulate digital presence and interactions (Q3 2026)
- Introduce financial transaction modeling (Q4 2026)
- Build 3D identity avatars (Q2 2027)
- Add voice and conversational AI support

### Final Phase: Unified Identity Representation
- Train a comprehensive model for identity screening.
- Launch a decentralized platform for collaborative validation and contribution.

---

## 🌍 Future Plans

We are continuously improving MIID to:
- Expand **face-based adversarial data generation** for enhanced AI benchmarking.
- Integrate **more complex biometric attack families** (document spoofing, voice, 3D avatars).
- Improve **fraud detection AI** using multi-modal data sources.

Join us in shaping the future of **identity verification and fraud prevention**.

📢 **Follow the project & contribute to our open-source development!**  
[Discord](https://discord.com/channels/799672011265015819/1351934165964296232) | [GitHub](https://github.com/yanez-compliance/MIID-subnet)

---

## 📜 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

_Built with ❤️ by the YANEZ-MIID Team_
