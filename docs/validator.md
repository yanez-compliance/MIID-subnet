# MIID Validator

This document explains how to run a MIID validator on the Bittensor network. The MIID subnet focuses on generating name variations for identity management and analysis.

## Overview

MIID validators:
1. Generate challenge queries containing names
2. Send requests to miners to generate spelling variations
3. Evaluate the quality and quantity of the variations returned
4. Assign scores to miners based on their performance
5. Set weights on the network to influence miner rewards

## Requirements

- Python 3.10 or higher
- Git
- A Bittensor wallet with TAO for staking
- A local LLM via Ollama (default: llama3.1:latest)
- Sufficient storage for challenge data and responses (minimum 10GB recommended)
- Reliable internet connection
- 8GB+ RAM (16GB recommended)
- Open port 8091 for miner-to-validator communication

## Installation

### Option 1: Automated Setup (Recommended)

1. First, clone the MIID repository:
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. Run the automated setup script:
```bash
bash scripts/validator/setup.sh
```

This script will:
- Install all system dependencies
- Install Ollama and pull the llama3.1 model
- Create a Python virtual environment (validator_env)
- Install the MIID package and Bittensor
- Set up all necessary configurations

3. Activate the virtual environment:
```bash
source validator_env/bin/activate
```

If you encounter any issues with python-venv during installation, the script will attempt to fix them automatically. However, you can also manually install it:
```bash
sudo apt-get update
sudo apt-get install python3-venv
# Or for specific Python versions:
sudo apt-get install python3.10-venv  # for Python 3.10
sudo apt-get install python3.11-venv  # for Python 3.11
```

### Option 2: Manual Installation

If you prefer to install components manually:

1. Clone the MIID repository:
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. Create and activate a virtual environment:
```bash
python3 -m venv validator_env
source validator_env/bin/activate
```

3. Install the MIID package and dependencies:
```bash
python -m pip install -e .
```

4. Install Ollama:
Visit [ollama.ai](https://ollama.ai) for installation instructions.

5. Pull the default LLM model:
```bash
ollama pull llama3.1:latest
```

## Installation Recommendations

- **For beginners**: Use the automated setup script (Option 1) for the smoothest experience.
- **For experienced users**: Either option works well. The setup script ensures consistent environments, while manual installation offers more control.
- **For production**: Use the setup script to ensure all dependencies are properly installed, then consider additional hardening:
  - Use a service manager like systemd or supervisor to ensure your validator runs continuously
  - Set up monitoring and alerts for downtime
  - Consider dedicated hardware for reliability
  - Implement regular backups of wallet keys
  - Configure proper firewall rules for port 8091

## Running a Validator

1. Stake to the subnet:
```bash
btcli stake add --netuid 322 --amount 100 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network test
```

3. Start your validator:
```bash
python neurons/validator.py --netuid 322 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network test
```

For detailed information about logging setup and management, see the [Logging Guide](logging.md).

## Configuration Options

You can configure your validator with the following command-line arguments:

- `--neuron.timeout`: Base timeout for miner requests in seconds (default: 120)
- `--neuron.sample_size`: Number of miners to query per step (default: 10)
- `--neuron.ollama_model_name`: The Ollama model to use for verification (default: llama3.1:latest)
- `--neuron.logging.debug`: Enable debug logging

Example with custom configuration:
```bash
python neurons/validator.py --netuid 322 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network test --neuron.timeout 180 --neuron.sample_size 15
```

## How It Works

### Query Generation

The validator uses a `QueryGenerator` to create challenge queries for miners:
1. Generates random names using the Faker library
2. Creates query templates with varying complexity
3. Sets expected parameters for variation count and quality

### Miner Selection

Each validation round:
1. Randomly selects a subset of miners to query
2. Processes miners in batches to avoid overwhelming the network
3. Uses an adaptive timeout based on query complexity

### Response Evaluation

The validator evaluates miners based on:
1. **Responsiveness**: Whether they respond at all
2. **Completeness**: Whether they provide variations for all requested names
3. **Quantity**: The number of variations provided (up to a reasonable limit)
4. **Quality**: The uniqueness and similarity of variations to the original names

### Scoring and Weighting

Miners are scored using:
1. Phonetic similarity metrics (how similar the variations sound)
2. Orthographic similarity metrics (how similar the variations look)
3. Response time and reliability

## Advanced Configuration

For advanced users, you can modify:

- `MIID/validator/query_generator.py`: Customize challenge creation
