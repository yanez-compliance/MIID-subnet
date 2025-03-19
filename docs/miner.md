# MIID Miner

This document explains how to run a MIID miner on the Bittensor network. The MIID subnet focuses on generating name variations for identity management and analysis.

## Overview

MIID miners receive requests from validators containing names and a query template. The miner:
1. Processes each name through a local LLM
2. Generates multiple spelling variations of each name
3. Returns these variations to the validator

## Requirements

- Python 3.10 or higher
- A Bittensor wallet with TAO for registration
- A local LLM via Ollama (default: llama3.1:latest)
- Sufficient storage for model weights (~10GB or more depending on model)
- At least 8GB RAM (16GB+ recommended)

## Installation

### Option 1: Automated Setup (Recommended)

For a streamlined installation process, use the provided setup script:

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
git clone https://github.com/yourusername/MIID-subnet.git
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

## Installation Recommendations

- **For beginners**: Use the automated setup script (Option 1) for the smoothest experience.
- **For experienced users**: Either option works well. The setup script ensures consistent environments, while manual installation offers more control.
- **For production**: Use the setup script to ensure all dependencies are properly installed, then consider additional hardening:
  - Use a service manager like systemd or supervisor
  - Set up monitoring and logging
  - Consider using a more powerful GPU for faster inference

## Running a Miner

1. Create or import a wallet:
```bash
btcli wallet new
# or
btcli wallet import
```

2. Register your miner to the subnet:
```bash
btcli subnet register --netuid 322 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network test
```

3. Start your miner:
```bash
python neurons/miner.py --netuid 322 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network test
```

## Configuration Options

You can configure your miner with the following command-line arguments:

- `--neuron.ollama_model_name`: The Ollama model to use (default: llama3.1:latest)
- `--neuron.logging.debug`: Enable debug logging
- `--neuron.log_responses`: Save miner responses for analysis
- `--neuron.response_cache_dir`: Directory to store response logs

Example with custom configuration:
```bash
python neurons/miner.py --netuid 322 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network test --neuron.ollama_model_name mistral:7b --neuron.logging.debug
```

## How It Works

1. The miner receives a request from a validator containing:
   - A list of names to generate variations for
   - A query template with a {name} placeholder

2. For each name, the miner:
   - Formats the query template with the name
   - Sends the formatted query to the local LLM
   - Extracts the generated variations from the LLM response

3. The miner processes all responses and returns a dictionary mapping each input name to a list of variations.

## Performance Tips

1. Use a powerful GPU if available to speed up LLM inference
2. Ensure your miner has reliable internet connectivity
3. Monitor your miner's logs for errors or performance issues
4. Consider using a high-quality LLM model for better variations
