#!/bin/bash
#
# setup.sh - Setup environment dependencies for MIID Miner
#
# This script installs required dependencies for running a MIID miner node,
# including Python, Bittensor, and Ollama with the llama3.1 model.

set -e  # Exit immediately on error

# ---------------------------------------------------------
# Error Handling and Helpers
# ---------------------------------------------------------

# Handles errors by printing a message and exiting.
handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

# Prints success messages in green.
success_msg() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

# Prints info messages in blue.
info_msg() {
  echo -e "\e[34m[INFO]\e[0m $1"
}

# Prints warning messages in yellow.
warn_msg() {
  echo -e "\e[33m[WARN]\e[0m $1"
}

# Attempts to install a package and (optionally) an alternative name.
try_install() {
  local PKG="$1"
  local ALT="$2"
  info_msg "Attempting to install '$PKG'..."
  if apt-cache show "$PKG" &>/dev/null; then
    sudo apt-get install -y "$PKG" && return 0
  fi

  if [ -n "$ALT" ]; then
    warn_msg "'$PKG' not found, trying alternative '$ALT'..."
    if apt-cache show "$ALT" &>/dev/null; then
      sudo apt-get install -y "$ALT" && return 0
    fi
  fi

  warn_msg "Neither '$PKG' nor '$ALT' is available on this system."
  return 1
}

# ---------------------------------------------------------
# Section 1: System Dependencies
# Updates apt, upgrades packages, installs sudo, etc.
# ---------------------------------------------------------
fix_python_apt() {
  info_msg "Checking python3-apt installation..."
  
  apt update && apt upgrade -y && apt install sudo
  # Try to remove python3-apt if it exists
  sudo apt-get remove -y python3-apt >/dev/null 2>&1 || true
  sudo apt-get autoremove -y >/dev/null 2>&1 || true
  
  # Clean apt cache
  sudo apt-get clean
  sudo rm -rf /var/lib/apt/lists/*
  
  # Update apt without using python3-apt
  sudo apt-get update -o APT::Update::Post-Invoke-Success="" || warn_msg "apt-get update encountered errors but continuing..."
  
  # Install python3-apt fresh
  sudo apt-get install -y python3-apt >/dev/null 2>&1 || warn_msg "Could not install python3-apt, but continuing anyway..."
}

install_system_dependencies() {
  info_msg "Installing system dependencies..."
  
  # Fix python3-apt first
  fix_python_apt
  
  # Update package list with error handling
  info_msg "Updating package list..."
  if ! sudo apt-get update -o APT::Update::Post-Invoke-Success="" 2>/dev/null; then
    warn_msg "apt-get update encountered errors but continuing..."
  fi
  
  # Upgrade packages with error handling
  info_msg "Upgrading packages..."
  if ! sudo apt-get upgrade -y 2>/dev/null; then
    warn_msg "apt-get upgrade encountered errors but continuing..."
  fi
  
  # Install required packages
  info_msg "Installing required packages..."
  local PACKAGES=(
    sudo
    curl
    wget
    git
  )
  
  for package in "${PACKAGES[@]}"; do
    if ! dpkg -l "$package" >/dev/null 2>&1; then
      info_msg "Installing $package..."
      if ! sudo apt-get install -y "$package" 2>/dev/null; then
        warn_msg "Failed to install $package, but continuing..."
      fi
    else
      info_msg "$package is already installed."
    fi
  done

  # general useful installs
  apt update -y && apt install sudo -y && apt install nano -y && apt install python3 -y && alias python=python3 && apt install curl -y
  sudo apt update -y && sudo apt install jq -y && sudo apt install npm -y && sudo npm install pm2 -g -y && pm2 update
  
  success_msg "System dependencies setup completed."
}

# ---------------------------------------------------------
# Section 2: Python 3.10+ and Libraries
# Installs Python, dev tools, cmake, etc.
# ---------------------------------------------------------
install_python() {
  info_msg "Installing Python and dependencies..."
  
  # Install Python packages with error handling
  local PYTHON_PACKAGES=(
    python3-pip
    python3-dev
    build-essential
    cmake
    python3-venv
  )
  
  for package in "${PYTHON_PACKAGES[@]}"; do
    if ! dpkg -l "$package" >/dev/null 2>&1; then
      info_msg "Installing $package..."
      if ! sudo apt-get install -y "$package" 2>/dev/null; then
        warn_msg "Failed to install $package, but continuing..."
      fi
    else
      info_msg "$package is already installed."
    fi
  done
  
  success_msg "Python dependencies installed successfully."
}

# ---------------------------------------------------------
# Section 3: Ollama Installation
# Installs Ollama using the official script
# ---------------------------------------------------------
install_ollama() {
  info_msg "Installing Ollama..."
  if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh || handle_error "Failed to install Ollama"
    success_msg "Ollama installed successfully."
  else
    info_msg "Ollama is already installed. Skipping."
  fi
  
  pm2 start ollama -- serve
  # Pull the required LLM model
  info_msg "Pulling llama3.1:latest model..."
  # Run ollama in background
  
  ollama pull llama3.1:latest || handle_error "Failed to pull llama3.1:latest model"
  success_msg "llama3.1:latest model pulled successfully."
}

# ---------------------------------------------------------
# Section 4: Virtual Environment Setup
# Creates a new venv with Python, then activates it.
# ---------------------------------------------------------
create_and_activate_venv() {
  local VENV_DIR="miner_env"
  info_msg "Creating virtual environment..."
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment"
    success_msg "Virtual environment created successfully."
  else
    info_msg "Virtual environment already exists. Skipping."
  fi

  info_msg "Activating virtual environment..."
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtual environment"
}

# ---------------------------------------------------------
# Section 5: Install Python Requirements
# Installs project dependencies from requirements.txt.
# ---------------------------------------------------------
install_python_requirements() {
  info_msg "Installing Python dependencies..."
  # Pin setuptools<82: v82+ drops pkg_resources and breaks isolated `pip install -e .`
  pip install --upgrade pip "setuptools>=68,<82" wheel || handle_error "Failed to upgrade pip, setuptools, and wheel"
  
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt || handle_error "Failed to install base Python dependencies"
  else
    warn_msg "requirements.txt not found. Installing core dependencies."
    pip install numpy pandas faker tqdm Levenshtein python-Levenshtein metaphone || handle_error "Failed to install core dependencies"
  fi
  
  success_msg "Python dependencies installed successfully."
}

# ---------------------------------------------------------
# Section 6: Install MIID in editable mode
# ---------------------------------------------------------
install_miid() {
  info_msg "Installing MIID package in editable mode..."
  if ! pip install -e .; then
    warn_msg "Editable install with build isolation failed; retrying with --no-build-isolation (venv setuptools)..."
    pip install -e . --no-build-isolation || handle_error "Failed to install MIID package"
  fi
  success_msg "MIID package installed successfully."
}

# ---------------------------------------------------------
# Section 6b: Phase 4 — AdaFace + diffusion model setup
# Clones AdaFace repo, downloads pretrained model, and
# optionally pre-downloads the FLUX.2-klein diffusion model.
# ---------------------------------------------------------
install_phase4_deps() {
  info_msg "Setting up Phase 4 (face image variation) dependencies..."

  # Install miner-only Python packages for full mode.
  if [ -f requirements-miner.txt ]; then
    info_msg "Installing miner-specific dependencies (diffusion models, face validation, timelock)..."
    pip install -r requirements-miner.txt || handle_error "Failed to install miner dependencies"
    if python -c "from MIID.miner.drand_encrypt import is_timelock_available; import sys; sys.exit(0 if is_timelock_available() else 1)" 2>/dev/null; then
      success_msg "Drand timelock encryption available (pinned versions in requirements-miner.txt)."
    else
      warn_msg "Timelock not importable on this Python/platform. Production: use Linux x86_64 and Python 3.10 venv so timelock wheels install. Miner will use raw-bytes sandbox fallback until then."
    fi
  else
    warn_msg "requirements-miner.txt not found. Miner image generation packages may be missing."
  fi

  # Clone AdaFace if not already present
  local ADAFACE_DIR="MIID/miner/AdaFace"
  if [ ! -d "$ADAFACE_DIR" ]; then
    info_msg "Cloning AdaFace repository..."
    git clone https://github.com/mk-minchul/AdaFace.git "$ADAFACE_DIR" || handle_error "Failed to clone AdaFace"
    success_msg "AdaFace cloned successfully."
  else
    info_msg "AdaFace already present. Skipping clone."
  fi

  # Download pretrained AdaFace model
  local ADAFACE_MODEL="$ADAFACE_DIR/pretrained/adaface_ir50_ms1mv2.ckpt"
  if [ ! -f "$ADAFACE_MODEL" ]; then
    info_msg "Downloading AdaFace pretrained model..."
    mkdir -p "$ADAFACE_DIR/pretrained"
    pip install gdown 2>/dev/null
    gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O "$ADAFACE_MODEL" || warn_msg "Failed to download AdaFace model via gdown. Download manually from: https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing and place at $ADAFACE_MODEL"
    if [ -f "$ADAFACE_MODEL" ]; then
      success_msg "AdaFace model downloaded."
    fi
  else
    info_msg "AdaFace model already present. Skipping download."
  fi

  # Pre-download FLUX.2-klein diffusion model (requires HF_TOKEN)
  if [ -n "$HF_TOKEN" ]; then
    info_msg "Pre-downloading FLUX.2-klein diffusion model (this may take a while)..."
    python -m MIID.miner.downloading_model && success_msg "FLUX.2-klein model downloaded." || warn_msg "Could not pre-download model. It will be downloaded on first miner run."
  else
    warn_msg "HF_TOKEN not set. Skipping diffusion model download. Set HF_TOKEN before running the miner."
  fi

  success_msg "Phase 4 setup completed."
}

# ---------------------------------------------------------
# Section 7: Install Bittensor
# Uses the official install script for Bittensor.
# ---------------------------------------------------------
install_bittensor() {
  info_msg "Installing Bittensor..."
  pip install bittensor || handle_error "Failed to install Bittensor"
  success_msg "Bittensor installed successfully."
}

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
main() {
  # Check if running as root and warn
  if [ "$(id -u)" = "0" ]; then
    warn_msg "Running as root is not recommended. Please run as a normal user with sudo privileges."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  fi
  
  install_system_dependencies
  install_python
  install_ollama
  create_and_activate_venv
  install_python_requirements
  install_miid
  install_bittensor

  # Ask whether to install Phase 4 (image generation) packages
  local INSTALL_MODE="${1:-}"
  if [ "$INSTALL_MODE" = "--full" ]; then
    info_msg "Full mode requested (--full flag). Installing Phase 4 packages..."
    install_phase4_deps
  elif [ "$INSTALL_MODE" = "--basic" ]; then
    info_msg "Basic mode requested (--basic flag). Skipping Phase 4 packages."
  else
    echo ""
    info_msg "Choose setup mode:"
    echo -e "   [B] Basic  — Name variations only (no GPU required)"
    echo -e "   [F] Full   — Name + face image variations (GPU recommended)"
    echo ""
    read -p "Enter B or F (default: F): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Bb]$ ]]; then
      info_msg "Basic mode selected. Skipping Phase 4 packages."
    else
      info_msg "Full mode selected. Installing Phase 4 packages..."
      install_phase4_deps
      INSTALL_MODE="--full"
    fi
  fi

  echo ""
  info_msg "============================================"
  info_msg "MIID Miner setup completed successfully!"
  info_msg "============================================"
  echo ""

  if [ "$INSTALL_MODE" = "--full" ]; then
    info_msg "Mode: FULL (name + image variations)"
    echo ""
    info_msg "Drand timelock uses pinned packages in requirements-miner.txt; wheels need Linux x86_64 + Python 3.10 (use python3.10 -m venv if needed)."
    echo ""
    info_msg "Before running the miner, set these environment variables:"
    echo -e "   export HF_TOKEN=\"hf_YOUR_TOKEN_HERE\"    # Get from huggingface.co/settings/tokens"
    echo -e "   export FLUX_DEVICE=\"cuda\"                # or \"mps\" for Apple Silicon, \"cpu\" for CPU-only"
  else
    info_msg "Mode: BASIC (name variations only)"
    echo ""
    info_msg "To upgrade to Full later, run:"
    echo -e "   source miner_env/bin/activate"
    echo -e "   pip install -r requirements-miner.txt"
    echo -e "   See docs/miner.md 'Upgrading from Basic to Full' for details."
  fi

  echo ""
  info_msg "To start using your miner:"
  echo -e "   1. Activate the virtual environment: source miner_env/bin/activate"
  echo -e "   2. Register your miner: btcli subnet register --netuid 54 --wallet.name your_wallet --wallet.hotkey your_hotkey --subtensor.network finney"
  echo -e "   3. Start your miner: python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name your_wallet --wallet.hotkey your_hotkey --logging.debug"
  echo ""
  info_msg "For the full guide, see: docs/miner.md"
  echo -e "\n----------------------------------------\n"
}

main "$@"