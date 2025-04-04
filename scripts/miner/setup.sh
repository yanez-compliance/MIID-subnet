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

# Attempts to install a package and (optionally) an alternative name.
try_install() {
  local PKG="$1"
  local ALT="$2"
  echo -e "\e[34m[INFO]\e[0m Attempting to install '$PKG'..."
  if apt-cache show "$PKG" &>/dev/null; then
    sudo apt-get install -y "$PKG" && return 0
  fi

  if [ -n "$ALT" ]; then
    echo -e "\e[33m[WARN]\e[0m '$PKG' not found, trying alternative '$ALT'..."
    if apt-cache show "$ALT" &>/dev/null; then
      sudo apt-get install -y "$ALT" && return 0
    fi
  fi

  echo -e "\e[33m[WARN]\e[0m Neither '$PKG' nor '$ALT' is available on this system."
  return 1
}

# ---------------------------------------------------------
# Section 1: System Dependencies
# Updates apt, upgrades packages, installs sudo, etc.
# ---------------------------------------------------------
install_system_dependencies() {
  echo -e "\e[34m[INFO]\e[0m Installing system dependencies..."
  sudo apt update -y || handle_error "Failed to update package list"
  sudo apt upgrade -y || handle_error "Failed to upgrade packages"
  sudo apt install -y sudo curl wget git || handle_error "Failed to install basic tools"
}

# ---------------------------------------------------------
# Section 2: Python 3.10+ and Libraries
# Installs Python, dev tools, cmake, etc.
# ---------------------------------------------------------
install_python() {
  echo -e "\e[34m[INFO]\e[0m Installing Python and dependencies..."
  try_install python3-pip
  try_install python3-dev
  try_install build-essential
  try_install cmake
}

# ---------------------------------------------------------
# Section 3: Ollama Installation
# Installs Ollama using the official script
# ---------------------------------------------------------
install_ollama() {
  echo -e "\e[34m[INFO]\e[0m Installing Ollama..."
  if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh || handle_error "Failed to install Ollama"
    success_msg "Ollama installed successfully."
  else
    echo -e "\e[32m[INFO]\e[0m Ollama is already installed. Skipping."
  fi
  
  # Pull the required LLM model
  echo -e "\e[34m[INFO]\e[0m Pulling llama3.1:latest model..."
  ollama pull llama3.1:latest || handle_error "Failed to pull llama3.1:latest model"
  success_msg "llama3.1:latest model pulled successfully."
}

# ---------------------------------------------------------
# Section 4: Virtual Environment Setup
# Creates a new venv with Python, then activates it.
# ---------------------------------------------------------
create_and_activate_venv() {
  local VENV_DIR="miner_env"
  echo -e "\e[34m[INFO]\e[0m Creating virtual environment..."
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment"
    success_msg "Virtual environment created successfully."
  else
    echo -e "\e[32m[INFO]\e[0m Virtual environment already exists. Skipping."
  fi

  echo -e "\e[34m[INFO]\e[0m Activating virtual environment..."
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtual environment"
}

# ---------------------------------------------------------
# Section 5: Install Python Requirements
# Installs project dependencies from requirements.txt.
# ---------------------------------------------------------
install_python_requirements() {
  echo -e "\e[34m[INFO]\e[0m Installing Python dependencies..."
  pip install --upgrade pip setuptools wheel || handle_error "Failed to upgrade pip, setuptools, and wheel"
  
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt || handle_error "Failed to install Python dependencies"
  else
    echo -e "\e[33m[WARN]\e[0m requirements.txt not found. Installing core dependencies."
    pip install numpy pandas faker tqdm Levenshtein python-Levenshtein metaphone || handle_error "Failed to install core dependencies"
  fi
  
  success_msg "Python dependencies installed successfully."
}

# ---------------------------------------------------------
# Section 6: Install MIID in editable mode
# ---------------------------------------------------------
install_miid() {
  echo -e "\e[34m[INFO]\e[0m Installing MIID package in editable mode..."
  pip install -e . || handle_error "Failed to install MIID package"
  success_msg "MIID package installed successfully."
}

# ---------------------------------------------------------
# Section 7: Install Bittensor
# Uses the official install script for Bittensor.
# ---------------------------------------------------------
install_bittensor() {
  echo -e "\e[34m[INFO]\e[0m Installing Bittensor..."
  pip install bittensor || handle_error "Failed to install Bittensor"
  success_msg "Bittensor installed successfully."
}

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
main() {
  install_system_dependencies
  install_python
  install_ollama
  create_and_activate_venv
  install_python_requirements
  install_miid
  install_bittensor
  
  echo -e "\e[34m[INFO]\e[0m MIID Miner setup completed successfully!"
  echo -e "\e[34m[INFO]\e[0m To start using your miner:"
  echo -e "   1. Activate the virtual environment: source miner_env/bin/activate"
  echo -e "   2. Start your miner: python neurons/miner.py --netuid 322 --wallet.name your_wallet --wallet.hotkey your_hotkey --subtensor.network test"
}

main "$@"