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
  
  # Pull the required LLM model
  info_msg "Pulling llama3.1:latest model..."
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
  pip install --upgrade pip setuptools wheel || handle_error "Failed to upgrade pip, setuptools, and wheel"
  
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt || handle_error "Failed to install Python dependencies"
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
  pip install -e . || handle_error "Failed to install MIID package"
  success_msg "MIID package installed successfully."
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
  
  info_msg "MIID Miner setup completed successfully!"
  info_msg "To start using your miner:"
  echo -e "   1. Activate the virtual environment: source miner_env/bin/activate"
  echo -e "   2. Start your miner: python neurons/miner.py --netuid 322 --wallet.name your_wallet --wallet.hotkey your_hotkey --subtensor.network test"
}

main "$@"