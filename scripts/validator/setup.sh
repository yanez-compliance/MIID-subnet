#!/bin/bash
#
# setup.sh - Setup environment dependencies for MIID Validator
#
# This script installs required dependencies for running a MIID validator node,
# including Python, Bittensor, and Ollama with the llama3.1 model.

set -e  # Exit immediately on error

# ---------------------------------------------------------
# Error Handling and Helpers
# ---------------------------------------------------------

handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

success_msg() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

info_msg() {
  echo -e "\e[34m[INFO]\e[0m $1"
}

warn_msg() {
  echo -e "\e[33m[WARN]\e[0m $1"
}

# ---------------------------------------------------------
# Section 1: System Dependencies
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

ensure_python_venv() {
  info_msg "Ensuring python3-venv is properly installed..."
  
  # Get Python version
  PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
  info_msg "Detected Python version: $PYTHON_VERSION"
  
  # Try installing python3-venv and version-specific venv
  if ! dpkg -l python3-venv >/dev/null 2>&1; then
    info_msg "Installing python3-venv packages..."
    if ! sudo apt-get install -y python3-venv; then
      warn_msg "Failed to install python3-venv, trying alternative methods..."
      
      # Try version-specific venv package
      if ! sudo apt-get install -y "python${PYTHON_VERSION}-venv"; then
        # Try installing both Python 3.10 and 3.11 venv packages
        if ! sudo apt-get install -y python3.10-venv python3.11-venv; then
          handle_error "Failed to install any Python venv package. Please install manually: sudo apt-get install python3-venv"
        fi
      fi
    fi
  fi
  
  # Verify venv module is working
  if ! python3 -c "import venv" 2>/dev/null; then
    handle_error "Python venv module is not working properly. Please try reinstalling python3-venv"
  fi
  
  success_msg "Python venv is properly installed."
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
  info_msg "Installing required base packages..."
  local PACKAGES=(
    sudo
    curl
    wget
    git
    python3-pip
    python3-dev
    build-essential
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
  
  # Ensure Python venv is properly installed
  ensure_python_venv
  
  # Additional installs (same as miner)
  info_msg "Installing additional packages (nano, jq, npm, pm2)..."
  sudo apt-get install -y nano jq npm || warn_msg "Failed to install some packages."
  sudo npm install -g pm2 || warn_msg "Failed to install pm2 globally."
  pm2 update || warn_msg "Failed to update pm2."
  
  success_msg "System dependencies setup completed."
}

# ---------------------------------------------------------
# Section 2: Ollama Installation
# ---------------------------------------------------------
install_ollama() {
  info_msg "Installing Ollama..."
  if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh || handle_error "Failed to install Ollama"
    success_msg "Ollama installed successfully."
  else
    info_msg "Ollama is already installed. Skipping."
  fi
  
  # Use pm2 to serve Ollama (mirrors miner script)
  pm2 start ollama -- serve || warn_msg "Failed to start Ollama with PM2, but continuing..."
  
  info_msg "Pulling llama3.1:latest model..."
  ollama pull llama3.1:latest || handle_error "Failed to pull llama3.1:latest model"
  success_msg "llama3.1:latest model pulled successfully."
}

# ---------------------------------------------------------
# Section 3: Virtual Environment Setup
# ---------------------------------------------------------
create_and_activate_venv() {
  local VENV_DIR="validator_env"
  info_msg "Creating virtual environment..."
  
  # Remove any existing failed/incomplete venv
  if [ -d "$VENV_DIR" ]; then
    info_msg "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
  fi
  
  # Try creating venv with different Python versions if needed
  if ! python3 -m venv "$VENV_DIR"; then
    warn_msg "Failed to create venv with python3, trying with specific Python version..."
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    
    if ! "python${PYTHON_VERSION}" -m venv "$VENV_DIR"; then
      # Try with Python 3.10 specifically
      if command -v python3.10 &>/dev/null && python3.10 -m venv "$VENV_DIR"; then
        success_msg "Virtual environment created with Python 3.10"
      # Try with Python 3.11
      elif command -v python3.11 &>/dev/null && python3.11 -m venv "$VENV_DIR"; then
        success_msg "Virtual environment created with Python 3.11"
      else
        handle_error "Failed to create virtual environment with any Python version"
      fi
    fi
  else
    success_msg "Virtual environment created successfully."
  fi

  info_msg "Activating virtual environment..."
  # shellcheck source=/dev/null
  if ! source "$VENV_DIR/bin/activate"; then
    handle_error "Failed to activate virtual environment"
  fi
  
  # Verify virtual environment is working
  if ! python -c "import sys; assert sys.prefix != sys.base_prefix" 2>/dev/null; then
    handle_error "Virtual environment activation failed"
  fi
  
  success_msg "Virtual environment is active and working properly."
}

# ---------------------------------------------------------
# Section 4: Install Python Requirements
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
# Section 5: Install MIID in editable mode
# ---------------------------------------------------------
install_miid() {
  info_msg "Installing MIID package in editable mode..."
  pip install -e . || handle_error "Failed to install MIID package"
  success_msg "MIID package installed successfully."
}

# ---------------------------------------------------------
# Section 6: Install Bittensor
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
  install_ollama
  create_and_activate_venv
  install_python_requirements
  install_miid
  install_bittensor
  
  info_msg "MIID Validator setup completed successfully!"
  info_msg "To start using your validator:"
  echo -e "   1. Activate the virtual environment: source validator_env/bin/activate"
  echo -e "   2. Start your validator: python neurons/validator.py --netuid 322 --wallet.name your_wallet --wallet.hotkey your_hotkey --subtensor.network test"
}

main "$@"
