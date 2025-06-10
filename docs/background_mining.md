# Running MIID Miners in the Background

This guide provides step-by-step instructions for running your MIID miner in the background using either tmux or pm2. This is useful for keeping your miner running even when you're not connected to the server.

> **Note**: Before following this guide, make sure you have completed the initial setup using the [Miner Guide](miner.md). This includes:
> - Running the setup script or manual installation
> - Creating your wallet
> - Registering your miner
> - Testing basic miner functionality

## Option 1: Using tmux

### Installation
```bash
# For Ubuntu/Debian
sudo apt update -y && \
sudo apt install -y sudo nano python3 curl tmux

# For CentOS/RHEL
sudo yum install -y sudo nano python3 curl tmux

# For macOS
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install tmux python3 curl

# Set Python3 as default
echo 'alias python=python3' >> ~/.zshrc
source ~/.zshrc

# For Intel Macs, you might need to install some additional packages
# brew install openssl readline sqlite3 xz zlib tcl-tk

# For Apple Silicon (M1/M2) Macs, you might need to install some additional packages
# brew install openssl readline sqlite3 xz zlib tcl-tk
```

### Step-by-Step Setup

1. Activate your Python virtual environment (created during initial setup):
```bash
source miner_env/bin/activate
```

2. Create a new tmux session:
```bash
tmux new -s miid_miner
```

3. Start your miner:
```bash
python neurons/miner.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

4. Detach from the tmux session:
   - Press `Ctrl+B` followed by `D`

### Managing Your Miner

- To reattach to the session:
```bash
tmux attach -t miid_miner
```

- To list all sessions:
```bash
tmux ls
```

- To kill the session:
```bash
tmux kill-session -t miid_miner
```

## Option 2: Using pm2 (Recommended for Production)

### Installation
```bash
# For Ubuntu/Debian
sudo apt update -y && \
sudo apt install -y sudo nano python3 curl jq npm

# For CentOS/RHEL
sudo yum install -y sudo nano python3 curl jq npm

# For macOS
# Install Node.js and npm using Homebrew
brew install node npm

# Set Python3 as default (if not already done)
echo 'alias python=python3' >> ~/.zshrc
source ~/.zshrc

# Install pm2 globally
sudo npm install pm2 -g -y && \
pm2 update
```

### Step-by-Step Setup

1. Create a start script:
```bash
# Create a directory for your scripts
mkdir -p ~/scripts
nano ~/scripts/start_miner.sh
```

2. Add the following content to start_miner.sh:
```bash
#!/bin/bash
source ~/miner_env/bin/activate
python ~/MIID-subnet/neurons/miner.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

3. Make the script executable:
```bash
chmod +x ~/scripts/start_miner.sh
```

4. Start the miner with pm2:
```bash
pm2 start ~/scripts/start_miner.sh --name miid_miner
```

### Managing Your Miner

- View miner status:
```bash
pm2 status
```

- View logs:
```bash
pm2 logs miid_miner
```

- Restart the miner:
```bash
pm2 restart miid_miner
```

- Stop the miner:
```bash
pm2 stop miid_miner
```

- Delete the miner process:
```bash
pm2 delete miid_miner
```

### Additional pm2 Features

1. Configure automatic startup:
```bash
pm2 startup
pm2 save
```

2. Monitor resource usage:
```bash
pm2 monit
```

3. Set up log rotation:
```bash
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
```

## Troubleshooting

### Common Issues

1. **Process not starting**
   - Check logs: `pm2 logs miid_miner` or `tmux attach -t miid_miner`
   - Verify Python environment is activated
   - Check wallet credentials

2. **Process keeps crashing**
   - Check system resources (memory, CPU)
   - Verify network connectivity
   - Check for port conflicts

3. **Can't connect to miner**
   - Verify the process is running
   - Check firewall settings
   - Verify port 8091 is open

### Monitoring

- For pm2: Use `pm2 monit` for real-time monitoring
- For tmux: Use system monitoring tools like `htop` or `top`

## Best Practices

1. **Regular Backups**
   - Backup your wallet and configuration files
   - Keep a record of your hotkey and coldkey

2. **Security**
   - Use strong passwords for your wallet
   - Keep your system updated
   - Monitor for unauthorized access

3. **Performance**
   - Monitor system resources
   - Set up proper logging
   - Configure automatic restarts on failure

4. **Maintenance**
   - Regularly check logs for errors
   - Update your miner software when new versions are released
   - Monitor network status and validator connections 