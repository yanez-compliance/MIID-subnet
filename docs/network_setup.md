# Network Setup Guide for MIID Miners

## Port Requirements

To ensure proper connectivity with the MIID subnet, miners need to open the following ports:

### Required Ports
- **Port 8091**: Primary miner-to-validator communication (Default port)
  - Protocol: TCP
  - Direction: Inbound
  - Purpose: Main communication port for validator queries
  - Note: This port can be configured using the `--axon.port` parameter when starting the miner

### Optional Ports (if using additional features)
- **Port 5000**: Used for local API access (if running local services)
- **Port 22**: SSH access (for remote management)

## Starting the Miner with Custom Port

You can specify the port when starting your miner using the `--axon.port` parameter:

```bash
btcli miner start \
    --netuid <netuid> \
    --wallet.name <wallet_name> \
    --wallet.hotkey <hotkey_name> \
    --axon.port 8091
```

Replace:
- `<netuid>` with your subnet ID
- `<wallet_name>` with your wallet name
- `<hotkey_name>` with your hotkey name

Note: While you can change the port using `--axon.port`, it's recommended to use the default port 8091 unless you have specific requirements.

## Network Configuration Steps

### 1. Firewall Configuration

#### Ubuntu/Debian (using UFW)
```bash
# Allow required ports
sudo ufw allow 8091/tcp

# Optional: Allow SSH
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

#### CentOS/RHEL (using firewalld)
```bash
# Allow required port
sudo firewall-cmd --permanent --add-port=8091/tcp

# Optional: Allow SSH
sudo firewall-cmd --permanent --add-port=22/tcp

# Reload firewall
sudo firewall-cmd --reload
```

### 2. Cloud Provider Configuration

If running on a cloud provider (AWS, GCP, Azure, etc.), you need to:

1. Configure Security Groups/Network ACLs to allow inbound traffic on port 8091
2. Ensure the instance's public IP is accessible
3. Configure any load balancers or NAT gateways to forward the required port

#### AWS Example
- Add inbound rule to your security group:
  - Type: Custom TCP
  - Port: 8091
  - Source: 0.0.0.0/0 (or restrict to validator IPs)

#### Azure Example
- Add inbound port rule in Network Security Group:
  - Priority: 100
  - Name: Allow-MIID-8091
  - Port: 8091
  - Protocol: TCP
  - Action: Allow

### 3. Testing Connectivity

Before starting your miner, test the required port:

```bash
# Test inbound port
nc -zv <your-ip> 8091
```

### 4. Troubleshooting

If you're experiencing connection issues:

1. Check if the required port is open:
   ```bash
   # Check inbound port
   sudo netstat -tulpn | grep 8091
   ```

2. Verify firewall rules:
   ```bash
   # For UFW
   sudo ufw status
   
   # For firewalld
   sudo firewall-cmd --list-all
   ```

3. Check cloud provider security groups
4. Ensure no other services are using port 8091
5. Verify your miner is actually listening on port 8091

## Common Issues and Solutions

### 1. Port Already in Use
If you see an error that port 8091 is already in use:
```bash
# Find the process using the port
sudo lsof -i :8091

# Kill the process if needed
sudo kill -9 <process-id>
```

### 2. Connection Timeout
If validators can't connect:
- Verify your public IP is correct
- Check if your ISP is blocking port 8091
- Ensure your cloud provider's security groups are properly configured

### 3. Firewall Blocking Connections
If the firewall is blocking connections:
- Double-check your firewall rules
- Try temporarily disabling the firewall for testing
- Ensure the rules are applied in the correct order

## Support

If you're still experiencing issues after following these steps:
1. Check the [Discord](https://discord.com/channels/799672011265015819/1351934165964296232) for community support
2. Contact the YANEZ-MIID team directly 

Remember: Always keep your firewall rules as restrictive as possible while still allowing necessary traffic. 