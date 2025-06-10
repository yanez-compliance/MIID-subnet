# Logging Setup and Management

This document explains how to set up and manage logs for both MIID validators and miners. Proper logging is essential for monitoring, debugging, and maintaining your MIID nodes.

## Basic Logging

Both validators and miners can save their logs to files using the following commands:

### Validator Logging
```bash
python neurons/validator.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name validator --wallet.hotkey validator_default --logging.debug > validator.log 2>&1
```

### Miner Logging
```bash
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --logging.debug > miner.log 2>&1
```

These commands will:
- Run the node with debug logging enabled
- Save all output (both stdout and stderr) to the respective log file
- Include the chain endpoint for better connectivity
- Use the specified wallet configuration

## Real-time Log Monitoring

You can monitor logs in real-time using:
```bash
# For validator logs
tail -f validator.log

# For miner logs
tail -f miner.log
```

## Advanced Log Management

For long-term log management, we recommend using logrotate to:
- Prevent log files from growing too large
- Archive old logs
- Maintain proper file permissions
- Ensure continuous monitoring with `tail -f`

### Installing logrotate
```bash
# For Ubuntu/Debian
sudo apt-get install logrotate

# For CentOS/RHEL
sudo yum install logrotate
```

### Creating Logrotate Configuration

1. Create configuration files:
```bash
# For validator
sudo nano /etc/logrotate.d/validator

# For miner
sudo nano /etc/logrotate.d/miner
```

2. Add the following configuration to both files (adjust paths as needed):
```
/path/to/your/logs/validator.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
    postrotate
        /usr/bin/killall -HUP tail
    endscript
}

/path/to/your/logs/miner.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
    postrotate
        /usr/bin/killall -HUP tail
    endscript
}
```

### Configuration Options Explained

- `daily`: Rotate logs every day
- `rotate 7`: Keep 7 days of logs
- `compress`: Compress old log files
- `delaycompress`: Don't compress the most recent rotated log
- `missingok`: Don't error if log file is missing
- `notifempty`: Don't rotate empty logs
- `create 0640 root root`: Create new log files with these permissions
- `postrotate`: Commands to run after rotation (ensures `tail -f` continues working)

## Log Analysis Tips

1. **Searching Logs**:
```bash
# Search for errors
grep -i "error" validator.log

# Search for specific patterns
grep "pattern" miner.log
```

2. **Viewing Recent Logs**:
```bash
# Last 100 lines
tail -n 100 validator.log

# Last hour of logs
grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')" miner.log
```

3. **Monitoring Log Growth**:
```bash
# Check log file size
ls -lh validator.log

# Monitor log growth
watch -n 60 "ls -lh validator.log"
```

## Best Practices

1. **Regular Monitoring**:
   - Check logs daily for errors or warnings
   - Set up alerts for critical errors
   - Monitor log file sizes

2. **Storage Management**:
   - Ensure sufficient disk space for logs
   - Adjust rotation settings based on available space
   - Consider using a separate partition for logs

3. **Security**:
   - Set appropriate file permissions
   - Consider encrypting sensitive log data
   - Regularly review log access patterns

4. **Performance**:
   - Use log rotation to prevent performance impact
   - Consider using log aggregation tools for large deployments
   - Monitor log writing performance

## Troubleshooting

1. **Log Rotation Issues**:
   - Check logrotate status: `logrotate -d /etc/logrotate.d/validator`
   - Verify permissions: `ls -l /path/to/logs/`
   - Check disk space: `df -h`

2. **Monitoring Issues**:
   - If `tail -f` stops working, check for log rotation
   - Verify log file exists and is being written to
   - Check file permissions and ownership

3. **Performance Issues**:
   - Monitor disk I/O during log writing
   - Consider using a faster storage medium for logs
   - Adjust log rotation frequency if needed 