# MIID Helpful Hints

This document provides quick reference guides and helpful hints for running MIID validators and miners. For more detailed information, please refer to the full documentation files.

## Quick Links to Documentation

- [Network Setup Guide](network_setup.md) - Setting up your environment
- [Background Mining Guide](background_mining.md) - Running miners in the background
- [Logging Guide](logging.md) - Understanding and managing logs
- [Validator Guide](validator.md) - Running and managing validators
- [Miner Guide](miner.md) - Running and managing miners

## Common Issues and Solutions

### Network Setup Issues

1. **Connection Problems**
   - Check your internet connection
   - Verify firewall settings
   - Ensure ports are properly forwarded
   - See [Network Setup Guide](network_setup.md) for detailed troubleshooting

2. **Wallet Issues**
   - Ensure your wallet has sufficient TAO for staking
   - Verify wallet hotkey is properly configured
   - Check wallet permissions

### Mining Issues

1. **Miner not working or validators cannot reach you**
   - Confirm your axon has a real IP and port on-chain, firewall/NAT is open, and `axon.external_ip` / `axon.external_port` match what validators use. To print your on-chain IP/port from any machine, run [`MIID/miner/active_miner_check/is_my_miner_alive.py`](../MIID/miner/active_miner_check/is_my_miner_alive.py) with `--hotkey` (SS58), then run these on **another** machine (substitute your IP and port for the example below):

```bash
nc -zv <IP> <PORT>
curl -v --max-time 10 "http://<IP>:<PORT>/IdentitySynapse"
```

   - **nc:** You want a line like `Connection to … port … [tcp/*] succeeded!` — that means the TCP port is reachable.
   - **curl:** You may get `HTTP/1.1 500 Internal Server Error` with a JSON `Internal Server Error` body; that is common when the caller is not a whitelisted validator and still means the axon responded (check miner logs if you expect a reject).

2. **Background Mining**
   - Use `pm2` or `tmux` for session management
   - Monitor resource usage
   - Check [Background Mining Guide](background_mining.md) for best practices

3. **Performance Issues**
   - Monitor CPU and memory usage
   - Check disk space availability
   - Verify network bandwidth

### Logging and Monitoring

1. **Understanding Logs**
   - Log levels: DEBUG, INFO, WARNING, ERROR
   - Log rotation and management
   - See [Logging Guide](logging.md) for detailed information

2. **Common Log Messages**
   - Score calculation details
   - Network connection status
   - Mining performance metrics

## Best Practices

### Validator Best Practices
- Regular monitoring of validator performance
- Proper staking management
- Network health checks
- See [Validator Guide](validator.md) for more details

### Miner Best Practices
- Resource optimization
- Regular updates
- Performance monitoring
- See [Miner Guide](miner.md) for more details

### Where to improve the miner (code map)

These are the main places miners execute work and can tune behavior (different image models, pipelines, or matching validator intent):

| Area | Path | Notes |
|------|------|--------|
| Image variation pipeline | [`MIID/miner/generate_variations.py`](../MIID/miner/generate_variations.py) | Model pool, env vars (`MIID_MODEL`, `HF_TOKEN`, etc.), and how variations are generated. |
| Per-model implementations | [`MIID/miner/models/`](../MIID/miner/models/) | Individual model loaders and `generate` helpers (e.g. FLUX, PuLID, Qwen). |
| Miner neuron (LLM + Phase 4 wiring) | [`neurons/miner.py`](../neurons/miner.py) | Axon, Ollama name variations, whitelist, and Phase 4 image request handling. |

**Prompts and “what the validator asks for”** (variation types, intensities, accessories, screen replay, etc.) are defined on the validator side. To see the exact wording and structure validators use when building image tasks, read:

- [`MIID/validator/image_variations.py`](../MIID/validator/image_variations.py)

**KAV (Known Address Variation) scoring** — if you are focused on maximizing online quality rewards, the grading logic lives in the validator reward module. Read how name, DOB, address, penalties, and blended ranking work here:

- [`MIID/validator/reward.py`](../MIID/validator/reward.py)

## Performance Optimization

### System Resources
- Recommended CPU: 4+ cores
- Minimum RAM: 8GB
- Storage: 50GB+ free space
- Network: Stable connection with 10Mbps+ bandwidth

### Configuration Tips
- Adjust batch sizes based on system capabilities
- Optimize timeout settings
- Configure appropriate logging levels

## Troubleshooting Guide

### Common Error Messages
1. **Connection Errors**
   - Check network configuration
   - Verify endpoint accessibility
   - Review firewall settings

2. **Performance Errors**
   - Monitor system resources
   - Check for resource bottlenecks
   - Review configuration settings

3. **Validation Errors**
   - Check scoring parameters
   - Verify data formats
   - Review network consensus

### Quick Fixes
1. **Restart Services**
   ```bash
   # For validators
   systemctl restart miid-validator
   
   # For miners
   systemctl restart miid-miner
   ```

2. **Check Logs**
   ```bash
   # View recent logs
   tail -f /var/log/miid/validator.log
   tail -f /var/log/miid/miner.log
   ```



## Additional Resources

- [Network Setup Guide](network_setup.md) - Detailed network configuration
- [Background Mining Guide](background_mining.md) - Mining optimization
- [Logging Guide](logging.md) - Log management and analysis
- [Validator Guide](validator.md) - Validator operations
- [Miner Guide](miner.md) - Miner operations

## Support

For additional support:
1. Check the documentation
2. Review logs for specific error messages
3. Consult the community forums
4. Contact support team [Discord](https://discord.com/channels/799672011265015819/1351934165964296232) | [GitHub](https://github.com/yanez-compliance/MIID-subnet)

Remember to always check the latest documentation for updates and new features. 