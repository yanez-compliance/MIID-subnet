```

```markdown:docs/testing.md
# Testing the MIID Subnet

This document describes how to run tests for the MIID subnet to verify functionality.

## Unit Tests

The MIID codebase includes unit tests in the `tests/` directory:

- `test_mock.py`: Tests for the mock implementation
- `test_template_validator.py`: Tests for the validator functionality
- `helpers.py`: Helper functions for testing

### Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_mock.py
```

## Mock Mode

You can run the subnet in mock mode for testing purposes without connecting to the Bittensor network:

```bash
python neurons/validator.py --mock
python neurons/miner.py --mock
```

## Manual Testing

You can manually test the miner functionality:

1. Start the Ollama server:
```bash
ollama serve
```

2. Test LLM queries directly:
```bash
python -c "import ollama; print(ollama.generate(model='llama3.1:latest', prompt='Generate 5 spelling variations for the name John Smith'))"
```

3. Test the variation extraction function:
```bash
python -c "from neurons.miner import Miner; m = Miner(); print(m.process_variations(['Respond', '---', 'Query-John Smith', '---', '1. Jon Smith\\n2. J. Smith\\n3. John Smyth\\n4. Johnn Smith\\n5. Jon Smyth'], 'test', './'))"
```
```

```markdown:docs/reward_system.md
# MIID Reward System

This document explains how miners are rewarded in the MIID subnet.

## Overview

Miners in the MIID subnet are rewarded based on the quality and quantity of name variations they provide. The reward system evaluates:

1. Response validity
2. Variation quantity
3. Phonetic similarity to original name
4. Orthographic similarity to original name
5. Overall diversity of variations

## Reward Calculation

The `get_name_variation_rewards` function in `MIID/validator/reward.py` handles reward calculation:

1. For each miner response:
   - Check if the response is valid (has variations)
   - For each name in the request:
     - Check if variations were provided for this name
     - Calculate quality score for the variations

2. Quality is determined by:
   - Phonetic similarity using algorithms like Soundex or Metaphone
   - Orthographic similarity using string distance metrics
   - Number of variations provided (up to an expected count)
   - Diversity among the variations

3. Final reward is an average of quality scores across all names

## Parameters

- `variation_count`: Expected number of variations per name
- `phonetic_similarity`: Configuration for phonetic similarity thresholds
- `orthographic_similarity`: Configuration for orthographic similarity thresholds

## Example

If a miner provides 5 high-quality variations for each of 3 requested names:
- Quality score might be 0.8-0.9 per name
- Overall reward would be ~0.85

If a miner provides only 1-2 low-quality variations:
- Quality score might be 0.2-0.3 per name
- Overall reward would be ~0.25

## Improving Your Rewards

To maximize rewards as a miner:

1. Ensure your LLM is well-tuned for generating name variations
2. Return the requested number of variations for each name
3. Make sure variations maintain phonetic similarity to the original
4. Provide diverse variations within reasonable spelling differences
5. Respond promptly to all validator requests
```

These documents should provide a comprehensive guide for users of your MIID subnet. They cover setup, configuration, protocol details, testing, and the reward system.

Would you like me to expand on any particular section or create any additional documentation files?