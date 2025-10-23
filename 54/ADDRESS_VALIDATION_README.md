# Address Validation Testing Scripts

This directory contains scripts to test address validation using the `_grade_address_variations` function from `MIID/validator/reward.py`.

## Scripts

### 1. test_address_validation.py

Main script for testing address validation with multiple modes.

#### Usage Modes

**Interactive Mode:**
```bash
python test_address_validation.py -i
```
Prompts you to enter a seed address and test addresses interactively.

**Test from JSON File:**
```bash
python test_address_validation.py -f example_address_test.json
```

**Test Single Address:**
```bash
python test_address_validation.py \
  -s "123 Main St, New York, NY 10001, United States" \
  -seed "456 5th Ave, New York, NY 10018, United States"
```

**Test Multiple Addresses:**
```bash
python test_address_validation.py \
  -a "addr1" "addr2" "addr3" \
  -seed "seed_address"
```

**Verbose Mode:**
```bash
python test_address_validation.py -f test.json -v
```

#### JSON File Format

```json
{
  "seed_addresses": [
    "456 5th Ave, New York, NY 10018, United States",
    "10 Downing Street, Westminster, London SW1A 2AA, United Kingdom"
  ],
  "variations": {
    "Name1": [
      ["name_variation", "dob_variation", "address_variation"],
      ["name_variation", "dob_variation", "address_variation"]
    ],
    "Name2": [
      ["name_variation", "dob_variation", "address_variation"]
    ]
  }
}
```

### 2. test_guam_addresses.py

Specialized script to test addresses from the Guam (GU) prefetched address data.

#### Usage

```bash
python test_guam_addresses.py
```

This script:
- Loads all addresses from `addgen/addrs/GU/*.json`
- Uses the first address as the seed
- Tests a subset of addresses (up to 10)
- Displays detailed validation results

### 3. example_address_test.json

Example JSON file showing the proper format for test data with addresses from US and UK.

## What the Validation Tests

### 1. Heuristic Checks

- **looks_like_address()**: Validates basic address structure
  - Minimum length (30 characters)
  - Presence of numbers (street numbers)
  - Presence of letters
  - Proper comma separators
  - No special characters like `, `, `:`, `%`, etc.

### 2. Region Validation

- **validate_address_region()**: Checks if address is in correct region
  - Extracts city and country from address
  - Validates city exists in country using geonamescache
  - Special handling for disputed regions (Luhansk, Crimea, etc.)

### 3. API Validation

- **Nominatim API**: Validates up to 5 random addresses
  - Queries OpenStreetMap Nominatim API
  - 1 second delay between requests to prevent rate limiting
  - Handles timeouts gracefully

## Scoring System

The function returns a score from 0.0 to 1.0:

- **0.0**: Failed heuristic or region validation
- **Reduced (0.6-0.8)**: All passed but some API timeouts (-0.2 per timeout)
- **1.0**: Perfect - all validations passed without timeouts

## Output Format

The scripts provide detailed output including:

- Overall score
- Total addresses tested
- Region match count
- Heuristic validation results
- Per-address validation details:
  - Whether it looks like an address
  - Whether region matches seed
  - Extracted city and country
  - Validation status
- API validation results:
  - Number of API calls made
  - Success/timeout/failure counts
  - Details of each API attempt

## Examples

### Test Single Address

```bash
./test_address_validation.py \
  -s "1600 Pennsylvania Avenue NW, Washington, DC 20500, United States" \
  -seed "1 First Street NE, Washington, DC 20543, United States"
```

### Test with Countries Having Few Addresses

```bash
# Test addresses from countries with < 20 addresses
./test_address_validation.py \
  -a "Vatican City, Vatican" \
       "Pitcairn Islands" \
       "Cocos Islands, Australia" \
  -seed "Vatican City, Vatican"
```

### Test Guam Addresses

```bash
./test_guam_addresses.py
```

## Requirements

- Python 3.6+
- bittensor
- geonamescache
- requests
- All other dependencies from MIID project

## Notes

- API validation is rate-limited (1 second between requests)
- The Nominatim API may timeout occasionally - this is handled gracefully
- Addresses should include at least: street, city, and country
- Address format is flexible but should have proper comma separation



