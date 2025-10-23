# Address Validation Testing - Summary

## Created Files

### 1. **test_address_validation.py** (Advanced)
Full-featured testing script with multiple modes, but has some conflicts with bittensor's argparse.

**Features:**
- Interactive mode
- JSON file input
- Single address testing
- Batch address testing
- Detailed validation reports

**Note:** Currently has argparse conflicts with bittensor. Use the simple script instead.

### 2. **simple_address_test.py** ✅ **RECOMMENDED**
Simple, working script that demonstrates address validation without argparse conflicts.

**Usage:**
```bash
cd /work/MIID-subnet/54
python simple_address_test.py
```

**What it does:**
- Tests 5 example addresses against a seed address
- Shows individual validation checks for each address
- Makes API calls to Nominatim for validation
- Displays detailed results

**Output includes:**
- Heuristic checks (looks like address, region match)
- City/country extraction
- Overall validation score
- Detailed per-address results
- API validation statistics

### 3. **test_guam_addresses.py**
Tests addresses from the Guam (GU) prefetched data.

**Usage:**
```bash
cd /work/MIID-subnet/54
python test_guam_addresses.py
```

### 4. **example_address_test.json**
Example JSON file showing the proper format for test data.

```json
{
  "seed_addresses": ["address1", "address2"],
  "variations": {
    "Name1": [["name", "dob", "address"], ...],
    "Name2": [["name", "dob", "address"], ...]
  }
}
```

### 5. **ADDRESS_VALIDATION_README.md**
Comprehensive documentation for all scripts and the validation system.

## How Address Validation Works

The `_grade_address_variations` function from `MIID/validator/reward.py` performs three levels of validation:

### Level 1: Heuristic Checks
- **Function:** `looks_like_address()`
- Checks:
  - Minimum 30 characters
  - At least 20 letters
  - At least 2 number groups (street numbers)
  - At least 2 commas
  - No special characters like `, `, `:`, `%`, etc.

### Level 2: Region Validation
- **Function:** `validate_address_region()`
- Extracts city and country from address
- Validates city exists in that country using geonamescache
- Compares against seed address region

### Level 3: API Validation
- **Function:** `check_with_nominatim()`
- Randomly samples up to 5 addresses
- Queries OpenStreetMap Nominatim API
- 1-second delay between requests (rate limiting)
- Handles timeouts gracefully

## Scoring System

| Score | Meaning |
|-------|---------|
| 0.0 | Failed heuristic or region validation |
| 0.6-0.8 | Passed but some API timeouts (-0.2 per timeout) |
| 1.0 | Perfect - all validations passed |

## Quick Start

### Test the Simple Script
```bash
cd /work/MIID-subnet/54
python simple_address_test.py
```

### Test Guam Addresses
```bash
cd /work/MIID-subnet/54
python test_guam_addresses.py
```

### Modify the Simple Script
Edit `simple_address_test.py` and change the test addresses:

```python
seed_addresses = [
    "Your seed address here"
]

test_addresses = [
    "Address 1 to test",
    "Address 2 to test",
    # Add more addresses...
]
```

## Example Output

```
================================================================================
SIMPLE ADDRESS VALIDATION TEST
================================================================================

Seed Address: 456 5th Ave, New York, NY 10018, United States

Testing 5 addresses:
  1. 123 Broadway, New York, NY 10001, United States
  2. 789 Madison Ave, New York, NY 10021, United States
  ...

INDIVIDUAL ADDRESS CHECKS
--------------------------------------------------------------------------------

1. 123 Broadway, New York, NY 10001, United States...
   Looks like address: True
   Region match: False
   Extracted: City='york', Country='united states'

...

📊 RESULTS:
   Overall Score: 0.000
   Total Addresses: 5
   Region Matches: 0
   Heuristic Perfect: False
   API Result: False
```

## Important Notes

1. **API Rate Limiting:** The Nominatim API has rate limits. The script includes 1-second delays between requests.

2. **City Extraction:** The city extraction logic looks for known cities in geonamescache. "New York" might be extracted as "york" depending on the address format.

3. **Region Matching:** Addresses must be in the same city or country as the seed address to pass validation.

4. **Dependencies:** Requires `unidecode` module:
   ```bash
   pip install unidecode
   ```

## Files Created

```
54/
├── test_address_validation.py       # Advanced script (has argparse issues)
├── simple_address_test.py          # ✅ Simple working script
├── test_guam_addresses.py          # Test Guam addresses
├── example_address_test.json       # Example JSON format
├── ADDRESS_VALIDATION_README.md    # Full documentation
└── ADDRESS_TEST_SUMMARY.md        # This file
```

## Next Steps

1. Run `simple_address_test.py` to see how it works
2. Modify the test addresses in `simple_address_test.py` to test your own data
3. Use `test_guam_addresses.py` as a template for testing other prefetched address data
4. Reference `ADDRESS_VALIDATION_README.md` for complete documentation



