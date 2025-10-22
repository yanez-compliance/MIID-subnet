# Address Similarity Filter Script

## Overview

This script filters and removes similar addresses from the `addrs/` directory using the address similarity detection logic from `cheat_detection.py`. It helps identify and clean up duplicate or near-duplicate addresses across different city JSON files.

## Features

- **Normalize addresses** using the same logic as `cheat_detection.py` (removes digits, spaces, punctuation, converts to lowercase)
- **Detect similar files** using overlap coefficient and Jaccard similarity metrics
- **Find duplicate addresses** within individual files
- **Multiple cleaning modes**:
  - Report-only (no changes)
  - Deduplicate within files
  - Remove similar files
  - Merge similar files

## Analysis Results

From the initial scan of `/work/MIID-subnet/54/addgen/addrs/`:

- **Total files scanned:** 29,267 JSON files
- **Files successfully loaded:** 29,001 files
- **Similar file pairs found:** 1,073 pairs
- **Files with internal duplicates:** 4,096 files
- **Total duplicate addresses:** 10,688 addresses

## Usage

### Basic Usage

```bash
# Report only (recommended first step)
python filter_similar_addresses.py --mode report_only

# Remove duplicate addresses within each file
python filter_similar_addresses.py --mode deduplicate

# Delete files that are too similar to others
python filter_similar_addresses.py --mode remove_similar_files

# Merge similar files together
python filter_similar_addresses.py --mode merge
```

### Advanced Options

```bash
# Specify custom directory
python filter_similar_addresses.py \
    --addrs-dir /path/to/addrs \
    --mode report_only

# Adjust similarity thresholds
python filter_similar_addresses.py \
    --mode report_only \
    --overlap-threshold 0.9 \
    --jaccard-threshold 0.8
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--addrs-dir` | `/work/MIID-subnet/54/addgen/addrs` | Path to the addresses directory |
| `--mode` | `report_only` | Action to take (see modes below) |
| `--overlap-threshold` | `0.8` | Overlap coefficient threshold for similarity (0.0-1.0) |
| `--jaccard-threshold` | `0.7` | Jaccard similarity threshold (0.0-1.0) |

## Modes

### 1. `report_only` (Default)
**Safest option - No changes made**

- Scans all files and analyzes similarity
- Reports findings without modifying any files
- Shows:
  - Similar file pairs with their metrics
  - Files containing duplicate addresses
  - Total statistics

**Use this first** to understand what will be changed.

### 2. `deduplicate`
**Removes duplicate addresses within each file**

- Keeps files separate
- Only removes exact duplicates within the same file
- Maintains at least one copy of each unique address per file
- Updates the `count` field in JSON
- **Safest cleanup option**

Example:
```
Before: ["123 Main St", "456 Oak Ave", "123 Main St"]
After:  ["123 Main St", "456 Oak Ave"]
```

### 3. `remove_similar_files`
**Deletes entire files that are too similar**

- Compares all files pairwise
- When two files are similar (above thresholds), keeps the one with more unique addresses
- **Caution:** This permanently deletes files

Example:
```
File A: 10 unique addresses
File B: 8 unique addresses, 80% similar to File A
Result: File B is deleted
```

### 4. `merge`
**Merges similar files together**

- Groups files with similar addresses
- Merges them into one file (keeps the largest)
- Preserves all unique addresses across the group
- Deletes the merged files

Example:
```
File A: ["123 Main St", "456 Oak Ave"]
File B: ["123 Main St", "789 Pine Rd"]
Result: File A: ["123 Main St", "456 Oak Ave", "789 Pine Rd"], File B deleted
```

## Similarity Metrics

The script uses two similarity metrics (same as `cheat_detection.py`):

### Overlap Coefficient
```
overlap = |A ∩ B| / min(|A|, |B|)
```
- Measures how much the smaller set is contained in the larger
- 1.0 = complete overlap
- Default threshold: 0.8

### Jaccard Similarity
```
jaccard = |A ∩ B| / |A ∪ B|
```
- Measures overall similarity between sets
- 1.0 = identical sets
- Default threshold: 0.7

Files are considered similar if **either** metric exceeds its threshold.

## Address Normalization

Addresses are normalized using this process:

1. Remove street numbers and ranges (e.g., "123-456", "123")
2. Remove spaces, commas, hyphens, semicolons
3. Convert to lowercase

Example:
```
Original:  "123 Main Street, Apt 4-B"
Normalized: "mainstreetapt"
```

This allows detection of addresses that are structurally the same but differ in formatting or house numbers.

## Example Workflow

### Step 1: Initial Analysis
```bash
python filter_similar_addresses.py --mode report_only
```

Review the output to see:
- How many similar files exist
- Which files are similar to each other
- How many duplicates are within files

### Step 2: Clean Duplicates Within Files (Safe)
```bash
python filter_similar_addresses.py --mode deduplicate
```

This removes obvious duplicates without deleting any files.

### Step 3: Handle Similar Files (If Needed)

**Option A - Delete redundant files:**
```bash
python filter_similar_addresses.py --mode remove_similar_files
```

**Option B - Merge similar files:**
```bash
python filter_similar_addresses.py --mode merge
```

### Step 4: Verify Results
```bash
python filter_similar_addresses.py --mode report_only
```

Check that the cleanup worked as expected.

## Examples of Similar Files Found

The script found many similar file pairs, including:

```
Kampong Pangkal Kalong, MY ↔ Peringat, MY
Overlap: 1.0000, Jaccard: 1.0000 (100% identical addresses)

Phú Quốc, VN ↔ Phu Quoc, VN  
Overlap: 1.0000, Jaccard: 1.0000 (same city, different spelling)

Igbor, NG ↔ Gboko, NG
Overlap: 1.0000, Jaccard: 1.0000 (different cities with identical addresses)
```

Many of these are likely:
- Same city with different spellings
- Neighboring cities sharing addresses
- Data collection artifacts

## Implementation Details

The script reuses core functions from `cheat_detection.py`:

- `normalize_address()` - Based on lines 309-313, 358-362
- `overlap_coefficient()` - From lines 85-89
- `jaccard()` - From lines 92-99
- Address comparison logic - From lines 462-492

## Safety Recommendations

1. **Always run `report_only` first** to understand what will change
2. **Backup your data** before using `remove_similar_files` or `merge` modes
3. **Start with `deduplicate`** mode as it's the safest
4. **Review the logs** to see what was changed
5. **Test on a subset** of data first if concerned

## Logging

The script provides detailed logging:
- INFO level: Progress and summary information
- WARNING level: Failed file loads or issues
- All logs include timestamps

Redirect to file for later review:
```bash
python filter_similar_addresses.py --mode deduplicate 2>&1 | tee cleanup.log
```

## Performance

- Scanning 29,000 files: ~2 seconds
- Similarity comparison: ~10 minutes (O(n²) comparison)
- Deduplication: ~1-2 seconds per 1000 files
- Memory usage: Moderate (loads all addresses into memory)

## Troubleshoリューション

### Script is slow
- Normal for large datasets (n² comparisons)
- Consider processing country-by-country for huge datasets

### Too many similar files found
- Increase thresholds: `--overlap-threshold 0.9 --jaccard-threshold 0.8`
- Review examples to verify they should be merged/deleted

### Files not being detected as similar
- Lower thresholds: `--overlap-threshold 0.7 --jaccard-threshold 0.6`
- Check if addresses have unique formats

## Files

- `filter_similar_addresses.py` - Main script
- `FILTER_ADDRESSES_README.md` - This documentation
- Output logs show all operations performed

## License

Same as parent project.

