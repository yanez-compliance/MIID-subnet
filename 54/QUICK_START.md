# Quick Start Guide: Address Filter Script

## TL;DR

```bash
cd /work/MIID-subnet/54

# 1. See what needs cleaning (SAFE - no changes)
python filter_similar_addresses.py --mode report_only

# 2. Clean duplicates within files (SAFE - recommended)
python filter_similar_addresses.py --mode deduplicate

# 3. (Optional) Handle similar files - see full README
```

## What This Script Does

Analyzes and cleans the address database using the same similarity detection logic from `cheat_detection.py`:

- ✅ Finds duplicate addresses within files
- ✅ Finds similar addresses across different city files  
- ✅ Removes duplicates safely
- ✅ Can merge or delete redundant files

## Current Status

From initial scan of your database:

| Metric | Count |
|--------|-------|
| Total JSON files | 29,267 |
| Files with data | 29,001 |
| **Similar file pairs** | **1,073** |
| **Files with internal duplicates** | **4,096** |
| **Total duplicate addresses** | **10,688** |

## Recommended Workflow

### Step 1: Analyze (1 minute)
```bash
python filter_similar_addresses.py --mode report_only
```

This shows you what will be changed without making any modifications.

### Step 2: Clean Duplicates (1 minute)
```bash
python filter_similar_addresses.py --mode deduplicate
```

**Safe operation** - Only removes exact duplicates within each file. No files are deleted.

Expected results:
- ✅ Removes ~10,688 duplicate addresses
- ✅ Updates 4,096 files
- ✅ No data loss (keeps one copy of each address)

### Step 3: Handle Similar Files (Optional)

If you want to also clean up similar files across different cities:

```bash
# Option A: Delete redundant files (keeps the one with more addresses)
python filter_similar_addresses.py --mode remove_similar_files

# Option B: Merge similar files together
python filter_similar_addresses.py --mode merge
```

⚠️ **Warning:** These operations delete files. Create a backup first:
```bash
cp -r addgen/addrs addgen/addrs_backup_$(date +%Y%m%d_%H%M%S)
```

## What Makes Addresses "Similar"?

Addresses are normalized by:
1. Removing house numbers (123, 456-789)
2. Removing spaces, commas, hyphens
3. Converting to lowercase

Example:
- `"123 Main Street, Apt 4-B"` → `"mainstreetapt"`
- `"456 Main Street, Apt 5-C"` → `"mainstreetapt"` (duplicate!)

Two files are "similar" if:
- **Overlap coefficient > 0.8** (80% of smaller set is in larger set)
- **OR Jaccard similarity > 0.7** (70% similarity overall)

## Files Created

| File | Purpose |
|------|---------|
| `filter_similar_addresses.py` | Main script |
| `FILTER_ADDRESSES_README.md` | Detailed documentation |
| `QUICK_START.md` | This file |
| `example_filter_usage.sh` | Interactive examples |

## Examples of What Gets Fixed

### Example 1: Duplicates Within File
**Before:**
```json
{
  "city": "New York",
  "addresses": [
    "123 Main St, New York, NY",
    "456 Oak Ave, New York, NY", 
    "123 Main St, New York, NY"  // Duplicate!
  ]
}
```

**After (deduplicate mode):**
```json
{
  "city": "New York",
  "addresses": [
    "123 Main St, New York, NY",
    "456 Oak Ave, New York, NY"
  ]
}
```

### Example 2: Similar Files
**Before:**
```
Phú Quốc, VN → ["123 Beach Rd", "456 Harbor St"]
Phu Quoc, VN  → ["123 Beach Rd", "456 Harbor St"]  // Same city, different spelling!
```

**After (merge mode):**
```
Phú Quốc, VN → ["123 Beach Rd", "456 Harbor St"]
(Second file deleted)
```

## Need Help?

- **Full documentation:** See `FILTER_ADDRESSES_README.md`
- **Interactive examples:** Run `./example_filter_usage.sh`
- **Command help:** `python filter_similar_addresses.py --help`

## Safety Tips

1. ✅ Always run `--mode report_only` first
2. ✅ Start with `--mode deduplicate` (safest)
3. ✅ Create backups before using `remove_similar_files` or `merge`
4. ✅ Test on one country first: `--addrs-dir addgen/addrs/US`

## Typical Run Time

- Report generation: ~10 minutes (for 29K files)
- Deduplication: ~1-2 minutes
- Merge/Remove: ~2-5 minutes

The similarity comparison is O(n²) so it takes time with many files.

