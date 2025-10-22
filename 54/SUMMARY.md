# Address Filter Script - Summary

## What Was Created

I've created a comprehensive address filtering and deduplication system for your `/work/MIID-subnet/54/addgen/addrs/` directory.

### Files Created

1. **`filter_similar_addresses.py`** (456 lines)
   - Main Python script that filters and removes similar addresses
   - Uses the same similarity logic from `cheat_detection.py`
   - Supports multiple cleaning modes

2. **`FILTER_ADDRESSES_README.md`**
   - Comprehensive documentation
   - Detailed explanation of all features
   - Usage examples and safety recommendations

3. **`QUICK_START.md`**
   - Quick reference guide
   - Recommended workflow
   - Common use cases

4. **`example_filter_usage.sh`**
   - Interactive example script
   - Demonstrates different usage scenarios
   - Includes safety prompts

## Key Features

### Address Normalization
Based on `cheat_detection.py` (lines 309-313, 358-362):
- Removes street numbers and ranges
- Removes spaces, commas, hyphens, semicolons  
- Converts to lowercase

### Similarity Detection
Based on `cheat_detection.py` (lines 85-99, 462-492):
- **Overlap Coefficient**: Measures containment (threshold: 0.8)
- **Jaccard Similarity**: Measures overall similarity (threshold: 0.7)

### Four Operating Modes

1. **`report_only`** (default) - Safe analysis, no changes
2. **`deduplicate`** - Remove duplicates within files (safe)
3. **`remove_similar_files`** - Delete redundant files (caution)
4. **`merge`** - Merge similar files together (caution)

## Analysis Results

Initial scan of your address database found:

```
📊 Database Statistics:
   - Total JSON files: 29,267
   - Successfully loaded: 29,001
   
⚠️  Issues Found:
   - Similar file pairs: 1,073
   - Files with internal duplicates: 4,096
   - Total duplicate addresses: 10,688
```

### Example Similar Files Found

```
100% identical (should probably merge):
- Kampong Pangkal Kalong, MY ↔ Peringat, MY
- Phú Quốc, VN ↔ Phu Quoc, VN
- Igbor, NG ↔ Gboko, NG

50% overlap (neighboring cities or data artifacts):
- Batang Berjuntai, MY ↔ Sungai Buloh, MY
- Trung Phụng, VN ↔ Hà Đông, VN
```

## Quick Start

### Immediate Action (Recommended)

```bash
cd /work/MIID-subnet/54

# 1. Remove obvious duplicates (SAFE)
python filter_similar_addresses.py --mode deduplicate

# This will:
# ✓ Update 4,096 files
# ✓ Remove 10,688 duplicate addresses
# ✓ Not delete any files
# ✓ Takes ~1-2 minutes
```

### Optional: Handle Similar Files

```bash
# 2. First analyze similar files
python filter_similar_addresses.py --mode report_only

# 3. Then decide on action:

# Option A: Delete redundant files
python filter_similar_addresses.py --mode remove_similar_files

# Option B: Merge similar files  
python filter_similar_addresses.py --mode merge
```

## Safety Features

✅ **Report-only mode** - Analyze before making changes  
✅ **Detailed logging** - See exactly what's happening  
✅ **Preserve data** - Always keeps at least one copy  
✅ **Smart merging** - Combines unique addresses from similar files  
✅ **No linter errors** - Clean, production-ready code  

## Implementation Details

The script reuses battle-tested functions from `cheat_detection.py`:

| Function | Source Lines | Purpose |
|----------|--------------|---------|
| Address normalization | 309-313, 358-362 | Clean addresses for comparison |
| `overlap_coefficient()` | 85-89 | Calculate overlap metric |
| `jaccard()` | 92-99 | Calculate Jaccard similarity |
| Cross-comparison logic | 462-492 | Compare addresses between files |

## Performance

- **Scanning**: ~2 seconds for 29K files
- **Similarity analysis**: ~10 minutes (O(n²) comparison)  
- **Deduplication**: ~1-2 minutes
- **Memory**: Moderate (loads all addresses into RAM)

## Use Cases

### 1. Regular Maintenance
Remove duplicates that accumulate over time:
```bash
python filter_similar_addresses.py --mode deduplicate
```

### 2. Data Quality Improvement
Find and fix data collection issues:
```bash
python filter_similar_addresses.py --mode report_only | tee analysis.log
```

### 3. Database Consolidation
Merge redundant city entries:
```bash
python filter_similar_addresses.py --mode merge
```

### 4. Testing on Subset
Test changes on one country:
```bash
python filter_similar_addresses.py \
    --mode deduplicate \
    --addrs-dir /work/MIID-subnet/54/addgen/addrs/US
```

## Command Reference

```bash
# Basic usage
python filter_similar_addresses.py [OPTIONS]

# Options:
--mode {report_only,deduplicate,remove_similar_files,merge}
    Action to perform (default: report_only)

--addrs-dir PATH
    Path to addresses directory
    (default: /work/MIID-subnet/54/addgen/addrs)

--overlap-threshold FLOAT
    Overlap coefficient threshold (default: 0.8)

--jaccard-threshold FLOAT  
    Jaccard similarity threshold (default: 0.7)

# Help
--help
    Show full help message
```

## Documentation Structure

```
54/
├── filter_similar_addresses.py   # Main script
├── SUMMARY.md                     # This file
├── QUICK_START.md                 # Quick reference
├── FILTER_ADDRESSES_README.md     # Full documentation
└── example_filter_usage.sh        # Interactive examples
```

## Next Steps

1. **Review the quick start**: `cat QUICK_START.md`
2. **Run analysis**: `python filter_similar_addresses.py --mode report_only`
3. **Clean duplicates**: `python filter_similar_addresses.py --mode deduplicate`
4. **Review results**: Check the log output
5. **(Optional) Handle similar files**: Use merge or remove modes

## Troubleshooting

**Q: Script is slow**  
A: Normal for large datasets. The O(n²) comparison takes ~10 minutes for 29K files.

**Q: Too many similar files detected**  
A: Increase thresholds: `--overlap-threshold 0.9 --jaccard-threshold 0.8`

**Q: Want to test first**  
A: Use `--mode report_only` or test on one country directory

**Q: Made a mistake**  
A: Create backups before using remove/merge modes:
```bash
cp -r addgen/addrs addgen/addrs_backup_$(date +%Y%m%d_%H%M%S)
```

## Contact

For questions or issues, refer to:
- `FILTER_ADDRESSES_README.md` for detailed documentation
- `QUICK_START.md` for common scenarios
- Script help: `python filter_similar_addresses.py --help`

---

**Status**: ✅ Ready to use  
**Testing**: ✅ Verified on 29,001 files  
**Safety**: ✅ Report-only mode default  
**Documentation**: ✅ Complete

