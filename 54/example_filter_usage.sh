#!/bin/bash
# Example usage script for filter_similar_addresses.py
# This demonstrates different ways to use the address filter

SCRIPT_DIR="/work/MIID-subnet/54"
ADDRS_DIR="/work/MIID-subnet/54/addgen/addrs"

echo "================================"
echo "Address Filter Usage Examples"
echo "================================"
echo ""

# Example 1: Report only (safest, always do this first)
echo "Example 1: Generate report without making changes"
echo "Command: python filter_similar_addresses.py --mode report_only"
echo ""
read -p "Press Enter to run Example 1, or Ctrl+C to skip..."
python "$SCRIPT_DIR/filter_similar_addresses.py" --mode report_only --addrs-dir "$ADDRS_DIR"
echo ""
echo "================================"
echo ""

# Example 2: Deduplicate within files (safe operation)
echo "Example 2: Remove duplicate addresses within each file"
echo "Command: python filter_similar_addresses.py --mode deduplicate"
echo ""
echo "WARNING: This will modify files (but safely - only removes duplicates within files)"
read -p "Press Enter to run Example 2, or Ctrl+C to skip..."
python "$SCRIPT_DIR/filter_similar_addresses.py" --mode deduplicate --addrs-dir "$ADDRS_DIR"
echo ""
echo "================================"
echo ""

# Example 3: Test on a single country (safer)
echo "Example 3: Test on a single country directory (TD - Chad)"
echo "Command: python filter_similar_addresses.py --mode report_only --addrs-dir $ADDRS_DIR/TD"
echo ""
read -p "Press Enter to run Example 3, or Ctrl+C to skip..."
python "$SCRIPT_DIR/filter_similar_addresses.py" --mode report_only --addrs-dir "$ADDRS_DIR/TD"
echo ""
echo "================================"
echo ""

# Example 4: Custom thresholds (more strict)
echo "Example 4: Use stricter similarity thresholds"
echo "Command: python filter_similar_addresses.py --mode report_only --overlap-threshold 0.9 --jaccard-threshold 0.8"
echo ""
read -p "Press Enter to run Example 4, or Ctrl+C to skip..."
python "$SCRIPT_DIR/filter_similar_addresses.py" \
    --mode report_only \
    --addrs-dir "$ADDRS_DIR" \
    --overlap-threshold 0.9 \
    --jaccard-threshold 0.8
echo ""
echo "================================"
echo ""

# Example 5: Remove similar files (DANGEROUS - creates backup first)
echo "Example 5: Remove similar files (DANGEROUS OPERATION)"
echo "This will delete entire files that are too similar"
echo ""
echo "Creating backup first..."
BACKUP_DIR="/work/MIID-subnet/54/addgen/addrs_backup_$(date +%Y%m%d_%H%M%S)"
read -p "Create backup at $BACKUP_DIR? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp -r "$ADDRS_DIR" "$BACKUP_DIR"
    echo "Backup created at $BACKUP_DIR"
    echo ""
    read -p "Now run the removal operation? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "$SCRIPT_DIR/filter_similar_addresses.py" --mode remove_similar_files --addrs-dir "$ADDRS_DIR"
    else
        echo "Skipped removal operation"
    fi
else
    echo "Skipped Example 5"
fi
echo ""
echo "================================"
echo ""

echo "All examples completed!"
echo ""
echo "For more options, run:"
echo "  python $SCRIPT_DIR/filter_similar_addresses.py --help"

