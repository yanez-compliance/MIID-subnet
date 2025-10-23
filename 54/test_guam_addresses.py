#!/usr/bin/env python3
"""
Test script to validate addresses from Guam (GU) using the address validation system.
This demonstrates how to test addresses from the prefetched address data.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MIID.validator.reward import _grade_address_variations


def load_guam_addresses():
    """Load addresses from Guam JSON files."""
    guam_dir = Path(__file__).parent / 'addgen' / 'addrs' / 'CC'
    
    if not guam_dir.exists():
        print(f"Error: Guam address directory not found: {guam_dir}")
        return None
    
    all_addresses = []
    
    # Load all JSON files from Guam directory
    for json_file in guam_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'addresses' in data:
                    all_addresses.extend(data['addresses'])
                    print(f"Loaded {len(data['addresses'])} addresses from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_addresses


def main():
    print("="*80)
    print("TESTING GUAM ADDRESSES")
    print("="*80)
    
    # Load Guam addresses
    addresses = load_guam_addresses()
    
    if not addresses:
        print("No addresses loaded")
        return 1
    
    print(f"\nTotal addresses loaded: {len(addresses)}")
    
    # Use first address as seed
    seed_address = "Cocos Islands"
    
    if not seed_address:
        print("No seed address available")
        return 1
    
    print(f"\nSeed address: {seed_address}")
    
    # Create test variations using a subset of addresses
    test_addresses = addresses
    
    # Create variations structure
    variations = {
        "TestPerson": [
            ["TestPerson", "1990-01-01", addr]
            for addr in test_addresses
        ]
    }
    
    seed_addresses = [seed_address]
    miner_metrics = {}
    
    # Run validation
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    result = _grade_address_variations(variations, seed_addresses, miner_metrics)
    
    # Print results
    print(f"\n📊 Overall Score: {result['overall_score']:.3f}")
    print(f"   Total Addresses Tested: {result.get('total_addresses', 0)}")
    print(f"   Region Matches: {result.get('region_matches', 0)}")
    print(f"   Heuristic Perfect: {result.get('heuristic_perfect', False)}")
    print(f"   API Result: {result.get('api_result', 'N/A')}")
    
    # Print detailed breakdown if available
    if 'detailed_breakdown' in result:
        breakdown = result['detailed_breakdown']
        
        if 'validation_results' in breakdown:
            print("\n" + "-"*80)
            print("DETAILED VALIDATION RESULTS")
            print("-"*80)
            
            for name, validations in breakdown['validation_results'].items():
                passed = sum(1 for v in validations if v['passed_validation'])
                failed = len(validations) - passed
                
                print(f"\n{name}:")
                print(f"  ✅ Passed: {passed}/{len(validations)}")
                print(f"  ❌ Failed: {failed}/{len(validations)}")
                
                # Show first few failures
                failures = [v for v in validations if not v['passed_validation']]
                if failures:
                    print(f"\n  First failures:")
                    for i, fail in enumerate(failures[:3], 1):
                        print(f"    {i}. {fail['address'][:60]}...")
                        print(f"       Looks like address: {fail['looks_like_address']}")
                        print(f"       Region match: {fail['region_match']}")
        
        if 'api_validation' in breakdown:
            api_val = breakdown['api_validation']
            print("\n" + "-"*80)
            print("API VALIDATION")
            print("-"*80)
            print(f"  Result: {api_val.get('api_result', 'N/A')}")
            print(f"  Total Calls: {api_val.get('total_calls', 0)}")
            print(f"  Successful: {api_val.get('nominatim_successful_calls', 0)}")
            print(f"  Timeouts: {api_val.get('nominatim_timeout_calls', 0)}")
            print(f"  Failed: {api_val.get('nominatim_failed_calls', 0)}")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

