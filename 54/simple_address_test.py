#!/usr/bin/env python3
"""
Simple script to test address validation.
Usage: python simple_address_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MIID.validator.reward import (
    _grade_address_variations,
    looks_like_address,
    validate_address_region,
    extract_city_country
)


def test_addresses():
    """Test a few example addresses."""
    
    print("="*80)
    print("SIMPLE ADDRESS VALIDATION TEST")
    print("="*80)
    
    # Test data
    seed_addresses = [
        "456 5th Ave, New York, NY 10018, United States"
    ]
    
    test_addresses = [
        "123 Broadway, New York, NY 10001, United States",
        "789 Madison Ave, New York, NY 10021, United States",
        "321 Park Ave, New York, NY 10022, United States",
        "100 Main St, Los Angeles, CA 90012, United States",  # Different city
        "This is not a valid address",  # Invalid
    ]
    
    # Create variations structure
    variations = {
        "TestPerson": [
            ["TestPerson", "1990-01-01", addr]
            for addr in test_addresses
        ]
    }
    
    print(f"\nSeed Address: {seed_addresses[0]}")
    print(f"\nTesting {len(test_addresses)} addresses:")
    for i, addr in enumerate(test_addresses, 1):
        print(f"  {i}. {addr}")
    
    # Test individual address properties
    print("\n" + "-"*80)
    print("INDIVIDUAL ADDRESS CHECKS")
    print("-"*80)
    
    for i, addr in enumerate(test_addresses, 1):
        print(f"\n{i}. {addr[:60]}...")
        
        # Check if looks like address
        looks_valid = looks_like_address(addr)
        print(f"   Looks like address: {looks_valid}")
        
        if looks_valid:
            # Check region
            region_match = validate_address_region(addr, seed_addresses[0])
            print(f"   Region match: {region_match}")
            
            # Extract city/country
            city, country = extract_city_country(addr)
            print(f"   Extracted: City='{city}', Country='{country}'")
    
    # Run full validation
    print("\n" + "="*80)
    print("FULL VALIDATION (including API calls)")
    print("="*80)
    print("\nNote: This will make API calls to Nominatim, please wait...")
    
    miner_metrics = {}
    result = _grade_address_variations(variations, seed_addresses, miner_metrics)
    
    # Print results
    print(f"\n📊 RESULTS:")
    print(f"   Overall Score: {result['overall_score']:.3f}")
    print(f"   Total Addresses: {result.get('total_addresses', 0)}")
    print(f"   Region Matches: {result.get('region_matches', 0)}")
    print(f"   Heuristic Perfect: {result.get('heuristic_perfect', False)}")
    print(f"   API Result: {result.get('api_result', 'N/A')}")
    
    # Show detailed breakdown
    if 'detailed_breakdown' in result and 'validation_results' in result['detailed_breakdown']:
        print("\n" + "-"*80)
        print("DETAILED RESULTS")
        print("-"*80)
        
        for name, validations in result['detailed_breakdown']['validation_results'].items():
            passed = sum(1 for v in validations if v['passed_validation'])
            print(f"\n{name}: {passed}/{len(validations)} passed")
            
            for i, val in enumerate(validations, 1):
                status = "✅" if val['passed_validation'] else "❌"
                print(f"  {status} Address {i}: {val['status']}")
    
    # Show API validation
    if 'detailed_breakdown' in result and 'api_validation' in result['detailed_breakdown']:
        api_val = result['detailed_breakdown']['api_validation']
        print("\n" + "-"*80)
        print("API VALIDATION")
        print("-"*80)
        print(f"  API Result: {api_val.get('api_result', 'N/A')}")
        print(f"  Total Calls: {api_val.get('total_calls', 0)}")
        print(f"  Successful: {api_val.get('nominatim_successful_calls', 0)}")
        print(f"  Timeouts: {api_val.get('nominatim_timeout_calls', 0)}")
        print(f"  Failed: {api_val.get('nominatim_failed_calls', 0)}")
        
        if 'api_attempts' in api_val:
            print("\n  API Attempts:")
            for attempt in api_val['api_attempts']:
                result_str = str(attempt['result'])
                print(f"    {attempt['attempt']}. {result_str} - {attempt['address'][:50]}...")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    try:
        test_addresses()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

