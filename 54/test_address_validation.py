#!/usr/bin/env python3
"""
Script to test address validation using the _grade_address_variations function from reward.py
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to import from MIID
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_addresses_from_json(json_file: str) -> Dict[str, Any]:
    """
    Load address test data from a JSON file.
    
    Expected JSON format:
    {
        "seed_addresses": ["address1", "address2", ...],
        "variations": {
            "Name1": [
                ["name_var1", "dob_var1", "address_var1"],
                ["name_var2", "dob_var2", "address_var2"]
            ],
            "Name2": [...]
        }
    }
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_test_data_from_addresses(addresses: List[str], seed_addresses: List[str]) -> Dict[str, Any]:
    """
    Create test data structure from a list of addresses.
    Creates one variation per address with dummy name and DOB.
    
    Args:
        addresses: List of addresses to test
        seed_addresses: List of seed addresses (one per name)
    
    Returns:
        Dictionary with variations and seed_addresses
    """
    variations = {}
    
    # Create a name for each seed address
    for i, seed_addr in enumerate(seed_addresses):
        if seed_addr:
            name = f"TestName{i+1}"
            variations[name] = []
            
            # Find addresses corresponding to this seed (in simple case, just one)
            # For testing purposes, we'll assign addresses round-robin
            start_idx = i * (len(addresses) // len(seed_addresses))
            end_idx = (i + 1) * (len(addresses) // len(seed_addresses))
            
            for addr in addresses[start_idx:end_idx]:
                variations[name].append([f"{name}_var", "1990-01-01", addr])
    
    return {
        "seed_addresses": seed_addresses,
        "variations": variations
    }


def print_validation_details(result: Dict[str, Any]):
    """Print detailed validation results in a readable format."""
    
    print("\n" + "="*80)
    print("ADDRESS VALIDATION RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL SCORE: {result['overall_score']:.3f}")
    print(f"   Total Addresses: {result.get('total_addresses', 0)}")
    print(f"   Region Matches: {result.get('region_matches', 0)}")
    print(f"   Heuristic Perfect: {result.get('heuristic_perfect', False)}")
    print(f"   API Result: {result.get('api_result', 'N/A')}")
    
    if 'detailed_breakdown' in result:
        breakdown = result['detailed_breakdown']
        
        # Print validation results per name
        if 'validation_results' in breakdown:
            print("\n" + "-"*80)
            print("DETAILED VALIDATION BY NAME")
            print("-"*80)
            
            for name, validations in breakdown['validation_results'].items():
                print(f"\n🔍 {name}:")
                
                for i, val_result in enumerate(validations, 1):
                    status_emoji = "✅" if val_result['passed_validation'] else "❌"
                    print(f"\n  {status_emoji} Address {i}:")
                    print(f"     Address: {val_result['address'][:80]}...")
                    print(f"     Seed: {val_result['seed_address'][:80]}...")
                    print(f"     Looks Like Address: {val_result['looks_like_address']}")
                    print(f"     Region Match: {val_result['region_match']}")
                    print(f"     Status: {val_result['status']}")
                    
                    # Extract and display city/country
                    if val_result['address']:
                        city, country = extract_city_country(val_result['address'])
                        print(f"     Extracted: City='{city}', Country='{country}'")
        
        # Print API validation results
        if 'api_validation' in breakdown:
            api_val = breakdown['api_validation']
            print("\n" + "-"*80)
            print("API VALIDATION RESULTS")
            print("-"*80)
            
            print(f"\n  API Result: {api_val.get('api_result', 'N/A')}")
            print(f"  Total Eligible Addresses: {api_val.get('total_eligible_addresses', 0)}")
            print(f"  Total API Calls: {api_val.get('total_calls', 0)}")
            
            if 'api_attempts' in api_val and api_val['api_attempts']:
                print("\n  API Call Details:")
                for attempt in api_val['api_attempts']:
                    result_emoji = "✅" if attempt['result'] == True else "⏱️" if attempt['result'] == "TIMEOUT" else "❌"
                    print(f"    {result_emoji} Attempt {attempt['attempt']}: {attempt['result']}")
                    print(f"       Address: {attempt['address'][:70]}...")
            
            # Print API statistics
            if api_val.get('total_calls', 0) > 0:
                print(f"\n  Nominatim Statistics:")
                print(f"    ✅ Successful: {api_val.get('nominatim_successful_calls', 0)}")
                print(f"    ⏱️ Timeouts: {api_val.get('nominatim_timeout_calls', 0)}")
                print(f"    ❌ Failed: {api_val.get('nominatim_failed_calls', 0)}")
    
    print("\n" + "="*80 + "\n")


def test_single_address(address: str, seed_address: str):
    """Test a single address with detailed output."""
    print(f"\n{'='*80}")
    print(f"TESTING SINGLE ADDRESS")
    print(f"{'='*80}")
    print(f"\nSeed Address: {seed_address}")
    print(f"Test Address: {address}")
    
    # Test heuristics
    print(f"\n📋 HEURISTIC CHECKS:")
    looks_valid = looks_like_address(address)
    print(f"  Looks Like Address: {looks_valid}")
    
    if looks_valid:
        region_match = validate_address_region(address, seed_address)
        print(f"  Region Match: {region_match}")
        
        city, country = extract_city_country(address)
        print(f"  Extracted City: '{city}'")
        print(f"  Extracted Country: '{country}'")
        
        seed_city, seed_country = extract_city_country(seed_address)
        print(f"  Seed City: '{seed_city}'")
        print(f"  Seed Country: '{seed_country}'")
    
    # Create test data and run full validation
    variations = {
        "TestName": [["TestName", "1990-01-01", address]]
    }
    seed_addresses = [seed_address]
    miner_metrics = {}
    
    print(f"\n🔍 RUNNING FULL VALIDATION (including API):")
    result = _grade_address_variations(variations, seed_addresses, miner_metrics)
    
    print_validation_details(result)


def test_address_list(addresses: List[str], seed_addresses: List[str]):
    """Test a list of addresses."""
    print(f"\n{'='*80}")
    print(f"TESTING ADDRESS LIST")
    print(f"{'='*80}")
    print(f"\nNumber of addresses: {len(addresses)}")
    print(f"Number of seed addresses: {len(seed_addresses)}")
    
    # Create test data
    test_data = create_test_data_from_addresses(addresses, seed_addresses)
    
    print(f"\nVariations per seed address:")
    for name, vars_list in test_data['variations'].items():
        print(f"  {name}: {len(vars_list)} variations")
    
    # Run validation
    miner_metrics = {}
    print(f"\n🔍 RUNNING VALIDATION:")
    result = _grade_address_variations(
        test_data['variations'],
        test_data['seed_addresses'],
        miner_metrics
    )
    
    print_validation_details(result)


def interactive_mode():
    """Interactive mode for testing addresses."""
    print("\n" + "="*80)
    print("INTERACTIVE ADDRESS VALIDATION")
    print("="*80)
    
    seed_address = input("\nEnter seed address: ").strip()
    if not seed_address:
        print("Error: Seed address cannot be empty")
        return
    
    print("\nEnter addresses to test (one per line, empty line to finish):")
    addresses = []
    while True:
        addr = input("> ").strip()
        if not addr:
            break
        addresses.append(addr)
    
    if not addresses:
        print("Error: No addresses provided")
        return
    
    print(f"\nTesting {len(addresses)} address(es)...")
    
    if len(addresses) == 1:
        test_single_address(addresses[0], seed_address)
    else:
        test_address_list(addresses, [seed_address])


def main():
    # Import these here to avoid bittensor argparse conflicts
    from MIID.validator.reward import (
        _grade_address_variations,
        looks_like_address,
        validate_address_region,
        extract_city_country
    )
    import bittensor as bt
    
    # Make these available to other functions
    globals()['_grade_address_variations'] = _grade_address_variations
    globals()['looks_like_address'] = looks_like_address
    globals()['validate_address_region'] = validate_address_region
    globals()['extract_city_country'] = extract_city_country
    globals()['bt'] = bt
    
    parser = argparse.ArgumentParser(
        description='Test address validation using _grade_address_variations from reward.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python test_address_validation.py -i
  
  # Test from JSON file
  python test_address_validation.py -f test_addresses.json
  
  # Test single address
  python test_address_validation.py -s "123 Main St, New York, United States" -seed "456 5th Ave, New York, United States"
  
  # Test list of addresses from command line
  python test_address_validation.py -a "addr1" "addr2" "addr3" -seed "seed_address"
        """
    )
    
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode')
    parser.add_argument('-f', '--file', type=str,
                        help='JSON file with test data')
    parser.add_argument('-s', '--single', type=str,
                        help='Single address to test')
    parser.add_argument('-a', '--addresses', nargs='+',
                        help='List of addresses to test')
    parser.add_argument('-seed', '--seed-address', type=str,
                        help='Seed address for validation')
    parser.add_argument('-seeds', '--seed-addresses', nargs='+',
                        help='List of seed addresses (one per name)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        bt.logging.set_trace(True)
        bt.logging.set_debug(True)
    
    try:
        if args.interactive:
            interactive_mode()
        
        elif args.file:
            print(f"Loading test data from: {args.file}")
            data = load_addresses_from_json(args.file)
            
            miner_metrics = {}
            result = _grade_address_variations(
                data['variations'],
                data['seed_addresses'],
                miner_metrics
            )
            
            print_validation_details(result)
        
        elif args.single:
            if not args.seed_address:
                print("Error: --seed-address required when using --single")
                return 1
            
            test_single_address(args.single, args.seed_address)
        
        elif args.addresses:
            if args.seed_addresses:
                seed_addresses = args.seed_addresses
            elif args.seed_address:
                seed_addresses = [args.seed_address]
            else:
                print("Error: --seed-address or --seed-addresses required when using --addresses")
                return 1
            
            test_address_list(args.addresses, seed_addresses)
        
        else:
            print("No test mode specified. Use -i, -f, -s, or -a")
            print("Run with --help for usage information")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

