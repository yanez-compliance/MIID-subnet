#!/usr/bin/env python3
"""
Script to analyze prefetched addresses in the addrs directory.
Shows address counts per country using geonamescache for country information.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import geonamescache

def analyze_addresses(addrs_dir):
    """
    Analyze the address directory and count addresses per country.
    
    Args:
        addrs_dir: Path to the addrs directory
    
    Returns:
        dict: Country statistics
    """
    addrs_path = Path(addrs_dir)
    
    if not addrs_path.exists():
        print(f"Error: Directory {addrs_dir} does not exist")
        return None
    
    # Initialize geonamescache
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    
    # Statistics
    country_stats = defaultdict(lambda: {
        'cities': 0,
        'total_addresses': 0,
        'country_name': 'Unknown'
    })
    
    # Iterate through country directories
    for country_dir in sorted(addrs_path.iterdir()):
        if not country_dir.is_dir():
            continue
        
        country_code = country_dir.name
        
        # Get country name from geonamescache
        country_name = 'Unknown'
        for code, info in countries.items():
            if info['iso'] == country_code:
                country_name = info['name']
                break
        
        country_stats[country_code]['country_name'] = country_name
        
        # Count JSON files (cities) in this country
        json_files = list(country_dir.glob('*.json'))
        country_stats[country_code]['cities'] = len(json_files)
        
        # Count total addresses
        total_addrs = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Count addresses in the file
                    if 'count' in data:
                        total_addrs += data['count']
                    elif 'addresses' in data:
                        total_addrs += len(data['addresses'])
            except Exception as e:
                print(f"Warning: Error reading {json_file}: {e}")
        
        country_stats[country_code]['total_addresses'] = total_addrs
    
    return dict(country_stats)


def print_statistics(stats):
    """Print statistics in a formatted table."""
    if not stats:
        print("No statistics to display")
        return
    
    # Calculate totals
    total_countries = len(stats)
    total_cities = sum(s['cities'] for s in stats.values())
    total_addresses = sum(s['total_addresses'] for s in stats.values())
    
    print("\n" + "="*80)
    print("ADDRESS DIRECTORY ANALYSIS")
    print("="*80)
    print(f"\nTotal Countries: {total_countries}")
    print(f"Total Cities: {total_cities}")
    print(f"Total Addresses: {total_addresses}")
    print("\n" + "-"*80)
    print(f"{'Code':<6} {'Country':<30} {'Cities':<10} {'Addresses':<12}")
    print("-"*80)
    
    # Sort by number of cities (descending)
    sorted_stats = sorted(stats.items(), 
                         key=lambda x: x[1]['cities'], 
                         reverse=True)
    
    for country_code, data in sorted_stats:
        country_name = data['country_name']
        cities = data['cities']
        addresses = data['total_addresses']
        
        print(f"{country_code:<6} {country_name:<30} {cities:<10} {addresses:<12}")
    
    print("-"*80)
    print(f"{'TOTAL':<6} {'':<30} {total_cities:<10} {total_addresses:<12}")
    print("="*80)


def main():
    # Get the script directory
    script_dir = Path(__file__).parent
    addrs_dir = script_dir / 'addrs'
    
    print(f"Analyzing addresses in: {addrs_dir}")
    
    # Analyze
    stats = analyze_addresses(addrs_dir)
    
    if stats:
        # Print statistics
        print_statistics(stats)
        
        # Additional insights
        print("\nTOP 10 COUNTRIES BY NUMBER OF CITIES:")
        print("-"*50)
        sorted_by_cities = sorted(stats.items(), 
                                 key=lambda x: x[1]['cities'], 
                                 reverse=True)[:10]
        for i, (code, data) in enumerate(sorted_by_cities, 1):
            print(f"{i:2}. {data['country_name']:<30} ({code}): {data['cities']} cities")
        
        print("\nTOP 10 COUNTRIES BY NUMBER OF ADDRESSES:")
        print("-"*50)
        sorted_by_addrs = sorted(stats.items(), 
                                key=lambda x: x[1]['total_addresses'], 
                                reverse=True)[:10]
        for i, (code, data) in enumerate(sorted_by_addrs, 1):
            print(f"{i:2}. {data['country_name']:<30} ({code}): {data['total_addresses']} addresses")


if __name__ == '__main__':
    main()

