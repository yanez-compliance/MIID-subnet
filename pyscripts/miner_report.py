#!/usr/bin/env python3
"""
Miner Report Script for Bittensor Subnet 54

This script fetches miner statistics from the API and displays them in a formatted table
with filtering and sorting capabilities.
"""

import argparse
import json
import requests
import sys
from typing import Dict, List, Optional, Any
from tabulate import tabulate
import bittensor as bt

subtensor = bt.subtensor()
metagraph = subtensor.metagraph(netuid=54)


def fetch_miners_stats() -> Optional[Dict]:
    """Fetch miners statistics from the API."""
    url = "https://miners-stats.yanezcompliance.net/miners_stats"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching miners stats: {e}", file=sys.stderr)
        return None


def get_coldkey_for_uid(uid: int) -> Optional[str]:
    """Get coldkey for a given UID using bittensor SDK."""
    try:
        if uid < len(metagraph.coldkeys):
            return metagraph.coldkeys[uid]
        else:
            return None
    except Exception as e:
        print(f"Error getting coldkey for UID {uid}: {e}", file=sys.stderr)
        return None


def parse_miner_data(data: Dict) -> List[Dict[str, Any]]:
    """Parse the miners data and extract relevant information."""
    miners = []
    
    for miner in data['miners']:
        try:
            uid = int(miner['miner_uid'])
            
            # Extract scores with defaults
            total_score = miner.get('score', 0.0)
            phonetic_score = miner.get('similarity_metrics', {}).get('phonetic_similarity', 0.0)
            orthographic_score = miner.get('similarity_metrics', {}).get('orthographic_similarity', 0.0)
            rule_score = miner.get('rule_compliance_score', 0.0)
            
            # Get coldkey
            coldkey = get_coldkey_for_uid(uid)
            
            miners.append({
                'uid': uid,
                'coldkey': coldkey or 'N/A',
                'score': total_score,
                'phonetic_score': phonetic_score,
                'orthographic_score': orthographic_score,
                'rule_score': rule_score
            })
            
        except (ValueError, KeyError) as e:
            print(f"Error parsing data for UID {uid}: {e}", file=sys.stderr)
            continue
    
    return miners


def apply_filters(miners: List[Dict[str, Any]], filters: Dict[str, str]) -> List[Dict[str, Any]]:
    """Apply filters to the miners list."""
    filtered_miners = miners.copy()
    
    for column, filter_value in filters.items():
        if not filter_value:
            continue
            
        try:
            if column == 'coldkey':
                filtered_miners = [m for m in filtered_miners if filter_value.lower() in m['coldkey'].lower()]
            elif column in ['score', 'phonetic_score', 'orthographic_score', 'rule_score', 'count']:
                # Support range filters like ">0.5", "<1.0", "0.5-1.0"
                if '-' in filter_value and not filter_value.startswith('-'):
                    # Range filter
                    min_val, max_val = map(float, filter_value.split('-'))
                    filtered_miners = [m for m in filtered_miners if min_val <= m[column] <= max_val]
                elif filter_value.startswith('>='):
                    threshold = float(filter_value[2:])
                    filtered_miners = [m for m in filtered_miners if m[column] >= threshold]
                elif filter_value.startswith('<='):
                    threshold = float(filter_value[2:])
                    filtered_miners = [m for m in filtered_miners if m[column] <= threshold]
                elif filter_value.startswith('>'):
                    threshold = float(filter_value[1:])
                    filtered_miners = [m for m in filtered_miners if m[column] > threshold]
                elif filter_value.startswith('<'):
                    threshold = float(filter_value[1:])
                    filtered_miners = [m for m in filtered_miners if m[column] < threshold]
                else:
                    # Exact match
                    threshold = float(filter_value)
                    filtered_miners = [m for m in filtered_miners if abs(m[column] - threshold) < 1e-6]
        except ValueError as e:
            print(f"Invalid filter value for {column}: {filter_value} ({e})", file=sys.stderr)
            continue
    
    return filtered_miners


def sort_miners(miners: List[Dict[str, Any]], sort_by: str, reverse: bool = True) -> List[Dict[str, Any]]:
    """Sort miners by the specified column."""
    valid_columns = ['coldkey', 'score', 'phonetic_score', 'orthographic_score', 'rule_score', 'count']
    
    if sort_by not in valid_columns:
        print(f"Invalid sort column: {sort_by}. Valid options: {', '.join(valid_columns)}", file=sys.stderr)
        sort_by = 'score'  # Default fallback
    
    try:
        return sorted(miners, key=lambda x: x[sort_by], reverse=reverse)
    except KeyError:
        print(f"Sort column {sort_by} not found in data", file=sys.stderr)
        return miners


def format_table(miners: List[Dict[str, Any]]) -> str:
    """Format the miners data as a table."""
    if not miners:
        return "No miners found matching the criteria."
    
    headers = ['Rank', 'Coldkey', 'Score', 'Phonetic Sim Score', 'Orthographic Sim Score', 'Rule Score', 'Count']
    
    table_data = []
    for rank, miner in enumerate(miners, 1):
        # Truncate coldkey for better display
        coldkey_display = miner['coldkey']
        if len(coldkey_display) > 12 and coldkey_display != 'N/A':
            coldkey_display = coldkey_display[:8] + '...'
        
        table_data.append([
            rank,
            coldkey_display,
            f"{miner['score']:.4f}",
            f"{miner['phonetic_score']:.4f}",
            f"{miner['orthographic_score']:.4f}",
            f"{miner['rule_score']:.4f}",
            f"{miner['count']}"
        ])
    
    return tabulate(table_data, headers=headers, tablefmt='grid')


def main():
    parser = argparse.ArgumentParser(description='Generate miner report for Bittensor Subnet 54')
    
    # Filter options
    parser.add_argument('--filter-coldkey', type=str, help='Filter by coldkey (substring match)')
    parser.add_argument('--filter-score', type=str, help='Filter by total score (supports >, <, >=, <=, range with -)')
    parser.add_argument('--filter-phonetic-score', type=str, help='Filter by phonetic similarity score')
    parser.add_argument('--filter-orthographic-score', type=str, help='Filter by orthographic similarity score')
    parser.add_argument('--filter-rule-score', type=str, help='Filter by rule-based score')
    parser.add_argument('--filter-count', type=str, help='Filter by count')
    # Sort options
    parser.add_argument('--sort-by', type=str, default='score', 
                       choices=['coldkey', 'score', 'phonetic_score', 'orthographic_score', 'rule_score', 'count'],
                       help='Column to sort by (default: score)')
    parser.add_argument('--ascending', action='store_true', help='Sort in ascending order (default: descending)')
    
    # Output options
    parser.add_argument('--limit', type=int, help='Limit number of results displayed')
    parser.add_argument('--json-output', action='store_true', help='Output results as JSON instead of table')
    
    args = parser.parse_args()
    
    # Fetch data
    print("Fetching miners statistics...", file=sys.stderr)
    data = fetch_miners_stats()
    if not data:
        sys.exit(1)
    
    # Parse miners data
    print("Processing miner data and fetching coldkeys...", file=sys.stderr)
    miners = parse_miner_data(data)
    aggregated_miners = {}
    for miner in miners:
        if miner['coldkey'] not in aggregated_miners:
            miner['count'] = 1
            aggregated_miners[miner['coldkey']] = miner
        else:
            aggregated_miners[miner['coldkey']]['score'] = aggregated_miners[miner['coldkey']]['score'] + miner['score']
            aggregated_miners[miner['coldkey']]['phonetic_score'] = aggregated_miners[miner['coldkey']]['phonetic_score'] + miner['phonetic_score']
            aggregated_miners[miner['coldkey']]['orthographic_score'] = aggregated_miners[miner['coldkey']]['orthographic_score'] + miner['orthographic_score']
            aggregated_miners[miner['coldkey']]['rule_score'] = aggregated_miners[miner['coldkey']]['rule_score'] + miner['rule_score']
            aggregated_miners[miner['coldkey']]['count'] = aggregated_miners[miner['coldkey']]['count'] + 1
    # Average the aggregated scores
    for coldkey, miner in aggregated_miners.items():
        miner['score'] = miner['score'] / miner['count']
        miner['phonetic_score'] = miner['phonetic_score'] / miner['count']
        miner['orthographic_score'] = miner['orthographic_score'] / miner['count']
        miner['rule_score'] = miner['rule_score'] / miner['count']
    # Convert aggregated_miners dict to list and sort by score
    miners = list(aggregated_miners.values())
    miners.sort(key=lambda x: x['score'], reverse=True)
    if not miners:
        print("No valid miner data found.")
        sys.exit(1)
    
    # Apply filters
    filters = {
        'coldkey': args.filter_coldkey,
        'score': args.filter_score,
        'phonetic_score': args.filter_phonetic_score,
        'orthographic_score': args.filter_orthographic_score,
        'rule_score': args.filter_rule_score,
        'count': args.filter_count
    }
    
    filtered_miners = apply_filters(miners, filters)
    
    # Sort miners
    sorted_miners = sort_miners(filtered_miners, args.sort_by, reverse=not args.ascending)
    
    # Apply limit
    if args.limit:
        sorted_miners = sorted_miners[:args.limit]
    
    # Output results
    if args.json_output:
        print(json.dumps(sorted_miners, indent=2))
    else:
        print(format_table(sorted_miners))


if __name__ == '__main__':
    main()
