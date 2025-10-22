#!/usr/bin/env python3
"""
Script to filter and remove similar addresses from the addrs directory.
Uses the address similarity detection logic from cheat_detection.py.
Includes address validation using logic from collect_city_info.py.
"""

import json
import re
import logging
import time
import requests
import geonamescache
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize geonamescache for validation
gc = geonamescache.GeonamesCache()
CITIES = gc.get_cities()
COUNTRIES = gc.get_countries()

# Address validation cache
ADDR_CHECKED_CACHE: Dict[str, bool] = {}

# Country name mapping from collect_city_info.py
COUNTRY_MAPPING = {
    "korea, south": "south korea",
    "korea, north": "north korea",
    "cote d ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "ivory coast": "ivory coast",
    "the gambia": "gambia",
    "gambia": "gambia",
    "netherlands": "the netherlands",
    "holland": "the netherlands",
    "congo, democratic republic of the": "democratic republic of the congo",
    "drc": "democratic republic of the congo",
    "congo, republic of the": "republic of the congo",
    "burma": "myanmar",
    "bonaire": "bonaire, saint eustatius and saba",
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "britain": "united kingdom",
    "uae": "united arab emirates",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "u.k.": "united kingdom",
}

SPECIAL_REGIONS = ["luhansk", "crimea", "donetsk", "west sahara"]


def normalize_address(address: str) -> str:
    """
    Normalize an address for comparison by removing digits, spaces, commas, hyphens, 
    semicolons, and converting to lowercase.
    Based on the logic from cheat_detection.py lines 309-313 and 358-362.
    """
    if not address or not address.strip():
        return ""
    
    # Remove street numbers and ranges (e.g., "123-456" or "123")
    normalized = re.sub(r'\d+-\d+|\d+', '', address)
    # Remove spaces, commas, hyphens, semicolons
    normalized = normalized.strip().replace(" ", "").replace(",", "").replace("-", "").replace(";", "")
    # Convert to lowercase
    normalized = normalized.lower()
    return normalized


# ============================================================================
# Address validation functions from collect_city_info.py
# ============================================================================

def looks_like_address(address: str) -> bool:
    """Check if a string looks like a valid address."""
    address = address.strip().lower()

    # Keep all letters (Latin and non-Latin) and numbers
    address_len = re.sub(r'[^\w]', '', address.strip(), flags=re.UNICODE)

    if len(address_len) < 30:
        logging.debug(f"Address has less than 30 characters: {address}")
        return False
    if len(address_len) > 300:  # maximum length check
        return False

    # Count letters (both Latin and non-Latin)
    letter_count = len(re.findall(r'[^\W\d]', address, flags=re.UNICODE))
    if letter_count < 20:
        logging.debug(f"Address has less than 20 letters: {address}")
        return False

    if re.match(r"^[^a-zA-Z]*$", address):  # no letters at all
        logging.debug(f"Address has no letters: {address}")
        return False
    if len(set(address)) < 5:  # all chars basically the same
        logging.debug(f"Address has all same chars: {address}")
        return False
        
    # Has at least two digit (street number)
    number_groups = re.findall(r"\d+", address)
    if len(number_groups) < 2:
        logging.debug(f"Address has less than 2 numbers: {address}")
        return False

    if address.count(",") < 2:
        logging.debug(f"Address has less than 2 commas: {address}")
        return False
    
    # Check for special characters that should not be in addresses
    special_chars = ['`', ':', '%', '$', '@', '*', '^', '[', ']', '{', '}']
    if any(char in address for char in special_chars):
        logging.debug(f"Address has special characters: {address}")
        return False
    return True


def extract_city_country(address: str) -> tuple:
    """Extract city and country from an address string."""
    if not address:
        return "", ""

    address = address.lower()
    
    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 2:
        return "", ""
    
    country = parts[-1]
    country = COUNTRY_MAPPING.get(country.lower(), country.lower())
    
    # If no country found, return empty
    if not country:
        return "", ""

    # Check each section from right to left (excluding the country)
    for i in range(2, len(parts) + 1):
        candidate_index = -i
        if abs(candidate_index) > len(parts):
            break
        
        candidate_part = parts[candidate_index]
        if not candidate_part:
            continue
            
        words = candidate_part.split()
        
        # Try different combinations of words (1-2 words max)
        for num_words in range(len(words)):
            current_word = words[num_words]

            # Try current word
            candidates = [current_word]

            # Also try current + previous (if exists)
            if num_words > 0:
                prev_plus_current = words[num_words - 1] + " " + words[num_words]
                candidates.append(prev_plus_current)

            for city_candidate in candidates:
                # Skip if contains numbers or is too short
                if any(char.isdigit() for char in city_candidate):
                    continue

                # Validate the city exists in the country
                if city_in_country(city_candidate, country):
                    return city_candidate, country

    return "", country


def city_in_country(city_name: str, country_name: str) -> bool:
    """Check if a city is in the specified country using geonamescache."""
    if not city_name or not country_name:
        return False
    
    try:
        city_name_lower = city_name.lower()
        country_name_lower = country_name.lower()
        
        # Find country code
        country_code = None
        for code, data in COUNTRIES.items():
            if data.get('name', '').lower() == country_name_lower:
                country_code = code
                break
        
        if not country_code:
            return False
        
        # Only check cities that are actually in the specified country
        city_words = city_name_lower.split()
        
        for city_id, city_data in CITIES.items():
            # Skip cities not in the target country
            if city_data.get("countrycode", "") != country_code:
                continue
                
            city_data_name = city_data.get("name", "").lower()
            
            # Check exact match first
            if city_data_name == city_name_lower:
                return True
            # Check first word match
            elif len(city_words) >= 2 and city_data_name.startswith(city_words[0]):
                return True
            # Check second word match
            elif len(city_words) >= 2 and city_words[1] in city_data_name:
                return True
        
        return False
        
    except Exception as e:
        logging.warning(f"Error checking city '{city_name}' in country '{country_name}': {str(e)}")
        return False


def validate_address_region(generated_address: str, seed_address: str) -> bool:
    """Validate that generated address has correct region from seed address."""
    if not generated_address or not seed_address:
        return False
    
    # Special handling for disputed regions not in geonames
    seed_lower = seed_address.lower()
    if seed_lower in SPECIAL_REGIONS:
        # If seed is a special region, check if that region appears in generated address
        gen_lower = generated_address.lower()
        return seed_lower in gen_lower
    
    # Extract city and country from both addresses
    gen_city, gen_country = extract_city_country(generated_address)
    seed_city, seed_country = seed_address.lower(), seed_address.lower()
    
    # If no city was extracted from generated address, it's an error
    if not gen_city:
        return False
    
    # If no country was extracted from generated address, it's an error
    if not gen_country:
        return False
    
    # Check if either city or country matches
    city_match = gen_city and seed_city and gen_city == seed_city
    country_match = gen_country and seed_country and gen_country == seed_country
    
    if not (city_match or country_match):
        logging.debug(f"No city or country match for generated address: {generated_address} and seed address: {seed_address}")
        return False
    
    # If we have both city and country, validate city is in country
    if gen_city and gen_country:
        return city_in_country(gen_city, gen_country)
    
    return True


def check_with_nominatim(address: str) -> bool:
    """Check if address exists in Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json"}
        response = requests.get(url, params=params, headers={"User-Agent": "address-checker"}, timeout=10)
        return len(response.json()) > 0
    except requests.exceptions.Timeout:
        logging.warning(f"API timeout for address: {address}")
        return False
    except:
        return False


def validate_address(addr_str: str, seed_address: str) -> bool:
    """
    Validate an address using logic from collect_city_info.py.
    
    Args:
        addr_str: The address string to validate
        seed_address: The seed address (typically country_name) to match against
        
    Returns:
        True if address is valid, False otherwise
    """
    val = ADDR_CHECKED_CACHE.get(addr_str)
    if val is True:
        return True
    if val in (False, "TIMEOUT"):
        return False

    # Step 1: quick shape check
    looks_like = looks_like_address(addr_str)

    # Step 2: region constraint
    region_match = False
    if looks_like:
        region_match = validate_address_region(addr_str, seed_address)

    if not looks_like:
        logging.debug(f"address does not look like an address: {addr_str}")
        ADDR_CHECKED_CACHE[addr_str] = False
        return False

    if not region_match:
        logging.debug(f"NO SEED MATCH: {addr_str} : {seed_address}")
        ADDR_CHECKED_CACHE[addr_str] = False
        return False
    return True
    try:
        result = check_with_nominatim(addr_str)
        time.sleep(1)  # gentle pace for the search call
    except Exception as e:
        logging.warning("validate_error addr=%r err=%s", addr_str, e)
        result = False

    ADDR_CHECKED_CACHE[addr_str] = bool(result)
    return bool(result)


def overlap_coefficient(a: Set[str], b: Set[str]) -> float:
    """
    Calculate overlap coefficient between two sets.
    From cheat_detection.py lines 85-89.
    """
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    return inter / float(min(len(a), len(b)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.
    From cheat_detection.py lines 92-99.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / float(union) if union else 1.0


def load_address_file(file_path: Path) -> Dict:
    """Load and parse a single address JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load {file_path}: {e}")
        return {}


def scan_all_addresses(addrs_dir: Path) -> Dict[str, Dict]:
    """
    Scan all address files in the directory.
    Returns: {file_path: {data, normalized_addresses}}
    """
    all_files = {}
    json_files = list(addrs_dir.rglob("*.json"))
    
    logging.info(f"Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        data = load_address_file(file_path)
        if not data or 'addresses' not in data:
            continue
        
        addresses = data.get('addresses', [])
        if not addresses:
            continue
        
        # Normalize all addresses in this file
        normalized = set()
        for addr in addresses:
            norm_addr = normalize_address(addr)
            if norm_addr:
                normalized.add(norm_addr)
        
        if normalized:
            all_files[str(file_path)] = {
                'data': data,
                'normalized_addresses': normalized,
                'original_addresses': addresses,
                'city': data.get('city', 'Unknown'),
                'country_code': data.get('country_code', 'Unknown')
            }
    
    logging.info(f"Successfully loaded {len(all_files)} files with addresses")
    return all_files


def find_similar_files(all_files: Dict[str, Dict], 
                       overlap_threshold: float = 0.8, 
                       jaccard_threshold: float = 0.7) -> List[Tuple[str, str, float, float]]:
    """
    Find pairs of files with similar addresses.
    Returns: List of (file1, file2, overlap, jaccard) tuples
    Based on cheat_detection.py lines 462-492.
    """
    similar_pairs = []
    file_paths = list(all_files.keys())
    
    logging.info(f"Comparing {len(file_paths)} files for similarity...")
    
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            file1 = file_paths[i]
            file2 = file_paths[j]
            
            addr_set1 = all_files[file1]['normalized_addresses']
            addr_set2 = all_files[file2]['normalized_addresses']
            
            if not addr_set1 or not addr_set2:
                continue
            
            # Calculate overlap and jaccard similarity
            overlap = overlap_coefficient(addr_set1, addr_set2)
            jac = jaccard(addr_set1, addr_set2)
            
            # If addresses are too similar, record it
            # Using same thresholds as cheat_detection.py line 489
            if overlap > overlap_threshold or jac > jaccard_threshold:
                similar_pairs.append((file1, file2, overlap, jac))
    
    logging.info(f"Found {len(similar_pairs)} similar file pairs")
    return similar_pairs


def remove_duplicate_addresses_within_file(addresses: List[str]) -> List[str]:
    """
    Remove duplicate addresses within a single file based on normalized form.
    """
    seen = set()
    unique_addresses = []
    
    for addr in addresses:
        norm_addr = normalize_address(addr)
        if norm_addr and norm_addr not in seen:
            seen.add(norm_addr)
            unique_addresses.append(addr)
    
    return unique_addresses


def validate_addresses_in_file(addresses: List[str], country_name: str) -> Tuple[List[str], int]:
    """
    Validate addresses in a file using the validate_address function.
    
    Args:
        addresses: List of addresses to validate
        country_name: The country name to use as seed_address for validation
        
    Returns:
        Tuple of (valid_addresses, invalid_count)
    """
    valid_addresses = []
    invalid_count = 0
    
    logging.info(f"Validating {len(addresses)} addresses for country: {country_name}")
    
    for idx, addr in enumerate(addresses, 1):
        if idx % 10 == 0:
            logging.info(f"  Progress: {idx}/{len(addresses)} addresses validated")
        
        try:
            is_valid = validate_address(addr, seed_address=country_name)
            
            if is_valid:
                valid_addresses.append(addr)
            else:
                invalid_count += 1
                logging.debug(f"Invalid address removed: {addr}")
        except Exception as e:
            logging.warning(f"Error validating address '{addr}': {e}")
            invalid_count += 1
    
    logging.info(f"Validation complete: {len(valid_addresses)} valid, {invalid_count} invalid")
    return valid_addresses, invalid_count


def clean_addresses_in_files(all_files: Dict[str, Dict], 
                             similar_pairs: List[Tuple[str, str, float, float]],
                             mode: str = 'deduplicate') -> Tuple[int, int, int]:
    """
    Clean up addresses based on similarity analysis.
    
    Modes:
    - 'deduplicate': Remove duplicate addresses within each file only
    - 'validate': Validate addresses and remove invalid ones using collect_city_info.py validation
    - 'remove_similar_files': Delete entire files that are too similar to others
    - 'merge': Keep one file from each similar pair, merge unique addresses
    
    Returns: (files_modified, addresses_removed, files_deleted)
    """
    files_modified = 0
    addresses_removed = 0
    files_deleted = 0
    
    if mode == 'validate':
        # Validate addresses and remove invalid ones
        for file_path, file_info in all_files.items():
            original_addresses = file_info['original_addresses']
            country_name = file_info['data'].get('country', '')
            
            if not country_name:
                logging.warning(f"No country name found in {file_path}, skipping validation")
                continue
            
            logging.info(f"\nProcessing file: {Path(file_path).name}")
            valid_addresses, invalid_count = validate_addresses_in_file(original_addresses, country_name)
            
            if invalid_count > 0:
                # Update the file
                data = file_info['data']
                data['addresses'] = valid_addresses
                data['count'] = len(valid_addresses)
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                addresses_removed += invalid_count
                files_modified += 1
                logging.info(f"Removed {invalid_count} invalid addresses from {Path(file_path).name}")
            else:
                logging.info(f"All addresses valid in {Path(file_path).name}")
    
    elif mode == 'deduplicate':
        # Remove duplicate addresses within each file
        for file_path, file_info in all_files.items():
            original_addresses = file_info['original_addresses']
            unique_addresses = remove_duplicate_addresses_within_file(original_addresses)
            
            if len(unique_addresses) < len(original_addresses):
                # Update the file
                data = file_info['data']
                data['addresses'] = unique_addresses
                data['count'] = len(unique_addresses)
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                removed = len(original_addresses) - len(unique_addresses)
                addresses_removed += removed
                files_modified += 1
                logging.info(f"Removed {removed} duplicate addresses from {Path(file_path).name}")
    
    elif mode == 'remove_similar_files':
        # Build a set of files to delete (keep the one with more unique addresses)
        files_to_delete = set()
        
        for file1, file2, overlap, jac in similar_pairs:
            if file1 in files_to_delete or file2 in files_to_delete:
                continue
            
            # Keep the file with more addresses
            count1 = len(all_files[file1]['normalized_addresses'])
            count2 = len(all_files[file2]['normalized_addresses'])
            
            if count1 >= count2:
                files_to_delete.add(file2)
                logging.info(f"Marking {Path(file2).name} for deletion (similar to {Path(file1).name}, overlap={overlap:.2f}, jaccard={jac:.2f})")
            else:
                files_to_delete.add(file1)
                logging.info(f"Marking {Path(file1).name} for deletion (similar to {Path(file2).name}, overlap={overlap:.2f}, jaccard={jac:.2f})")
        
        # Delete marked files
        for file_path in files_to_delete:
            try:
                Path(file_path).unlink()
                files_deleted += 1
                addresses_removed += len(all_files[file_path]['original_addresses'])
                logging.info(f"Deleted {Path(file_path).name}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}: {e}")
    
    elif mode == 'merge':
        # Group similar files and merge them
        # Build a graph of similar files
        from collections import defaultdict
        
        graph = defaultdict(set)
        for file1, file2, overlap, jac in similar_pairs:
            graph[file1].add(file2)
            graph[file2].add(file1)
        
        visited = set()
        groups = []
        
        # Find connected components (groups of similar files)
        def dfs(node, group):
            visited.add(node)
            group.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)
        
        for file_path in graph:
            if file_path not in visited:
                group = []
                dfs(file_path, group)
                groups.append(group)
        
        # For each group, keep one file and merge unique addresses
        for group in groups:
            if len(group) <= 1:
                continue
            
            # Choose the file with most addresses as the base
            group.sort(key=lambda f: len(all_files[f]['normalized_addresses']), reverse=True)
            base_file = group[0]
            
            # Collect all unique addresses from the group
            all_norm_addresses = set()
            all_original_addresses = []
            
            for file_path in group:
                file_info = all_files[file_path]
                for addr in file_info['original_addresses']:
                    norm_addr = normalize_address(addr)
                    if norm_addr and norm_addr not in all_norm_addresses:
                        all_norm_addresses.add(norm_addr)
                        all_original_addresses.append(addr)
            
            # Update the base file
            data = all_files[base_file]['data']
            data['addresses'] = all_original_addresses
            data['count'] = len(all_original_addresses)
            
            with open(base_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            files_modified += 1
            logging.info(f"Merged {len(group)} files into {Path(base_file).name} with {len(all_original_addresses)} unique addresses")
            
            # Delete the other files in the group
            for file_path in group[1:]:
                try:
                    Path(file_path).unlink()
                    files_deleted += 1
                    logging.info(f"Deleted merged file {Path(file_path).name}")
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")
    
    return files_modified, addresses_removed, files_deleted


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter and remove similar addresses from the addrs directory'
    )
    parser.add_argument(
        '--addrs-dir',
        type=str,
        default='/work/MIID-subnet/54/addgen/addrs',
        help='Path to the addrs directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['deduplicate', 'validate', 'remove_similar_files', 'merge', 'report_only'],
        default='report_only',
        help='Action to take: deduplicate (remove dupes within files), validate (remove invalid addresses), remove_similar_files (delete similar files), merge (merge similar files), report_only (just report similarities)'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.8,
        help='Overlap coefficient threshold for considering files similar (default: 0.8)'
    )
    parser.add_argument(
        '--jaccard-threshold',
        type=float,
        default=0.7,
        help='Jaccard similarity threshold for considering files similar (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    addrs_dir = Path(args.addrs_dir)
    if not addrs_dir.exists():
        logging.error(f"Directory not found: {addrs_dir}")
        return
    
    logging.info(f"Scanning addresses in {addrs_dir}")
    all_files = scan_all_addresses(addrs_dir)
    
    if not all_files:
        logging.info("No address files found")
        return
    
    logging.info(f"Loaded {len(all_files)} files")
    
    # # Find similar files
    # similar_pairs = find_similar_files(
    #     all_files, 
    #     overlap_threshold=args.overlap_threshold,
    #     jaccard_threshold=args.jaccard_threshold
    # )
    
    # if similar_pairs:
    #     logging.info("\n" + "="*80)
    #     logging.info("SIMILAR FILES DETECTED:")
    #     logging.info("="*80)
    #     for file1, file2, overlap, jac in similar_pairs[:20]:  # Show top 20
    #         logging.info(f"\nFile 1: {Path(file1).name} ({all_files[file1]['city']}, {all_files[file1]['country_code']})")
    #         logging.info(f"File 2: {Path(file2).name} ({all_files[file2]['city']}, {all_files[file2]['country_code']})")
    #         logging.info(f"Overlap: {overlap:.4f}, Jaccard: {jac:.4f}")
        
    #     if len(similar_pairs) > 20:
    #         logging.info(f"\n... and {len(similar_pairs) - 20} more similar pairs")
    # else:
    #     logging.info("No similar file pairs found")
    
    # Count duplicate addresses within files
    total_duplicates = 0
    files_with_duplicates = 0
    for file_path, file_info in all_files.items():
        original_count = len(file_info['original_addresses'])
        unique_count = len(file_info['normalized_addresses'])
        if unique_count < original_count:
            duplicates = original_count - unique_count
            total_duplicates += duplicates
            files_with_duplicates += 1
    
    logging.info("\n" + "="*80)
    logging.info("DUPLICATE ADDRESSES WITHIN FILES:")
    logging.info("="*80)
    logging.info(f"Files with duplicate addresses: {files_with_duplicates}")
    logging.info(f"Total duplicate addresses: {total_duplicates}")
    
    if args.mode == 'report_only':
        logging.info("\n" + "="*80)
        logging.info("REPORT ONLY MODE - No changes made")
        logging.info("Use --mode to actually modify files:")
        logging.info("  --mode deduplicate: Remove duplicate addresses within each file")
        logging.info("  --mode validate: Remove invalid addresses using validation logic")
        logging.info("  --mode remove_similar_files: Delete files that are too similar")
        logging.info("  --mode merge: Merge similar files together")
        logging.info("="*80)
        return
    
    # Apply cleaning
    logging.info("\n" + "="*80)
    logging.info(f"APPLYING MODE: {args.mode}")
    logging.info("="*80)
    similar_pairs = []
    files_modified, addresses_removed, files_deleted = clean_addresses_in_files(
        all_files, similar_pairs, mode=args.mode
    )
    
    logging.info("\n" + "="*80)
    logging.info("SUMMARY:")
    logging.info("="*80)
    logging.info(f"Files modified: {files_modified}")
    logging.info(f"Addresses removed: {addresses_removed}")
    logging.info(f"Files deleted: {files_deleted}")
    logging.info("="*80)


if __name__ == '__main__':
    main()

