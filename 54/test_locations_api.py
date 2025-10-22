#!/usr/bin/env python3
"""
Test script for the Address Allocation Service API
Tests the /locations endpoint and other API endpoints.
"""

import requests
import json
import sys
from typing import Dict, List, Optional
from datetime import datetime


class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level:7} | {message}")
    
    def test_result(self, test_name: str, passed: bool, details: str = ""):
        status = "✓ PASS" if passed else "✗ FAIL"
        self.test_results.append((test_name, passed, details))
        if passed:
            self.passed += 1
            self.log(f"{status} - {test_name}", "SUCCESS")
        else:
            self.failed += 1
            self.log(f"{status} - {test_name}: {details}", "ERROR")
        if details and passed:
            self.log(f"  {details}", "INFO")
    
    def test_health(self):
        """Test GET /health endpoint"""
        self.log("Testing GET /health...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.test_result(
                    "Health Check",
                    True,
                    f"Status: {data.get('status')}, Base: {data.get('addr_base')}"
                )
                return True
            else:
                self.test_result("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.test_result("Health Check", False, str(e))
            return False
    
    def test_stats(self):
        """Test GET /stats endpoint"""
        self.log("Testing GET /stats...")
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                total = data.get('total_addresses', 0)
                countries = len(data.get('countries', {}))
                rounds = data.get('rounds', 0)
                self.test_result(
                    "Stats Endpoint",
                    True,
                    f"Countries: {countries}, Total Addresses: {total}, Rounds: {rounds}"
                )
                return data
            else:
                self.test_result("Stats Endpoint", False, f"Status code: {response.status_code}")
                return None
        except Exception as e:
            self.test_result("Stats Endpoint", False, str(e))
            return None
    
    def test_locations_basic(self):
        """Test POST /locations with a simple request"""
        self.log("Testing POST /locations (basic)...")
        
        payload = {
            "round_id": "test_round_001",
            "seed_addresses": ["US", "UK", "Japan"],
            "per_seed": 2
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/locations",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                warnings = data.get('warnings', [])
                
                # Check structure
                if len(results) == 3:
                    details = []
                    for r in results:
                        seed = r.get('seed')
                        cc = r.get('country_code')
                        allocated = r.get('allocated', [])
                        exhausted = r.get('exhausted', False)
                        details.append(f"{seed} -> {cc}: {len(allocated)} addresses")
                    
                    self.test_result(
                        "Locations Basic",
                        True,
                        f"Allocated for 3 seeds: {'; '.join(details)}"
                    )
                    
                    if warnings:
                        self.log(f"  Warnings: {warnings}", "WARNING")
                    
                    return data
                else:
                    self.test_result(
                        "Locations Basic",
                        False,
                        f"Expected 3 results, got {len(results)}"
                    )
            else:
                self.test_result(
                    "Locations Basic",
                    False,
                    f"Status code: {response.status_code}, Body: {response.text[:200]}"
                )
        except Exception as e:
            self.test_result("Locations Basic", False, str(e))
        
        return None
    
    def test_locations_various_formats(self):
        """Test /locations with various seed formats"""
        self.log("Testing POST /locations (various formats)...")
        
        payload = {
            "round_id": "test_round_013",
            "seed_addresses": [
                "luhansk",           # ISO2 code
                "donetsk",           # ISO2 code
                "crimea",           # ISO2 code
                "west sahara",           # ISO2 code
                # "Russia",      # Country name
                # "FR",           # ISO2 code
                # "Tokyo, Japan", # City, Country format
                # "India",        # Country name
            ],
            "per_seed": 10
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/locations",
                json=payload,
                timeout=10
            )
            self.log(f"Response: {response.json()}")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                success_count = sum(1 for r in results if not r.get('exhausted', False))
                details = f"Successfully allocated for {success_count}/{len(results)} seeds"
                
                self.test_result("Locations Various Formats", True, details)
                
                # Show details
                for r in results:
                    self.log(f"  {r['seed']:20} -> {r.get('country_code', 'N/A'):3} : {len(r.get('allocated', []))} addresses", "INFO")
                
                return data
            else:
                self.test_result(
                    "Locations Various Formats",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            self.test_result("Locations Various Formats", False, str(e))
        
        return None
    
    def test_locations_persistence(self):
        """Test that allocations persist across calls (no duplicate addresses)"""
        self.log("Testing POST /locations (persistence check)...")
        
        round_id = "test_persistence_001"
        seed = "BR"
        
        # First allocation
        payload1 = {
            "round_id": round_id,
            "seed_addresses": [seed],
            "per_seed": 5
        }
        
        try:
            response1 = requests.post(f"{self.base_url}/locations", json=payload1, timeout=10)
            if response1.status_code != 200:
                self.test_result("Locations Persistence", False, "First request failed")
                return
            
            data1 = response1.json()
            addrs1 = set(data1['results'][0]['allocated'])
            
            # Second allocation (same round, same seed)
            payload2 = {
                "round_id": round_id,
                "seed_addresses": [seed],
                "per_seed": 5
            }
            
            response2 = requests.post(f"{self.base_url}/locations", json=payload2, timeout=10)
            if response2.status_code != 200:
                self.test_result("Locations Persistence", False, "Second request failed")
                return
            
            data2 = response2.json()
            addrs2 = set(data2['results'][0]['allocated'])
            
            # Check for no overlap
            overlap = addrs1 & addrs2
            
            if len(overlap) == 0:
                self.test_result(
                    "Locations Persistence",
                    True,
                    f"No duplicate addresses across 2 requests (10 unique addresses allocated)"
                )
            else:
                self.test_result(
                    "Locations Persistence",
                    False,
                    f"Found {len(overlap)} duplicate addresses: {list(overlap)[:3]}"
                )
            
        except Exception as e:
            self.test_result("Locations Persistence", False, str(e))
    
    def test_locations_high_volume(self):
        """Test /locations with higher per_seed value"""
        self.log("Testing POST /locations (high volume)...")
        
        payload = {
            "round_id": "test_high_volume_001",
            "seed_addresses": ["IN", "RU"],
            "per_seed": 20
        }
        
        try:
            response = requests.post(f"{self.base_url}/locations", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                total_allocated = sum(len(r.get('allocated', [])) for r in results)
                expected = 40  # 2 seeds × 20
                
                if total_allocated >= expected * 0.8:  # Allow 80% threshold
                    self.test_result(
                        "Locations High Volume",
                        True,
                        f"Allocated {total_allocated}/{expected} addresses"
                    )
                else:
                    self.test_result(
                        "Locations High Volume",
                        False,
                        f"Only allocated {total_allocated}/{expected} addresses"
                    )
            else:
                self.test_result("Locations High Volume", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.test_result("Locations High Volume", False, str(e))
    
    def test_rounds_list(self):
        """Test GET /rounds endpoint"""
        self.log("Testing GET /rounds...")
        
        try:
            response = requests.get(f"{self.base_url}/rounds", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                rounds = data.get('rounds', {})
                total = data.get('total_rounds', 0)
                
                self.test_result(
                    "Rounds List",
                    True,
                    f"Found {total} round(s): {list(rounds.keys())[:5]}"
                )
                return data
            else:
                self.test_result("Rounds List", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.test_result("Rounds List", False, str(e))
        
        return None
    
    def test_round_detail(self, round_id: str = "test_round_001"):
        """Test GET /rounds/{round_id} endpoint"""
        self.log(f"Testing GET /rounds/{round_id}...")
        
        try:
            response = requests.get(f"{self.base_url}/rounds/{round_id}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                seeds = data.get('seeds', {})
                total_seeds = data.get('total_seeds', 0)
                total_used = data.get('total_used', 0)
                
                self.test_result(
                    "Round Detail",
                    True,
                    f"Round {round_id}: {total_seeds} seeds, {total_used} addresses used"
                )
                return data
            elif response.status_code == 404:
                self.test_result("Round Detail", True, f"Round {round_id} not found (expected if not created)")
            else:
                self.test_result("Round Detail", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.test_result("Round Detail", False, str(e))
        
        return None
    
    def test_invalid_requests(self):
        """Test error handling with invalid requests"""
        self.log("Testing error handling...")
        
        # Test 1: Empty seed_addresses
        try:
            payload = {
                "round_id": "test_error_001",
                "seed_addresses": [],
                "per_seed": 1
            }
            response = requests.post(f"{self.base_url}/locations", json=payload, timeout=5)
            
            if response.status_code == 400:
                self.test_result("Error Handling (Empty Seeds)", True, "Correctly rejected empty seeds")
            else:
                self.test_result("Error Handling (Empty Seeds)", False, f"Expected 400, got {response.status_code}")
        except Exception as e:
            self.test_result("Error Handling (Empty Seeds)", False, str(e))
        
        # Test 2: Invalid country
        try:
            payload = {
                "round_id": "test_error_002",
                "seed_addresses": ["INVALID_COUNTRY_XYZ"],
                "per_seed": 1
            }
            response = requests.post(f"{self.base_url}/locations", json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                warnings = data.get('warnings', [])
                if warnings:
                    self.test_result("Error Handling (Invalid Country)", True, "Warning generated for invalid country")
                else:
                    self.test_result("Error Handling (Invalid Country)", False, "No warning for invalid country")
            else:
                self.test_result("Error Handling (Invalid Country)", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.test_result("Error Handling (Invalid Country)", False, str(e))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0:.1f}%")
        print("="*80)
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for name, passed, details in self.test_results:
                if not passed:
                    print(f"  ✗ {name}: {details}")
        
        return self.failed == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test the Address Allocation Service API")
    parser.add_argument(
        "--url",
        default="http://localhost:9999",
        help="Base URL of the API (default: http://localhost:9999)"
    )
    parser.add_argument(
        "--test",
        choices=["all", "health", "stats", "locations", "rounds", "errors"],
        default="all",
        help="Which test(s) to run"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    print("="*80)
    print(f"ADDRESS ALLOCATION SERVICE API TESTER")
    print(f"Target: {args.url}")
    print("="*80)
    print()
    
    # Run tests based on selection
    if args.test in ["all", "health"]:
        if not tester.test_health():
            print("\n⚠️  Service not available. Exiting.")
            sys.exit(1)
        print()
    
    if args.test in ["all", "stats"]:
        tester.test_stats()
        print()
    
    if args.test in ["all", "locations"]:
        tester.test_locations_various_formats()
        print()
    
    # if args.test in ["all", "rounds"]:
    #     tester.test_rounds_list()
    #     print()
    #     tester.test_round_detail()
    #     print()
    
    # if args.test in ["all", "errors"]:
    #     tester.test_invalid_requests()
    #     print()
    
    # # Print summary
    # success = tester.print_summary()
    # sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

