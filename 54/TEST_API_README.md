# Address Allocation Service API Testing

This directory contains a comprehensive test script for the Address Allocation Service API.

## Files

- `test_locations_api.py` - Main test script
- `service.py` - The FastAPI service implementation
- `addgen/` - Directory containing prefetched addresses

## Prerequisites

```bash
pip install requests fastapi uvicorn geonamescache
```

## Starting the Service

First, start the Address Allocation Service:

```bash
cd /work/MIID-subnet/54
uvicorn service:app --host 0.0.0.0 --port 8000
```

Or with custom environment variables:

```bash
cd /work/MIID-subnet/54
ADDGEN_ADDRS_DIR="addgen/addrs" \
STATE_DIR="state" \
LOG_LEVEL="INFO" \
uvicorn service:app --host 0.0.0.0 --port 8000
```

## Running Tests

### Run All Tests

```bash
python3 test_locations_api.py
```

### Run Specific Test Categories

```bash
# Health check only
python3 test_locations_api.py --test health

# Statistics endpoint
python3 test_locations_api.py --test stats

# Location allocation tests
python3 test_locations_api.py --test locations

# Rounds management tests
python3 test_locations_api.py --test rounds

# Error handling tests
python3 test_locations_api.py --test errors
```

### Test Against Different URL

```bash
python3 test_locations_api.py --url http://192.168.1.100:8000
```

## What the Tests Cover

### 1. Health Check (`test_health`)
- Tests `GET /health` endpoint
- Verifies service is running and responsive

### 2. Statistics (`test_stats`)
- Tests `GET /stats` endpoint
- Checks pool sizes and round counts

### 3. Basic Location Allocation (`test_locations_basic`)
- Tests `POST /locations` with simple country codes
- Verifies address allocation for US, UK, Japan
- Checks response structure

### 4. Various Seed Formats (`test_locations_various_formats`)
- Tests different seed formats:
  - ISO2 codes (CN, FR)
  - Country names (Germany, India)
  - City, Country format (Tokyo, Japan)
- Verifies country inference logic

### 5. Persistence Check (`test_locations_persistence`)
- Tests that addresses are unique per round/seed
- Verifies no duplicate addresses across multiple requests
- Ensures state persistence works correctly

### 6. High Volume (`test_locations_high_volume`)
- Tests allocation of many addresses (20 per seed)
- Verifies system handles larger requests

### 7. Rounds List (`test_rounds_list`)
- Tests `GET /rounds` endpoint
- Lists all tracked rounds

### 8. Round Detail (`test_round_detail`)
- Tests `GET /rounds/{round_id}` endpoint
- Shows seed usage for a specific round

### 9. Error Handling (`test_invalid_requests`)
- Tests with empty seed lists (should return 400)
- Tests with invalid country codes (should warn)
- Verifies proper error responses

## Test Output Example

```
================================================================================
ADDRESS ALLOCATION SERVICE API TESTER
Target: http://localhost:8000
================================================================================

[10:30:45] SUCCESS | ✓ PASS - Health Check
[10:30:45] INFO    |   Status: ok, Base: /work/MIID-subnet/54/addgen/addrs

[10:30:45] SUCCESS | ✓ PASS - Stats Endpoint
[10:30:45] INFO    |   Countries: 131, Total Addresses: 7763, Rounds: 0

[10:30:46] SUCCESS | ✓ PASS - Locations Basic
[10:30:46] INFO    |   Allocated for 3 seeds: US -> US: 2 addresses; UK -> GB: 2 addresses; Japan -> JP: 2 addresses

...

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 11
Passed: 11
Failed: 0
Success Rate: 100.0%
================================================================================
```

## API Endpoint Reference

### POST /locations
Allocate addresses for given seeds.

**Request:**
```json
{
  "round_id": "round_001",
  "seed_addresses": ["US", "Japan", "Germany"],
  "per_seed": 5
}
```

**Response:**
```json
{
  "round_id": "round_001",
  "results": [
    {
      "seed": "US",
      "country_code": "US",
      "allocated": ["123 Main St, New York...", ...],
      "exhausted": false,
      "warning": null
    }
  ],
  "warnings": []
}
```

### GET /stats
Get pool statistics.

**Response:**
```json
{
  "countries": {"US": 100, "JP": 220, ...},
  "total_addresses": 7763,
  "rounds": 5
}
```

### GET /rounds
List all tracked rounds.

**Response:**
```json
{
  "rounds": {"round_001": 3, "round_002": 5},
  "total_rounds": 2
}
```

### GET /rounds/{round_id}
Get details for a specific round.

**Response:**
```json
{
  "round_id": "round_001",
  "seeds": {"us": 5, "japan": 5},
  "total_seeds": 2,
  "total_used": 10
}
```

### POST /rounds/reset
Reset round allocations.

**Request (reset specific round):**
```json
{
  "round_id": "round_001"
}
```

**Request (reset all):**
```json
{}
```

## Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/stats

# Allocate addresses
curl -X POST http://localhost:8000/locations \
  -H "Content-Type: application/json" \
  -d '{
    "round_id": "test_001",
    "seed_addresses": ["US", "UK", "Japan"],
    "per_seed": 3
  }'

# List rounds
curl http://localhost:8000/rounds

# Get round details
curl http://localhost:8000/rounds/test_001
```

## Troubleshooting

### Service Not Running
If you get connection errors, make sure the service is running:
```bash
curl http://localhost:8000/health
```

If not, start it with:
```bash
cd /work/MIID-subnet/54
uvicorn service:app --host 0.0.0.0 --port 8000
```

### Pool Empty for Country
If you get warnings about empty pools, you may need to prefetch more addresses for that country.

### Tests Failing
1. Check service is running: `curl http://localhost:8000/health`
2. Check logs in the service console
3. Verify address files exist: `ls -la addgen/addrs/`
4. Check state directory is writable: `ls -la state/`

## Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed or service unavailable



