#!/usr/bin/env python3
# test_addr_service.py
# Functional test for addr_service.py API

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


def jprint(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


@dataclass
class ApiResult:
    ok: bool
    data: Optional[dict] = None
    status: Optional[int] = None
    error: Optional[str] = None


class AddrApi:
    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.sess = requests.Session()
        self.sess.headers.update({"Content-Type": "application/json"})

    def _get(self, path: str) -> ApiResult:
        try:
            r = self.sess.get(self.base + path, timeout=self.timeout)
            return ApiResult(ok=r.ok, data=r.json() if r.content else None, status=r.status_code,
                             error=None if r.ok else r.text)
        except Exception as e:
            return ApiResult(ok=False, error=str(e))

    def _post(self, path: str, payload: dict) -> ApiResult:
        try:
            r = self.sess.post(self.base + path, data=json.dumps(payload), timeout=self.timeout)
            return ApiResult(ok=r.ok, data=r.json() if r.content else None, status=r.status_code,
                             error=None if r.ok else r.text)
        except Exception as e:
            return ApiResult(ok=False, error=str(e))

    # endpoints
    def health(self):           return self._get("/health")
    def stats(self):            return self._get("/stats")
    def reload(self):           return self._post("/reload", {})
    def locations(self, round_id: str, seeds: List[str], per_seed: int):
        return self._post("/locations", {"round_id": round_id, "seed_addresses": seeds, "per_seed": per_seed})
    def list_rounds(self):      return self._get("/rounds")
    def round_detail(self, rid):return self._get(f"/rounds/{rid}")
    def round_reset(self, rid: Optional[str] = None):
        return self._post("/rounds/reset", {"round_id": rid} if rid else {})


def assert_true(cond: bool, msg: str):
    if not cond:
        print(f"❌ {msg}")
        sys.exit(2)
    print(f"✅ {msg}")


def main():
    parser = argparse.ArgumentParser(description="Test the Address Allocation Service")
    parser.add_argument("--base-url", default="http://localhost:9999", help="Service base URL")
    parser.add_argument("--seeds", default="Japan,US", help="Comma-separated list of seed strings")
    parser.add_argument("--per-seed", type=int, default=1, help="Addresses per seed per call")
    parser.add_argument("--iterations", type=int, default=3, help="How many repeated /locations calls")
    parser.add_argument("--round-id", default=None, help="Round id to use (default auto)")
    parser.add_argument("--stress-exhaust", action="store_true",
                        help="Also try to exhaust a seed's pool (requests many)")
    parser.add_argument("--verbose", action="store_true", help="Print API responses")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    assert_true(len(seeds) > 0, "Parsed at least one seed from --seeds")

    round_id = args.round_id or f"t-{int(time.time())}"
    print(f"▶ Using base_url={args.base_url} round_id={round_id} seeds={seeds} per_seed={args.per_seed} iterations={args.iterations}")

    api = AddrApi(args.base_url)

    # 1) health
    res = api.health()
    assert_true(res.ok, "/health is OK")
    if args.verbose: jprint(res.data)

    # 2) stats (before)
    res = api.stats()
    assert_true(res.ok, "/stats is OK")
    if args.verbose: jprint(res.data)
    before_total = res.data.get("total_addresses", 0)

    # 3) reload (should keep working even if nothing changes)
    res = api.reload()
    assert_true(res.ok, "/reload succeeded")
    if args.verbose: jprint(res.data)

    # 4) Repeated /locations calls: ensure NO duplicates for (round_id, seed)
    seen: Dict[str, set] = {s.lower(): set() for s in seeds}
    duplicates_found = False

    for i in range(args.iterations):
        res = api.locations(round_id, seeds, args.per_seed)
        assert_true(res.ok, f"/locations call {i+1}/{args.iterations} is OK")
        if args.verbose: jprint(res.data)

        payload = res.data or {}
        results = payload.get("results", [])
        # Verify structure
        assert_true(len(results) == len(seeds), "Got one result per seed")

        for item in results:
            seed = item["seed"]
            alloc = item.get("allocated", []) or []
            exhausted = item.get("exhausted", False)

            # track duplicates across calls for SAME seed
            skey = seed.lower()
            for addr in alloc:
                print(f"addr: {addr}")
                if addr in seen[skey]:
                    print(f"❌ Duplicate detected for seed={seed!r} address={addr!r} on iteration {i+1}")
                    duplicates_found = True
                seen[skey].add(addr)

            if exhausted:
                # Not a hard failure (pool may be small), but make noise
                print(f"⚠️  Exhausted for seed={seed!r} on iteration {i+1}")

    assert_true(not duplicates_found, "No duplicate addresses returned for the same (round_id, seed)")

    # # 5) Rounds list & detail
    # res = api.list_rounds()
    # assert_true(res.ok, "/rounds list OK")
    # rounds = res.data.get("rounds", {})
    # assert_true(round_id in rounds, "Our round_id is present in /rounds listing")

    # res = api.round_detail(round_id)
    # assert_true(res.ok, f"/rounds/{round_id} detail OK")
    # if args.verbose: jprint(res.data)
    # detail_used_total = res.data.get("total_used", 0)
    # assert_true(detail_used_total >= 0, "Round detail shows non-negative used count")

    # # 6) Optional stress: try to exhaust a single seed by asking a lot
    # if args.stress_exhaust:
    #     stress_seed = seeds[0]
    #     print(f"▶ Stressing seed={stress_seed!r} to try exhaustion")
    #     res = api.locations(round_id, [stress_seed], per_seed=10_000)
    #     assert_true(res.ok, "Stress /locations call OK")
    #     if args.verbose: jprint(res.data)
    #     ritem = res.data["results"][0]
    #     if ritem["exhausted"]:
    #         print("⚠️  Exhaustion reported as expected for stress case")
    #     else:
    #         print("ℹ️  Stress call did not exhaust the pool (that’s fine if you have a huge pool)")

    # # 7) Reset the round and verify we can allocate again for same seeds
    # res = api.round_reset(round_id)
    # assert_true(res.ok, f"/rounds/reset round_id={round_id} OK")

    # # After reset, allocations should start fresh (no used state for this round)
    # res = api.locations(round_id, seeds, args.per_seed)
    # assert_true(res.ok, "Post-reset /locations is OK")
    # if args.verbose: jprint(res.data)
    # print("✅ Round reset re-enabled allocations")

    # # 8) Final stats
    # res = api.stats()
    # assert_true(res.ok, "Final /stats OK")
    # after_total = res.data.get("total_addresses", 0)
    # print(f"▶ Pool total addresses (before/after reload): {before_total} / {after_total}")

    print("\n🎉 ALL TESTS PASSED")


if __name__ == "__main__":
    main()
