# file: tools/adaptive_expand_phonetic_buckets.py
#!/usr/bin/env python3
"""
Phonetic-aware variant expansion with single-pass BFS, per-layer caps, and strict timeout.

Buckets: vpool[ldiff][orth_level 0..3][phon_class P0..P7] -> TopK set
- ldiff bucket is |len(variant) - len(seed)|, clamped to [0..len_max].
- orth_level via similarity bounds; phon_class via (soundex, metaphone, nysiis) equality pattern.

Design:
- Fixed radius R = min(4, len(seed)-1).
- One BFS up to R; timeout checked on every neighbor gen and bucket insert.
- Per-layer caps to bound explosion for long names.
- Alphabet/edits tuned for Soundex+Metaphone+NYSIIS.

Requirements: jellyfish (mandatory). python-Levenshtein (optional).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Iterable
from tabulate import tabulate
import random
import os

CACHE_DIR = Path(os.getenv("NVGEN_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")))
CACHE_DIR.mkdir(exist_ok=True)

BUCKET_K = int(os.getenv("NVGEN_BUCKET_K", "500"))

try:
    import jellyfish  # soundex, metaphone, nysiis
except Exception as exc:
    raise ImportError("Install 'jellyfish' (pip install jellyfish)") from exc

try:
    import Levenshtein as _lev  # optional speed-up for orth_sim
except Exception:
    _lev = None


# ---------------- Similarity (orthographic)
def orth_sim(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    if _lev:
        d = _lev.distance(a, b)
        m = max(len(a), len(b)) or 1
        return 1.0 - (d / m)
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


ORTH_BOUNDS: Dict[int, Tuple[float, float]] = {
    0: (0.70, 1.00),
    1: (0.50, 0.69),
    2: (0.20, 0.49),
    3: (0.00, 0.19),
}

def orth_level(x: float) -> int:
    # Optimized with direct comparisons (faster than loop)
    if x >= 0.70:
        return 0
    elif 0.69 >= x >= 0.50:
        return 1
    elif 0.49 >= x >= 0.20:
        return 2
    elif 0.19 >= x >= 0.00:
        return 3
    return None # for some gaps 


# ---------------- Phonetic class (P0..P7 by equality pattern)
@dataclass(frozen=True)
class SeedCodes:
    sdx: str
    met: str
    nys: str

def seed_codes(name: str) -> SeedCodes:
    return SeedCodes(
        sdx=jellyfish.soundex(name),
        met=jellyfish.metaphone(name),
        nys=jellyfish.nysiis(name),
    )

def phon_class(seed: SeedCodes, v: str) -> int:
    # Calculate phonetic codes directly with optimized bit packing
    se = jellyfish.soundex(v)   == seed.sdx
    me = jellyfish.metaphone(v) == seed.met
    ne = jellyfish.nysiis(v)    == seed.nys
    
    # Direct bit packing without dictionary lookup
    idx = (4 if se else 0) + (2 if me else 0) + (1 if ne else 0)
    
    # Direct mapping array (faster than dict lookup)
    mapping = [7, 6, 5, 3, 4, 2, 1, 0]  # [0,1,2,3,4,5,6,7] -> [7,6,5,3,4,2,1,0]
    return mapping[idx]


# ---------------- Phonetic-aware alphabet
VOWELS = tuple("aeiou")
YHW = tuple("yhw")
SDX_GROUPS: List[Tuple[str, ...]] = [
    tuple("bfpv"),            # 1
    tuple("cgjkqsxz"),        # 2
    tuple("dt"),              # 3
    tuple("l"),               # 4
    tuple("mn"),              # 5
    tuple("r"),               # 6
]
SDX_MAP: Dict[str, int] = {c: i for i, grp in enumerate(SDX_GROUPS) for c in grp}

def sdx_group(c: str) -> Tuple[str, ...] | tuple():
    i = SDX_MAP.get(c)
    return SDX_GROUPS[i] if i is not None else tuple()


def make_replacement_map(base_alpha: str) -> Dict[str, List[str]]:
    """
    Replacement candidates per char:
      - Vowels -> other vowels + y/h/w
      - Consonants -> same Soundex group + y/h/w
      - Extras affecting Metaphone/NYSIIS: c<->k<->q<->s/x, g<->j, t<->d
    """
    alpha = list(dict.fromkeys(base_alpha.lower()))
    repl: Dict[str, List[str]] = {}
    extra_pairs = {
        "c": ("k","q","s","x"),
        "k": ("c","q"),
        "q": ("k","c"),
        "g": ("j",),
        "j": ("g",),
        "s": ("x","c"),
        "x": ("s",),
        "t": ("d",),
        "d": ("t",),
    }
    for ch in alpha:
        opts: List[str] = []
        if ch in VOWELS:
            opts = [v for v in VOWELS if v != ch] + list(YHW)
        else:
            opts = [c for c in sdx_group(ch) if c != ch] + list(YHW) + [e for e in extra_pairs.get(ch, ()) if e != ch]
        # dedup preserve order
        seen = set(); seq: List[str] = []
        for o in opts:
            if o not in seen:
                seen.add(o); seq.append(o)
        repl[ch] = seq
    return repl

def contextual_variants(w: str) -> Iterable[str]:
    n = len(w)
    lower = w
    # start-boundary transforms (Metaphone/NYSIIS)
    if lower.startswith(("kn", "gn", "pn")):
        yield lower[1:]                  # "knight" -> "night"
    if lower.startswith("wr"):
        yield "r" + lower[2:]            # "write" -> "rite"
    if lower.startswith("x"):
        yield "s" + lower[1:]            # "xeno" -> "seno"

    # PH <-> F (Metaphone; NYSIIS PH->FF)
    for i in range(n - 1):
        if lower[i:i+2] == "ph":
            yield lower[:i] + "f" + lower[i+2:]
    for i in range(n):
        if lower[i] == "f":
            yield lower[:i] + "ph" + lower[i+1:]

    # SCH -> SS/SH (NYSIIS)
    for i in range(n - 2):
        if lower[i:i+3] == "sch":
            yield lower[:i] + "ss" + lower[i+3:]
            yield lower[:i] + "sh" + lower[i+3:]

    # CK -> K; C->S before I/E/Y; G->J before I/E/Y (Metaphone)
    for i in range(n - 1):
        if lower[i:i+2] == "ck":
            yield lower[:i] + "k" + lower[i+2:]
    for i in range(n - 1):
        c, nxt = lower[i], lower[i+1]
        if c == "c" and nxt in "iey":
            yield lower[:i] + "s" + lower[i+1:]
        if c == "g" and nxt in "iey":
            yield lower[:i] + "j" + lower[i+1:]

    # X -> KS (Metaphone), S -> X (rarely, but explore)
    for i in range(n):
        if lower[i] == "x":
            yield lower[:i] + "ks" + lower[i+1:]
        if lower[i] == "s":
            yield lower[:i] + "x" + lower[i+1:]

    # Endings per NYSIIS
    if lower.endswith(("ee", "ie")):
        yield lower[:-2] + "y"
    for suf in ("dt", "rt", "rd", "nt", "nd"):
        if lower.endswith(suf):
            yield lower[:-2] + "d"

    # MAC -> MCC (NYSIIS)
    if lower.startswith("mac"):
        yield "mcc" + lower[3:]

# ---------------- Phonetic-aware neighboring (timeout & caps)
def neighbors_once_phonetic(
    w: str,
    repl_map: Dict[str, List[str]],
) -> Iterator[str]:
    n = len(w)
    # insertions near vowels + y/h/w
    for i in range(n + 1):
        left = w[i-1] if i-1 >= 0 else ""
        right = w[i] if i < n else ""
        local: List[str] = []
        if left in VOWELS or right in VOWELS:
            local.extend(VOWELS)
        local.extend(YHW)
        cnt = 0
        for ch in local:
            yield w[:i] + ch + w[i:]
            cnt += 1

    # deletions of y/h/w
    for i in range(n):
        if w[i] in YHW:
            yield w[:i] + w[i+1:]

    # replacements
    for i in range(n):
        ci = w[i]
        cand = repl_map.get(ci, ())
        if not cand:
            continue
        wi, wj = w[:i], w[i+1:]
        cnt = 0
        for ch in cand:
            if ch != ci:
                yield wi + ch + wj
                cnt += 1
    # contextual (digraph + boundary) edits
    for v in contextual_variants(w):
        yield v

    # limited transpositions: if either is vowel
    for i in range(n-1):
        a, b = w[i], w[i+1]
        if a != b and (a in VOWELS or b in VOWELS):
            yield w[:i] + b + a + w[i+2:]

    # duplication (stutter)
    for i in range(1, n):
        yield w[:i] + w[i-1] + w[i:]
    
    return
    # common multi-char transforms affecting metaphone/nysiis (bounded)
    # ph -> f
    idx = w.find("ph")
    if idx != -1:
        yield w[:idx] + "f" + w[idx+2:]
    # insert 'h' after 'p' not already 'ph'
    for i in range(n-1):
        if w[i] == "p" and w[i+1] != "h":
            yield w[:i+1] + "h" + w[i+1:]
            break  # only once per word
    # initial rules
    if w.startswith(("kn","gn","pn","wr")):
        yield w[1:]
    if w.startswith("wh"):
        yield "w" + w[2:]
    if w.startswith("ae"):
        yield w[1:]
    if w.startswith("x"):
        yield "s" + w[1:]
    # nysiis: mac->mcc, sch->sss
    if w.startswith("mac"):
        yield "mcc" + w[3:]
    if w.startswith("sch"):
        yield "sss" + w[3:]
    # suffix group -> d
    for suf in ("dt","rt","rd","nt","nd"):
        if w.endswith(suf):
            yield w[:-2] + "d"
            break


# ---------------- Top-K buckets
class TopK:
    __slots__ = ("k", "heap", "seen")
    def __init__(self, k: int) -> None:
        self.k = k
        self.heap: List[Tuple[float, str]] = []
        self.seen: Set[str] = set()
    def add(self, v: str, score: float) -> None:
        if v in self.seen:
            return
        neg = -score
        if len(self.heap) < self.k:
            heappush(self.heap, (neg, v)); self.seen.add(v); return
        if neg > self.heap[0][0]:
            return
        _, worst = heappop(self.heap); self.seen.remove(worst)
        heappush(self.heap, (neg, v)); self.seen.add(v)
    def to_set(self) -> Set[str]:
        return set(self.seen)


# ---------------- Single-pass BFS up to fixed R (timeout & layer caps)
def bfs_layers(
    seed: str,
    R: int,
    repl_map: Dict[str, List[str]],
    timeout_at: Optional[float],
) -> List[Set[str]]:
    """
    Returns exact-distance layers [1..R]. Enforces:
      - timeout checks on every yield/add.
    """
    if R <= 0:
        return []
    # Remove all special characters from seed to normalize for BFS
    import re
    seed_ = re.sub(r'[^a-zA-Z]', 'a', seed)
    seen: Set[str] = {seed_}
    layers: List[Set[str]] = [set()] * (R + 1)  # index 0 unused
    prev = {seed_}
    seed_ph = seed_codes(seed)
    max_ld = 2
    vpool = [[[set() for p in range(8)] for o in range(4)] for ld in range(max_ld + 1)]

    for depth in range(1, R + 1):
        cur: Set[str] = set()
        # print(f"Seed: {seed}, BFS depth {depth}, Time: {time.time() - start_time}")
        
        for w in prev:
            count_parent = 0
            for v in neighbors_once_phonetic(w, repl_map):
                if timeout_at and time.time() > timeout_at:
                    print(f"seed: {seed}, Timeout at depth {depth}, time: {time.time()}")
                    return vpool, layers[:depth]
                if v in seen:
                    continue
                ld = abs(len(v) - len(seed))
                if ld > max_ld:
                    continue
                o = orth_level(orth_sim(seed, v))
                p = phon_class(seed_ph, v)
                if o is None or len(vpool[ld][o][p]) == BUCKET_K:
                    seen.add(v)
                    continue
                vpool[ld][o][p].add(v)
                seen.add(v)
                cur.add(v)
        layers[depth] = cur
        if not cur:
            return vpool, layers[:depth]
        prev = cur
    return vpool, layers


def get_cache_file(name: str) -> Path:
    """Get cache file path for a name"""
    return CACHE_DIR / f"{name}.txt"


def load_pool_from_cache(name: str) -> Optional[Dict]:
    """Load pool data from cache file"""
    cache_file = get_cache_file(name)
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading cache for {name}: {e}")
        return None


def save_pool_to_cache(name: str, pool: Dict, stats: Dict):
    """Save pool data to cache file, appending to existing data if present"""
    cache_file = get_cache_file(name)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                "name": name,
                "pool": pool,
                "stats": stats
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving cache for {name}: {e}")


# ---------------- Expansion into buckets (direct push; strict timeout)
def expand_into_buckets(
    name: str,
    *,
    alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    bucket_k: int = BUCKET_K,
    timeout_seconds: float = 300.0,
) -> Tuple[Dict[int, Dict[int, Dict[int, Set[str]]]], Dict[str, int]]:
    """
    Fixed R = min(4, len(name)-1). Buckets vpool[ldiff][orth][phonP].
    """
    cache_data = load_pool_from_cache(name)
    if cache_data:
        if "pool" not in cache_data or "stats" not in cache_data:
            return None, None
        return cache_data["pool"], cache_data["stats"]
    
    start = time.time()
    timeout_at = start + timeout_seconds if timeout_seconds else None

    seed = name.lower()
    R = len(name) * 2
    repl_map = make_replacement_map(alphabet)
    seed_ph = seed_codes(seed)

    # allocate buckets for ld in [0..len_max]
    vpool, layers = bfs_layers(seed, R, repl_map, timeout_at)
    elapsed_sec = time.time() - start

    # Generate additional layer with same-length variations using only different characters
    used_chars = set(seed.lower())
    available_chars = [c for c in alphabet if c not in used_chars]
    if available_chars:  # Only generate if we have available characters
        max_try = bucket_k * 10
        try_count = 0
        
        while len(vpool[0][3][7]) < bucket_k and try_count < max_try:
            # Generate a variation of the same length using only different characters
            variation = ''.join(random.choice(available_chars) for _ in range(len(seed)))
            o = orth_level(orth_sim(seed, variation))
            p = phon_class(seed_ph, variation)
            vpool[0][o][p].add(variation)
            try_count += 1
    
    # materialize - ensure we always return at least one layer structure
    out = [[[sorted(vpool[ld][o][p]) for p in range(8)] for o in range(4)] for ld in range(len(vpool))]
    stats = {
        "timeout": timeout_seconds,
        "elapsed_sec": f"{round(elapsed_sec, 2)}s",
    }
    save_pool_to_cache(name, out, stats)
    return out, stats

# ---------------- CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phonetic-aware expansion into vpool[ld][orth][phonP] with timeout-safe BFS.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--name", help="Single name.")
    src.add_argument("--names", help="Comma-separated names.")
    src.add_argument("--file", type=Path, help="File with one name per line.")
    ap.add_argument("--alphabet", default="abcdefghijklmnopqrstuvwxyz", help="Edit alphabet (default: a-z).")
    ap.add_argument("--bucket-k", type=int, default=BUCKET_K)
    ap.add_argument("--timeout-seconds", type=float, default=300.0)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out", type=Path)
    return ap.parse_args()

def load_names(args: argparse.Namespace) -> List[str]:
    if args.name:
        return [args.name]
    if args.names:
        return [n.strip() for n in args.names.split(",") if n.strip()]
    if args.file:
        return [ln.strip() for ln in args.file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []

def main() -> None:
    args = parse_args()
    names = load_names(args)
    results: List[Dict[str, object]] = []
    for nm in names:
        pool, stats = expand_into_buckets(
            name=nm,
            # alphabet=args.alphabet,
            # bucket_k=args.bucket_k,
            # timeout_seconds=args.timeout_seconds,
        )
        payload = {
            "name": nm,
            "stats": stats,
            "pool": pool
        }
        results.append(payload)

        if args.summary:
            print(f"== {nm} ==")
            print(f"stats: {stats}")
            
            # Check if pool has data
            if not pool:
                print("No variations generated (empty pool)")
            else:
                for ld in range(len(pool)):
                    print(f"\nname={nm}, level={ld}")
                    # Data rows
                    table_data = []
                    for orth in range(4):  # Fixed 4 orthographic levels
                        row = [f"O{orth}"]
                        
                        for phon in range(8):  # Fixed 8 phonetic levels
                            count = 0
                            try:
                                if ld < len(pool) and orth < len(pool[ld]) and phon < len(pool[ld][orth]):
                                    count = len(pool[ld][orth][phon])
                            except (IndexError, TypeError) as e:
                                count = 0
                            row.append(str(count) if count > 0 else "")
                        
                        table_data.append(row)
                    
                    # Print table using tabulate
                    headers = ["Orth", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
                    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))


    final = results if len(results) > 1 else results[0]
    if args.json_out:
        args.json_out.write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(f"Wrote JSON to {args.json_out}")
    elif args.json or not args.summary:
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
