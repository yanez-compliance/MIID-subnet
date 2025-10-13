import re
import json
import logging
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
import numpy as np


_LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",  # was l
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
    "#": "h",
    "%": "o",
    "^": "",
    "&": "and",
    "*": "",
    ")": "",
    "(": "c", # was ""
    "-": "",
    "_": "",
    "+": "t", # was ""
    "=": "",
    "{": "",
    "}": "",
    "[": "",
    "]": "",
    "|": "i", # was ""
    ":": "",
    ";": "",
    "'": "",
    '"': "",
    ",": "",
    ".": "",
    "/": "",
    "\\": "",
    "?": "",
    ">": "",
    "<": "",
    # Additional mappings
    "ä": "a",
    "€": "e",
    "ë": "e",
    "°": "o",
    "ö": "o",
    "§": "s",
    "¢": "c",
    "k": "c",
})


def normalize_variation(text: str, aggressive: bool = True) -> str:
    """Return a canonical form of a variation for robust equality/overlap checks.

    - Lowercase
    - Strip spaces
    - Optionally apply leetspeak mapping and remove punctuation/digits
    """
    if text is None:
        return ""
    normalized = text.strip().lower()
    # Collapse whitespace
    normalized = re.sub(r"\s+", "", normalized)
    if aggressive:
        normalized = normalized.translate(_LEET_MAP)
        # Remove any lingering non-letters
        normalized = re.sub(r"[^a-z]", "", normalized)
    return normalized


def build_normalized_set(variations: List[str]) -> Set[str]:
    return {normalize_variation(v) for v in variations if v}


def overlap_coefficient(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    return inter / float(min(len(a), len(b)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / float(union) if union else 1.0


def hash_signature(variations_by_name: Dict[str, List[str]]) -> str:
    """Stable signature for an entire response to detect exact copies across miners.
    Canonicalizes and sorts per-name sets before hashing.
    """
    parts: List[str] = []
    for name in sorted(variations_by_name.keys()):
        canon = sorted(build_normalized_set(variations_by_name[name]))
        parts.append(name.lower())
        parts.extend(canon)
    # Simple deterministic hash: djb2
    h = 5381
    for ch in "|".join(parts):
        h = ((h << 5) + h) + ord(ch)
        h &= 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def _try_json_load(text: str) -> Dict[str, List[str]]:
    try:
        obj = json.loads(text)
        # The cheating files may have varying shapes; extract name->list recursively
        result: Dict[str, List[str]] = {}
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        result[k] = v
                    else:
                        walk(v)
            elif isinstance(o, list):
                for item in o:
                    walk(item)
        walk(obj)
        if result:
            return result
    except Exception:
        pass
    return {}


_PAIR_RE = re.compile(r"\s*\"(?P<name>[^\"]+)\"\s*:\s*\[(?P<body>.*?)\]", re.DOTALL)


def _tolerant_extract_name_lists(text: str) -> Dict[str, List[str]]:
    """Extract name -> list-of-strings from malformed JSON-like files.
    Looks for "name": [ "a", "b", ... ] patterns.
    """
    result: Dict[str, List[str]] = {}
    for m in _PAIR_RE.finditer(text):
        name = m.group("name").strip()
        body = m.group("body")
        items = re.findall(r"\"([^\"]+)\"", body)
        if items:
            result[name] = items
    return result


def load_cheat_corpus(paths: List[str]) -> Dict[str, Set[str]]:
    """Load cheating corpora from a list of file paths.
    Returns a mapping name -> canonical set of known-cheat variations.
    Robust to malformed JSON.
    """
    combined: Dict[str, Set[str]] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        data = _try_json_load(text)
        if not data:
            data = _tolerant_extract_name_lists(text)
        for name, vars_list in data.items():
            canon = {normalize_variation(v) for v in vars_list}
            if not canon:
                continue
            combined.setdefault(name.lower(), set()).update(canon)
    return combined


def corpus_overlap_score(
    miner_variations_by_name: Dict[str, List[str]],
    cheat_corpus: Dict[str, Set[str]]
) -> float:
    """Compute the average overlap ratio with the cheat corpus across provided names.
    Overlap for a name = |normalized ∩ corpus[name]| / max(1, |normalized|)
    Returns 0 if none of the names exist in corpus.
    """
    scores: List[float] = []
    for name, vars_list in miner_variations_by_name.items():
        canon = build_normalized_set(vars_list)
        corpus_set = cheat_corpus.get(name.lower())
        if not corpus_set or not canon:
            continue
        inter = len(canon.intersection(corpus_set))
        scores.append(inter / float(len(canon)))
    return sum(scores) / len(scores) if scores else 0.0


def pairwise_similarity_metrics(
    normalized_sets_by_miner: List[Dict[str, Set[str]]]
) -> List[Tuple[float, float]]:
    """
    Return list of (max_avg_overlap, max_avg_jaccard) per miner against others.
    Similarity is calculated as the average of per-name metrics.
    """
    n = len(normalized_sets_by_miner)
    results: List[Tuple[float, float]] = [(0.0, 0.0) for _ in range(n)]

    for i in range(n):
        max_avg_overlap = 0.0
        max_avg_jaccard = 0.0
        miner_i_sets = normalized_sets_by_miner[i]
        
        if not miner_i_sets:
            continue

        for j in range(n):
            if i == j:
                continue
            
            miner_j_sets = normalized_sets_by_miner[j]
            if not miner_j_sets:
                continue

            overlap_scores = []
            jaccard_scores = []
            
            # Compare sets for each name present in both miners
            common_names = miner_i_sets.keys() & miner_j_sets.keys()
            if not common_names:
                continue

            for name in common_names:
                set_i = miner_i_sets[name]
                set_j = miner_j_sets[name]
                overlap_scores.append(overlap_coefficient(set_i, set_j))
                jaccard_scores.append(jaccard(set_i, set_j))

            avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
            avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

            if avg_overlap > max_avg_overlap:
                max_avg_overlap = avg_overlap
            if avg_jaccard > max_avg_jaccard:
                max_avg_jaccard = avg_jaccard
        
        results[i] = (max_avg_overlap, max_avg_jaccard)
        
    return results


def detect_cheating_patterns(
    responses: List[Dict[str, List[List[str]]]],  # List of dictionaries with variations
    uids: List[int],
    rewards: np.ndarray,
    seed_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Analyzes miner responses to detect cheating patterns like collusion, copying, and excessive special characters.
    Now handles the new format where each variation is [name_var, dob_var, address_var].

    Returns a dictionary of numpy arrays, each with length equal to number of miners:
    - duplication_penalties
    - signature_penalties
    - collusion_penalties
    - special_char_penalties
    - address_duplication_penalties
    - special_char_counts
    - total_variations_counts
    - special_char_ratios
    """
    num_miners = len(responses)
    duplication_penalties = np.zeros(num_miners)
    special_char_penalties = np.zeros(num_miners)
    signature_penalties = np.zeros(num_miners)
    collusion_penalties = np.zeros(num_miners)
    address_duplication_penalties = np.zeros(num_miners)
    special_char_counts = np.zeros(num_miners, dtype=int)
    total_variations_counts = np.zeros(num_miners, dtype=int)
    special_char_ratios = np.zeros(num_miners)
    all_normalized_sets = []
    all_address_sets = []  # New: track address variations for cross-miner comparison
    all_signatures = []

    for i in range(num_miners):
        variations = responses[i]  # Each response is already a dictionary

        if variations:
            special_char_variations_count = 0
            total_variations_count = 0
            all_addresses = []
            
            for name in variations:
                name_variations = variations.get(name, [])
                if name_variations and len(name_variations) > 0:
                    # name_variations is a list of [name_var, dob_var, address_var] arrays
                    # We want the name variations (index 0 of each array)
                    name_vars = [var[0] for var in name_variations if len(var) > 0 and var[0]]
                    total_variations_count += len(name_vars)
                    for var in name_vars:
                        # Count as special char only if it has excessive non-alphanumeric characters
                        # Allow common punctuation like periods, hyphens, apostrophes
                        special_chars = sum(1 for c in var if not c.isalnum() and not c.isspace() and c not in ".-'")
                        if special_chars > 2:  # Only penalize if more than 2 special chars
                            special_char_variations_count += 1
                    
                    # Extract address variations (index 2 of each [name_var, dob_var, address_var] array)
                    address_vars = [var[2] for var in name_variations if len(var) > 2 and var[2]]
                    all_addresses.extend([
                        re.sub(r'\d+', '', addr).strip().replace(" ", "").replace(",", "").lower()
                        for addr in address_vars if addr and addr.strip()
                    ])

            if total_variations_count > 0:
                special_char_ratio = special_char_variations_count / total_variations_count
                special_char_counts[i] = special_char_variations_count
                total_variations_counts[i] = total_variations_count
                special_char_ratios[i] = special_char_ratio
                if special_char_ratio > 0.5:
                    penalty = (special_char_ratio - 0.5) / 0.5
                    special_char_penalties[i] = min(penalty, 1.0)
            
            # Count duplicates within this miner's addresses
            if all_addresses:
                unique_addresses = set(all_addresses)
                duplicate_count = len(all_addresses) - len(unique_addresses)
                if duplicate_count > 0:
                    # Apply penalty based on duplicate ratio, normalized and capped at 0.2
                    duplicate_ratio = duplicate_count / len(all_addresses)
                    # Normalize: scale the ratio to 0-1 range, then scale to 0-0.2 range
                    normalized_penalty = duplicate_ratio * 0.2
                    address_duplication_penalties[i] = min(normalized_penalty, 0.2)
        
        if not variations:
            all_normalized_sets.append(None)
            all_address_sets.append(None)
            all_signatures.append(None)
            continue

        miner_normalized_sets: Dict[str, Set[str]] = {}
        miner_address_sets: Dict[str, Set[str]] = {}  # Track address variations by name
        miner_map_for_signature: Dict[str, list] = {}
        has_any_variations = False
        
        for name in seed_names:
            if name in variations and variations[name]:
                has_any_variations = True
                name_variations = variations.get(name, [])
                # Extract name variations (index 0 of each [name_var, dob_var, address_var] array)
                canon_list = [var[0] for var in name_variations if len(var) > 0 and var[0]]
                miner_map_for_signature[name] = canon_list
                miner_normalized_sets[name] = build_normalized_set(canon_list)
                
                # Extract address variations (index 2 of each [name_var, dob_var, address_var] array)
                address_list = [var[2] for var in name_variations if len(var) > 2 and var[2]]
                # Normalize addresses for comparison (remove spaces, commas, convert to lowercase)
                normalized_addresses = [
                    addr.strip().replace(" ", "").replace(",", "").lower()
                    for addr in address_list if addr and addr.strip()
                ]
                miner_address_sets[name] = set(normalized_addresses)

        if not has_any_variations:
            all_normalized_sets.append(None)
            all_address_sets.append(None)
            all_signatures.append(None)
            continue
        
        all_normalized_sets.append(miner_normalized_sets)
        all_address_sets.append(miner_address_sets)
        all_signatures.append(hash_signature(miner_map_for_signature))

    fmt15 = [f"{r:.15f}" for r in rewards]
    buckets_exact: Dict[str, List[int]] = {}
    for idx, key in enumerate(fmt15):
        if rewards[idx] > 0.0:
            buckets_exact.setdefault(key, []).append(idx)

    buckets_near: Dict[int, List[int]] = {}
    for idx, r in enumerate(rewards):
        if r > 0.0:
            key = int(round(r * 10000))
            buckets_near.setdefault(key, []).append(idx)

    COLLUSION_GROUP_SIZE_THRESHOLD = 5
    for bucket_indices in buckets_exact.values():
        if len(bucket_indices) > COLLUSION_GROUP_SIZE_THRESHOLD:
            if rewards[bucket_indices[0]] < 0.95:
                logging.warning(f"Large collusion group detected with {len(bucket_indices)} members with score < 0.95. Applying direct penalty.")
                penalty_value = 0.75
                for i in bucket_indices:
                    collusion_penalties[i] = max(collusion_penalties[i], penalty_value)

    def penalize_group(indices: List[int], strict: bool) -> None:
        if len(indices) < 2:
            return
        
        valid_indices = [i for i in indices if all_normalized_sets[i] is not None]
        if len(valid_indices) < 2:
            return

        group_sets = [all_normalized_sets[i] for i in valid_indices]
        metrics_local = pairwise_similarity_metrics(group_sets)
        
        for k, (max_avg_overlap, max_avg_jaccard) in enumerate(metrics_local):
            i = valid_indices[k]
            if strict:
                thr_overlap, thr_jacc = 0.75, 0.70
            else:
                thr_overlap, thr_jacc = 0.80, 0.70
            
            if max_avg_overlap > thr_overlap or max_avg_jaccard > thr_jacc:
                overlap_pen = max(0.0, (max_avg_overlap - thr_overlap) / max(1e-6, 1.0 - thr_overlap))
                jaccard_pen = max(0.0, (max_avg_jaccard - thr_jacc) / max(1e-6, 1.0 - thr_jacc))
                penalty = min(1.0, max(overlap_pen, jaccard_pen))
                duplication_penalties[i] = max(duplication_penalties[i], penalty)

    sig_to_indices: Dict[str, List[int]] = {}
    for idx, sig in enumerate(all_signatures):
        if not sig:
            continue
        sig_to_indices.setdefault(sig, []).append(idx)
    for indices in sig_to_indices.values():
        if len(indices) > 1:
            # Only penalize miners with a reward greater than 0
            valid_indices = [idx for idx in indices if rewards[idx] > 0]
            if len(valid_indices) > 1:
                for idx in valid_indices:
                    signature_penalties[idx] = max(signature_penalties[idx], 0.8)

    for idxs in buckets_exact.values():
        penalize_group(idxs, strict=True)
    for idxs in buckets_near.values():
        penalize_group(idxs, strict=False)

    logging.info("Performing cross-group similarity check on all valid miners.")
    valid_indices = [i for i, s in enumerate(all_normalized_sets) if isinstance(s, dict) and s]
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            idx1 = valid_indices[i]
            idx2 = valid_indices[j]
            set1 = all_normalized_sets[idx1]
            set2 = all_normalized_sets[idx2]

            common_names = set1.keys() & set2.keys()
            if not common_names:
                overlap = 0.0
                jac = 0.0
            else:
                overlap_scores = [overlap_coefficient(set1[name], set2[name]) for name in common_names]
                jaccard_scores = [jaccard(set1[name], set2[name]) for name in common_names]
                overlap = sum(overlap_scores) / len(overlap_scores)
                jac = sum(jaccard_scores) / len(jaccard_scores)

            if overlap > 0.95 or jac > 0.90:
                penalty = 0.5
                duplication_penalties[idx1] = max(duplication_penalties[idx1], penalty)
                duplication_penalties[idx2] = max(duplication_penalties[idx2], penalty)

    # Check for name similarity between miners (cross-miner name duplication)
    logging.info("Performing cross-miner name similarity check.")
    valid_indices = [i for i, s in enumerate(all_normalized_sets) if isinstance(s, dict) and s]
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            idx1 = valid_indices[i]
            idx2 = valid_indices[j]
            set1 = all_normalized_sets[idx1]
            set2 = all_normalized_sets[idx2]

            common_names = set1.keys() & set2.keys()
            if not common_names:
                overlap = 0.0
                jac = 0.0
            else:
                overlap_scores = [overlap_coefficient(set1[name], set2[name]) for name in common_names]
                jaccard_scores = [jaccard(set1[name], set2[name]) for name in common_names]
                overlap = sum(overlap_scores) / len(overlap_scores)
                jac = sum(jaccard_scores) / len(jaccard_scores)

            if overlap > 0.95 or jac > 0.90:
                penalty = 0.5
                duplication_penalties[idx1] = max(duplication_penalties[idx1], penalty)
                duplication_penalties[idx2] = max(duplication_penalties[idx2], penalty)

    # Check for address similarity between miners
    logging.info("Performing cross-miner address similarity check.")
    valid_address_indices = [i for i, s in enumerate(all_address_sets) if isinstance(s, dict) and s]
    for i in range(len(valid_address_indices)):
        for j in range(i + 1, len(valid_address_indices)):
            idx1 = valid_address_indices[i]
            idx2 = valid_address_indices[j]
            addr_set1 = all_address_sets[idx1]
            addr_set2 = all_address_sets[idx2]

            common_names = addr_set1.keys() & addr_set2.keys()
            if not common_names:
                continue

            # Check address similarity for each common name
            for name in common_names:
                addresses1 = addr_set1[name]
                addresses2 = addr_set2[name]
                
                if not addresses1 or not addresses2:
                    continue
                
                # Calculate overlap and jaccard for addresses
                overlap = overlap_coefficient(addresses1, addresses2)
                jac = jaccard(addresses1, addresses2)
                
                # If addresses are too similar, apply penalty
                if overlap > 0.8 or jac > 0.7:
                    penalty = min(0.6, max(overlap, jac) * 0.8)  # Scale penalty based on similarity
                    address_duplication_penalties[idx1] = max(address_duplication_penalties[idx1], penalty)
                    address_duplication_penalties[idx2] = max(address_duplication_penalties[idx2], penalty)

    return {
        "duplication_penalties": duplication_penalties,
        "signature_penalties": signature_penalties,
        "collusion_penalties": collusion_penalties,
        "special_char_penalties": special_char_penalties,
        "address_duplication_penalties": address_duplication_penalties,
        "special_char_counts": special_char_counts,
        "total_variations_counts": total_variations_counts,
        "special_char_ratios": special_char_ratios,
    }


