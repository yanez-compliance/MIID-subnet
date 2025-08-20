import re
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


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


