from typing import List, Tuple, Iterable, Dict, Set, Any, Optional
from itertools import combinations
import bittensor as bt
try:
    from typing import SimpleNamespace
except Exception as e:
    from types import SimpleNamespace
import asyncio
from tabulate import tabulate
import json
import random
import numpy as np
import httpx
import time

from MIID.miner.custom_logger import CustomLogger
from MIID.miner.generate_possible_count_pairs_v4 import generate_all_possible_count_pairs_v4
from MIID.miner.rule_based_transformations import RULE_BASED_TRANSFORMATIONS
from MIID.miner.rule_based_transformations import RULE_BASED_TRANSFORMATIONS_COMBINED
from MIID.validator.reward import calculate_variation_quality, calculate_orthographic_similarity
from MIID.validator.rule_evaluator import evaluate_rule_compliance
from MIID.validator.rule_evaluator import RULE_EVALUATORS
from MIID.validator.reward import get_name_variation_rewards


def calculate_orthographic_level(name: str, variation: str) -> int:
    """Calculate the orthographic similarity level of a variation"""
    orthographic_boundaries = {
                0: (0.70, 1.00),    # High orthographic similarity
                1: (0.50, 0.69),   # Medium orthographic similarity  
                2: (0.20, 0.49),       # Low orthographic similarity
                3: (0.0, 0.19)       # Too low orthographic similarity
            }
    sim = calculate_orthographic_similarity(name, variation)
    for level, (min_val, max_val) in orthographic_boundaries.items(): 
        if min_val <= sim <= max_val:
            return level
    return None # Too low orthographic similarity

def get_effective_rules(name: str, selected_rules: List[str]) -> List[str]:
    compliant_variations, _ = evaluate_rule_compliance(name, [""], selected_rules)
    return list(compliant_variations.keys())

def get_name_parts(name: str) -> Tuple[str, str]:
    name_parts = name.split()
    if len(name_parts) < 2:
        first_name = name
        last_name = None
    else:
        i = name.find(" ")
        first_name = name[:i]
        last_name = name[i+1:]

    return first_name, last_name

def generate_one_rule_based_variation(rules, name, rng, max_attempts = 1000):
    attempts = 0
    while attempts < max_attempts:
        try:
            attempts +=1
            key = "_".join(sorted(rules))
            variation = RULE_BASED_TRANSFORMATIONS_COMBINED[key](name, rng)
            if not variation.strip() or variation == name:
                continue
            is_valid = True
            for r in rules:
                if not RULE_EVALUATORS[r](name, variation):
                    is_valid = False
                    break
            if is_valid:
                return variation
        except Exception:
            pass
    return None

def get_cand_minrequired_rule_varset(name: str, effective_rules: List[str]) -> List[Set[str]]:
    checked_rules = set()
    cand_minrequired_rule_varset = []
    rng = random.Random(hash(name))
    for r in range(len(effective_rules), 0, -1):
        for comb in combinations(effective_rules, r):
            key = "_".join(sorted(comb))
            if not key in RULE_BASED_TRANSFORMATIONS_COMBINED or any([cr in checked_rules for cr in comb]):
                continue
            max_attempts = 200
            group_rule_variations = set()
            while max_attempts > 0:
                max_attempts -= 1
                variation = generate_one_rule_based_variation(comb, name, rng)
                exists = False
                for v in cand_minrequired_rule_varset:
                    if variation in v:
                        exists = True
                        break
                if exists:
                    continue
                if variation:
                    group_rule_variations.add(variation)
                    if len(group_rule_variations) >= 15:
                        break
                else:
                    break
            if group_rule_variations:
                checked_rules.update(comb)
                cand_minrequired_rule_varset.append(group_rule_variations)
    return cand_minrequired_rule_varset

def variation_matches_any_rule(name: str, variation: str, rules: List[str]) -> bool:
    """Check if a variation matches any of the specified rule-based transformations"""
    if not rules:
        return False
    
    for rule in rules:
        if rule in RULE_EVALUATORS:
            try:
                if RULE_EVALUATORS[rule](name, variation):
                    return True
            except Exception:
                # If rule evaluation fails, assume it doesn't match
                continue
    return False


def _fetch_nvgen_pool(
    name: str,
    logger: CustomLogger,
    timeout_s: float = 100.0,
    retries: int = 5,
    backoff_base: float = 0.5,
) -> Optional[List[List[List[List[str]]]]]:
    """Get pool from nvgen service; returns nested pools or None."""
    if httpx is None:
        logger.warning("httpx not available; skipping nvgen fetch")
        return None
    from MIID.miner.pool_generator import expand_into_buckets
    return expand_into_buckets(name, timeout_seconds=timeout_s)
    url = f"http://localhost:8001/pool?original_name={name}&timeout={int(timeout_s)}"
    with httpx.Client(timeout=timeout_s + 10.0) as client:
        for attempt in range(1, retries + 1):
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    pools = data.get("pools")
                    if isinstance(pools, list):
                        return pools  # type: ignore
                    logger.debug("nvgen response missing 'pools' key or wrong type")
                else:
                    logger.debug(f"nvgen HTTP {resp.status_code} for {name}")
            except Exception as e:  # pragma: no cover
                logger.debug(f"nvgen exception for {name}: {e}")
            if attempt < retries:
                delay = backoff_base * (2 ** (attempt - 1))
                logger.info(f"Retry {attempt}/{retries} in {delay:.2f}s for {name}")
                time.sleep(delay)
    return None


def gen_pool(name, effective_rules, logger: CustomLogger, timeout_s: float = 100.0):
    """
    Build vpools as a 1(l) × 4(o) × 8(p) matrix of sets.
    - Pull base variants from nvgen (if available).
    - Enforce per-(l,o,p,lowercase) quota: at most `per_lowercase_limit` variants per lowercase string.
    - Fill with fast case-variants until quota is met (no exponential toggling).
    """
    if not name:
        return []

    max_l_level = len(name)
    nonrule_vpools: Optional[List[List[List[Set[str]]]]] = None

    # -----------------------------------------------------------------------
    # Fetch from nvgen
    # -----------------------------------------------------------------------
    try:
        pools_raw, _ = _fetch_nvgen_pool(name, logger, timeout_s=timeout_s)
        if pools_raw:
            # normalize into sets
            max_l_level = len(pools_raw)
            nonrule_vpools = [
                [[set() for _ in range(8)] for _ in range(4)]
                for _ in range(max_l_level)
            ]
            for l_level, o_levels in enumerate(pools_raw):
                if l_level >= max_l_level:
                    break
                nonempty = False
                for o_level, p_classes in enumerate(o_levels[:4]):
                    for p_level, variants in enumerate(p_classes[:8]):
                        # normalize to list of strings
                        if not isinstance(variants, list) or not variants:
                            continue
                        nonempty = True
                        for v in variants:
                            if not isinstance(v, str):
                                continue
                            if effective_rules and variation_matches_any_rule(name, v, effective_rules):
                                continue
                            if len(nonrule_vpools[l_level][o_level][p_level]) >= 20:
                                break
                            # diagnose miscalculated orthographic level
                            o_target = calculate_orthographic_level(name, v)
                            if o_target is None:
                                logger.warning(f"Miscalculated orthographic level: {v} -> {o_target}")
                                continue
                            nonrule_vpools[l_level][o_target][p_level].add(v)
                if not nonempty:
                    # stop expanding further l-levels if this one is empty
                    break
    except Exception as e:  # pragma: no cover
        import traceback
        traceback.print_exc()
        logger.warning(f"nvgen fetch failed for {name}: {e}")

    if not nonrule_vpools:
        nonrule_vpools = [
            [[set() for _ in range(8)] for _ in range(4)]
            for _ in range(max_l_level)
        ]

    # -----------------------------------------------------------------------
    # Build new vpools with lowercase quotas and fast case fill
    # -----------------------------------------------------------------------
    L = len(nonrule_vpools)
    vpools_out: List[List[List[Set[str]]]] = [
        [
            [
                nonrule_vpools[l_level][o_level][p_level].copy() for p_level in range(8)
            ] for o_level in range(4)
        ] for l_level in range(L)
    ]
    nonrule_vpools[0][3][7].add(name)
    # 1) generate one sample case variants for each variantion
    for l_level in range(L):
        for o_level in range(4):
            for p_level in range(8):
                for variantion in nonrule_vpools[l_level][o_level][p_level]:
                    src = variantion
                    for i in range(len(src)):
                        # abcdefg -> ABCdefg
                        cased = src[:i+1].upper() + src[i+1:]
                        cased_str = "".join(cased)
                        o_target = calculate_orthographic_level(name, cased_str)
                        if o_target is None:
                            continue
                        if len(vpools_out[l_level][o_target][p_level]) >= 20:
                            continue
                        if effective_rules and variation_matches_any_rule(name, cased_str, effective_rules):
                            continue
                        vpools_out[l_level][o_target][p_level].add(cased_str)
    # 2) generate case variants for each variantion to fill up the quota
    for l_level in range(L):
        for o_level in range(4):
            for p_level in range(8):
                # generate case variants for each variantion
                for variantion in nonrule_vpools[l_level][o_level][p_level]:
                    for i in range(1, len(variantion) + 1):
                        # combinations of i elements from src
                        for cand in combinations(range(len(variantion)), i):
                            cased = list(variantion)
                            for c in cand:
                                cased[c] = cased[c].upper()
                            cased_str = "".join(cased)
                            o_target = calculate_orthographic_level(name, cased_str)
                            if o_target is None:
                                continue
                            if len(vpools_out[l_level][o_target][p_level]) >= 20:
                                break
                            if effective_rules and variation_matches_any_rule(name, cased_str, effective_rules):
                                continue
                            vpools_out[l_level][o_target][p_level].add(cased_str)
    
    return vpools_out

def generate_additional_rule_varset(
    original_name: str,
    effective_rules: Set[str],
    cand_minrequired_rule_varset: List[Set[str]],
    rule_count: int,
    logger: CustomLogger) -> Set[str]:
    attempts = 0
    max_attempts = 1000
    rng = random.Random(hash(original_name))
    # additional_count = rule_count - len(minimum_required_rule_variations_set_list)
    # if additional_count <= 0:
    #     return set()
    additional_rule_varset = set()
    while attempts < max_attempts:
        if not effective_rules:
            break
        rule = rng.choice(effective_rules)
        attempts += 1
        try:
            variation = RULE_BASED_TRANSFORMATIONS[rule](original_name, rng)
            is_valid = RULE_EVALUATORS[rule](original_name, variation)
            if (is_valid and
                variation != original_name and 
                len(variation.strip()) > 0 and
                not any([variation in varset for varset in cand_minrequired_rule_varset])):
                additional_rule_varset.add(variation)
        except Exception:
            continue
    return additional_rule_varset

def counts_in_matrix48(
    name: str,
    pool: List[str],
    O: List[int],
    P: List[int],
    nonrule_count: int,
    logger: CustomLogger
) -> Dict[str, int]:
    if name is None:
        return None
    counts_in_matrix48 = {}
    # initilize max_l_level = len(name) and update from variant pool
    max_l_level = len(pool)
    # select variations from each pool according to
    # orthographic and phonetic similarity using max flow algorithm
    O = O.copy()
    P = P.copy()

    x = []
    pidx_levels = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7]
    ]
    # start from the lowest level of length
    max_flow = []
    for l_level in range(max_l_level):
        # calc matrix flow in level l_level
        # construct MaxU matrix 4(p) * 4(o)
        MaxU = [[None]*4 for _ in range(4)]
        for o_level in range(4):
            for p_level in range(4):
                MaxU[o_level][p_level] = sum(
                    len(pool[l_level][o_level][pidx])
                    for pidx in pidx_levels[p_level]
                    )

        # solve max flow problem
        from MIID.miner.matrix_flow import solve_maxflow, max_transport44
        # mf, flow = solve_maxflow(MaxU, P, O)
        mf, flow = max_transport44(MaxU, P, O)
        

        # let's update the remains of O and P for the next length level.
        for o_level in range(4):
            O[o_level] = O[o_level] - sum(mf[o_level])
        for p_level in range(4):
            P[p_level] = P[p_level] - sum(mf[o_level][p_level] for o_level in range(4))
        # Check if all O values are zero
        # append last column to x: 4(p) * 4(o) -> 4(p) * 4(o)
        for o_level in range(4):
            mf[o_level].append(0)
        max_flow.append(mf)

    # if othographic(sum of row) not solved successfully,
    # complete the orthographic count of matrix_flow with the variants in the order of m, f, l, t-f
    for o_level in range(4):
        if O[o_level] > 0:
            logger.warning(f"Complete {O[o_level]} for othographic level {o_level}")
            for l_level in range(max_l_level):
                for p_level in (3, 1, 2, 0):
                    remains = sum(len(pool[l_level][o_level][pidx]) for pidx in pidx_levels[p_level]) - max_flow[l_level][o_level][p_level]
                    if remains >= O[o_level]:
                        max_flow[l_level][o_level][p_level] += O[o_level]
                        O[o_level] = 0
                        break
                    else:
                        max_flow[l_level][o_level][p_level] += remains
                        O[o_level] -= remains

    # if still not completed nonrule count,
    # complete the remains with ortho_level = 3 row to ensure nonrule count
    remains = nonrule_count - sum(
        max_flow[l_level][o_level][p_level]
        for p_level in range(4)
        for o_level in range(4)
        for l_level in range(max_l_level)
    )
    if remains > 0:
        for l_level in range(max_l_level):
            # for p_level in [1, 2, 0, 3]:
            for p_level in [3, 1, 2, 0]:
                count = sum(
                    len(pool[l_level][3][pidx])
                    for pidx in pidx_levels[p_level]
                ) - max_flow[l_level][3][p_level]
                if count >= remains:
                    max_flow[l_level][3][p_level] += remains
                    remains = 0
                    break # need for double break
                else:
                    max_flow[l_level][3][p_level] += count
                    remains -= count
    logger.warning(f"Remains: {remains}")
    max_flow2 = [[[0 for _ in range(8)] for _ in range(4)] for _ in range(max_l_level)]
    for l_level in range(len(max_flow)):
        for p_level in [1, 2, 0, 3]:
            all_zero = True
            O = [0] * 4
            for o_level in range(4):
                O[o_level] = max_flow[l_level][o_level][p_level]
                if O[o_level] != 0:
                    all_zero = False
            if all_zero:
                continue
            MaxU = [[0 for _ in range(len(pidx_levels[p_level]))] for _ in range(4)]
            for o_level in range(4):
                for idx, pidx in enumerate(pidx_levels[p_level]):
                    MaxU[o_level][idx] = len(pool[l_level][o_level][pidx])
            logger.debug(f"p_level: {p_level}, O: {O}")
            for row in MaxU:
                logger.debug(f"{row}")
            
            # # v1.0 : divergent algorithm
            # from MIID.miner.matrix_flow import maxflow_then_maxdisp_int
            # xx, _ = maxflow_then_maxdisp_int(MaxU, O)

            # v2.0: normalize algorithm
            from MIID.miner.matrix_flow import solve_integer_diverse
            xx, _ = solve_integer_diverse(MaxU, O)

            for o_level in range(4):
                logger.debug(f"{xx[o_level]}")
            for o_level in range(4):
                for idx, pidx in enumerate(pidx_levels[p_level]):
                    max_flow2[l_level][o_level][pidx] = xx[o_level][idx]
    logger.debug(f"final:")
    for l_level in range(max_l_level):
        logger.debug(f"name: {name}, l_level: {l_level}")
        logger.debug(f"-" * 50)
        for o_level in range(4):
            logger.debug(f"{max_flow2[l_level][o_level]}")
    return max_flow2

def calculate_nonrule_variations_count(
    name: str,
    name_pool: List[str],
    nonrule_count: int,
    phonetic_similarity: float,
    orthographic_similarity: float,
    effective_rules: Set[str],
    logger: CustomLogger) -> Set[str]:
    if not name:
        return []
    if nonrule_count <= 0:
        return []

    # Use provided similarity distributions or defaults
    phonetic_dist = phonetic_similarity if phonetic_similarity else {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    orthographic_dist = orthographic_similarity if orthographic_similarity else {"Light": 0.3, "Medium": 0.4, "Far": 0.3}

    # calculate the number of variations for each orthographic similarity level
    # to complete the nonrule_count, the rest will be assigned to the too-far level
    O = [0] * 4
    for key in orthographic_dist.keys():
        if key == "Light":
            O[0] = int(orthographic_dist["Light"] * nonrule_count)
        elif key == "Medium":
            O[1] = int(orthographic_dist["Medium"] * nonrule_count)
        elif key == "Far":
            O[2] = int(orthographic_dist["Far"] * nonrule_count)
    O[3] = nonrule_count - sum(O)
    # calculate the number of variations for each phonetic similarity level
    P = [0] * 4
    # if phonetic_dist["Light"] < 0.4 and phonetic_dist["Medium"] < 0.4 and phonetic_dist["Far"] < 0.4:
    #     phonetic_dist["Light"] = 0.3
    #     phonetic_dist["Medium"] = 0.3
    #     phonetic_dist["Far"] = 0.3
    for key in phonetic_dist.keys():
        if key == "Light":
            P[0] = int(phonetic_dist["Light"] * nonrule_count)
        elif key == "Medium":
            P[1] = int(phonetic_dist["Medium"] * nonrule_count)
        elif key == "Far":
            P[2] = int(phonetic_dist["Far"] * nonrule_count)
    # to complete the nonrule_count, the rest will be assigned to the medium level
    P[1] += nonrule_count - sum(P)

    # from MIID.miner.cal_phonetic_similarity import optimal_n0_n1_n2
    # P[0], P[1], P[2] = optimal_n0_n1_n2(nonrule_count, phonetic_dist)
    # P[3] = 0


    logger.debug(f"O: {O}")
    logger.debug(f"P: {P}")

    # select variations from each pool according to
    # orthographic and phonetic similarity using max flow algorithm
    max_flow2 = counts_in_matrix48(
        name,
        name_pool,
        O,
        P,
        nonrule_count,
        logger
    )
    return max_flow2


def get_name_variation_rewards_exclude_phonetic(
    seed_names: List[str],
    responses: List[Dict[str, List[str]]],
    uids: List[int],
    variation_count: int = 10,
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    rule_based: Dict[str, Any] = None
) -> Tuple[np.ndarray, List[Dict]]:

    from MIID.validator.reward import get_name_variation_rewards
    from MIID.validator.reward import get_name_part_weights
    name_part_weights = get_name_part_weights(seed_names[0])

    rewards, detailed_metrics =  get_name_variation_rewards(
        None, seed_names, responses, uids, variation_count, phonetic_similarity, orthographic_similarity, rule_based)
    if not detailed_metrics:
        return [0], [None]
    if not detailed_metrics[0]['name_metrics'].values():
        return [0], [None]
    name_metrics = list(detailed_metrics[0]['name_metrics'].values())[0]
    
    try:
        if 'last_name' in name_metrics:
            rewards -= name_metrics['first_name']['metrics']['similarity']['phonetic'] * 0.24 * name_part_weights['first_name_weight'] + name_metrics['last_name']['metrics']['similarity']['phonetic'] * 0.24 * name_part_weights['last_name_weight']
        else:
            rewards -= name_metrics['first_name']['metrics']['similarity']['phonetic'] * 0.24 * name_part_weights['first_name_weight']
    except Exception as e:
        # print(json.dumps(name_metrics, indent=4))
        return [0], [None]
    return rewards, detailed_metrics

class AnswerCandidate:
    def __init__(self, name: str,
                 minimal_rule_based_count: int,
                 additional_rule_based_count: int,
                 duplicated_rule_based_count: int,
                 base_count: int,
                 effective_rules: Set[str],
                 cand_minrequired_rule_varset: List[Set[str]],
                 additional_rule_varset: Set[str],
                 nonrule_count_matrix: Dict[str, List[List[List[int]]]],
                 nonrule_name_pools: Dict[str, List[List[List[List[str]]]]],
                 query_params: Dict[str, Any],
                 scores: float,
                 metric: Dict[str, Any]):
        self.name = name
        self.first_name, self.last_name = get_name_parts(name)
        self.minimal_rule_based_count = minimal_rule_based_count
        self.additional_rule_based_count = additional_rule_based_count
        self.duplicated_rule_based_count = duplicated_rule_based_count
        self.base_count = base_count
        self.effective_rules = effective_rules
        self.cand_minrequired_rule_varset = cand_minrequired_rule_varset
        self.additional_rule_varset = additional_rule_varset
        self.nonrule_count_matrix = nonrule_count_matrix
        self.nonrule_name_pools = nonrule_name_pools
        self.query_params = query_params
        self.scores = scores
        self.metric = metric
        self.rng = random.Random(hash(name))
        self.bucket = None
        
    def get_next_answer(self, noisy_count: int = 0, bucket_no: int = 0) -> Set[str]:
        # sample_variations is not set, but list as rule-based variations can be duplicated.
        if self.bucket:
            return self.bucket[self.rng.randint(0, len(self.bucket)-1)].get_next_answer(noisy_count, bucket_no)
        sample_variations = []
        # add minimum required rule variations
        if len(self.cand_minrequired_rule_varset) > 0:
            count = 0
            for varset in self.cand_minrequired_rule_varset:
                if count >= self.minimal_rule_based_count:
                    break
                count += 1
                sample_variations.append(list(varset)[self.rng.randint(0, len(varset) - 1)])
        # add additional rule variations
        additionals = list(self.additional_rule_varset.copy())
        if len(self.cand_minrequired_rule_varset) > 0:
            for varset in self.cand_minrequired_rule_varset:
                additionals.extend([v for v in varset if v not in sample_variations])
        sample_variations.extend(self.rng.sample(additionals, min(len(additionals), self.additional_rule_based_count)))
        sample_variations_no_specials = list([v for v in sample_variations if "." not in v])
        if sample_variations_no_specials:
            duplications = [self.rng.choice(sample_variations_no_specials) for _ in range(self.duplicated_rule_based_count)]
            sample_variations.extend(duplications)
        # add nonrule variations
        
        def select_variations(pool, count_matrix):
            selected_variations = []
            for l_level in range(len(pool)):
                for o_level in range(4):
                    for p_level in range(8):
                        if len(pool[l_level][o_level][p_level]) >= count_matrix[l_level][o_level][p_level]:
                            selected_variations.extend(self.rng.sample(pool[l_level][o_level][p_level], count_matrix[l_level][o_level][p_level]))
            return selected_variations
        def add_noisy_to_count_mat(pool, count_matrix, noisy_count):
            result = [[[0 for _ in range(8)] for _ in range(4)] for _ in range(len(pool))]
            for l_level in range(0, len(pool)):
                for o_level in range(4):
                    for p_level in range(8):
                        result[l_level][o_level][p_level] = count_matrix[l_level][o_level][p_level]
                        if l_level == 0:
                            continue
                        if (result[l_level - 1][o_level][p_level] > 0 and
                            len(pool[l_level][o_level][p_level]) - result[l_level][o_level][p_level] > 0):
                            max_noisy = min(
                                result[l_level - 1][o_level][p_level],
                                len(pool[l_level][o_level][p_level]) - result[l_level][o_level][p_level],
                                noisy_count
                            )
                            result[l_level - 1][o_level][p_level] -= max_noisy
                            result[l_level][o_level][p_level] += max_noisy
                            noisy_count -= max_noisy
            return result
        # import copy
        count_matrix = {}
        # copy.deepcopy(self.nonrule_count_matrix[self.name])
        first_noisy_count = noisy_count if not self.last_name else noisy_count//2
        last_noisy_count = noisy_count - first_noisy_count
        
        count_matrix[self.first_name] = add_noisy_to_count_mat(self.nonrule_name_pools[self.first_name], self.nonrule_count_matrix[self.first_name], first_noisy_count)
        first_variations = select_variations(self.nonrule_name_pools[self.first_name], count_matrix[self.first_name])
        nonrule_variations = []
        if not self.last_name:
            nonrule_variations = first_variations
        else:
            count_matrix[self.last_name] = add_noisy_to_count_mat(self.nonrule_name_pools[self.last_name], self.nonrule_count_matrix[self.last_name], last_noisy_count)
            last_variations = list(select_variations(self.nonrule_name_pools[self.last_name], count_matrix[self.last_name]))
            for i, v in enumerate(first_variations):
                if i < len(last_variations):
                    v = v + " " + last_variations[i]
                    if self.effective_rules and variation_matches_any_rule(self.name, v, self.effective_rules):
                        continue
                    nonrule_variations.append(v)
        sample_variations.extend(list(nonrule_variations))
        return sample_variations

    def get_scores(self) -> float:
        return self.scores
    def get_metric(self) -> Dict[str, Any]:
        return self.metric


def try_once(
    original_name: str,
    first_name: str,
    last_name: str,
    name_pools: Dict[str, List[str]],
    expected_total_count: int,
    minimal_rule_based_count: int,
    additional_rule_based_count: int,
    duplicated_rule_based_count: int,
    base_count: int,
    rule_percentage: float,
    selected_rules: List[str],
    effective_rules: Set[str],
    cand_minrequired_rule_varset: List[Set[str]],
    additional_rule_varset: Set[str],
    phonetic_similarity: float,
    orthographic_similarity: float,
    query_params: Dict[str, Any],
    logger: CustomLogger) -> Tuple[Set[str], Set[str]]:

    count_matrix = {}

    for name in [first_name, last_name]:
        count_matrix[name] = calculate_nonrule_variations_count(
            name,
            name_pools[name],
            base_count,
            phonetic_similarity,
            orthographic_similarity,
            effective_rules,
            logger
        )

    cand = AnswerCandidate(
        name=original_name,
        minimal_rule_based_count=minimal_rule_based_count,
        additional_rule_based_count=additional_rule_based_count,
        duplicated_rule_based_count=duplicated_rule_based_count,
        base_count=base_count,
        effective_rules=effective_rules,
        cand_minrequired_rule_varset=cand_minrequired_rule_varset,
        additional_rule_varset=additional_rule_varset,
        nonrule_count_matrix=count_matrix,
        nonrule_name_pools=name_pools,
        query_params=query_params,
        scores=0.0,
        metric={}
    )
    responses = {}
    responses = [SimpleNamespace(
        variations={original_name: list(cand.get_next_answer())}
    )]
    # Calculate rule-based metadata
    rule_based = {"selected_rules": selected_rules, "rule_percentage": rule_percentage * 100}
    debug_level = bt.logging.get_level()
    bt.logging.setLevel('CRITICAL')
    if (
        # (phonetic_similarity["Light"] < 0.5 and phonetic_similarity["Medium"] < 0.5 and phonetic_similarity["Far"] < 0.5) or
        ("Light" in phonetic_similarity and phonetic_similarity["Light"] == 1.0) or
        # (phonetic_similarity["Medium"] == 1.0) or
        ("Far" in phonetic_similarity and phonetic_similarity["Far"] == 1.0)
        ):
        scores, metric = get_name_variation_rewards(
            None,
            seed_names=[original_name], 
            responses=responses,
            uids=[0],
            variation_count=expected_total_count,
            phonetic_similarity=phonetic_similarity,
            orthographic_similarity=orthographic_similarity,
            rule_based=rule_based,
        )
    else:
        scores, metric = get_name_variation_rewards_exclude_phonetic(
            seed_names=[original_name], 
            responses=responses,
            uids=[0],
            variation_count=expected_total_count,
            phonetic_similarity=phonetic_similarity,
            orthographic_similarity=orthographic_similarity,
            rule_based=rule_based,
        )
    bt.logging.setLevel(debug_level)
    logger.debug(f"scores: {scores}")
    cand.scores = scores[0]
    cand.metric = metric[0]
    return cand

def generate_name_variations(
    original_name: str,
    query_params: Dict[str, Any],
    key: str = None,
    timeout: int = 100
) -> List[str]:

    expected_total_count=int(query_params.get("variation_count") or 0)
    rule_percentage=float(query_params.get("rule_percentage") or 0.0)
    selected_rules=list(query_params.get("selected_rules") or [])
    phonetic_similarity=query_params.get("phonetic_similarity")
    orthographic_similarity=query_params.get("orthographic_similarity")

    logger = CustomLogger(name=original_name, output_file=f"logs/{key}/{original_name}.log", use_stdout=False)
    logger.info(f"Generating variations for {original_name}")
    logger.info(f"-" * 100)
    
    effective_rules = get_effective_rules(original_name, selected_rules)
    logger.info(f"Effective rules: {effective_rules}")
    logger.info(f"-" * 100)

    # generate rule-based variations at most one per each effective rule
    cand_minrequired_rule_varset = get_cand_minrequired_rule_varset(original_name, effective_rules)
    logger.info(f"Min Required Rule Variation Set: {cand_minrequired_rule_varset}")
    logger.info(f"-" * 100)

    timeout_at = time.time() + timeout
    
    # 1) Rule-based portion
    additional_rule_varset = generate_additional_rule_varset(
        original_name,
        effective_rules,
        cand_minrequired_rule_varset,
        0,
        logger
    )
    # get name parts
    (first_name, last_name) = get_name_parts(original_name)
    logger.info(f"First name: {first_name}")
    logger.info(f"Last name: {last_name}")
    logger.info(f"-" * 100)

    # get pool of nonrule variations for first and last name
    try:
        import concurrent.futures
        from itertools import repeat
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            tasks = list(executor.map(gen_pool, [first_name, last_name], repeat(effective_rules), repeat(logger), repeat(timeout / 3 * 2)))
            name_pools = {first_name: tasks[0], last_name: tasks[1]}
    except Exception as e:
        logger.warning(f"Failed to generate variant pools for {original_name}: {e}")
        name_pools = None
        return []

    # # Get all possible rule count pairs
    # rule_count_pairs = generate_possible_count_pairs(
    #     expected_total_count,
    #     rule_percentage,
    #     orthographic_similarity,
    #     len(effective_rules)
    # )
    rule_count_pairs = generate_all_possible_count_pairs_v4(
        expected_total_count,
        len(cand_minrequired_rule_varset),
        rule_percentage,
        phonetic_similarity
    )
    logger.info(f"Rule count pairs: {rule_count_pairs}")
    
    best_cand = None
    bucket_cand = {}
    scores = []

    from tqdm import tqdm
    # for i, (rule_count, nonrule_count) in tqdm(enumerate(rule_count_pairs), desc=f"Processing {original_name}"):
    for i, (minimal_rule_based_count, additional_rule_based_count, duplicated_rule_based_count, base_count, total_count) in enumerate(rule_count_pairs):
        logger.info(
            f"-" * 30 +
            f"Trying rule_count_pair {i+1} / {len(rule_count_pairs)}: minimal_rule_based_count/additional_rule_based_count/duplicated_rule_based_count/base_count: {minimal_rule_based_count}/{additional_rule_based_count}/{duplicated_rule_based_count}/{base_count}/{total_count}" +
            "-" * 30
        )
        if time.time() > timeout_at:
            break
        cand = try_once(
            original_name,
            first_name,
            last_name,
            name_pools,
            expected_total_count,
            minimal_rule_based_count,
            additional_rule_based_count,
            duplicated_rule_based_count,
            base_count,
            rule_percentage,
            selected_rules,
            effective_rules,
            cand_minrequired_rule_varset,
            additional_rule_varset,
            phonetic_similarity,
            orthographic_similarity,
            query_params,
            logger
        )
        # if not best_cand:
        #     best_cand = cand
        #     continue
        if not cand:
            continue
        logger.debug(f"\n{json.dumps(cand.metric, indent=4)}")
        # precision from 4->2 : orthographic's minor difference is not significant, instead focus on phonetic similarity
        fmt2 = f"{cand.scores:.2f}" # 2 decimal places
        if fmt2 not in bucket_cand:
            bucket_cand[fmt2] = []
        bucket_cand[fmt2].append(cand)
        if fmt2 not in scores:
            scores.append(fmt2)
        # if cand.scores - best_cand.scores >= 0.001:
        #     best_cand = cand
    scores.sort()
    best_score = scores[-1]
    best_bucket = bucket_cand[best_score]
    logger.debug(f"Best score: {best_score}")
    logger.debug(f"-" * 100)
    try:
        logger.debug(f"Best score achieved: {best_score}")
        # logger.debug(f"\n{json.dumps(best_bucket, indent=4)}")
        # logger.debug(f"Best selected: {best_cand.minimal_rule_based_count}/{best_cand.additional_rule_based_count}/{best_cand.duplicated_rule_based_count}/{best_cand.base_count}")
        # logger.debug(f"Min Required Rule Variation Set: {best_cand.cand_minrequired_rule_varset}")
        # logger.debug(f"Additional Rule Variation Set: {best_cand.additional_rule_varset}")
        # logger.debug(f"Metric: \n{json.dumps(best_cand.metric, indent=4)}")
        # logger.debug(f"-" * 100)
        # best_bucket.query_params = query_params
    except Exception as e:
        pass
    best_cand = AnswerCandidate(
        name=original_name,
        minimal_rule_based_count=0,
        additional_rule_based_count=0,
        duplicated_rule_based_count=0,
        base_count=0,
        effective_rules=None,
        cand_minrequired_rule_varset=None,
        additional_rule_varset=None,
        nonrule_count_matrix=None,
        nonrule_name_pools=None,
        query_params=query_params,
        scores=0.0,
        metric=None
    )
    # Get the last 3 items from best_bucket
    best_bucket = best_bucket[-10:]
    best_cand.bucket=best_bucket
    logger.flush()
    return best_cand
