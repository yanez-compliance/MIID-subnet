import math
from typing import Dict, Tuple

def generate_all_possible_count_pairs(C: int, Pr: float) -> list[Tuple[int, int]]:
    """
    Generate all possible (rule_count, nonrule_count) pairs where the sum is in range [int(C/3), int(3*C)].
    
    INPUT:
    C: target count, int
    Pr: Route percent, 0~1 float (not used in this function)
    
    OUTPUT:
    List of (rule_count, nonrule_count) tuples
    """
    
    min_sum = int(C / 4)
    max_sum = min(20, int(3 * C))
    
    pairs = []
    
    for total in range(min_sum, max_sum + 1):
        for rule_count in range(0, min(total - 1, int(total * Pr) + 2)):
            nonrule_count = total - rule_count
            pairs.append((rule_count, nonrule_count))
    
    return pairs
    
Ordered = {
  "343": {
    "1": { "level": "OK",    "counts": [0,1,0] },
    "2": { "level": "OK",    "counts": [1,1,0] },
    "3": { "level": "Great", "counts": [1,1,1] },
    "4": { "level": "Good",  "counts": [1,2,1] },
    "5": { "level": "OK",    "counts": [2,2,1] },
    "6": { "level": "Good",  "counts": [2,2,2] },
    "7": { "level": "Great", "counts": [2,3,2] },
    "8": { "level": "OK",    "counts": [2,3,3] },
    "9": { "level": "OK",    "counts": [3,4,2] },
    "10":{ "level": "Best",  "counts": [3,4,3] },
    "11":{ "level": "OK",    "counts": [3,5,3] },
    "12":{ "level": "OK",    "counts": [4,5,3] },
    "13":{ "level": "Great", "counts": [4,5,4] },
    "14":{ "level": "Good",  "counts": [4,6,4] },
    "15":{ "level": "Best",  "counts": [5,6,4] }
  },
  "262": {
    "1": { "level": "Great", "counts": [0,1,0] },
    "2": { "level": "OK",    "counts": [0,1,1] },
    "3": { "level": "OK",    "counts": [1,2,0] },
    "4": { "level": "Great", "counts": [1,2,1] },
    "5": { "level": "Best",  "counts": [1,3,1] },
    "6": { "level": "Great", "counts": [1,4,1] },
    "7": { "level": "OK",    "counts": [2,4,1] },
    "8": { "level": "OK",    "counts": [2,5,1] },
    "9": { "level": "Great", "counts": [2,5,2] },
    "10":{ "level": "Best",  "counts": [2,6,2] },
    "11":{ "level": "Great", "counts": [2,7,2] },
    "12":{ "level": "OK",    "counts": [2,7,3] },
    "13":{ "level": "OK",    "counts": [3,8,2] },
    "14":{ "level": "Great", "counts": [3,8,3] },
    "15":{ "level": "Best",  "counts": [3,9,3] }
  },
  "136": {
    "1": { "level": "Good",  "counts": [0,0,1] },
    "2": { "level": "Good",  "counts": [0,1,1] },
    "3": { "level": "Great", "counts": [0,1,2] },
    "4": { "level": "OK",    "counts": [0,1,3] },
    "5": { "level": "OK",    "counts": [1,1,3] },
    "6": { "level": "OK",    "counts": [1,2,3] },
    "7": { "level": "Great", "counts": [1,2,4] },
    "8": { "level": "Good",  "counts": [1,2,5] },
    "9": { "level": "Good",  "counts": [1,3,5] },
    "10":{ "level": "Best",  "counts": [1,3,6] },
    "11":{ "level": "Good",  "counts": [1,3,7] },
    "12":{ "level": "Good",  "counts": [1,4,7] },
    "13":{ "level": "Great", "counts": [1,4,8] },
    "14":{ "level": "OK",    "counts": [1,4,9] },
    "15":{ "level": "Great", "counts": [2,5,8] }
  },
  "550": {
    "1": { "level": "OK",    "counts": [1,0,0] },
    "2": { "level": "Best",  "counts": [1,1,0] },
    "3": { "level": "OK",    "counts": [2,1,0] },
    "4": { "level": "Best",  "counts": [2,2,0] },
    "5": { "level": "OK",    "counts": [3,2,0] },
    "6": { "level": "Best",  "counts": [3,3,0] },
    "7": { "level": "OK",    "counts": [4,3,0] },
    "8": { "level": "Best",  "counts": [4,4,0] },
    "9": { "level": "OK",    "counts": [5,4,0] },
    "10":{ "level": "Best",  "counts": [5,5,0] },
    "11":{ "level": "OK",    "counts": [6,5,0] },
    "12":{ "level": "Best",  "counts": [6,6,0] },
    "13":{ "level": "OK",    "counts": [7,6,0] },
    "14":{ "level": "Best",  "counts": [7,7,0] },
    "15":{ "level": "OK",    "counts": [8,7,0] }
  },
  "154": {
    "1": { "level": "Avoid", "counts": [0,1,0] },
    "2": { "level": "Great", "counts": [0,1,1] },
    "3": { "level": "Avoid", "counts": [0,2,1] },
    "4": { "level": "OK",    "counts": [0,2,2] },
    "5": { "level": "Good",  "counts": [1,2,2] },
    "6": { "level": "OK",    "counts": [1,3,2] },
    "7": { "level": "Avoid", "counts": [1,3,3] },
    "8": { "level": "Great", "counts": [1,4,3] },
    "9": { "level": "Avoid", "counts": [1,5,3] },
    "10":{ "level": "Best",  "counts": [1,5,4] },
    "11":{ "level": "Avoid", "counts": [1,6,4] },
    "12":{ "level": "Great", "counts": [1,6,5] },
    "13":{ "level": "Avoid", "counts": [2,6,5] },
    "14":{ "level": "OK",    "counts": [2,7,5] },
    "15":{ "level": "Best",  "counts": [2,8,5] }
  },
  "0*0": {
    "1": { "level": "Best", "counts": [0,1,0] },
    "2": { "level": "Best", "counts": [0,2,0] },
    "3": { "level": "Best", "counts": [0,3,0] },
    "4": { "level": "Best", "counts": [0,4,0] },
    "5": { "level": "Best", "counts": [0,5,0] },
    "6": { "level": "Best", "counts": [0,6,0] },
    "7": { "level": "Best", "counts": [0,7,0] },
    "8": { "level": "Best", "counts": [0,8,0] },
    "9": { "level": "Best", "counts": [0,9,0] },
    "10":{ "level": "Best", "counts": [0,10,0] },
    "11":{ "level": "Best", "counts": [0,11,0] },
    "12":{ "level": "Best", "counts": [0,12,0] },
    "13":{ "level": "Best", "counts": [0,13,0] },
    "14":{ "level": "Best", "counts": [0,14,0] },
    "15":{ "level": "Best", "counts": [0,15,0] }
  },
  "730": {
    "1": { "level": "OK",     "counts": [1,0,0] },
    "2": { "level": "Poorer", "counts": [1,1,0] },
    "3": { "level": "Great",  "counts": [2,1,0] },
    "4": { "level": "Good",   "counts": [3,1,0] },
    "5": { "level": "Worst",  "counts": [4,1,0] },
    "6": { "level": "Good",   "counts": [4,2,0] },
    "7": { "level": "Great",  "counts": [5,2,0] },
    "8": { "level": "Poorer", "counts": [6,2,0] },
    "9": { "level": "OK",     "counts": [6,3,0] },
    "10":{ "level": "Best",   "counts": [7,3,0] },
    "11":{ "level": "OK",     "counts": [8,3,0] },
    "12":{ "level": "Poorer", "counts": [8,4,0] },
    "13":{ "level": "Great",  "counts": [9,4,0] },
    "14":{ "level": "Good",   "counts": [10,4,0] },
    "15":{ "level": "Great",  "counts": [11,4,0] }
  },
  "00*": {
    "1": { "level": "Best", "counts": [0,0,1] },
    "2": { "level": "Best", "counts": [0,0,2] },
    "3": { "level": "Best", "counts": [0,0,3] },
    "4": { "level": "Best", "counts": [0,0,4] },
    "5": { "level": "Best", "counts": [0,0,5] },
    "6": { "level": "Best", "counts": [0,0,6] },
    "7": { "level": "Best", "counts": [0,0,7] },
    "8": { "level": "Best", "counts": [0,0,8] },
    "9": { "level": "Best", "counts": [0,0,9] },
    "10":{ "level": "Best", "counts": [0,0,10] },
    "11":{ "level": "Best", "counts": [0,0,11] },
    "12":{ "level": "Best", "counts": [0,0,12] },
    "13":{ "level": "Best", "counts": [0,0,13] },
    "14":{ "level": "Best", "counts": [0,0,14] },
    "15":{ "level": "Best", "counts": [0,0,15] }
  },
  "*00": {
    "1": { "level": "Best", "counts": [1,0,0] },
    "2": { "level": "Best", "counts": [2,0,0] },
    "3": { "level": "Best", "counts": [3,0,0] },
    "4": { "level": "Best", "counts": [4,0,0] },
    "5": { "level": "Best", "counts": [5,0,0] },
    "6": { "level": "Best", "counts": [6,0,0] },
    "7": { "level": "Best", "counts": [7,0,0] },
    "8": { "level": "Best", "counts": [8,0,0] },
    "9": { "level": "Best", "counts": [9,0,0] },
    "10":{ "level": "Best", "counts": [10,0,0] },
    "11":{ "level": "Best", "counts": [11,0,0] },
    "12":{ "level": "Best", "counts": [12,0,0] },
    "13":{ "level": "Best", "counts": [13,0,0] },
    "14":{ "level": "Best", "counts": [14,0,0] },
    "15":{ "level": "Best", "counts": [15,0,0] }
  }
}

# Define ranking order
LEVEL_ORDER = {
    "Best": 1,
    "Great": 2,
    "Good": 3,
    "OK": 4,
    "Poorer": 5,
    "Worst": 6,
    "Avoid": 7
}

def order_candidates(config_key, candidate_N, Ordered):
    """
    Given a config key and list of candidate N values,
    return them sorted by quality level.
    """
    results = []
    for n in candidate_N:
        if str(n) not in Ordered[config_key]:
            continue
        entry = Ordered[config_key][str(n)]
        level = entry["level"]
        counts = entry["counts"]
        rank = LEVEL_ORDER.get(level, 99)
        results.append((n, level, counts, rank))
    if not results:
        return candidate_N
    # Sort by rank (lower is better), then N
    results.sort(key=lambda x: (x[3], x[0]), reverse=True)
    return results


def encode_config_key(target):
    """
    Convert target dict {"Light": L, "Medium": M, "Far": F}
    into shorthand string: 3 digits, '*' if 1.0.
    """
    L = round(target.get("Light", 0.0), 1)
    M = round(target.get("Medium", 0.0), 1)
    F = round(target.get("Far", 0.0), 1)

    digits = []
    for val in (L, M, F):
        if val == 1.0:
            digits.append("*")
        else:
            digits.append(str(int(val * 10)))

    return "".join(digits)


def generate_all_possible_count_pairs_v6(expected_count, minimum_rule_based_count, rule_percent, phonetic_similarity) -> list[Tuple[int, int]]:
    expected_base_count = expected_count * (1.0 - rule_percent)
    base_tolerance = 0.2  # 20% base tolerance
    tolerance = base_tolerance + (0.05 * (expected_base_count // 10))  # Add 5% per 10 expected variations
    tolerance = min(tolerance, 0.4)  # Cap at 40% maximum tolerance
    
    tolerance_range = expected_base_count * tolerance
    lower_bound = max(1, expected_base_count - tolerance_range)  # Ensure at least 1 variation required
    upper_bound = expected_base_count + tolerance_range
    lower_bound = math.ceil(lower_bound)
    upper_bound = math.floor(upper_bound)
    # if lower_bound <= actual_count <= upper_bound:
    #     count_score = 1.0
    
    pairs = []
    # strategy1: focus on expected_base_count
    base_range_count_score_1 = list(range(lower_bound, upper_bound + 1))
    # increase lookup range
    base_range_count_score_0_9 = list(range(3, lower_bound)) + list(range(upper_bound + 1, 20))
    # strategy2: focus on phonetic similarity
    min_phonetic_similarity = min(phonetic_similarity.values())
    config_key = encode_config_key(phonetic_similarity)
    ordered_list = order_candidates(config_key, base_range_count_score_0_9, Ordered) + order_candidates(config_key, base_range_count_score_1, Ordered)
    
    print("upgrade to v6.1")
    for base_count, level, counts, rank in ordered_list:
        for rule_based_count in range(0, int(expected_count * 1.2) + 1):
            if rule_based_count + base_count > int(expected_count * 1.2):
                break
            _minimum_rule_based_count = minimum_rule_based_count
            additional_rule_based_count = rule_based_count - _minimum_rule_based_count
            if additional_rule_based_count < 0:
                continue
            duplicated_rule_based_count = 0
            pairs.append(
                (
                    _minimum_rule_based_count,
                    additional_rule_based_count,
                    duplicated_rule_based_count,
                    base_count,
                    _minimum_rule_based_count + additional_rule_based_count + duplicated_rule_based_count + base_count
                )
            )
        
    return pairs



def generate_possible_count_pairs(C: int, Pr: float, O: Dict[str, float], Ce: int) -> list[Tuple[int, int]]:
    """
    Calculate top 10 optimal rule count (Cr) and none rule count (Cn) based on constraints and optimization criteria.
    
    INPUT:
    C: target count, int
    Pr: Route percent, 0~1 float
    O: distribution dict, dictionary {'Light': 0.7, 'Medium': 0.2, 'Far': 0.1}
    Ce: effective rule count, int
    
    OUTPUT:
    List of (Cr, Cn) tuples, sorted by score descending, max 10 items
    Where:
    Cr: rule count, int
    Cn: none rule count, int
    
    Constraints:
    R1: int(0.5 * C) < Cr + Cn <= int(1.5 * C)
    
    Optimization criteria (when R1 satisfied):
    - Weight 0.4: min(O values, except 0.0) * Cn >= 1 (for orthograph)
    - Weight 0.3: Cr = int((Cr + Cn) * Pr) (for rule)
    - Weight 0.2: Cr > Ce (for rule)
    - Weight 0.1: Cr + Cn close to range <C-R, C+R>
      here R = max(1, C*(1-Pr)*0.2)
    """
    
    # Calculate bounds for R1
    min_total = int(0.5 * C) + 1  # > int(0.5 * C)
    max_total = int(3 * C)      # <= int(1.5 * C)
    
    # Get minimum non-zero value from O
    non_zero_values = [v for v in O.values() if v > 0.0]
    min_o_value = min(non_zero_values) if non_zero_values else 0.1
    
    # Calculate range R for the fourth criterion
    R = max(1, int(C * (1 - Pr) * 0.2))
    target_min = C - R
    target_max = C + R
    
    candidates = []
    
    # Try all possible combinations within the valid range
    for total in range(min_total, max_total + 1):
        for cr_candidate in range(0, total + 1):
            cn_candidate = total - cr_candidate
            
            # Calculate optimization score
            score = 0.0
            
            # Criterion 1 (weight 0.4): min(O values, except 0.0) * Cn >= 1 (for orthograph)
            if min_o_value * cn_candidate >= 1:
                score += 0.4
                
            # Criterion 2 (weight 0.3): Cr = int((Cr + Cn) * Pr) (for rule)
            expected_cr = int(total * Pr)
            if cr_candidate == expected_cr:
                score += 0.3
                
            # Criterion 3 (weight 0.2): Cr > Ce (for rule)
            if cr_candidate > Ce:
                score += 0.2
                
            # Criterion 4 (weight 0.1): Cr + Cn close to range <C-R, C+R>
            if target_min <= total <= target_max:
                score += 0.1
            
            # Store detailed scoring for filtering
            criteria_scores = (
                1 if min_o_value * cn_candidate >= 1 else 0,  # orthograph
                1 if cr_candidate == expected_cr else 0,      # rule match
                1 if cr_candidate > Ce else 0,                # rule count
                1 if target_min <= total <= target_max else 0 # count score (binary)
            )
            
            candidates.append((score, cr_candidate, cn_candidate, criteria_scores))
    
    # Group candidates by their first 3 criteria (orthograph, rule match, rule count)
    # For each group, keep the 3 with best count score
    groups = {}
    for score, cr, cn, (orth, rule_match, rule_count, count_score) in candidates:
        key = (orth, rule_match, rule_count)
        if key not in groups:
            groups[key] = []
        groups[key].append((score, cr, cn, (orth, rule_match, rule_count, count_score)))
    
    # For each group, keep only the top 3 by score
    filtered_candidates = []
    for key, group_candidates in groups.items():
        # Get all items where at least 2 out of the first 3 criteria are 1
        orth, rule_match, rule_count = key
        criteria_sum = orth + rule_match + rule_count
        
        if criteria_sum >= 2:
            # Sort by score descending and take all items
            group_candidates.sort(reverse=True)
            filtered_candidates.extend(group_candidates)
        else:
            # For other cases, sort by score descending and take only top 5
            group_candidates.sort(reverse=True)
            filtered_candidates.extend(group_candidates[:5])
    # Remove records where cr or cn is 0
    filtered_candidates = [c for c in filtered_candidates if c[1] > 0 and c[2] > 0]
    filtered_candidates.sort(reverse=True)
    return [(cr, cn) for _, cr, cn, _ in filtered_candidates[:27]]


def validate_solution(C: int, Pr: float, O: Dict[str, float], Ce: int, Cr: int, Cn: int) -> Dict[str, any]:
    """
    Validate if the solution satisfies all constraints and criteria.
    """
    results = {}
    
    # Check R1: int(0.5 * C) < Cr + Cn <= int(1.5 * C)
    min_total = int(0.5 * C)
    max_total = int(1.5 * C)
    results['R1'] = min_total < (Cr + Cn) <= max_total
    
    # Check optimization criteria
    non_zero_values = [v for v in O.values() if v > 0.0]
    min_o_value = min(non_zero_values) if non_zero_values else 0.1
    
    # Calculate range R for the fourth criterion
    R = max(1, int(C * (1 - Pr) * 0.2))
    target_min = C - R
    target_max = C + R
    
    # Criterion 1 (weight 0.4): min(O values, except 0.0) * Cn >= 1 (for orthograph)
    results['criterion_1_orthograph'] = min_o_value * Cn >= 1
    
    # Criterion 2 (weight 0.3): Cr = int((Cr + Cn) * Pr) (for rule)
    expected_cr = int((Cr + Cn) * Pr)
    results['criterion_2_rule_match'] = Cr == expected_cr
    
    # Criterion 3 (weight 0.2): Cr > Ce (for rule)
    results['criterion_3_rule_count'] = Cr > Ce
    
    # Criterion 4 (weight 0.1): Cr + Cn close to range <C-R, C+R>
    results['criterion_4_in_range'] = target_min <= (Cr + Cn) <= target_max
    results['criterion_4_range'] = f"[{target_min}, {target_max}]"
    
    # Calculate total score
    score = 0.0
    if results['criterion_1_orthograph']:
        score += 0.4
    if results['criterion_2_rule_match']:
        score += 0.3
    if results['criterion_3_rule_count']:
        score += 0.2
    if results['criterion_4_in_range']:
        score += 0.1
    
    results['total_score'] = score
    
    return results


# Example usage and test
if __name__ == "__main__":
    # Test case 1
    # Test case 2
    C = 15
    Pr = 0.51
    O = {'Light': 0.5, 'Medium': 0.5}
    Ce = 1
    
    
    solutions = calculate_rule_counts(C, Pr, O, Ce)
    
    print(f"Test Case 1:")
    print(f"Input: C={C}, Pr={Pr}, O={O}, Ce={Ce}")
    print("\nAll Solutions (Cr, Cn):")
    for i, (cr, cn) in enumerate(solutions, 1):
        validation = validate_solution(C, Pr, O, Ce, cr, cn)
        print(f"\nSolution {i}:")
        print(f"Cr={cr}, Cn={cn}")
        print(f"Validation: {validation}")
    print()