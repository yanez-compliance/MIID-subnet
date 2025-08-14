"""
Standalone test for query validation logic without full dependencies.
This test simulates the validation behavior to demonstrate how the system
handles problematic queries from miners.
"""

import re
from typing import Dict, List, Tuple, Any

def validate_query_template_logic(
    query_template: str,
    labels: Dict[str, Any] = None
) -> Tuple[bool, str, List[str]]:
    """
    Simplified version of validate_query_template for testing.
    This demonstrates the validation logic without external dependencies.
    """
    if not query_template:
        return False, "Query template is empty", []

    # Require at least one {name} placeholder
    placeholder_count = query_template.count("{name}")
    if placeholder_count < 1:
        return False, "Query template must contain at least one {name} placeholder", []

    # Collect non-blocking issues
    issues: List[str] = []

    lowered = query_template.lower()
    
    # Soft checks: absence will be treated as an issue (not a hard error)
    if "phonetic" not in lowered:
        issues.append("Mention phonetic similarity requirements.")
    if "orthographic" not in lowered:
        issues.append("Mention orthographic similarity requirements.")
    if "rule" not in lowered and "transformation" not in lowered:
        issues.append("Mention rule-based transformation requirement.")

    # Label-aware checks to detect missing numbers/levels
    if labels:
        # Variation count
        variation_count = labels.get("variation_count")
        if isinstance(variation_count, int):
            if str(variation_count) not in query_template:
                issues.append(f"Specify exact number of variations: {variation_count}.")

        # Helper to verify percentages and levels
        def compute_expected_percentages(sim_config: Dict[str, float]) -> List[Tuple[str, int]]:
            expected: List[Tuple[str, int]] = []
            for level, frac in sim_config.items():
                try:
                    pct = int(frac * 100)
                    expected.append((level, pct))
                except Exception:
                    continue
            return expected

        def find_percent(text: str, percent: int) -> bool:
            # Look for standalone percentage tokens like "20%" (avoid matching 120% etc.)
            return re.search(rf"(?<!\d){percent}%", text) is not None

        # Phonetic similarity checks
        phonetic_cfg = labels.get("phonetic_similarity") or {}
        if isinstance(phonetic_cfg, dict) and phonetic_cfg:
            for level, pct in compute_expected_percentages(phonetic_cfg):
                if not find_percent(query_template, pct):
                    issues.append(f"Indicate {pct}% share for phonetic '{level}'.")
                if level.lower() not in lowered:
                    issues.append(f"State the phonetic level: {level}.")

        # Orthographic similarity checks
        orthographic_cfg = labels.get("orthographic_similarity") or {}
        if isinstance(orthographic_cfg, dict) and orthographic_cfg:
            for level, pct in compute_expected_percentages(orthographic_cfg):
                if not find_percent(query_template, pct):
                    issues.append(f"Indicate {pct}% share for orthographic '{level}'.")
                if level.lower() not in lowered:
                    issues.append(f"State the orthographic level: {level}.")

        # Rule-based percentage
        rule_meta = labels.get("rule_based") or {}
        rule_pct = rule_meta.get("percentage") if isinstance(rule_meta, dict) else None
        if isinstance(rule_pct, int):
            rule_descriptions_for_this_query = rule_meta.get("rule_descriptions", {}) if isinstance(rule_meta, dict) else {}
            descriptions_list = []
            if isinstance(rule_descriptions_for_this_query, dict):
                descriptions_list = [d for d in rule_descriptions_for_this_query.values() if isinstance(d, str) and d]

            # Reveal the percentage if missing
            percent_present = find_percent(query_template, rule_pct)
            if not percent_present:
                issues.append(f"Approximately {rule_pct}% of the variations should follow rule-based transformations.")

            # Reveal labels (only the ones missing from the query) if any are missing
            if descriptions_list:
                missing_labels = [desc for desc in descriptions_list if desc.lower() not in lowered]
                if missing_labels:
                    issues.append(f"Apply these rule-based transformations: {'; '.join(missing_labels)}.")

            # Duplicate percentage ambiguity -> explicit clarification
            if query_template.count(f"{rule_pct}%") > 1:
                issues.append(
                    (
                        f"We want {rule_pct}% of the name variations to be rule-based. "
                        "Each variation should have at least one transformation rule applied—some may have only one rule, while others may have multiple. "
                        "Importantly, all listed rules must be represented across the set of rule-based name variations."
                    )
                )

    # Simulate judge LLM analysis for specific problematic patterns
    if "additionally" in lowered:
        # The "Additionally" pattern often causes confusion
        if "additionally, generate" in lowered:
            issues.append("Clarify if 'Additionally' transformations are part of existing variations or separate.")
        elif "additionally, include" in lowered and "100%" in query_template:
            issues.append("Clarify the distribution of levels for the 100% requirement.")
    
    # Check for truncated queries
    if query_template.rstrip().endswith(("genera", "transfor", "varia", "includ")):
        issues.append("Query appears truncated - complete the transformation description.")
    
    # Check for percentage math issues
    percentages = re.findall(r'(\d+)%', query_template)
    if percentages:
        total = sum(int(p) for p in percentages)
        if total > 150:  # Allow some overlap but flag excessive totals
            issues.append("The percentages may exceed reasonable bounds - clarify the distribution.")

    # Deduplicate while preserving order
    seen = set()
    deduped_issues: List[str] = []
    for item in issues:
        if item not in seen:
            deduped_issues.append(item)
            seen.add(item)

    return True, "Query template is acceptable with clarifications", deduped_issues


def test_problematic_queries():
    """Test the problematic queries from miners"""
    
    print("="*70)
    print("TESTING QUERY VALIDATION WITH PROBLEMATIC MINER QUERIES")
    print("="*70)
    
    # Test 1: First problematic query with confusing "Additionally" section
    print("\n[TEST 1] Query with confusing 'Additionally' section")
    print("-"*70)
    
    query1 = (
        "Generate exactly 5 execution vectors (name variations) for each target identity {name}, "
        "ensuring phonetic similarity: \n- Implement 70% Light variations by adding or removing one "
        "consonant at a time from the original name, and changing vowel sounds to similar ones "
        "(e.g., 'a' to 'e');\n- Implement 30% Medium variations by replacing one syllable with a "
        "similar-sounding word (e.g., 'Smith' to 'Schmidt').\nFor orthographic similarity:\n"
        "- Implement 60% Light variations by changing the case of individual letters (e.g., 'John' to 'jOhN');\n"
        "- Implement 30% Medium variations by transposing adjacent letters or adding/removing a diacritic mark;\n"
        "- Implement 10% Far variations by using a completely different spelling for the same name, "
        "while keeping its phonetic pronunciation intact.\nAdditionally, generate variations that: \n"
        "- Duplicate a random letter in the original name (e.g., 'John' to 'JoJhn'); \n"
        "- Change one or more letters to their similar-looking counterparts (e.g., 'S' to '5')."
    )
    
    labels1 = {
        "variation_count": 5,
        "phonetic_similarity": {"Light": 0.7, "Medium": 0.3},
        "orthographic_similarity": {"Light": 0.6, "Medium": 0.3, "Far": 0.1},
        "rule_based": {"percentage": 20}
    }
    
    is_valid, error_msg, issues = validate_query_template_logic(query1, labels1)
    
    print(f"Query excerpt: {query1[:150]}...")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
    
    if issues:
        clarified = query1 + "\nHint: " + "; ".join(issues)
        print(f"\nClarified ending: ...{clarified[-200:]}")
    
    # Test 2: Query with 100% orthographic but unclear distribution
    print("\n[TEST 2] Query with 100% orthographic similarity ambiguity")
    print("-"*70)
    
    query2 = (
        'Generate exactly 15 execution vectors (name variations) for each target identity {name}, '
        'ensuring phonetic similarity:\n20% of the variations should sound-alike names that are '
        'Lightly similar,\n60% of the variations should sound-alike names that are '
        'Medium similar,\n20% of the variations should sound-alike names that are '
        'Far similar.\nAdditionally, include orthographic similarity: 100% of the '
        'variations should have visually similar spellings.\nFurthermore, approximately 37% of '
        'the variations should follow these rule-based transformations:\nReplace spaces with '
        'special characters in the name,\nUse first name initial with last name for a portion '
        'of the variations.'
    )
    
    labels2 = {
        "variation_count": 15,
        "phonetic_similarity": {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
        "orthographic_similarity": {"Light": 0.33, "Medium": 0.34, "Far": 0.33},
        "rule_based": {"percentage": 37}
    }
    
    is_valid, error_msg, issues = validate_query_template_logic(query2, labels2)
    
    print(f"Query excerpt: {query2[:150]}...")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
    
    # Test 3: Truncated query
    print("\n[TEST 3] Truncated query")
    print("-"*70)
    
    query3 = (
        "Generate exactly 6 variations of the name {name}, ensuring phonetic similarity "
        "(sound-alike names): 20% with slight pronunciation changes, 60% with moderate "
        "sound shifts, and 20% with significant vocal modifications. Ensure orthographic "
        "similarity (visually similar spellings) for 60% of the variations: 20% with "
        "minimal letter substitutions, 30% with some letter rearrangements, 10% with "
        "additional letter additions. In addition to the above phonetic and orthographic "
        "similarities, generate an additional 34% of variations that follow rule-based "
        "transformations: Additionally, genera"
    )
    
    labels3 = {
        "variation_count": 6,
        "phonetic_similarity": {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
        "orthographic_similarity": {"Light": 0.2, "Medium": 0.3, "Far": 0.1},
        "rule_based": {"percentage": 34}
    }
    
    is_valid, error_msg, issues = validate_query_template_logic(query3, labels3)
    
    print(f"Query ending: ...{query3[-100:]}")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
    
    # Test 4: Simple/clean query template
    print("\n[TEST 4] Simple query template (should be clear)")
    print("-"*70)
    
    query4 = (
        "The following name is the seed name to generate variations for: {name}. "
        "Generate 15 variations of the name, ensuring phonetic similarity: "
        "50% Medium, and orthographic similarity: 50% Medium, and also include 30% "
        "of variations that follow: Replace vowels with numbers (a->4, e->3)"
    )
    
    labels4 = {
        "variation_count": 15,
        "phonetic_similarity": {"Medium": 0.5},
        "orthographic_similarity": {"Medium": 0.5},
        "rule_based": {"percentage": 30}
    }
    
    is_valid, error_msg, issues = validate_query_template_logic(query4, labels4)
    
    print(f"Query: {query4}")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found ({len(issues)}):")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  None - Query is clear!")
    
    # Test 5: Query without {name} placeholder
    print("\n[TEST 5] Query missing {name} placeholder (should fail)")
    print("-"*70)
    
    query5 = "Generate 10 variations of John Smith"
    
    is_valid, error_msg, issues = validate_query_template_logic(query5, None)
    
    print(f"Query: {query5}")
    print(f"Structurally valid: {is_valid}")
    print(f"Error: {error_msg}")
    
    # Test 6: Query with multiple {name} placeholders is now valid
    print("\n[TEST 6] Query with multiple {name} placeholders (should now be VALID)")
    print("-"*70)
    
    query6 = "Generate variations of {name} based on {name}"
    
    is_valid, error_msg, issues = validate_query_template_logic(query6, None)
    
    print(f"Query: {query6}")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found: {issues if issues else 'None'}")
    
    # Test 7: Query with ambiguous rule percentages
    print("\n[TEST 7] Query with ambiguous rule percentages (should be clarified)")
    print("-"*70)

    query7 = """
"Generate 13 execution vectors for each target identity {name}, including phonetic similarity variations:
10% of variations should be 'Light' similarities (e.g. variants that sound similar but not identical, such as 'Smith' becoming 'Schmidt')
30% of variations should be 'Medium' similarities (e.g. variants that are clearly related but have some differences in pronunciation, such as 'Johnson' becoming 'Jenson')
60% of variations should be 'Far' similarities (e.g. variants that sound very different but still share some phonetic characteristics, such as 'Williams' becoming 'Villiers')

Also, generate 100% of variations for orthographic similarity:
Ensure all variations are visually similar to the original name {name}, with possible typos or slight alterations

Additionally, generate variations that perform these rule-based transformations:
Swap adjacent consonants (e.g. 'John' becoming 'Hohn') in approximately 26% of variations
Insert a random letter (e.g. 'Emily' becoming 'Emlyie') in approximately 26% of variations"
"""
    labels7 = {
        "variation_count": 13,
        "phonetic_similarity": {"Light": 0.1, "Medium": 0.3, "Far": 0.6},
        "orthographic_similarity": {"Light": 1.0}, # Assuming 100% Light
        "rule_based": {"percentage": 26}
    }
    
    is_valid, error_msg, issues = validate_query_template_logic(query7, labels7)
    
    print(f"Query excerpt: {query7[:150]}...")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
    
    # Check that our specific clarification was added
    expected_clarification = "We want 26% of the name variations to be rule-based"
    assert any(expected_clarification in issue for issue in issues), f"Expected clarification '{expected_clarification}' not found in issues."

    # Test 8: Query missing rule percentage but includes rule descriptions
    print("\n[TEST 8] Query with rule descriptions but MISSING percentage (should be clarified)")
    print("-"*70)

    query8 = """
"Generate 6 execution vectors for each target identity {name}, ensuring phonetic similarity: 30% of variations are Light (e.g. 'Johne', 'Joane'), 70% are Medium (e.g. 'Jonas', 'Janie'), and orthographic similarity: 33% are Light (e.g. 'Johny', 'Janeo'), 34% are Medium (e.g. 'Johnson', 'Jennyson'), and 33% are Far (e.g. 'Jeanne', 'Jany'). Additionally, generate variations that perform the following rule-based transformations: Delete a random letter (e.g. 'Jo hn' becomes 'Jo h'), Replace spaces with special characters (e.g. 'John Smith' becomes 'Jon_Smith!'), and Swap adjacent syllables (e.g. 'John-Doe' becomes 'Johnd-Oe'). Include all of these requirements in the generated execution vectors for each target identity {name}."
"""
    labels8 = {
        "variation_count": 6,
        "phonetic_similarity": {"Light": 0.3, "Medium": 0.7},
        "orthographic_similarity": {"Light": 0.33, "Medium": 0.34, "Far": 0.33},
        "rule_based": {
            "percentage": 40,  # The query is missing this
            "rule_descriptions": {
                "delete_random_letter": "Delete a random letter",
                "replace_spaces_with_special_characters": "Replace spaces with special characters",
                "swap_adjacent_syllables": "Swap adjacent syllables"
            }
        }
    }
    
    is_valid, error_msg, issues = validate_query_template_logic(query8, labels8)
    
    print(f"Query excerpt: {query8[:150]}...")
    print(f"Structurally valid: {is_valid}")
    print(f"Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
    
    # Check that our specific clarification for this new scenario was added
    expected_clarification = "Approximately 40% of the variations should follow rule-based transformations."
    assert any(expected_clarification in issue for issue in issues), f"Expected clarification '{expected_clarification}' not found in issues."


if __name__ == "__main__":
    # A simple way to run all tests
    test_problematic_queries()
