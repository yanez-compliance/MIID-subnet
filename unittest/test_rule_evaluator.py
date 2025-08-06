import unittest

# Dynamically import the standalone rule_evaluator module without 
# executing the heavy MIID package __init__ (which depends on bittensor).
import importlib.util
import pathlib
import sys
import types

# ------------------------------------------------------------------
# Mock bittensor so the rule_evaluator can import it safely.
# ------------------------------------------------------------------
mock_bt = types.ModuleType("bittensor")
class _DummyLog:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
mock_bt.logging = _DummyLog()
sys.modules["bittensor"] = mock_bt

# Path to the rule_evaluator file
RULE_EVAL_PATH = pathlib.Path(__file__).resolve().parents[1] / "MIID" / "validator" / "rule_evaluator.py"

spec = importlib.util.spec_from_file_location("rule_evaluator", RULE_EVAL_PATH)
rule_evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule_evaluator)

# Bring the required functions into the local namespace for readability
is_double_letter_replaced = rule_evaluator.is_double_letter_replaced
is_adjacent_consonants_swapped = rule_evaluator.is_adjacent_consonants_swapped
is_space_replaced_with_special_chars = rule_evaluator.is_space_replaced_with_special_chars
is_vowel_replaced = rule_evaluator.is_vowel_replaced
is_consonant_replaced = rule_evaluator.is_consonant_replaced


class TestRuleEvaluator(unittest.TestCase):
    """Unit-tests for the heuristic rule-evaluator functions."""

    # ------------------------------------------------------------------
    # replace_double_letters_with_single_letter
    # ------------------------------------------------------------------
    def test_double_letter_replaced_true(self):
        self.assertTrue(is_double_letter_replaced("williams", "wiliams"))

    def test_double_letter_replaced_false_no_double_in_original(self):
        self.assertFalse(is_double_letter_replaced("declan", "declan"))

    def test_double_letter_replaced_false_wrong_variation(self):
        # Variation length not reduced by exactly one
        self.assertFalse(is_double_letter_replaced("williams", "williams"))

    # ------------------------------------------------------------------
    # swap_adjacent_consonants
    # ------------------------------------------------------------------
    def test_adjacent_consonants_swapped_true(self):
        # swap the adjacent consonants "cl" -> "lc"
        self.assertTrue(is_adjacent_consonants_swapped("declan", "delcan"))

    def test_adjacent_consonants_swapped_false(self):
        self.assertFalse(is_adjacent_consonants_swapped("declan", "declan"))

    # ------------------------------------------------------------------
    # replace_spaces_with_random_special_characters
    # ------------------------------------------------------------------
    def test_space_replaced_with_special_chars_true(self):
        self.assertTrue(is_space_replaced_with_special_chars("declan williams", "declan_williams"))

    def test_space_replaced_with_special_chars_false(self):
        # Still contains spaces -> should be false
        self.assertFalse(is_space_replaced_with_special_chars("declan williams", "declan williams"))

    # ------------------------------------------------------------------
    # replace_random_vowel_with_random_vowel
    # ------------------------------------------------------------------
    def test_vowel_replaced_true(self):
        self.assertTrue(is_vowel_replaced("declan", "diclan"))  # e -> i

    def test_vowel_replaced_false(self):
        self.assertFalse(is_vowel_replaced("declan", "deklan"))  # consonant change

    # ------------------------------------------------------------------
    # replace_random_consonant_with_random_consonant
    # ------------------------------------------------------------------
    def test_consonant_replaced_true(self):
        self.assertTrue(is_consonant_replaced("declan", "deklan"))  # c -> k

    def test_consonant_replaced_false(self):
        self.assertFalse(is_consonant_replaced("declan", "declan"))


if __name__ == "__main__":
    unittest.main()
