"""
Unit tests for QueryGenerator validation and clarification system.
Tests the enhanced validate_query_template function with real problematic queries from miners.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

# Add the parent directory to the path to import MIID modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MIID.validator.query_generator import QueryGenerator
from MIID.validator.rule_extractor import get_rule_template_and_metadata


class TestQueryGeneratorValidation(unittest.TestCase):
    """Test suite for QueryGenerator validation and clarification functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config object
        self.mock_config = Mock()
        self.mock_config.neuron = Mock()
        self.mock_config.neuron.use_default_query = False
        self.mock_config.neuron.ollama_url = "http://localhost:11434"
        self.mock_config.neuron.ollama_model_name = "llama3.1:latest"
        self.mock_config.neuron.ollama_request_timeout = 30
        self.mock_config.neuron.ollama_fallback_models = ["llama3.2:latest"]
        self.mock_config.neuron.ollama_fallback_timeouts = [45, 60]
        
        # Judge LLM configs
        self.mock_config.neuron.ollama_judge_model = "tinyllama:latest"
        self.mock_config.neuron.ollama_judge_timeout = 10
        self.mock_config.neuron.ollama_judge_fallback_models = ["llama3.2:latest"]
        self.mock_config.neuron.ollama_judge_fallback_timeouts = [15, 20]
        
        # Create QueryGenerator instance
        self.query_gen = QueryGenerator(self.mock_config)
    
    def test_problematic_query_1(self):
        """Test the first problematic query from miner feedback"""
        # The confusing query from the miner
        query = (
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
        
        # Expected labels for this query
        labels = {
            "variation_count": 5,
            "phonetic_similarity": {"Light": 0.7, "Medium": 0.3},
            "orthographic_similarity": {"Light": 0.6, "Medium": 0.3, "Far": 0.1},
            "rule_based": {"percentage": 20}  # Assuming some rule-based percentage
        }
        
        # Mock the ollama client to simulate judge response
        with patch('ollama.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Simulate judge finding the "Additionally" part unclear
            mock_client.generate.return_value = {
                'response': json.dumps({
                    "issues": [
                        "Clarify if the 'Additionally' transformations are part of the 10% Far variations or separate",
                        "Specify the percentage for rule-based transformations"
                    ]
                })
            }
            
            # Validate the query
            is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
            
            # Should be structurally valid (has one {name})
            self.assertTrue(is_valid, f"Query should be structurally valid: {error_msg}")
            
            # Should have issues identified
            self.assertGreater(len(issues), 0, "Should identify issues in the confusing query")
            
            print(f"\n[Test 1] Problematic Query 1 - Issues found: {issues}")
    
    def test_problematic_query_2(self):
        """Test the second problematic query about orthographic settings"""
        query = (
            'Here is the Threat Scenario generation query based on the specifications:\n\n'
            '"Generate exactly 15 execution vectors (name variations) for each target identity {name}, '
            'ensuring phonetic similarity:\n20% of the variations should sound-alike names that are '
            'Lightly similar to {name},\n60% of the variations should sound-alike names that are '
            'Medium similar to {name},\n20% of the variations should sound-alike names that are '
            'Far similar to {name}.\nAdditionally, include orthographic similarity: 100% of the '
            'variations should have visually similar spellings.\nFurthermore, approximately 37% of '
            'the variations should follow these rule-based transformations:\nReplace spaces with '
            'special characters in the name,\nUse first name initial with last name for a portion '
            'of the variations."'
        )
        
        labels = {
            "variation_count": 15,
            "phonetic_similarity": {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
            "orthographic_similarity": {"Light": 0.33, "Medium": 0.34, "Far": 0.33},  # Assumed distribution
            "rule_based": {"percentage": 37}
        }
        
        with patch('ollama.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Judge identifies the orthographic ambiguity
            mock_client.generate.return_value = {
                'response': json.dumps({
                    "issues": [
                        "Specify the distribution of orthographic similarity levels (Light/Medium/Far) for the 100%",
                        "Clarify which orthographic level applies to the variations"
                    ]
                })
            }
            
            is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
            
            self.assertTrue(is_valid, f"Query should be structurally valid: {error_msg}")
            self.assertGreater(len(issues), 0, "Should identify orthographic distribution issue")
            
            print(f"\n[Test 2] Problematic Query 2 - Issues found: {issues}")
    
    def test_problematic_query_3_truncated(self):
        """Test the third problematic query that was truncated"""
        query = (
            "Generate exactly 6 variations of the name {name}, ensuring phonetic similarity "
            "(sound-alike names): 20% with slight pronunciation changes, 60% with moderate "
            "sound shifts, and 20% with significant vocal modifications. Ensure orthographic "
            "similarity (visually similar spellings) for 60% of the variations: 20% with "
            "minimal letter substitutions, 30% with some letter rearrangements, 10% with "
            "additional letter additions. In addition to the above phonetic and orthographic "
            "similarities, generate an additional 34% of variations that follow rule-based "
            "transformations: Additionally, genera"  # Truncated
        )
        
        labels = {
            "variation_count": 6,
            "phonetic_similarity": {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
            "orthographic_similarity": {"Light": 0.2, "Medium": 0.3, "Far": 0.1},
            "rule_based": {"percentage": 34}
        }
        
        with patch('ollama.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Judge identifies truncation and math issues
            mock_client.generate.return_value = {
                'response': json.dumps({
                    "issues": [
                        "Query appears truncated - complete the rule-based transformation description",
                        "Clarify if the 34% rule-based is additional to or part of the existing variations",
                        "The percentages may exceed 100% - clarify the distribution"
                    ]
                })
            }
            
            is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
            
            self.assertTrue(is_valid, f"Query should be structurally valid: {error_msg}")
            self.assertGreater(len(issues), 0, "Should identify truncation and percentage issues")
            
            print(f"\n[Test 3] Truncated Query - Issues found: {issues}")
    
    def test_simple_query_template(self):
        """Test the simple/default query template"""
        # Generate rule template for testing
        rule_percentage = 30
        rule_template, rule_metadata = get_rule_template_and_metadata(rule_percentage)
        
        # The simple template from the code
        clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "
        variation_count = 15
        phonetic_spec = "50% Medium"
        orthographic_spec = "50% Medium"
        
        simple_query = (
            f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}, "
            f"ensuring phonetic similarity: {phonetic_spec}, and orthographic similarity: "
            f"{orthographic_spec}, and also include {rule_percentage}% of variations that "
            f"follow: {rule_template}"
        )
        
        labels = {
            "variation_count": variation_count,
            "phonetic_similarity": {"Medium": 0.5},
            "orthographic_similarity": {"Medium": 0.5},
            "rule_based": {**(rule_metadata or {}), "percentage": rule_percentage}
        }
        
        with patch('ollama.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Judge finds it mostly clear but suggests minor improvements
            mock_client.generate.return_value = {
                'response': json.dumps({
                    "issues": []  # Simple template should be clear
                })
            }
            
            is_valid, error_msg, issues = self.query_gen.validate_query_template(simple_query, labels)
            
            self.assertTrue(is_valid, f"Simple query should be valid: {error_msg}")
            
            # The simple template might still have some minor issues from static checks
            print(f"\n[Test 4] Simple Query Template - Issues found: {issues}")
            
            # If there are issues, verify they get appended correctly
            if issues:
                clarified_query = simple_query + " Hint: " + "; ".join(issues)
                self.assertIn("Hint:", clarified_query)
                print(f"Clarified query would be: {clarified_query[:200]}...")
    
    def test_query_without_name_placeholder(self):
        """Test that queries without {name} placeholder are rejected"""
        query = "Generate 10 variations of the name John Smith"  # Missing {name}
        
        labels = {
            "variation_count": 10,
            "phonetic_similarity": {"Medium": 1.0},
            "orthographic_similarity": {"Medium": 1.0},
            "rule_based": {"percentage": 30}
        }
        
        # No need to mock ollama for this test - should fail before judge
        is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
        
        self.assertFalse(is_valid, "Query without {name} should be invalid")
        self.assertIn("must contain exactly one {name} placeholder", error_msg)
        
        print(f"\n[Test 5] No placeholder - Error: {error_msg}")
    
    def test_query_with_multiple_name_placeholders(self):
        """Test that queries with multiple {name} placeholders are rejected"""
        query = "Generate variations of {name} based on {name}"  # Two {name}
        
        labels = {
            "variation_count": 10,
            "phonetic_similarity": {"Medium": 1.0},
            "orthographic_similarity": {"Medium": 1.0},
            "rule_based": {"percentage": 30}
        }
        
        is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
        
        self.assertFalse(is_valid, "Query with multiple {name} should be invalid")
        self.assertIn("must contain exactly one {name} placeholder", error_msg)
        
        print(f"\n[Test 6] Multiple placeholders - Error: {error_msg}")
    
    def test_clarification_appending(self):
        """Test that clarifications are properly appended to queries"""
        # A query missing several key elements
        query = "Generate variations of the name {name}."
        
        labels = {
            "variation_count": 15,
            "phonetic_similarity": {"Light": 0.3, "Medium": 0.5, "Far": 0.2},
            "orthographic_similarity": {"Light": 0.4, "Medium": 0.4, "Far": 0.2},
            "rule_based": {"percentage": 25}
        }
        
        with patch('ollama.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Judge identifies many missing elements
            mock_client.generate.return_value = {
                'response': json.dumps({
                    "issues": [
                        "Specify the exact number of variations",
                        "Include phonetic similarity distribution",
                        "Include orthographic similarity distribution",
                        "Specify rule-based transformation percentage"
                    ]
                })
            }
            
            is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
            
            self.assertTrue(is_valid, "Query should be structurally valid")
            self.assertGreater(len(issues), 3, "Should identify multiple missing elements")
            
            # Test appending clarifications
            clarified_query = query + " Hint: " + "; ".join(issues)
            
            self.assertIn("Hint:", clarified_query)
            self.assertIn("15", clarified_query)  # Should mention the variation count
            
            print(f"\n[Test 7] Clarification Test")
            print(f"Original: {query}")
            print(f"Issues: {issues}")
            print(f"Clarified: {clarified_query}")
    
    def test_judge_fallback_mechanism(self):
        """Test that the judge fallback mechanism works when primary fails"""
        query = "Generate exactly 10 variations of {name} with phonetic and orthographic similarities."
        
        labels = {
            "variation_count": 10,
            "phonetic_similarity": {"Medium": 1.0},
            "orthographic_similarity": {"Medium": 1.0},
            "rule_based": {"percentage": 30}
        }
        
        with patch('ollama.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Simulate primary judge failing, fallback succeeding
            mock_client.generate.side_effect = [
                Exception("Connection timeout"),  # Primary fails
                Exception("Connection timeout"),  # Primary with longer timeout fails
                {  # Fallback model succeeds
                    'response': json.dumps({
                        "issues": ["Specify the distribution of similarity levels"]
                    })
                }
            ]
            
            is_valid, error_msg, issues = self.query_gen.validate_query_template(query, labels)
            
            self.assertTrue(is_valid, "Should still validate despite primary judge failure")
            
            # Should have called generate multiple times (fallback mechanism)
            self.assertGreater(mock_client.generate.call_count, 1, "Should have tried fallback")
            
            print(f"\n[Test 8] Judge Fallback - Attempts: {mock_client.generate.call_count}")
            print(f"Issues found after fallback: {issues}")


def run_tests():
    """Run all tests and display results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQueryGeneratorValidation)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
