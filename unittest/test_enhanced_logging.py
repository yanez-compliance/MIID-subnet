"""
Test script to demonstrate enhanced logging for query validation.
This shows how failed queries, reasons, and hints are logged.
"""

import sys
import os

# Add the parent directory to the path to import MIID modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_logging():
    """Demonstrate the enhanced logging functionality"""
    
    print("="*80)
    print("ENHANCED LOGGING DEMONSTRATION")
    print("="*80)
    
    print("\n📋 This shows how the enhanced logging will display:")
    print("   - Failed queries with reasons")
    print("   - Issues found by validation")
    print("   - Hints added to clarify queries")
    print("   - Judge LLM success/failure")
    
    print("\n" + "="*80)
    print("EXAMPLE LOG OUTPUT")
    print("="*80)
    
    print("""
🔍 Attempting judge with model: llama3.2:latest and timeout: 60s
✅ Judge succeeded with model: llama3.2:latest and timeout: 60s
   Judge found issues: ['Clarify if 'Additionally' transformations are part of existing variations or separate.']

❌ LLM 'llama3.1:latest' generated INVALID template:
   Failed Query: Generate variations of John Smith
   Reason: Query template must contain exactly one {name} placeholder
   Trying next model/timeout.

⚠️  LLM 'llama3.2:latest' generated query with issues - added clarifications:
   Original Query: Generate exactly 5 variations of {name}, ensuring phonetic similarity...
   Issues Found: ['Specify exact number of variations: 5.', 'Mention rule-based transformation requirement.']
   Added Hints:  Hint: Specify exact number of variations: 5.; Mention rule-based transformation requirement.

✅ LLM 'llama3.1:latest' generated CLEAN query (no issues found)
✅ Successfully generated query with model: llama3.1:latest and timeout: 90s
   Final Query: Generate exactly 15 variations of {name}, ensuring phonetic similarity...

💥 All models and timeouts failed. Falling back to a simple template.
⚠️  Simple template also has issues - added clarifications:
   Issues Found: ['Mention rule-based transformation requirement.']
   Added Hints: Hint: Mention rule-based transformation requirement.
🔄 Using fallback simple template: Generate 15 variations of {name}... Hint: Mention rule-based transformation requirement.
""")
    
    print("\n" + "="*80)
    print("LOG LEVELS AND MEANINGS")
    print("="*80)
    
    print("""
🔍 INFO: Judge LLM attempts and successful validations
✅ INFO: Successful operations (clean queries, successful judge)
⚠️  WARNING: Issues found that need clarification
❌ ERROR: Failed queries, invalid templates, errors
💥 ERROR: Complete failures, fallbacks
⏰ WARNING: Timeout events
""")
    
    print("\n" + "="*80)
    print("BENEFITS OF ENHANCED LOGGING")
    print("="*80)
    
    print("""
1. 🔍 TRANSPARENCY: See exactly why queries failed
2. 🛠️  DEBUGGING: Easy to identify problematic patterns
3. 📊 MONITORING: Track judge LLM success rates
4. 🎯 QUALITY: Monitor query clarity improvements
5. ⚡ TROUBLESHOOTING: Quick identification of timeout/error patterns
""")

if __name__ == "__main__":
    test_enhanced_logging()
