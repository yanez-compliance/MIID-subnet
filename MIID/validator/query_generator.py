import random
import bittensor as bt
import ollama
from typing import Dict, Any, Tuple, List
import os
import re
import json

# Make sure this import is outside any function or conditional blocks
from faker import Faker  # Ensure this is always imported

# --- Hint formatting helpers ---
def _dedupe_list(items: List[str] | None) -> List[str]:
    if not items:
        return []
    seen: set[str] = set()
    ordered: List[str] = []
    for it in items:
        if it not in seen:
            ordered.append(it)
            seen.add(it)
    return ordered

def _append_hint_section(template: str, tag: str, items: List[str]) -> str:
    items = _dedupe_list(items)
    if not items:
        return template
    section = "\n[" + tag + "]: " + "; ".join(items)
    # Avoid double-appending identical section
    if section in template:
        return template
    return template + section

def _get_keywords_from_rule_desc(description: str) -> List[str]:
    """Extracts keywords from a rule description for flexible matching."""
    # Lowercase, remove punctuation and parentheses content
    processed = description.lower()
    processed = re.sub(r'\(.*?\)', '', processed) # remove content in parens
    processed = re.sub(r'[^\w\s]', '', processed)  # remove punctuation
    
    # Define simple stopwords, can be expanded if needed
    stopwords = {'a', 'an', 'the', 'to', 'of', 'and', 'with', 'etc'}
    
    keywords = [word for word in processed.split() if word not in stopwords]
    return keywords



# List of Latin-script locales to generate names from (basic Latin characters only, no accents)
LATIN_LOCALES = ['en_US', 'en_GB', 'en_CA', 'en_AU']

# Add import for rule-based functionality
from MIID.validator.rule_extractor import get_rule_template_and_metadata

# Constants for query generation
SIMILARITY_LEVELS = ["Light", "Medium", "Far"]
DEFAULT_VARIATION_COUNT = 15
DEFAULT_ORTHOGRAPHIC_SIMILARITY = "Light"
DEFAULT_PHONETIC_SIMILARITY = "Light"
DEFAULT_QUERY = False  # Use simple default query instead of complex LLM-generated one

def _run_judge_model(
    client: ollama.Client, 
    model: str, 
    prompt: str,
    strict_mode: bool,
    # Pass context instead of relying on scope
    soft_issue_map: Dict[str, str],
    phonetic_expected_tokens: List[str],
    orthographic_expected_tokens: List[str],
    variation_count: int,
    rule_pct_val: int,
    rule_descs_list: List[str],
    ambiguity_sentence: str
) -> Tuple[List[str], Dict[str, Any]]:
    """Helper to run a single judge model call and parse the output."""
    text = ""
    parsed = None
    llm_issues = []
    
    resp = client.generate(model=model, prompt=prompt)
    text = resp.get('response', '').strip()
    
    # Try multiple JSON extraction strategies
    try:
        parsed = json.loads(text)
        llm_issues.append("Strategy 1 (Direct JSON): SUCCESS")
    except json.JSONDecodeError as e:
        llm_issues.append(f"Strategy 1 (Direct JSON): FAILED - {str(e)}")
        
        # Strategy 2: Extract JSON from markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                llm_issues.append("Strategy 2 (Markdown Code Block): SUCCESS")
            except json.JSONDecodeError as e:
                llm_issues.append(f"Strategy 2 (Markdown Code Block): FAILED - {str(e)}")
        
        # Strategy 3: Find JSON object in text (expecting a 'present' key)
        if not parsed:
            json_match = re.search(r'\{[^{}]*"present"[^{}]*\{[\s\S]*?\}\s*\}', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    llm_issues.append("Strategy 3 (Pattern Match): SUCCESS")
                except json.JSONDecodeError as e:
                    llm_issues.append(f"Strategy 3 (Pattern Match): FAILED - {str(e)}")
            else:
                llm_issues.append("Strategy 3 (Pattern Match): NO MATCH FOUND")
        
        # Strategy 4: Last resort - try to extract any JSON-like structure
        if not parsed:
            # Look for any JSON object that might contain issues
            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
            llm_issues.append(f"Strategy 4 (General JSON): Found {len(json_objects)} potential JSON objects")
            for i, obj_str in enumerate(json_objects):
                try:
                    temp_parsed = json.loads(obj_str)
                    if isinstance(temp_parsed, dict) and 'present' in temp_parsed:
                        parsed = temp_parsed
                        llm_issues.append(f"Strategy 4 (General JSON): SUCCESS with object {i+1}")
                        break
                except json.JSONDecodeError as e:
                    llm_issues.append(f"Strategy 4 (General JSON): Object {i+1} FAILED - {str(e)}")
                    continue
            if not parsed:
                llm_issues.append("Strategy 4 (General JSON): ALL OBJECTS FAILED")
    
    # Extract structured missing fields and map to canonical issue strings
    if parsed and isinstance(parsed, dict):
        present = parsed.get('present', {})
        mapped_issues: List[str] = []

        if isinstance(present, dict):
            # Soft checks -> map to explicit expected hints when possible
            present_soft = set(present.get('soft', []))
            expected_soft = set(soft_issue_map.keys())
            for key in expected_soft - present_soft:
                if key == 'phonetic' and phonetic_expected_tokens:
                    mapped_issues.append(f"Phonetic similarity: {', '.join(sorted(list(phonetic_expected_tokens)))}.")
                elif key == 'orthographic' and orthographic_expected_tokens:
                    mapped_issues.append(f"Orthographic similarity: {', '.join(sorted(list(orthographic_expected_tokens)))}.")
                elif key == 'rule':
                    # Only add rule soft check if neither the specific rule percentage nor any rule descriptions are detected
                    present_rp = present.get('rule_percentage')
                    present_rd = present.get('rule_descriptions', [])
                    if (isinstance(rule_pct_val, int) and present_rp == rule_pct_val) or (isinstance(present_rd, list) and len(present_rd) > 0):
                        # Treat 'rule' as implicitly satisfied via other fields; do not add a generic soft issue
                        pass
                    else:
                        # Provide more specific rule information when available
                        if isinstance(rule_pct_val, int) and rule_descs_list:
                            mapped_issues.append(f"Apply these rule-based transformations: {'; '.join(rule_descs_list)}.")
                        elif isinstance(rule_pct_val, int):
                            mapped_issues.append(f"Approximately {rule_pct_val}% of the variations should follow rule-based transformations.")
                        elif rule_descs_list:
                            mapped_issues.append(f"Apply these rule-based transformations: {'; '.join(rule_descs_list)}.")
                        else:
                            mapped_issues.append(soft_issue_map[key])
                else:
                    # Fallback to generic message only when no specific data is available
                    mapped_issues.append(soft_issue_map[key])

            # Variation count
            present_vc = present.get('variation_count')
            if isinstance(variation_count, int) and present_vc != variation_count:
                mapped_issues.append(f"Exact number of variations: {variation_count}.")
            
            # Phonetic tokens
            present_phon = set(present.get('phonetic_tokens', []))
            expected_phon = set(phonetic_expected_tokens)
            missing_phon_tokens = expected_phon - present_phon
            if missing_phon_tokens:
                phonetic_expected_str = ", ".join(sorted(list(expected_phon)))
                mapped_issues.append(f"Phonetic similarity: {phonetic_expected_str}.")
            
            # Orthographic tokens
            present_ortho = set(present.get('orthographic_tokens', []))
            expected_ortho = set(orthographic_expected_tokens)
            missing_ortho_tokens = expected_ortho - present_ortho
            if missing_ortho_tokens:
                orthographic_expected_str = ", ".join(sorted(list(expected_ortho)))
                mapped_issues.append(f"Orthographic similarity: {orthographic_expected_str}.")

            # Rule percentage
            present_rp = present.get('rule_percentage')
            if isinstance(rule_pct_val, int) and present_rp != rule_pct_val:
                mapped_issues.append(
                    f"Approximately {rule_pct_val}% of the variations should follow rule-based transformations."
                )

            # Rule descriptions
            present_rd = set(present.get('rule_descriptions', []))
            expected_rd = set(rule_descs_list)
            missing_rd = expected_rd - present_rd
            if missing_rd:
                mapped_issues.append(
                    f"Apply these rule-based transformations: {'; '.join(sorted(list(missing_rd)))}."
                )
            
            # # Ambiguity clarification (judge can signal under key 'rule_ambiguity')
            if present.get('rule_ambiguity') and isinstance(ambiguity_sentence, str):
                mapped_issues.append(ambiguity_sentence)

        # Final deduplication within the judge model to prevent any remaining duplicates
        seen_issues = set()
        deduped_mapped_issues = []
        for issue in mapped_issues:
            if issue not in seen_issues:
                deduped_mapped_issues.append(issue)
                seen_issues.add(issue)
        
        llm_issues = deduped_mapped_issues
    else:
        # If all JSON parsing failed
        if strict_mode:
            raise ValueError(f"Invalid JSON response in strict mode: {text[:200]}...")
        else:
            # Lenient mode: do not try to mine free-text; keep issues empty to avoid non-canonical phrasing
            llm_issues = []
    return llm_issues, parsed


class QueryGenerator:
    """
    Responsible for generating queries and challenges for the name variation validator.
    
    This class handles:
    1. Generating query templates (either default or complex)
    2. Creating sets of test names
    3. Managing the configuration of similarity types and variation counts
    """
    
    def __init__(self, config):
        """Initialize the query generator with the validator's config"""
        self.config = config
        
        # Allow config to override the DEFAULT_QUERY setting
        self.use_default_query = getattr(
            self.config.neuron if hasattr(self.config, 'neuron') else self.config, 
            'use_default_query', 
            DEFAULT_QUERY
        )
        # Cache last successful judge model/timeout for faster subsequent validations
        self.last_successful_judge_model: str | None = None
        self.last_successful_judge_timeout: int | None = None
        
        # Cache last successful generation model/timeout for faster query creation
        self.last_successful_generation_model: str | None = None
        self.last_successful_generation_timeout: int | None = None
        
        # Load sanctioned individuals
        self.sanctioned_individuals = []
        try:
            # Construct the path to the JSON file relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'Sanc_I_DOB_address.json')
            bt.logging.info(f"Loading sanctioned individuals from: {json_path}")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.sanctioned_individuals = json.load(f)
                bt.logging.info(f"Loaded {len(self.sanctioned_individuals)} sanctioned individuals.")
            else:
                bt.logging.error(f"Sanctioned individuals file not found at: {json_path}")
        except Exception as e:
            bt.logging.error(f"Error loading sanctioned individuals: {e}")

        # Load sanctioned countries
        self.sanctioned_countries = []
        try:
            # Construct the path to the JSON file relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'sanctioned_countries.json')
            bt.logging.info(f"Loading sanctioned countries from: {json_path}")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.sanctioned_countries = json.load(f)
                    self.sanctioned_countries = self.sanctioned_countries["sanctionedCountries"]
                bt.logging.info(f"Loaded {len(self.sanctioned_countries)} sanctioned countries.")
            else:
                bt.logging.error(f"Sanctioned countries file not found at: {json_path}")
        except Exception as e:
            bt.logging.error(f"Error loading sanctioned countries: {e}")

        # Flag to control whether to use the LLM judge (default False, but auto-enables for complex queries)
        self.use_judge_model = getattr(
            self.config.neuron if hasattr(self.config, 'neuron') else self.config,
            'use_judge_model',
            False
        )
        
        # Judge strict mode (default is lenient)
        self.judge_strict_mode = getattr(
            self.config.neuron if hasattr(self.config, 'neuron') else self.config,
            'judge_strict_mode',
            False
        )
        
        # Whether to run judge even when static checks pass
        self.judge_on_static_pass = getattr(
            self.config.neuron if hasattr(self.config, 'neuron') else self.config,
            'judge_on_static_pass',
            False
        )
        
        # Judge failure threshold for auto-disable suggestion
        self.judge_failure_threshold = getattr(
            self.config.neuron if hasattr(self.config, 'neuron') else self.config,
            'judge_failure_threshold',
            10
        )
        
        # Auto-enable judge for complex query generation (when not using default query)
        if not self.use_default_query:
            original_setting = self.use_judge_model
            self.use_judge_model = True
            if not original_setting:
                bt.logging.info("📊 Auto-enabling LLM judge because complex query generation is active (use_default_query=False)")
        
        bt.logging.info(f"Use LLM judge model: {self.use_judge_model} (strict mode: {self.judge_strict_mode}, judge_on_static_pass: {self.judge_on_static_pass}, failure threshold: {self.judge_failure_threshold})")

        bt.logging.debug(f"⚙️ use_default_query: {self.use_default_query}")
        bt.logging.info(f"QueryGenerator initialized with use_default_query={self.use_default_query}")

        # Validator will ensure Ollama models are available. No local pulls here.

    def validate_query_template(
        self,
        query_template: str,
        labels: Dict[str, Any] = None,
    ) -> Tuple[bool, str, List[str], List[str], List[str], str, int, Dict[str, Any]]:
        """
        Validate that a query template is structurally valid and semantically covers
        required specifications from labels. Returns issues for missing/unclear parts
        so we can append minimal clarifications to the prompt without revealing the
        entire specification.
            
        Returns:
            Tuple[bool, str, List[str]]: (structurally_valid, error_message, issues)
                - structurally_valid: True only if the template is safe to use (e.g., exactly one {name})
                - error_message: Blocking error if not structurally valid
                - issues: Non-blocking issues (missing/unclear bits) for append-only clarifications
        """
        # Ensure these are always defined for all code paths
        successful_judge_model: str | None = None
        successful_judge_timeout: int | None = None
        llm_issues: List[str] = []

        if not query_template:
            return False, "Query template is empty", [], [], [], successful_judge_model, successful_judge_timeout, {}

        # Require at least one {name} placeholder
        placeholder_count = query_template.count("{name}")
        if placeholder_count == 0:
            # Add a direct hint for missing {name} placeholder
            static_issues = ["Here is the name that you should generate variations of: {name}"]
            return False, "Query template must contain at least one {name} placeholder", static_issues, static_issues, [], successful_judge_model, successful_judge_timeout, {}

        # Collect non-blocking issues
        static_issues: List[str] = []

        lowered = query_template.lower()

        # Label-aware checks to detect missing numbers/levels
        if labels:
            # Combined hint generation for cleaner output
            phonetic_hint = None
            orthographic_hint = None
            rule_hint = None
            ambiguity_hint = None
            variation_count_hint = None
            
            # Variation count - just check if the exact number is present
            variation_count = labels.get("variation_count")
            if isinstance(variation_count, int):
                variation_count_str = str(variation_count)
                # Simply check if the exact number appears in the query
                if variation_count_str not in query_template:
                    variation_count_hint = f"Exact number of variations: {variation_count}."

            # Helper to verify percentages and levels
            def compute_expected_percentages(sim_config: Dict[str, float]) -> List[Tuple[str, int]]:
                expected: List[Tuple[str, int]] = []
                for level, frac in sim_config.items():
                    try:
                        # Match how we render elsewhere: int(frac*100)
                        pct = int(frac * 100)
                        expected.append((level, pct))
                    except Exception:
                        continue
                return expected

            def find_percent(text: str, percent: int) -> bool:
                # Look for standalone percentage tokens like "20%" (avoid matching 120% etc.)
                return re.search(rf"(?<!\d){percent}%", text) is not None

            # Phonetic similarity checks - flexible pattern matching
            phonetic_cfg = labels.get("phonetic_similarity") or {}
            if isinstance(phonetic_cfg, dict) and phonetic_cfg:
                expected_phonetic_tokens = []
                all_found = True
                for level, frac in phonetic_cfg.items():
                    pct = int(frac * 100)
                    token = f"{pct}% {level}"
                    expected_phonetic_tokens.append(token)
                    
                    # More flexible checking - look for percentage and level in proximity
                    # Check if both percentage and level appear in the query (case-insensitive for level)
                    pct_found = f"{pct}%" in query_template
                    level_found = level.lower() in query_template.lower()
                    
                    # Also check for patterns like "70% Light variations" or "incorporating 70% Light"
                    pattern_found = any([
                        f"{pct}% {level}" in query_template,
                        f"{pct}% {level.lower()}" in query_template.lower(),
                        f"incorporating {pct}% {level}" in query_template,
                        f"{pct}% {level} variation" in query_template.lower(),
                    ])
                    
                    if not (pct_found and level_found) and not pattern_found:
                        all_found = False
                        break
                
                if not all_found:
                    phonetic_hint = f"Phonetic similarity: {', '.join(expected_phonetic_tokens)}."

            # Orthographic similarity checks - flexible pattern matching
            orthographic_cfg = labels.get("orthographic_similarity") or {}
            if isinstance(orthographic_cfg, dict) and orthographic_cfg:
                expected_orthographic_tokens = []
                all_found = True
                for level, frac in orthographic_cfg.items():
                    pct = int(frac * 100)
                    token = f"{pct}% {level}"
                    expected_orthographic_tokens.append(token)
                    
                    # More flexible checking - look for percentage and level in proximity
                    # Check if both percentage and level appear in the query (case-insensitive for level)
                    pct_found = f"{pct}%" in query_template
                    level_found = level.lower() in query_template.lower()
                    
                    # Also check for patterns like "70% Light variations" or "incorporating 70% Light"
                    pattern_found = any([
                        f"{pct}% {level}" in query_template,
                        f"{pct}% {level.lower()}" in query_template.lower(),
                        f"incorporating {pct}% {level}" in query_template,
                        f"{pct}% {level} variation" in query_template.lower(),
                    ])
                    
                    if not (pct_found and level_found) and not pattern_found:
                        all_found = False
                        break
                
                if not all_found:
                    orthographic_hint = f"Orthographic similarity: {', '.join(expected_orthographic_tokens)}."

            # Rule-based checks - check for exact percentage and exact rule descriptions
            rule_meta = labels.get("rule_based") or {}
            rule_pct = rule_meta.get("percentage") if isinstance(rule_meta, dict) else None
            if isinstance(rule_pct, int):
                rule_descriptions_for_this_query = rule_meta.get("rule_descriptions", {}) if isinstance(rule_meta, dict) else {}
                descriptions_list: List[str] = []
                if isinstance(rule_descriptions_for_this_query, dict):
                    descriptions_list = [d for d in rule_descriptions_for_this_query.values() if isinstance(d, str) and d]
                
                rule_issues = []
                
                # 1) Check if the exact percentage is present in rule-based context
                # Look for the percentage in contexts that indicate rule-based transformations
                rule_context_patterns = [
                    f"{rule_pct}%",
                    f"approximately {rule_pct}%",
                    f"about {rule_pct}%",
                    f"{rule_pct}% of",
                    f"{rule_pct}% should",
                    f"{rule_pct}% will"
                ]
                rule_pct_found = any(pattern in query_template.lower() for pattern in rule_context_patterns)
                
                if not rule_pct_found:
                    rule_issues.append(f"Approximately {rule_pct}% of the variations should follow rule-based transformations.")
                
                # 2) Check if the rule descriptions are semantically present using keyword matching
                if descriptions_list:
                    missing_rules = []
                    query_lower = query_template.lower()
                    for desc in descriptions_list:
                        keywords = _get_keywords_from_rule_desc(desc)
                        # Check if all keywords are present in the query
                        if not all(keyword in query_lower for keyword in keywords):
                            missing_rules.append(desc)
                    
                    if missing_rules:
                        rule_issues.append(f"Apply these rule-based transformations: {'; '.join(missing_rules)}.")

                if rule_issues:
                    rule_hint = " ".join(rule_issues)
                
                # 3) Ambiguity: percentage mentioned multiple times in potentially confusing contexts (e.g., per-rule). Add explicit clarification
                # Only trigger if percentage appears in contexts that could be ambiguous (not just in clarification text)
                rule_pct_occurrences = query_template.count(f"{rule_pct}%")
                if rule_pct_occurrences > 1:
                    # Check if the multiple occurrences are in potentially confusing contexts
                    # Look for patterns like "X% per rule" or "X% for each transformation" which could be ambiguous
                    confusing_patterns = [
                        f"{rule_pct}% per",
                        f"{rule_pct}% for each",
                        f"{rule_pct}% of each",
                        f"{rule_pct}% per rule",
                        f"{rule_pct}% per transformation"
                    ]
                    has_confusing_context = any(pattern in query_template.lower() for pattern in confusing_patterns)
                    
                    if has_confusing_context:
                        ambiguity_hint = (
                            f"We want {rule_pct}% of the name variations to be rule-based. "
                            "Each variation should have at least one transformation rule applied—some may have only one rule, while others may have multiple. "
                            "Importantly, all listed rules must be represented across the set of rule-based name variations."
                        )
            
            # Add hints to issues list if they are not None
            if variation_count_hint: static_issues.append(variation_count_hint)
            if phonetic_hint: static_issues.append(phonetic_hint)
            if orthographic_hint: static_issues.append(orthographic_hint)
            if rule_hint: static_issues.append(rule_hint)
            if ambiguity_hint: static_issues.append(ambiguity_hint)


        # Early return if static checks passed and judge is not required
        if not static_issues:
            bt.logging.debug(f"🔍 Static checks found no issues. Judge enabled: {self.use_judge_model}, judge_on_static_pass: {getattr(self, 'judge_on_static_pass', False)}")
            # Skip judge if it's disabled entirely OR if we don't require judge on static pass
            if (not self.use_judge_model) or (not getattr(self, 'judge_on_static_pass', False)):
                bt.logging.info("✅ Static checks passed - skipping LLM judge")
                return True, "Query template is acceptable (static checks only)", static_issues, static_issues, [], None, None, {}

        # If static checks found issues, create a temporary template with hints for the judge.
        template_for_judge = query_template
        if static_issues:
            hint_suffix = "\n[STATIC VALIDATION HINTS]: " + "; ".join(static_issues)
            template_for_judge = query_template + hint_suffix
            bt.logging.info("🕵️‍♀️ Static analysis found issues. Appending hints before sending to judge.")
            # bt.logging.debug(f"   Original template: {query_template}")
            # bt.logging.debug(f"   Template for judge: {template_for_judge}")

        # Mandatory LLM judge with robust fallbacks
        llm_issues = []

        # Prepare structured expectations for the judge so we can reconstruct
        # issues using canonical phrasing (explicit expected specs rather than generic mentions).
        soft_issue_map = {
            "phonetic": "Specify phonetic similarity requirements.",
            "orthographic": "Specify orthographic similarity requirements.",
            "rule": "Specify rule-based transformation requirements.",
        }

        variation_count = labels.get("variation_count") if labels else None
        phonetic_cfg = labels.get("phonetic_similarity") if labels else None
        orthographic_cfg = labels.get("orthographic_similarity") if labels else None
        rule_meta = labels.get("rule_based") if labels else None

        # Expected tokens: ["60% Light", ...]
        phonetic_expected_tokens: List[str] = []
        orthographic_expected_tokens: List[str] = []
        rule_pct_val: int | None = None
        rule_descs_list: List[str] = []
        ambiguity_sentence: str | None = None

        if isinstance(phonetic_cfg, dict) and phonetic_cfg:
            try:
                for level, frac in phonetic_cfg.items():
                    phonetic_expected_tokens.append(f"{int(frac * 100)}% {level}")
            except Exception:
                phonetic_expected_tokens = []

        if isinstance(orthographic_cfg, dict) and orthographic_cfg:
            try:
                for level, frac in orthographic_cfg.items():
                    orthographic_expected_tokens.append(f"{int(frac * 100)}% {level}")
            except Exception:
                orthographic_expected_tokens = []

        if isinstance(rule_meta, dict) and rule_meta:
            rp = rule_meta.get("percentage")
            if isinstance(rp, int):
                rule_pct_val = rp
                ambiguity_sentence = (
                    f"We want {rp}% of the name variations to be rule-based. "
                    "Each variation should have at least one transformation rule applied—some may have only one rule, while others may have multiple. "
                    "Importantly, all listed rules must be represented across the set of rule-based name variations."
                )
            rule_descriptions_for_this_query = rule_meta.get("rule_descriptions", {})
            if isinstance(rule_descriptions_for_this_query, dict):
                rule_descs_list = [d for d in rule_descriptions_for_this_query.values() if isinstance(d, str) and d]
        
        # Initialize successful judge tracking to safe defaults
        successful_judge_model = None
        successful_judge_timeout = None
        neuron_cfg = getattr(self.config, 'neuron', self.config)
        primary_judge_model = getattr(neuron_cfg, 'ollama_judge_model', 'mistral:latest')
        judge_fallback_models = getattr(neuron_cfg, 'ollama_judge_fallback_models', [])
        # Prefer cached last success first, then primary, then fallbacks (deduped)
        candidate_models = []
        if self.last_successful_judge_model:
            candidate_models.append(self.last_successful_judge_model)
        candidate_models.append(primary_judge_model)
        candidate_models.extend(judge_fallback_models)
        # Deduplicate while preserving order
        seen_models = set()
        judge_models_to_try = []
        for m in candidate_models:
            if m and m not in seen_models:
                judge_models_to_try.append(m)
                seen_models.add(m)

        primary_judge_timeout = getattr(neuron_cfg, 'ollama_judge_timeout', 60)
        judge_fallback_timeouts = getattr(neuron_cfg, 'ollama_judge_fallback_timeouts', [])
        # Prefer cached last success first, then primary, then fallbacks (deduped)
        candidate_timeouts = []
        if isinstance(self.last_successful_judge_timeout, int):
            candidate_timeouts.append(self.last_successful_judge_timeout)
        candidate_timeouts.append(primary_judge_timeout)
        candidate_timeouts.extend(judge_fallback_timeouts)
        seen_t = set()
        judge_timeouts_to_try = []
        for t in candidate_timeouts:
            if t is not None and t not in seen_t:
                judge_timeouts_to_try.append(t)
                seen_t.add(t)
        judge_timeouts_to_try.sort()
        
        bt.logging.debug(f"🧪 Judge selection order -> models: {judge_models_to_try}, timeouts: {judge_timeouts_to_try}")

        judge_success = False
        llm_issues = [] # Ensure llm_issues is defined
        
        processed_models = set()

        def attempt_judge(model, timeout):
            nonlocal judge_success, llm_issues, successful_judge_model, successful_judge_timeout
            try:
                bt.logging.info(f"🔍 Attempting judge with model: {model} and timeout: {timeout}s")
                client = ollama.Client(host=neuron_cfg.ollama_url, timeout=timeout)
                
                # Provide the judge with structured expectations, asking it to mark what is missing.
                judge_prompt = (
                    "You are a strict but intelligent validator. Your task is to check which of the required SPECIFICATIONS are met by the query TEMPLATE. You must analyze the TEMPLATE semantically.\n\n"
                    "CRITICAL: When you find a specification in the TEMPLATE, you MUST return it EXACTLY as it appears in the SPECIFICATIONS list below.\n\n"
                    f"TEMPLATE:\n{template_for_judge}\n\n"
                    f"SPECIFICATIONS:\n"
                    f"- soft: {json.dumps(list(soft_issue_map.keys()), ensure_ascii=False)}\n"
                    f"- variation_count: {json.dumps(variation_count, ensure_ascii=False)}\n"
                    f"- phonetic_tokens: {json.dumps(phonetic_expected_tokens, ensure_ascii=False)}\n"
                    f"- orthographic_tokens: {json.dumps(orthographic_expected_tokens, ensure_ascii=False)}\n"
                    f"- rule_percentage: {json.dumps(rule_pct_val, ensure_ascii=False)}\n"
                    f"- rule_descriptions: {json.dumps(rule_descs_list, ensure_ascii=False)}\n\n"
                    "OUTPUT RULES:\n"
                    "1. For each specification, check if it's semantically present in the TEMPLATE.\n"
                    "2. If present, include it in your JSON response EXACTLY as shown in SPECIFICATIONS.\n"
                    "3. For phonetic_tokens and orthographic_tokens: if the TEMPLATE mentions these requirements (even with different wording), return ALL tokens from the SPECIFICATIONS list.\n"
                    "4. Example: If SPECIFICATIONS has phonetic_tokens: [\"100% Light\"] and the TEMPLATE mentions \"100% Light phonetic similarity\", return phonetic_tokens: [\"100% Light\"] in your response.\n\n"
                    "EXAMPLE OF FULL RESPONSE:\n"
                    "Step 1: variation_count - The TEMPLATE asks for 12 variations. Specification is 12. PRESENT.\n"
                    "Step 2: phonetic_tokens - The TEMPLATE mentions \"100% Light\" phonetic. Specification has [\"100% Light\"]. PRESENT.\n"
                    "Step 3: orthographic_tokens - The TEMPLATE mentions \"33% Far, 33% Light, 34% Medium\". Specifications match. PRESENT.\n\n"
                    "```json\n"
                    "{\n"
                    "  \"present\": {\n"
                    "    \"variation_count\": 12,\n"
                    "    \"phonetic_tokens\": [\"100% Light\"],\n"
                    "    \"orthographic_tokens\": [\"33% Far\", \"33% Light\", \"34% Medium\"],\n"
                    "    \"rule_percentage\": 34,\n"
                    "    \"rule_descriptions\": [\"Replace spaces in {name} with special characters\"]\n"
                    "  }\n"
                    "}\n"
                    "```\n\n"
                    "YOUR RESPONSE (start with step-by-step reasoning, then end with the final JSON block):"
                )
                
                issues, _ = _run_judge_model(
                    client, model, judge_prompt, self.judge_strict_mode,
                    soft_issue_map, phonetic_expected_tokens, orthographic_expected_tokens,
                    variation_count, rule_pct_val, rule_descs_list, ambiguity_sentence
                )
                
                llm_issues = issues
                judge_success = True
                self.last_successful_judge_model = model
                self.last_successful_judge_timeout = timeout
                successful_judge_model = model
                successful_judge_timeout = timeout
                bt.logging.info(f"✅ Judge succeeded with model: {model}, timeout: {timeout}s")
                return "SUCCESS"
            except Exception as e:
                bt.logging.warning(f"❌ Judge failed: model={model}, timeout={timeout}s, error={e}")
                if "timed out" in str(e).lower():
                    return "TIMEOUT"
                return "FATAL"

        # Step 1: Try cached model with forward-only timeouts
        if self.last_successful_judge_model in judge_models_to_try:
            model = self.last_successful_judge_model
            processed_models.add(model)
            
            start_timeout = self.last_successful_judge_timeout or 0
            timeouts = [t for t in judge_timeouts_to_try if t >= start_timeout]
            
            for timeout in timeouts:
                result = attempt_judge(model, timeout)
                if result == "SUCCESS":
                    break
                if result == "FATAL":
                    break 
            if judge_success:
                 pass # It will skip the next loop

        # Step 2: Try other models
        if not judge_success:
            for model in [m for m in judge_models_to_try if m not in processed_models]:
                for timeout in judge_timeouts_to_try:
                    result = attempt_judge(model, timeout)
                    if result == "SUCCESS":
                        break
                    if result == "FATAL":
                        break 
                if judge_success:
                    break
        
        # Merge static and judge issues, deduplicate while preserving order
        merged_for_dedup = []
        merged_for_dedup.extend(static_issues or [])
        merged_for_dedup.extend(llm_issues or [])

        seen = set()
        deduped_issues: List[str] = []
        for it in merged_for_dedup:
            if it not in seen:
                deduped_issues.append(it)
                seen.add(it)

        # Log deduplication results for debugging
        if static_issues or llm_issues:
            bt.logging.debug(f"🔍 Deduplication: {len(static_issues or [])} static issues + {len(llm_issues or [])} judge issues = {len(deduped_issues)} final issues")
            if len(static_issues or []) + len(llm_issues or []) != len(deduped_issues):
                bt.logging.info(f"🎯 Deduplication removed {len(static_issues or []) + len(llm_issues or []) - len(deduped_issues)} duplicate issues")

        # Log final validation results
        if deduped_issues:
            bt.logging.warning(f"⚠️  Final validation found {len(deduped_issues)} issues:")
            for i, issue in enumerate(deduped_issues, 1):
                bt.logging.warning(f"   {i}. {issue}")
        else:
            bt.logging.info(f"✅ Final validation: No issues found - query is clear")

        validation_details = {
            "static_issues": static_issues,
            "judge_model": successful_judge_model,
            "judge_timeout": successful_judge_timeout,
            "judge_issues": llm_issues,
            "final_issues": deduped_issues,
        }

        return True, "Query template is acceptable with clarifications", deduped_issues, static_issues, llm_issues, successful_judge_model, successful_judge_timeout, validation_details

    # Model pulling is centralized in the validator. No duplicate logic here.
    
    async def generate_complex_query(
        self,
        model_name: str,
        variation_count: int = 10,
        phonetic_similarity: Dict[str, float] = None,
        orthographic_similarity: Dict[str, float] = None,
        dob_similarity: Dict[str, float] = None,
        use_default: bool = False,
        rule_percentage: int = 30
    ) -> Tuple[str, Dict[str, Any], str, int, str, int, Dict[str, Any]]:
        """Generate a query template based on specified parameters"""
        # Default similarity preferences if none provided
        if phonetic_similarity is None:
            phonetic_similarity = {"Medium": 1.0}
        if orthographic_similarity is None:
            orthographic_similarity = {"Medium": 1.0}
        if dob_similarity is None:
            dob_similarity = {"year+month only": 1.0}
        
        # Generate rule-based template and metadata
        rule_template, rule_metadata = get_rule_template_and_metadata(rule_percentage)
        
        # Create the labels dictionary from the parameters
        labels = {
            "variation_count": variation_count,
            "phonetic_similarity": phonetic_similarity,
            "orthographic_similarity": orthographic_similarity,
            "dob_similarity": dob_similarity,
            "rule_based": {**(rule_metadata or {}), "percentage": rule_percentage},  # include percentage for validation
        }
        # Format the similarity specifications for the prompt
        phonetic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in phonetic_similarity.items()])
        orthographic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in orthographic_similarity.items()])
        dob_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in dob_similarity.items()])
        # bt.logging.debug(f"🤖 Generating query with: {variation_count} variations, " +
        #             f"phonetic similarity: {phonetic_spec}, " +
        #             f"orthographic similarity: {orthographic_spec}")
        # bt.logging.debug(f"⚖️ Rule-based requirement: {rule_percentage}% of variations should follow: {rule_template}")

        clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "  
        # Add a clarifying sentence at the beginning to make it clear this is the seed name
        simple_template = f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}, ensuring phonetic similarity: {phonetic_spec}, and orthographic similarity: {orthographic_spec}, and also include {rule_percentage}% of variations that follow: {rule_template}"
        # bt.logging.debug(f"📝 Simple template: {simple_template}")
        
        # Add address and DOB requirements if provided
        address_prefix = f" The following address is the seed country/city to generate address variations for: {{address}}. Generate unique real addresses within the specified country/city for each variation. "
        simple_template = simple_template + address_prefix
    
        dob_prefix = f" The following date of birth is the seed DOB to generate variations for: {{dob}}. Generate DOB variations with phonetic similarity patterns: {dob_spec}"
        simple_template = simple_template + dob_prefix
        
        if use_default:
            bt.logging.info("Using default query template (skipping complex query generation)")
            #clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "
            # Ensure the default template includes the rule_percentage
            # default_template = (
            #     f"{clarifying_prefix}Give me {DEFAULT_VARIATION_COUNT} comma separated alternative spellings "
            #     f"of the name {{name}}. Include 50% of them should Medium sound similar to the original name and 50% "
            #     f"should be Medium orthographically similar. Approximately {rule_percentage}% of the variations "
            #     f"should follow these rule-based transformations: {rule_template}. Provide only the names."
            # )
            # labels = {
            #     "variation_count": DEFAULT_VARIATION_COUNT,
            #     "phonetic_similarity": {"Medium": 0.5},
            #     "orthographic_similarity": {"Medium": 0.5},
            #     "rule_based": {**(rule_metadata or {}), "percentage": rule_percentage}
            # }

            # # Validate and minimally clarify
            # bt.logging.info(f"📝 Pre-judge default template: {simple_template}")
            # # Validate and minimally clarify the default template too
            # _ok, _msg, deduped_issues, static_issues_from_val, llm_issues_from_val, successful_judge_model, successful_judge_timeout, validation_details = self.validate_query_template(simple_template, labels)
            # if deduped_issues:
            #     # Append a single combined validation hint section in the query text
            #     simple_template = _append_hint_section(simple_template, "VALIDATION HINTS", deduped_issues)
                
            #     bt.logging.warning(f"⚠️  Default template has issues - added clarifications:")
            #     if static_issues_from_val:
            #         bt.logging.warning(f"   Static Issues Found: {static_issues_from_val}")
            #     if llm_issues_from_val:
            #         bt.logging.warning(f"   Judge Issues Found: {llm_issues_from_val}")
            # else:
            #     bt.logging.info(f"✅ Default template is clean (no issues found)")

            # bt.logging.debug(f"📄 Using default query template: {simple_template}")
            
            generation_log = {
                "decision": "Used default query as configured.",
                "final_template": simple_template,
                "labels": labels,
                "validation": validation_details,
                "attempts": []  # No attempts for default query
            }
            return simple_template, labels, None, None, successful_judge_model, successful_judge_timeout, generation_log
        
    
        
        # Define the prompt with specific parameters and security terminology
        address_requirement = ""
        dob_requirement = ""
        
        address_requirement = f"""
    5. ADDRESS REQUIREMENTS: The following address is the seed country/city to generate address variations for: {address}. Generate unique real addresses within the specified country/city for each variation. Each variation must have a different, realistic address within the same country/city."""
        
        dob_requirement = f"""
    6. DOB REQUIREMENTS: The following date of birth is the seed DOB to generate variations for: {dob}. Generate DOB variations with these patterns: {dob_spec}"""

        # Only include address/dob in example if they exist
        example_parts = [
            f"Generate {variation_count} variations of {{name}}",
            f"ensuring phonetic similarity ({phonetic_spec}) and orthographic similarity ({orthographic_spec})",
            f"Approximately {rule_percentage}% of the total variations should follow these rule-based transformations: {rule_template}"
        ]

        example_parts.append(f"Address variations for: {{address}}")
        example_parts.append(f"DOB variations for: {{dob}}")

        example_format = ". ".join(example_parts) + "."
        
        prompt = f"""Generate a Threat Scenario generation query with these specifications:

        THREAT SCENARIO CONTEXT:
        You are developing execution vectors to test an sanctions screening system. The regulators expect sanctions screening systems to detect these entities even when these may be trying to obfuscate themselves by providing variation of their names, instead of their actual names.
        These execution vectors will help identify potential vulnerabilities in these systems.
        
        EXECUTION VECTOR SPECIFICATIONS:
        1. Generate exactly {variation_count} execution vectors (name variations) for each target identity
        2. For phonetic similarity (sound-alike names), implement: {phonetic_spec}
        3. For orthographic similarity (visually similar spellings), implement: {orthographic_spec}
        4. For DOB similarity (date of birth variations), implement: {dob_spec}
        5. IMPORTANT: Approximately {rule_percentage}% of the total variations should follow the rule-based transformations below. This percentage applies to the entire group of transformations, not to each one individually. All listed transformations must be represented across the set of rule-based variations.
        Transformations: {rule_template}{address_requirement}{dob_requirement}
        
        IMPORTANT FORMATTING REQUIREMENTS:
        1. The query MUST use {{name}} as a placeholder for the target name.
        2. Format as a natural language request that explicitly states all requirements.
        3. Include both the similarity requirements (phonetic and orthographic) AND the rule-based transformation requirements in the query.
        4. Do not write any SQL queries or any code. Return text only.
        
        CRITICAL: Return ONLY the query template text. Do not include any explanations, analysis, or commentary about the query. Do not say "this query meets requirements" or similar phrases. Just return the actual query template that miners will receive.
        
        Example format: {example_format}
        
        YOUR RESPONSE (query template only):"""
        
        
        # Get the list of models to try: primary + fallbacks, prioritizing last successful
        primary_model = model_name
        fallback_models = getattr(self.config.neuron, 'ollama_fallback_models', [])
        candidate_models = []
        if self.last_successful_generation_model:
            candidate_models.append(self.last_successful_generation_model)
        candidate_models.append(primary_model)
        candidate_models.extend(fallback_models)
        seen_models = set()
        models_to_try = [m for m in candidate_models if m and not (m in seen_models or seen_models.add(m))]

        # Get the list of timeouts to try, prioritizing last successful
        primary_timeout = self.config.neuron.ollama_request_timeout
        fallback_timeouts = getattr(self.config.neuron, 'ollama_fallback_timeouts', [])
        candidate_timeouts = []
        if self.last_successful_generation_timeout:
            candidate_timeouts.append(self.last_successful_generation_timeout)
        candidate_timeouts.append(primary_timeout)
        candidate_timeouts.extend(fallback_timeouts)
        seen_timeouts = set()
        timeouts_to_try = [t for t in candidate_timeouts if t is not None and not (t in seen_timeouts or seen_timeouts.add(t))]
        timeouts_to_try.sort()
        bt.logging.debug(f"🧪 Generation selection order -> models: {models_to_try}, timeouts: {timeouts_to_try}")

        last_model_query_template: str | None = None
        last_successful_judge_model: str | None = None
        last_successful_judge_timeout: int | None = None
        generation_log = { "attempts": [], "decision": "No successful generation.", "final_template": None, "labels": labels }
        
        processed_models = set()

        # Iterate models; enforce forward-only timeouts for cached model
        for model in models_to_try:
            if self.last_successful_generation_model and model == self.last_successful_generation_model and isinstance(self.last_successful_generation_timeout, int):
                current_timeouts = [t for t in timeouts_to_try if t >= self.last_successful_generation_timeout]
            else:
                current_timeouts = timeouts_to_try
            for timeout in current_timeouts:
                attempt_log = {
                    "model": model,
                    "timeout": timeout,
                    "status": "failed",
                    "raw_template": None,
                    "validation": None,
                    "repair_attempt": None
                }
                try:
                    bt.logging.debug(f"🤖 Attempting to generate query with model: {model} and timeout: {timeout}s")
                    # Configure the client with the timeout
                    client = ollama.Client(host=self.config.neuron.ollama_url, timeout=timeout)
                    # Generate the query using Ollama
                    response = client.generate(model=model, prompt=prompt)
                    query_template = response['response'].strip()
                    # Track last attempted model template
                    last_model_query_template = query_template
                    attempt_log["raw_template"] = query_template

                    # Validate and minimally clarify the generated template
                    # bt.logging.debug(f"📝 Pre-judge LLM-generated query: {query_template}")
                    is_valid, error_msg, deduped_issues, static_issues_from_val, llm_issues_from_val, successful_judge_model, successful_judge_timeout, validation_details = self.validate_query_template(query_template, labels)
                    attempt_log["validation"] = validation_details

                    # Persist the successful judge config for later returns
                    last_successful_judge_model = successful_judge_model
                    last_successful_judge_timeout = successful_judge_timeout

                    if not is_valid:
                        bt.logging.error(f"❌ LLM '{model}' generated INVALID template:")
                        # bt.logging.error(f"   Failed Query: {query_template}")
                        # bt.logging.error(f"   Reason: {error_msg}")
                        # Optionally attempt a one-shot repair instead of regeneration
                        if getattr(self.config.neuron, 'enable_repair_prompt', False):
                            try:
                                bt.logging.info("🛠️  Attempting one-shot repair of invalid template using repair prompt")
                                repair_client = ollama.Client(host=self.config.neuron.ollama_url, timeout=self.config.neuron.ollama_request_timeout)
                                
                                # Combine static and LLM-judged issues for a comprehensive repair prompt
                                all_issues_for_repair = deduped_issues
                                
                                repair_log = { "attempted": True, "prompt_issues": all_issues_for_repair, "repaired_template": None, "status": "failed" }

                                repair_prompt = (
                                    "You are a helpful assistant. The following query template is invalid or incomplete.\n"
                                    "Given the template, labels, and detected issues, produce a corrected template that:\n"
                                    "- Includes exactly one {name} placeholder\n"
                                    "- Satisfies the labels\n"
                                    "- Is concise and declarative\n\n"
                                    f"TEMPLATE:\n{query_template}\n\n"
                                    f"LABELS (JSON):\n{json.dumps(labels or {}, ensure_ascii=False)}\n\n"
                                    f"ISSUES:\n{'; '.join(all_issues_for_repair) if all_issues_for_repair else 'N/A'}\n\n"
                                    "Return ONLY the repaired template text."
                                )
                                repair_resp = repair_client.generate(model=model, prompt=repair_prompt)
                                repaired = repair_resp.get('response', '').strip()
                                repair_log["repaired_template"] = repaired
                                if repaired:
                                    # Validate repaired template quickly
                                    _ok2, _msg2, issues2, _static_issues2, _llm_issues2, _, _, _ = self.validate_query_template(repaired, labels)
                                    if issues2:
                                        # Append single combined validation hints to repaired template
                                        repaired = _append_hint_section(repaired, "VALIDATION HINTS", issues2)
                                        bt.logging.warning(f"⚠️  Repaired template still has issues - added clarifications")
                                        bt.logging.warning(f"   Post-Repair Issues: {issues2}")
                                    bt.logging.info("✅ Using repaired template")
                                    repair_log["status"] = "success"
                                    
                                    # Cache successful generation config
                                    self.last_successful_generation_model = model
                                    self.last_successful_generation_timeout = timeout
                                    bt.logging.info(f"📌 Cached generation preference -> model: {model}, timeout: {timeout}s")
                                    
                                    attempt_log["status"] = "success_after_repair"
                                    attempt_log["repair_attempt"] = repair_log
                                    generation_log["attempts"].append(attempt_log)
                                    generation_log["decision"] = "Used repaired template from this attempt."
                                    generation_log["final_template"] = repaired
                                    # Add validation details to generation_log for consistency
                                    generation_log["validation"] = validation_details
                                    return repaired, labels, model, timeout, last_successful_judge_model, last_successful_judge_timeout, generation_log
                            except Exception as rep_e:
                                bt.logging.error(f"Repair attempt failed: {rep_e}")
                                repair_log["status"] = f"failed_with_error: {rep_e}"
                            
                            attempt_log["repair_attempt"] = repair_log

                        if getattr(self.config.neuron, 'regenerate_on_invalid', False):
                            bt.logging.error(f"   Trying next model/timeout.")
                            attempt_log["status"] = "failed_invalid_template"
                            generation_log["attempts"].append(attempt_log)
                            continue  # Try next timeout or model
                        else:
                            # Append available issues as hints (if any were produced before invalidation)
                            if deduped_issues:
                                query_template = _append_hint_section(query_template, "VALIDATION HINTS", deduped_issues)
                                bt.logging.warning(f"⚠️  Invalid template, appended clarifications and proceeding as requested")
                                if static_issues_from_val:
                                    bt.logging.warning(f"   Static Issues Found: {static_issues_from_val}")
                                if llm_issues_from_val:
                                    bt.logging.warning(f"   Judge Issues Found: {llm_issues_from_val}")
                            # Proceed with the current (invalid) query plus hints; miner may still handle
                            bt.logging.info(f"Proceeding without regeneration due to --neuron.regenerate_on_invalid=False")
                            # bt.logging.info(f"✅ Successfully generated query with model: {model} and timeout: {timeout}s (proceeding despite invalid)")
                            # bt.logging.info(f"   Final Query: {query_template}")
                            
                            # Cache successful generation config
                            self.last_successful_generation_model = model
                            self.last_successful_generation_timeout = timeout
                            bt.logging.info(f"📌 Cached generation preference -> model: {model}, timeout: {timeout}s")
                            
                            attempt_log["status"] = "proceeded_with_invalid_template"
                            generation_log["attempts"].append(attempt_log)
                            generation_log["decision"] = "Proceeded with invalid template from this attempt after adding hints."
                            generation_log["final_template"] = query_template
                            # Add validation details to generation_log for consistency
                            generation_log["validation"] = validation_details
                            return query_template, labels, model, timeout, last_successful_judge_model, last_successful_judge_timeout, generation_log

                    if deduped_issues:
                        # Append a single combined validation hint section in the query text
                        query_template = _append_hint_section(query_template, "VALIDATION HINTS", deduped_issues)
                        
                        bt.logging.warning(f"⚠️  LLM '{model}' generated query with issues - added clarifications:")
                        # Original query omitted; template now includes hint sections
                        if static_issues_from_val:
                            bt.logging.warning(f"   Static Issues Found: {static_issues_from_val}")
                        if llm_issues_from_val:
                            bt.logging.warning(f"   Judge Issues Found: {llm_issues_from_val}")
                    else:
                        bt.logging.info(f"✅ LLM '{model}' generated CLEAN query (no issues found)")

                    bt.logging.info(f"Successfully generated query with model: {model} and timeout: {timeout}s")
                    
                    # Cache successful generation config
                    self.last_successful_generation_model = model
                    self.last_successful_generation_timeout = timeout
                    bt.logging.debug(f"💾 Cached generation preference -> model: {model}, timeout: {timeout}s")
                    
                    # bt.logging.debug(f"📄 Final Query: {query_template}")
                    
                    attempt_log["status"] = "success"
                    generation_log["attempts"].append(attempt_log)
                    generation_log["decision"] = "Used template from this attempt."
                    generation_log["final_template"] = query_template
                    # Add validation details to generation_log for consistency
                    generation_log["validation"] = validation_details
                    return query_template, labels, model, timeout, last_successful_judge_model, last_successful_judge_timeout, generation_log

                except Exception as e:
                    bt.logging.error(f"❌ Failed to generate query with model: {model} and timeout: {timeout}s")
                    bt.logging.error(f"   Error: {e}")
                    if "timed out" in str(e).lower():
                        bt.logging.warning("⏰ Timeout occurred. Trying next timeout for this model.")
                        attempt_log["status"] = f"failed_with_timeout: {e}"
                        generation_log["attempts"].append(attempt_log)
                        continue # Move to next timeout
                    else:
                        bt.logging.error(f"💥 An unexpected error occurred. Trying next model.")
                        attempt_log["status"] = f"failed_with_error: {e}"
                        generation_log["attempts"].append(attempt_log)
                        break # break from timeout loop, and try next model.
        
        # Fallback logic remains the same
        bt.logging.error("💥 All models and timeouts failed.")
        # Prefer returning the last LLM-generated template with appended hints
        if last_model_query_template:
            bt.logging.info(f"📝 Finalizing with last LLM-generated template")
            _ok, _msg, deduped_issues, static_issues_from_val, llm_issues_from_val, final_judge_model, final_judge_timeout, _ = self.validate_query_template(last_model_query_template, labels)
            if deduped_issues:
                # Append a single combined validation hint section in the query text
                last_model_query_template = _append_hint_section(last_model_query_template, "VALIDATION HINTS", deduped_issues)
                bt.logging.warning(f"⚠️  Final LLM template has issues - added clarifications:")
                if static_issues_from_val:
                    bt.logging.warning(f"   Static Issues Found: {static_issues_from_val}")
                if llm_issues_from_val:
                    bt.logging.warning(f"   Judge Issues Found: {llm_issues_from_val}")
            else:
                bt.logging.info(f"✅ Final LLM template is clean (no issues found)")
            bt.logging.info(f"🔄 Returning last LLM-generated template without regeneration")
            generation_log["decision"] = "Used last successfully generated (but unvalidated) template as a fallback."
            generation_log["final_template"] = last_model_query_template
            # Create validation details for fallback case
            fallback_validation = {
                "static_issues": static_issues_from_val or [],
                "judge_model": final_judge_model,
                "judge_timeout": final_judge_timeout,
                "judge_issues": llm_issues_from_val or [],
                "final_issues": deduped_issues or []
            }
            generation_log["validation"] = fallback_validation
            return last_model_query_template, labels, None, None, final_judge_model, final_judge_timeout, generation_log
        
        # If we never received an LLM template at all, fall back to simple_template, but validate and append hints
        bt.logging.warning("No LLM-generated template available; using simple fallback template")
        bt.logging.info(f"🔄 Validating simple fallback template")
        _ok, _msg, deduped_issues, static_issues_from_val, llm_issues_from_val, final_judge_model, final_judge_timeout, _ = self.validate_query_template(simple_template, labels)
        if deduped_issues:
            # Append a single combined validation hint section in the query text
            simple_template = _append_hint_section(simple_template, "VALIDATION HINTS", deduped_issues)
        generation_log["decision"] = "Fell back to simple template after all LLM generation attempts failed."
        generation_log["final_template"] = simple_template
        # Create validation details for simple fallback case
        simple_fallback_validation = {
            "static_issues": static_issues_from_val or [],
            "judge_model": final_judge_model,
            "judge_timeout": final_judge_timeout,
            "judge_issues": llm_issues_from_val or [],
            "final_issues": deduped_issues or []
        }
        generation_log["validation"] = simple_fallback_validation
        return simple_template, labels, None, None, final_judge_model, final_judge_timeout, generation_log
    
    async def build_queries(self) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any], str, int, str, int, Dict[str, Any]]:
        """Build challenge queries for miners"""
        try:
            bt.logging.debug("🔄 Building test queries for miners")
            
            # Set up query parameters - randomly select different configurations
            # for each validation round to test miners on various tasks
            
            # 1. Determine variation count (between 5-DEFAULT_VARIATION_COUNT)
            variation_count = random.randint(5, DEFAULT_VARIATION_COUNT)
            
            # 2. Set up phonetic similarity distribution with weighted selection
            phonetic_configs_with_weights = [
                # Balanced distribution - high weight for balanced testing
                ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
                # Focus on Medium similarity - most common real-world scenario
                ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
                # Focus on Far similarity - important for edge cases
                ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
                # Light-Medium mix - moderate weight
                ({"Light": 0.5, "Medium": 0.5}, 0.12),
                # Medium-Far mix - moderate weight
                ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
                # Only Medium similarity - common case
                ({"Medium": 1.0}, 0.08),
                # High Light but not 100% - reduced frequency
                ({"Light": 0.7, "Medium": 0.3}, 0.05),
                # Only Far similarity - edge case
                ({"Far": 1.0}, 0.03),
                # Only Light similarity - reduced frequency
                ({"Light": 1.0}, 0.02),
            ]
            
            # 3. Set up orthographic similarity distribution with weighted selection
            orthographic_configs_with_weights = [
                # Balanced distribution - high weight for balanced testing
                ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
                # Focus on Medium similarity - most common real-world scenario
                ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
                # Focus on Far similarity - important for edge cases
                ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
                # Light-Medium mix - moderate weight
                ({"Light": 0.5, "Medium": 0.5}, 0.12),
                # Medium-Far mix - moderate weight
                ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
                # Only Medium similarity - common case
                ({"Medium": 1.0}, 0.08),
                # High Light but not 100% - reduced frequency
                ({"Light": 0.7, "Medium": 0.3}, 0.05),
                # Only Far similarity - edge case
                ({"Far": 1.0}, 0.03),
                # Only Light similarity - reduced frequency
                ({"Light": 1.0}, 0.02),
            ]

            # 4. Set up DOB similarity distribution with weighted selection
            # Update lines 1217-1236 to match the prompt specification:
            DOB_configs_with_weights = [
                # Balanced distribution
                ({"±1 day": 0.2, "±3 days": 0.2, "±30 days": 0.2, "±90 days": 0.2, "±365 days": 0.1, "year+month only": 0.1}, 0.25),
                # Focus on close variations
                ({"±1 day": 0.3, "±3 days": 0.3, "±30 days": 0.2, "±90 days": 0.1, "±365 days": 0.05, "year+month only": 0.05}, 0.20),
                # Focus on medium variations
                ({"±1 day": 0.1, "±3 days": 0.2, "±30 days": 0.3, "±90 days": 0.2, "±365 days": 0.15, "year+month only": 0.05}, 0.15),
                # Focus on far variations
                ({"±1 day": 0.05, "±3 days": 0.1, "±30 days": 0.2, "±90 days": 0.3, "±365 days": 0.3, "year+month only": 0.05}, 0.15),
                # Only close variations
                ({"±1 day": 0.5, "±3 days": 0.5}, 0.10),
                # Only medium variations
                ({"±30 days": 0.5, "±90 days": 0.5}, 0.08),
                # Only far variations
                ({"±365 days": 0.7, "year+month only": 0.3}, 0.05),
                # Year+month only
                ({"year+month only": 1.0}, 0.02),
            ]
            
            # Helper function for weighted random selection
            def weighted_random_choice(configs_with_weights):
                configs, weights = zip(*configs_with_weights)
                return random.choices(configs, weights=weights, k=1)[0]
            
            # Select configurations using weighted random selection
            phonetic_config = weighted_random_choice(phonetic_configs_with_weights)
            orthographic_config = weighted_random_choice(orthographic_configs_with_weights)
            dob_config = weighted_random_choice(DOB_configs_with_weights)
            
            # 4. Randomly choose rule_percentage for this query (e.g. 10-60%)
            rule_percentage = random.randint(10, 60)
            
            if self.use_default_query:
                bt.logging.info("Using default query template")
                variation_count = 10
                phonetic_config = {"Medium": 0.5}
                orthographic_config = {"Medium": 0.5}
                dob_config = {"year+month only": 0.5}
                rule_percentage = 30  # fallback for default
            
            # Generate a complex query template
            model_name = getattr(self.config.neuron, 'ollama_model_name', "llama3.1:latest")
            query_template, query_labels, successful_model, successful_timeout, successful_judge_model, successful_judge_timeout, generation_log = await self.generate_complex_query(
                model_name=model_name,
                variation_count=variation_count,
                phonetic_similarity=phonetic_config,
                orthographic_similarity=orthographic_config,
                dob_similarity=dob_config,
                use_default=self.use_default_query,
                rule_percentage=rule_percentage
            )
            
            # Get judge settings from the validation process
            # successful_judge_model = None
            # successful_judge_timeout = None
            
            # Generate test names using a mix of sanctioned and generated names
            seed_identities_with_labels = []
            
            # Ensure sample_size exists and has a valid value
            sample_size = getattr(self.config.seed_names, 'sample_size', 15)

            # Number of positive samples to take from the sanctioned list: 1/3 of sample_size, rounded down
            positive_sample_count = sample_size // 3
            
            # Track names to avoid duplicates across positive and negative lists
            seen_names = set()

            # 1. Add positive samples from the sanctioned list (ensure unique full names)
            if self.sanctioned_individuals:
                max_attempts = positive_sample_count * 10
                attempts = 0
                while (
                    len([n for n in seed_identities_with_labels if isinstance(n, dict) and n.get("label") == "positive"]) < positive_sample_count
                    and attempts < max_attempts
                ):
                    person = random.choice(self.sanctioned_individuals)
                    first_name = str(person.get("FirstName", "")).strip()
                    last_name = str(person.get("LastName", "")).strip()
                    dob = str(person.get("DOB", "")).strip()
                    address = str(person.get("Country_Residence", ""))
                    if first_name and last_name:
                        full_name = f"{first_name} {last_name}".lower()
                        if full_name not in seen_names:
                            seed_identities_with_labels.append({"name": full_name, "dob": dob, "address": address, "label": "positive"})
                            seen_names.add(full_name)
                            # bt.logging.info(f"Added positive sample: {full_name}")
                    attempts += 1
                current_positives = len([n for n in seed_identities_with_labels if isinstance(n, dict) and n.get("label") == "positive"]) 
                if current_positives < positive_sample_count:
                    bt.logging.warning(
                        f"Could not collect {positive_sample_count} unique positive samples; using {current_positives}."
                    )
            else:
                bt.logging.warning("Sanctioned individuals list is empty. No positive samples will be added.")
            

            # 2. Add high risk samples from the sanctioned Countries list

            # Number of positive samples to take from the sanctioned list: 1/3 of sample_size, rounded down
            high_risk_sample_count = sample_size // 3
            fake = Faker(LATIN_LOCALES)


            generated_names_high_risk = []
            while len(generated_names_high_risk) < high_risk_sample_count:
                first_name = fake.first_name().lower()
                last_name = fake.last_name().lower()
                dob = fake.date_of_birth(minimum_age=18, maximum_age=100).strftime("%Y-%m-%d")
                address = random.choice(self.sanctioned_countries)
                name = f"{first_name} {last_name}"
                if (name not in generated_names_high_risk and name not in seen_names and 
                    3 <= len(first_name) <= 20 and 
                    3 <= len(last_name) <= 20):
                    generated_names_high_risk.append(name)
                    seen_names.add(name)
                    # bt.logging.debug(f"📝 Generated negative two-part name: {name}")
            
            # Add generated names to the list with "negative" label
            for name in generated_names_high_risk:
                seed_identities_with_labels.append({"name": name, "dob": dob, "address": address, "label": "High Risk"})
            
        
            
            # 3. Add negative samples generated by Faker (always two-part names)
            negative_sample_count = sample_size - len(seed_identities_with_labels)
            
            generated_names = []
            while len(generated_names) < negative_sample_count:
                first_name = fake.first_name().lower()
                last_name = fake.last_name().lower()
                dob = fake.date_of_birth(minimum_age=18, maximum_age=100).strftime("%Y-%m-%d")
                
                address = fake.country()
                while address not in self.sanctioned_countries:
                    address = fake.country()

                name = f"{first_name} {last_name}"
                if (name not in generated_names and name not in seen_names and 
                    3 <= len(first_name) <= 20 and 
                    3 <= len(last_name) <= 20):
                    generated_names.append(name)
                    seen_names.add(name)
                    # bt.logging.debug(f"📝 Generated negative two-part name: {name}")
            
            # Add generated names to the list with "negative" label
            for name in generated_names:
                seed_identities_with_labels.append({"name": name, "dob": dob, "address": address, "label": "negative"})
            
            # Shuffle the list to mix positive and negative samples
            random.shuffle(seed_identities_with_labels)
            
            # Log the final list of seed names with their labels for traceability
            log_output = [f"'{item['name']}' ({item['label']})" for item in seed_identities_with_labels]
            # bt.logging.debug(f"📋 Generated {len(seed_identities_with_labels)} test names: [{', '.join(log_output)}]")
            
            # bt.logging.debug(f"📄 Query template: {query_template}")
            # bt.logging.debug(f"📋 Query labels: {query_labels}")
            
            # The function now returns a list of dictionaries, so we extract just the names for the return
            seed_names = [item['name'] for item in seed_identities_with_labels]
            return seed_identities_with_labels, query_template, query_labels, successful_model, successful_timeout, successful_judge_model, successful_judge_timeout, generation_log
            
        except Exception as e:
            bt.logging.error(f"Error building queries: {str(e)}")
            
            # Fallback to simple defaults
            variation_count = DEFAULT_VARIATION_COUNT
            phonetic_config = {"Medium": 0.5}
            orthographic_config = {"Medium": 0.5}
            # Fallback rule-based percentage
            rp = 30
            # Generate rule-based template and metadata for fallback
            rule_template, rule_metadata = get_rule_template_and_metadata(rp)
            
            # Build labels first
            query_labels = {
                "variation_count": variation_count,
                "phonetic_similarity": phonetic_config,
                "orthographic_similarity": orthographic_config,
                "rule_based": {**rule_metadata, "percentage": rp} if isinstance(rule_metadata, dict) else {"percentage": rp}
            }

            # Add clarifying sentence to fallback template
            clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "
            query_template = f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}, ensuring phonetic similarity: {phonetic_config}, and orthographic similarity: {orthographic_config}, and also include {rp}% of variations that follow: {rule_template}."
            
            # # Validate and minimally clarify the fallback template
            # _ok, _msg, issues, _llm_issues, _, _, _ = self.validate_query_template(query_template, query_labels)
            # if issues:
            #     # Organize hints for exception fallback
            #     exception_hints = []
            #     static_issues = [i for i in issues if i not in _llm_issues] if _llm_issues else issues
            #     judge_only_issues = [i for i in _llm_issues if i not in issues] if _llm_issues else []
                
            #     if static_issues:
            #         exception_hints.append("\n[EXCEPTION FALLBACK STATIC HINTS]: " + "; ".join(static_issues))
            #     if judge_only_issues:
            #         exception_hints.append("\n[EXCEPTION FALLBACK JUDGE HINTS]: " + "; ".join(judge_only_issues))
                
            #     suffix = "".join(exception_hints) if exception_hints else "\n[EXCEPTION FALLBACK HINTS]: " + "; ".join(issues)
            #     query_template = query_template + suffix
            
            # Generate fallback names (always two-part names)
            fake = Faker(LATIN_LOCALES)
            seed_names = []
            
            # Use the same sample size for fallback
            fallback_sample_size = getattr(self.config.seed_names, 'sample_size', 15)
            
            while len(seed_names) < fallback_sample_size:
                first_name = fake.first_name().lower()
                last_name = fake.last_name().lower()
                name = f"{first_name} {last_name}"
                
                # Randomly decide whether to add a title (1/10 chance)
                # TODO: add title to the query template in V1.2
                # if random.choice([True] + [False] * 9):
                #     prefix = fake.prefix().replace('.', '').lower()
                #     name = f"{prefix} {name}"
                
                if name not in seed_names:
                    seed_names.append(name)
            
            # In fallback, all names are "negative"
            seed_names_with_labels = [{"name": name, "label": "negative"} for name in seed_names]
            
            # bt.logging.info(f"Using fallback: {len(seed_names)} test names")
            # bt.logging.debug(f"📄 Query template: {query_template}")
            # bt.logging.debug(f"📋 Query labels: {query_labels}")
            
            fallback_log = {
                "decision": "Used fallback query generation due to an exception.",
                "error": str(e),
                "final_template": query_template,
                "labels": query_labels,
            }
            return seed_names_with_labels, query_template, query_labels, None, None, None, None, fallback_log
