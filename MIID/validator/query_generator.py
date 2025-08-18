import random
import bittensor as bt
import ollama
from typing import Dict, Any, Tuple, List
import os
import re
import json

# Make sure this import is outside any function or conditional blocks
from faker import Faker  # Ensure this is always imported

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
            json_path = os.path.join(current_dir, 'SanctionedIndividulas.json')
            bt.logging.info(f"Loading sanctioned individuals from: {json_path}")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.sanctioned_individuals = json.load(f)
                bt.logging.info(f"Loaded {len(self.sanctioned_individuals)} sanctioned individuals.")
            else:
                bt.logging.error(f"Sanctioned individuals file not found at: {json_path}")
        except Exception as e:
            bt.logging.error(f"Error loading sanctioned individuals: {e}")

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

        bt.logging.info(f"use_default_query: {self.use_default_query}#########################################")
        bt.logging.info(f"QueryGenerator initialized with use_default_query={self.use_default_query}")
    
    def validate_query_template(
        self,
        query_template: str,
        labels: Dict[str, Any] = None,
    ) -> Tuple[bool, str, List[str], List[str], str, int]:
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

        if not query_template:
            return False, "Query template is empty", [], [], successful_judge_model, successful_judge_timeout

        # Require at least one {name} placeholder
        placeholder_count = query_template.count("{name}")
        if placeholder_count == 0:
            return False, "Query template must contain at least one {name} placeholder", [], [], successful_judge_model, successful_judge_timeout

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
                        # Match how we render elsewhere: int(frac*100)
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
                missing_phonetic = []
                for level, pct in compute_expected_percentages(phonetic_cfg):
                    # If either the percent or level indicator is absent, request clarification
                    if not find_percent(query_template, pct):
                        missing_phonetic.append(f"{pct}% {level}")
                    # Allow synonyms like 'lightly' for Light; keep it soft by only checking when level token is fully missing
                    elif level.lower() not in lowered:
                        # Only add if not already covered by a more general phrase
                        missing_phonetic.append(f"{pct}% {level}")
                
                if missing_phonetic:
                    issues.append(f"Phonetic similarity: {', '.join(missing_phonetic)}.")

            # Orthographic similarity checks
            orthographic_cfg = labels.get("orthographic_similarity") or {}
            if isinstance(orthographic_cfg, dict) and orthographic_cfg:
                missing_orthographic = []
                for level, pct in compute_expected_percentages(orthographic_cfg):
                    if not find_percent(query_template, pct):
                        missing_orthographic.append(f"{pct}% {level}")
                    elif level.lower() not in lowered:
                        missing_orthographic.append(f"{pct}% {level}")
                
                if missing_orthographic:
                    issues.append(f"Orthographic similarity: {', '.join(missing_orthographic)}.")

            # Rule-based percentage
            rule_meta = labels.get("rule_based") or {}
            rule_pct = rule_meta.get("percentage") if isinstance(rule_meta, dict) else None
            if isinstance(rule_pct, int):
                rule_descriptions_for_this_query = rule_meta.get("rule_descriptions", {}) if isinstance(rule_meta, dict) else {}
                descriptions_list: List[str] = []
                if isinstance(rule_descriptions_for_this_query, dict):
                    descriptions_list = [d for d in rule_descriptions_for_this_query.values() if isinstance(d, str) and d]
                
                # Detect presence of percentage token in the query
                percent_present = find_percent(query_template, rule_pct)
                
                # 1) If percentage is missing, reveal the percentage explicitly
                if not percent_present:
                    issues.append(f"Approximately {rule_pct}% of the variations should follow rule-based transformations.")
                
                # 2) If any of the specific labels are missing from the query text, add labels-only hint
                if descriptions_list:
                    def _label_present(desc: str) -> bool:
                        """Generic check for presence of a rule label in the query text."""
                        if not desc:
                            return False
                        d_low = desc.lower()
                        # 1. Get the canonical core phrase (before any parentheses)
                        canonical_core = d_low.split('(')[0].strip()

                        # 2. Direct check first for performance and simple cases.
                        if canonical_core in lowered:
                            return True

                        # 3. If direct check fails, use a more robust word-based check.
                        stopwords = {'a', 'an', 'the', 'of', 'in', 'with', 'for', 'to', 'is', 'are', 'etc'}
                        canonical_words = {word for word in re.split(r'[^a-z]+', canonical_core) if word and word not in stopwords}

                        if not canonical_words:
                            return False # Cannot validate on empty set of words

                        # Tokenize the query once for efficiency (if not already done)
                        query_words = set(re.split(r'[^a-z]+', lowered))

                        # Check if for each canonical word, a word in the query starts with it.
                        for c_word in canonical_words:
                            if not any(q_word.startswith(c_word) for q_word in query_words):
                                return False # A significant word is missing
                        
                        return True # All significant words were found

                    missing_labels = [desc for desc in descriptions_list if not _label_present(desc)]
                    if missing_labels:
                        issues.append(f"Apply these rule-based transformations: {'; '.join(missing_labels)}.")
                
                # 3) Ambiguity: percentage mentioned multiple times (e.g., per-rule). Add explicit clarification
                if query_template.count(f"{rule_pct}%") > 1:
                    issues.append(
                        (
                            f"We want {rule_pct}% of the name variations to be rule-based. "
                            "Each variation should have at least one transformation rule applied—some may have only one rule, while others may have multiple. "
                            "Importantly, all listed rules must be represented across the set of rule-based name variations."
                        )
                    )

        # Early return if static checks passed and judge is not required
        if not issues:
            # Skip judge if it's disabled entirely OR if we don't require judge on static pass
            if (not self.use_judge_model) or (not getattr(self, 'judge_on_static_pass', False)):
                bt.logging.info("✅ Static checks passed - skipping LLM judge")
                return True, "Query template is acceptable (static checks only)", issues, [], None, None

        # Mandatory LLM judge with robust fallbacks
        llm_issues = []
        # Initialize successful judge tracking to safe defaults
        successful_judge_model = None
        successful_judge_timeout = None
        neuron_cfg = getattr(self.config, 'neuron', self.config)
        primary_judge_model = getattr(neuron_cfg, 'ollama_judge_model', 'llama3.1:latest')
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

        judge_success = False
        for judge_model in judge_models_to_try:
            for judge_timeout in judge_timeouts_to_try:
                try:
                    bt.logging.info(f"🔍 Attempting judge with model: {judge_model} and timeout: {judge_timeout}s")
                    client = ollama.Client(host=neuron_cfg.ollama_url, timeout=judge_timeout)
                    
                    judge_prompt = (
                        "You are a strict validator. Analyze the query template against the provided labels and return ONLY a valid JSON object.\n\n"
                        f"TEMPLATE:\n{query_template}\n\n"
                        f"LABELS (JSON):\n{json.dumps(labels or {}, ensure_ascii=False)}\n\n"
                        "INSTRUCTIONS:\n"
                        "1. Return ONLY a JSON object with an 'issues' array\n"
                        "2. List only MINIMAL missing or unclear elements that need clarification\n"
                        "3. Do not restate what is already clearly present\n"
                        "4. If no issues found, return: {\"issues\": []}\n"
                        "5. Do not include any text before or after the JSON\n\n"
                        "EXAMPLE OUTPUT:\n"
                        "{\"issues\": [\"Specify exact number of variations: 10.\", \"Mention phonetic similarity requirements.\"]}\n\n"
                        "RESPONSE (JSON only):"
                    )
                    
                    # Log the prompt being sent (for debugging)
                    bt.logging.debug(f"🔍 Judge prompt preview: {judge_prompt[:200]}...")
                    resp = client.generate(model=judge_model, prompt=judge_prompt)
                    text = resp.get('response', '').strip()
                    
                    # Log the raw response for debugging (truncated if too long)
                    if len(text) > 500:
                        bt.logging.debug(f"🔍 Judge raw response (truncated): {text[:500]}...")
                    else:
                        bt.logging.debug(f"🔍 Judge raw response: {text}")
                    
                    # Try multiple JSON extraction strategies
                    parsed = None
                    llm_issues = []
                    extraction_debug = []
                    
                    # Strategy 1: Direct JSON parsing
                    try:
                        parsed = json.loads(text)
                        extraction_debug.append("Strategy 1 (Direct JSON): SUCCESS")
                    except json.JSONDecodeError as e:
                        extraction_debug.append(f"Strategy 1 (Direct JSON): FAILED - {str(e)}")
                        
                        # Strategy 2: Extract JSON from markdown code blocks
                        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                        if code_block_match:
                            try:
                                parsed = json.loads(code_block_match.group(1))
                                extraction_debug.append("Strategy 2 (Markdown Code Block): SUCCESS")
                            except json.JSONDecodeError as e:
                                extraction_debug.append(f"Strategy 2 (Markdown Code Block): FAILED - {str(e)}")
                        
                        # Strategy 3: Find JSON object in text
                        if not parsed:
                            json_match = re.search(r'\{[^{}]*"issues"[^{}]*\[[^\]]*\]\s*\}', text)
                            if json_match:
                                try:
                                    parsed = json.loads(json_match.group(0))
                                    extraction_debug.append("Strategy 3 (Pattern Match): SUCCESS")
                                except json.JSONDecodeError as e:
                                    extraction_debug.append(f"Strategy 3 (Pattern Match): FAILED - {str(e)}")
                            else:
                                extraction_debug.append("Strategy 3 (Pattern Match): NO MATCH FOUND")
                        
                        # Strategy 4: Last resort - try to extract any JSON-like structure
                        if not parsed:
                            # Look for any JSON object that might contain issues
                            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
                            extraction_debug.append(f"Strategy 4 (General JSON): Found {len(json_objects)} potential JSON objects")
                            for i, obj_str in enumerate(json_objects):
                                try:
                                    temp_parsed = json.loads(obj_str)
                                    if isinstance(temp_parsed, dict) and 'issues' in temp_parsed:
                                        parsed = temp_parsed
                                        extraction_debug.append(f"Strategy 4 (General JSON): SUCCESS with object {i+1}")
                                        break
                                except json.JSONDecodeError as e:
                                    extraction_debug.append(f"Strategy 4 (General JSON): Object {i+1} FAILED - {str(e)}")
                                    continue
                            if not parsed:
                                extraction_debug.append("Strategy 4 (General JSON): ALL OBJECTS FAILED")
                    
                    # Extract issues from parsed JSON
                    if parsed and isinstance(parsed, dict):
                        llm_issues = parsed.get('issues', [])
                        if not isinstance(llm_issues, list):
                            llm_issues = []
                    else:
                        # If all JSON parsing failed, handle based on strict mode
                        if self.judge_strict_mode:
                            # In strict mode, fail on JSON parsing errors
                            raise ValueError(f"Invalid JSON response in strict mode: {text[:200]}...")
                        else:
                            # In lenient mode, try to extract issues from text
                            bt.logging.warning(f"🔧 JSON parsing failed for judge response, attempting text extraction (lenient mode)")
                            bt.logging.info(f"🔧 Text extraction debug info:")
                            bt.logging.info(f"   Raw text length: {len(text)} characters")
                            bt.logging.info(f"   Text preview: {text[:200]}...")
                            
                            # Look for common issue patterns in the text
                            issue_patterns = [
                                r'(?:you should|must|need to).*?(?:\.|$)',
                                r'(?:specify|mention|require|add|include|clarify).*?(?:\.|$)',
                                r'(?:missing|unclear|incomplete).*?(?:\.|$)'
                            ]
                            extracted_issues = []
                            for i, pattern in enumerate(issue_patterns):
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                bt.logging.info(f"   Pattern {i+1} matches: {len(matches)}")
                                for match in matches:
                                    if len(match.strip()) > 10:
                                        extracted_issues.append(match.strip())
                                        bt.logging.info(f"     - Found: {match.strip()}")
                            
                            if extracted_issues:
                                llm_issues = extracted_issues[:3]  # Limit to 3 most relevant
                                bt.logging.info(f"🔧 Extracted {len(llm_issues)} issues from text: {llm_issues}")
                            else:
                                # If no issues found in text, assume no issues
                                llm_issues = []
                                bt.logging.info(f"🔧 No issues extracted from text, assuming query is acceptable")
                    if isinstance(llm_issues, list):
                        judge_success = True
                        # Cache successful model/timeout for next time
                        self.last_successful_judge_model = judge_model
                        self.last_successful_judge_timeout = judge_timeout
                        # Record successful judge configuration for caller
                        successful_judge_model = judge_model
                        successful_judge_timeout = judge_timeout
                        bt.logging.info(f"📌 Cached judge preference -> model: {judge_model}, timeout: {judge_timeout}s")
                        bt.logging.info(f"✅ Judge succeeded with model: {judge_model} and timeout: {judge_timeout}s")
                        if llm_issues:
                            bt.logging.info(f"   Judge found issues: {llm_issues}")
                        else:
                            bt.logging.info(f"   Judge found no issues - query is clear")
                        break
                except Exception as e:
                    error_msg = str(e)
                    bt.logging.warning(f"❌ Judge failed with model: {judge_model} and timeout: {judge_timeout}s. Error: {error_msg}")
                    
                    # Enhanced error logging for JSON parsing issues
                    if "invalid json" in error_msg.lower() or "jsondecodeerror" in error_msg.lower():
                        bt.logging.error(f"🔍 JSON Parsing Debug Info:")
                        bt.logging.error(f"   Raw response length: {len(text)} characters")
                        bt.logging.error(f"   Response preview: {text[:300]}...")
                        if len(text) > 300:
                            bt.logging.error(f"   Response end: ...{text[-100:]}")
                        
                        # Show what parsing strategies were attempted
                        bt.logging.error(f"   JSON extraction attempts:")
                        for debug_line in extraction_debug:
                            bt.logging.error(f"     - {debug_line}")
                        
                        # Check if response contains JSON-like content
                        if '{' in text and '}' in text:
                            bt.logging.error(f"     - Contains JSON brackets: YES")
                            # Try to find where the JSON might be
                            brace_positions = []
                            for i, char in enumerate(text):
                                if char == '{':
                                    brace_positions.append(f"opening at pos {i}")
                                elif char == '}':
                                    brace_positions.append(f"closing at pos {i}")
                            if brace_positions:
                                bt.logging.error(f"     - Brace positions: {brace_positions[:10]}...")
                        else:
                            bt.logging.error(f"     - Contains JSON brackets: NO")
                        
                        # Check for common LLM response patterns
                        if text.lower().startswith('i apologize') or text.lower().startswith('sorry'):
                            bt.logging.error(f"     - Response type: APOLOGY (model may be refusing task)")
                        elif '```' in text:
                            bt.logging.error(f"     - Response type: MARKDOWN CODE BLOCK")
                        elif text.strip().startswith('{'):
                            bt.logging.error(f"     - Response type: STARTS WITH JSON")
                        else:
                            bt.logging.error(f"     - Response type: NATURAL LANGUAGE")
                    
                    if "timed out" in error_msg.lower():
                        bt.logging.warning("⏰ Judge timeout - trying next timeout/model")
                        continue
                    else:
                        bt.logging.error(f"💥 Judge error - trying next model")
                        break
            if judge_success:
                break

        if not judge_success:
            bt.logging.error("💥 All judge models and timeouts failed. Proceeding with static checks only.")
            llm_issues = []
            successful_judge_model = None
            successful_judge_timeout = None
            
            # If judge consistently fails, consider disabling it for future runs
            if not hasattr(self, '_judge_failure_count'):
                self._judge_failure_count = 0
            self._judge_failure_count += 1
            
            # After threshold consecutive failures, suggest disabling judge
            if self._judge_failure_count >= self.judge_failure_threshold:
                bt.logging.warning(f"⚠️  Judge has failed {self._judge_failure_count} times consecutively (threshold: {self.judge_failure_threshold}). Consider setting --neuron.use_judge_model=False to disable LLM judge validation.")
        else:
            # Reset failure count on success
            if hasattr(self, '_judge_failure_count'):
                self._judge_failure_count = 0

        # for it in llm_issues:
        #     if isinstance(it, str) and it not in issues:
        #         issues.append(it)

        # Deduplicate while preserving order
        seen = set()
        deduped_issues: List[str] = []
        for it in issues:
            if it not in seen:
                deduped_issues.append(it)
                seen.add(it)

        # Log final validation results
        if deduped_issues:
            bt.logging.warning(f"⚠️  Final validation found {len(deduped_issues)} issues:")
            for i, issue in enumerate(deduped_issues, 1):
                bt.logging.warning(f"   {i}. {issue}")
        else:
            bt.logging.info(f"✅ Final validation: No issues found - query is clear")

        return True, "Query template is acceptable with clarifications", deduped_issues, llm_issues, successful_judge_model, successful_judge_timeout
    
    async def generate_complex_query(
        self,
        model_name: str,
        variation_count: int = 10,
        phonetic_similarity: Dict[str, float] = None,
        orthographic_similarity: Dict[str, float] = None,
        use_default: bool = False,
        rule_percentage: int = 30
    ) -> Tuple[str, Dict[str, Any], str, int, str, int]:
        """Generate a query template based on specified parameters"""
        # Default similarity preferences if none provided
        if phonetic_similarity is None:
            phonetic_similarity = {"Medium": 1.0}
        if orthographic_similarity is None:
            orthographic_similarity = {"Medium": 1.0}
        
        # Generate rule-based template and metadata
        rule_template, rule_metadata = get_rule_template_and_metadata(rule_percentage)
        
        # Create the labels dictionary from the parameters
        labels = {
            "variation_count": variation_count,
            "phonetic_similarity": phonetic_similarity,
            "orthographic_similarity": orthographic_similarity,
            "rule_based": {**(rule_metadata or {}), "percentage": rule_percentage}  # include percentage for validation
        }
        # Format the similarity specifications for the prompt
        phonetic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in phonetic_similarity.items()])
        orthographic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in orthographic_similarity.items()])
        
        bt.logging.info(f"Generating query with: {variation_count} variations, " +
                    f"phonetic similarity: {phonetic_spec}, " +
                    f"orthographic similarity: {orthographic_spec}")
        bt.logging.info(f"Rule-based requirement: {rule_percentage}% of variations should follow: {rule_template}")

        clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "  
        # Add a clarifying sentence at the beginning to make it clear this is the seed name
        simple_template = f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}, ensuring phonetic similarity: {phonetic_spec}, and orthographic similarity: {orthographic_spec}, and also include {rule_percentage}% of variations that follow: {rule_template}"
        bt.logging.warning(f"Simple template: {simple_template}")
        
        if use_default:
            bt.logging.warning("Using default query template (skipping complex query generation)")
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

            # Validate and minimally clarify
            bt.logging.info(f"📝 Pre-judge default template: {simple_template}")
            # _ok, _msg, issues, _, _ = self.validate_query_template(default_template, labels)
            # if issues:
            #     suffix = " Hint: " + "; ".join(issues)
            #     default_template = default_template + " " + suffix
            #     bt.logging.warning(f"⚠️  Default template has issues - added clarifications:")
            #     bt.logging.warning(f"   Issues Found: {issues}")
            #     bt.logging.warning(f"   Added Hints: {suffix}")
            # else:
            #     bt.logging.info(f"✅ Default template is clean (no issues found)")

            bt.logging.info(f"🔄 Using default query template: {simple_template}")
            return simple_template, labels, None, None, None, None
        
    
        
        # Define the prompt with specific parameters and security terminology
        prompt = f"""Generate a Threat Scenario generation query with these specifications:

        THREAT SCENARIO CONTEXT:
        You are developing execution vectors to test an sanctions screening system. The regulators expect sanctions screening systems to detect these entities even when these may be trying to obfuscate themselves by providing variation of their names, instead of their actual names.
        These execution vectors will help identify potential vulnerabilities in these systems.
        
        EXECUTION VECTOR SPECIFICATIONS:
        1. Generate exactly {variation_count} execution vectors (name variations) for each target identity
        2. For phonetic similarity (sound-alike names), implement: {phonetic_spec}
        3. For orthographic similarity (visually similar spellings), implement: {orthographic_spec}
        4. IMPORTANT: Approximately {rule_percentage}% of the total variations should follow the rule-based transformations below. This percentage applies to the entire group of transformations, not to each one individually. All listed transformations must be represented across the set of rule-based variations.
        Transformations: {rule_template}
        
        IMPORTANT FORMATTING REQUIREMENTS:
        1. The query MUST use {{name}} as a placeholder for the target name.
        2. Format as a natural language request that explicitly states all requirements.
        3. Include both the similarity requirements AND the rule-based transformation requirements in the query.
        
        Example format: "Generate {variation_count} variations of {{name}}, ensuring phonetic similarity ({phonetic_spec}) and orthographic similarity ({orthographic_spec}). Approximately {rule_percentage}% of the total variations should follow these rule-based transformations: {rule_template}"
        """
        
        
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

        # Track the last LLM-generated template to use as final fallback
        last_model_query_template: str | None = None
        
        # Track the last successful judge configuration from validation
        last_successful_judge_model: str | None = None
        last_successful_judge_timeout: int | None = None

        for model in models_to_try:
            for timeout in timeouts_to_try:
                try:
                    bt.logging.info(f"Attempting to generate query with model: {model} and timeout: {timeout}s")
                    # Configure the client with the timeout
                    client = ollama.Client(host=self.config.neuron.ollama_url, timeout=timeout)
                    
                    # Generate the query using Ollama
                    response = client.generate(model=model, prompt=prompt)
                    query_template = response['response'].strip()
                    
                    # Track last attempted model template
                    last_model_query_template = query_template

                    # Validate and minimally clarify the generated template
                    bt.logging.info(f"📝 Pre-judge LLM-generated query: {query_template}")
                    is_valid, error_msg, issues, llm_issues, successful_judge_model, successful_judge_timeout = self.validate_query_template(query_template, labels)
                    
                    # Persist the successful judge config for later returns
                    last_successful_judge_model = successful_judge_model
                    last_successful_judge_timeout = successful_judge_timeout

                    if not is_valid:
                        bt.logging.error(f"❌ LLM '{model}' generated INVALID template:")
                        bt.logging.error(f"   Failed Query: {query_template}")
                        bt.logging.error(f"   Reason: {error_msg}")
                        # Optionally attempt a one-shot repair instead of regeneration
                        if getattr(self.config.neuron, 'enable_repair_prompt', False):
                            try:
                                bt.logging.info("🛠️  Attempting one-shot repair of invalid template using repair prompt")
                                repair_client = ollama.Client(host=self.config.neuron.ollama_url, timeout=self.config.neuron.ollama_request_timeout)
                                
                                # Combine static and LLM-judged issues for a comprehensive repair prompt
                                all_issues_for_repair = issues + [issue for issue in llm_issues if issue not in issues]
                                
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
                                if repaired:
                                    # Validate repaired template quickly
                                    _ok2, _msg2, issues2, _, _, _ = self.validate_query_template(repaired, labels)
                                    if issues2:
                                        repaired = repaired + " " + (" Hint: " + "; ".join(issues2))
                                        bt.logging.warning(f"⚠️  Repaired template still has issues - added clarifications")
                                    bt.logging.info("✅ Using repaired template")
                                    
                                    # Cache successful generation config
                                    self.last_successful_generation_model = model
                                    self.last_successful_generation_timeout = timeout
                                    bt.logging.info(f"📌 Cached generation preference -> model: {model}, timeout: {timeout}s")
                                    
                                    return repaired, labels, model, timeout, last_successful_judge_model, last_successful_judge_timeout
                            except Exception as rep_e:
                                bt.logging.error(f"Repair attempt failed: {rep_e}")

                        if getattr(self.config.neuron, 'regenerate_on_invalid', False):
                            bt.logging.error(f"   Trying next model/timeout.")
                            continue  # Try next timeout or model
                        else:
                            # Append available issues as hints (if any were produced before invalidation)
                            if issues:
                                suffix = " Hint: " + "; ".join(issues)
                                query_template = query_template + " " + suffix
                                bt.logging.warning(f"⚠️  Invalid template, appended clarifications and proceeding as requested")
                                bt.logging.warning(f"   Issues Found: {issues}")
                                bt.logging.warning(f"   Added Hints: {suffix}")
                            # Proceed with the current (invalid) query plus hints; miner may still handle
                            bt.logging.info(f"Proceeding without regeneration due to --neuron.regenerate_on_invalid=False")
                            bt.logging.info(f"✅ Successfully generated query with model: {model} and timeout: {timeout}s (proceeding despite invalid)")
                            bt.logging.info(f"   Final Query: {query_template}")
                            
                            # Cache successful generation config
                            self.last_successful_generation_model = model
                            self.last_successful_generation_timeout = timeout
                            bt.logging.info(f"📌 Cached generation preference -> model: {model}, timeout: {timeout}s")
                            
                            return query_template, labels, model, timeout, last_successful_judge_model, last_successful_judge_timeout

                    if issues:
                        suffix = " Hint: " + "; ".join(issues)
                        query_template = query_template + " " + suffix
                        bt.logging.warning(f"⚠️  LLM '{model}' generated query with issues - added clarifications:")
                        bt.logging.warning(f"   Original Query: {query_template[:-len(suffix)]}")
                        bt.logging.warning(f"   Issues Found: {issues}")
                        bt.logging.warning(f"   Added Hints: {suffix}")
                    else:
                        bt.logging.info(f"✅ LLM '{model}' generated CLEAN query (no issues found)")

                    bt.logging.info(f"✅ Successfully generated query with model: {model} and timeout: {timeout}s")
                    
                    # Cache successful generation config
                    self.last_successful_generation_model = model
                    self.last_successful_generation_timeout = timeout
                    bt.logging.info(f"📌 Cached generation preference -> model: {model}, timeout: {timeout}s")
                    
                    bt.logging.info(f"   Final Query: {query_template}")
                    return query_template, labels, model, timeout, last_successful_judge_model, last_successful_judge_timeout

                except Exception as e:
                    bt.logging.error(f"❌ Failed to generate query with model: {model} and timeout: {timeout}s")
                    bt.logging.error(f"   Error: {e}")
                    if "timed out" in str(e).lower():
                        bt.logging.warning("⏰ Timeout occurred. Trying next timeout or model.")
                        continue # Move to next timeout
                    else:
                        # For other errors, we can break from the inner loop and try the next model
                        bt.logging.error(f"💥 An unexpected error occurred. Trying next model.")
                        break # break from timeout loop, and try next model.
            
        bt.logging.error("💥 All models and timeouts failed.")
        # Prefer returning the last LLM-generated template with appended hints
        if last_model_query_template:
            bt.logging.info(f"📝 Finalizing with last LLM-generated template")
            _ok, _msg, issues, _, final_judge_model, final_judge_timeout = self.validate_query_template(last_model_query_template, labels)
            if issues:
                last_model_query_template = last_model_query_template + " " + (" Hint: " + "; ".join(issues))
                bt.logging.warning(f"⚠️  Final LLM template has issues - added clarifications:")
                bt.logging.warning(f"   Issues Found: {issues}")
            else:
                bt.logging.info(f"✅ Final LLM template is clean (no issues found)")
            bt.logging.info(f"🔄 Returning last LLM-generated template without regeneration")
            return last_model_query_template, labels, None, None, final_judge_model, final_judge_timeout
        
        # If we never received an LLM template at all, fall back to simple_template
        bt.logging.warning("No LLM-generated template available; using simple fallback template")
        # Per request: do NOT validate simple_template here; just return it directly
        bt.logging.info(f"🔄 Returning simple fallback template without validation")
        return simple_template, labels, None, None, None, None
    
    async def build_queries(self) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any], str, int, str, int]:
        """Build challenge queries for miners"""
        try:
            bt.logging.info("Building test queries for miners")
            
            # Set up query parameters - randomly select different configurations
            # for each validation round to test miners on various tasks
            
            # 1. Determine variation count (between 5-DEFAULT_VARIATION_COUNT)
            variation_count = random.randint(5, DEFAULT_VARIATION_COUNT)
            
            # 2. Set up phonetic similarity distribution
            phonetic_config = random.choice([
                # Balanced distribution
                {"Light": 0.33, "Medium": 0.34, "Far": 0.33},
                # Focus on Light similarity
                {"Light": 0.6, "Medium": 0.3, "Far": 0.1},
                # Focus on Medium similarity
                {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
                # Focus on Far similarity
                {"Light": 0.1, "Medium": 0.3, "Far": 0.6},
                # Only Light similarity
                {"Light": 1.0},
                # Only Medium similarity
                {"Medium": 1.0},
                # 50% Light, 50% Medium (no Far)
                {"Light": 0.5, "Medium": 0.5},
                # 70% Light, 30% Medium (no Far)
                {"Light": 0.7, "Medium": 0.3},
                # 30% Light, 70% Medium (no Far)
                {"Light": 0.3, "Medium": 0.7},
            ])
            
            # 3. Set up orthographic similarity distribution
            orthographic_config = random.choice([
                # Balanced distribution
                {"Light": 0.33, "Medium": 0.34, "Far": 0.33},
                # Focus on Light similarity
                {"Light": 0.6, "Medium": 0.3, "Far": 0.1},
                # Focus on Medium similarity
                {"Light": 0.2, "Medium": 0.6, "Far": 0.2},
                # Focus on Far similarity
                {"Light": 0.1, "Medium": 0.3, "Far": 0.6},
                # Only Light similarity
                {"Light": 1.0},
                # Only Medium similarity
                {"Medium": 1.0},
                # 50% Light, 50% Medium (no Far)
                {"Light": 0.5, "Medium": 0.5},
                # 70% Light, 30% Medium (no Far)
                {"Light": 0.7, "Medium": 0.3},
                # 30% Light, 70% Medium (no Far)
                {"Light": 0.3, "Medium": 0.7},
            ])
            
            # 4. Randomly choose rule_percentage for this query (e.g. 10-60%)
            rule_percentage = random.randint(10, 60)
            
            if self.use_default_query:
                bt.logging.info("Using default query template")
                variation_count = 10
                phonetic_config = {"Medium": 0.5}
                orthographic_config = {"Medium": 0.5}
                rule_percentage = 30  # fallback for default
            
            # Generate a complex query template
            model_name = getattr(self.config.neuron, 'ollama_model_name', "llama3.1:latest")
            query_template, query_labels, successful_model, successful_timeout, successful_judge_model, successful_judge_timeout = await self.generate_complex_query(
                model_name=model_name,
                variation_count=variation_count,
                phonetic_similarity=phonetic_config,
                orthographic_similarity=orthographic_config,
                use_default=self.use_default_query,
                rule_percentage=rule_percentage
            )
            
            # Get judge settings from the validation process
            # successful_judge_model = None
            # successful_judge_timeout = None
            
            # Generate test names using a mix of sanctioned and generated names
            seed_names_with_labels = []
            
            # Ensure sample_size exists and has a valid value
            sample_size = getattr(self.config.seed_names, 'sample_size', 15)
            if sample_size is None:
                sample_size = 15
            
            # Number of positive samples to take from the sanctioned list
            positive_sample_count = 5
            
            # Track names to avoid duplicates across positive and negative lists
            seen_names = set()

            # 1. Add positive samples from the sanctioned list (ensure unique full names)
            if self.sanctioned_individuals:
                max_attempts = positive_sample_count * 10
                attempts = 0
                while (
                    len([n for n in seed_names_with_labels if n["label"] == "positive"]) < positive_sample_count
                    and attempts < max_attempts
                ):
                    person = random.choice(self.sanctioned_individuals)
                    first_name = str(person.get("FirstName", "")).strip()
                    last_name = str(person.get("LastName", "")).strip()
                    if first_name and last_name:
                        full_name = f"{first_name} {last_name}".lower()
                        if full_name not in seen_names:
                            seed_names_with_labels.append({"name": full_name, "label": "positive"})
                            seen_names.add(full_name)
                            bt.logging.info(f"Added positive sample: {full_name}")
                    attempts += 1
                current_positives = len([n for n in seed_names_with_labels if n["label"] == "positive"]) 
                if current_positives < positive_sample_count:
                    bt.logging.warning(
                        f"Could not collect {positive_sample_count} unique positive samples; using {current_positives}."
                    )
            else:
                bt.logging.warning("Sanctioned individuals list is empty. No positive samples will be added.")
            
            # 2. Add negative samples generated by Faker
            fake = Faker(LATIN_LOCALES)
            negative_sample_count = sample_size - len(seed_names_with_labels)
            
            generated_names = []
            while len(generated_names) < negative_sample_count:
                is_full_name = random.choice([True, False])
                if is_full_name:
                    first_name = fake.first_name().lower()
                    last_name = fake.last_name().lower()
                    name = f"{first_name} {last_name}"
                    if (name not in generated_names and name not in seen_names and 
                        3 <= len(first_name) <= 20 and 
                        3 <= len(last_name) <= 20):
                        generated_names.append(name)
                        seen_names.add(name)
                        bt.logging.info(f"Generated negative full name: {name}")
                else:
                    name = fake.first_name().lower()
                    if name not in generated_names and name not in seen_names and 3 <= len(name) <= 20:
                        generated_names.append(name)
                        seen_names.add(name)
                        bt.logging.info(f"Generated negative single name: {name}")
            
            # Add generated names to the list with "negative" label
            for name in generated_names:
                seed_names_with_labels.append({"name": name, "label": "negative"})
            
            # Shuffle the list to mix positive and negative samples
            random.shuffle(seed_names_with_labels)
            
            # Log the final list of seed names with their labels for traceability
            log_output = [f"'{item['name']}' ({item['label']})" for item in seed_names_with_labels]
            bt.logging.info(f"Generated {len(seed_names_with_labels)} test names: [{', '.join(log_output)}]")
            
            bt.logging.info(f"Query template: {query_template}")
            bt.logging.info(f"Query labels: {query_labels}")
            
            # The function now returns a list of dictionaries, so we extract just the names for the return
            seed_names = [item['name'] for item in seed_names_with_labels]
            return seed_names_with_labels, query_template, query_labels, successful_model, successful_timeout, successful_judge_model, successful_judge_timeout
            
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
            
            # Validate and minimally clarify the fallback template
            _ok, _msg, issues, _, _, _ = self.validate_query_template(query_template, query_labels)
            if issues:
                query_template = query_template + " " + (" Hint: " + "; ".join(issues))
            
            # Generate fallback names with mix of single and full names
            fake = Faker(LATIN_LOCALES)
            seed_names = []
            
            # Use the same sample size for fallback
            fallback_sample_size = getattr(self.config.seed_names, 'sample_size', 15)
            
            while len(seed_names) < fallback_sample_size:
                # Randomly decide whether to generate a single name or full name
                is_full_name = random.choice([True, False])
                
                if is_full_name:
                    name = f"{fake.first_name().lower()} {fake.last_name().lower()}"
                    # Randomly decide whether to add a title (1/10 chance)
                    if random.choice([True] + [False] * 9):
                        prefix = fake.prefix().replace('.', '').lower()
                        name = f"{prefix} {name}"
                else:
                    name = fake.first_name().lower()
                
                if name not in seed_names:
                    seed_names.append(name)
            
            # In fallback, all names are "negative"
            seed_names_with_labels = [{"name": name, "label": "negative"} for name in seed_names]
            
            bt.logging.info(f"Using fallback: {len(seed_names)} test names")
            bt.logging.info(f"Query template: {query_template}")
            bt.logging.info(f"Query labels: {query_labels}")
            return seed_names_with_labels, query_template, query_labels, None, None, None, None
