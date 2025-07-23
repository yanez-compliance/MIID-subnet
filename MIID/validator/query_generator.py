import random
import bittensor as bt
import ollama
from typing import Dict, Any, Tuple, List
import os

# Make sure this import is outside any function or conditional blocks
from faker import Faker  # Ensure this is always imported

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
        
        bt.logging.info(f"use_default_query: {self.use_default_query}#########################################")
        bt.logging.info(f"QueryGenerator initialized with use_default_query={self.use_default_query}")
    
    def validate_query_template(self, query_template: str) -> Tuple[bool, str]:
        """
        Validate that a query template contains exactly one {name} placeholder and is properly formatted.
        
        Args:
            query_template: The query template to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not query_template:
            return False, "Query template is empty"
        
        # Check for {name} placeholder
        if "{name}" not in query_template:
            return False, "Query template is missing {name} placeholder"
        
        # Check for required keywords
        required_keywords = ["phonetic", "orthographic", "rule"]
        missing_keywords = [kw for kw in required_keywords if kw not in query_template.lower()]
        
        if missing_keywords:
            return False, f"Query template missing required keywords: {', '.join(missing_keywords)}"
        
        return True, "Query template is valid"
    
    async def generate_complex_query(
        self,
        model_name: str,
        variation_count: int = 10,
        phonetic_similarity: Dict[str, float] = None,
        orthographic_similarity: Dict[str, float] = None,
        use_default: bool = False,
        rule_percentage: int = 30
    ) -> Tuple[str, Dict[str, Any], str, int]:
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
            "rule_based": rule_metadata  # Add rule-based metadata
        }
        
        # If use_default flag is True, skip LLM and use default template
        if use_default:
            bt.logging.warning("Using default query template (skipping complex query generation)")
            clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "
            # Ensure the default template includes the rule_percentage
            default_template = (
                f"{clarifying_prefix}Give me {DEFAULT_VARIATION_COUNT} comma separated alternative spellings "
                f"of the name {{name}}. Include 50% of them should Medium sound similar to the original name and 50% "
                f"should be Medium orthographically similar. Approximately {rule_percentage}% of the variations "
                f"should follow these rule-based transformations: {rule_template}. Provide only the names."
            )
            labels = {
                "variation_count": DEFAULT_VARIATION_COUNT,
                "phonetic_similarity": {"Medium": 0.5},
                "orthographic_similarity": {"Medium": 0.5},
                "rule_based": rule_metadata
            }
            
            bt.logging.warning(f"Use default query template: {default_template}")
            return default_template, labels, None, None
        
        # Format the similarity specifications for the prompt
        phonetic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in phonetic_similarity.items()])
        orthographic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in orthographic_similarity.items()])
        
        bt.logging.info(f"Generating query with: {variation_count} variations, " +
                    f"phonetic similarity: {phonetic_spec}, " +
                    f"orthographic similarity: {orthographic_spec}")
        bt.logging.info(f"Rule-based requirement: {rule_percentage}% of variations should follow: {rule_template}")
        
        # Define the prompt with specific parameters and security terminology
        prompt = f"""Generate a Threat Scenario generation query with these specifications:

        THREAT SCENARIO CONTEXT:
        You are developing execution vectors to test an sanctions screening system. The regulators expect sanctions screening systems to detect these entities even when these may be trying to obfuscate themselves by providing variation of their names, instead of their actual names.
        These execution vectors will help identify potential vulnerabilities in these systems.
        
        EXECUTION VECTOR SPECIFICATIONS:
        1. Generate exactly {variation_count} execution vectors (name variations) for each target identity
        2. For phonetic similarity (sound-alike names), implement: {phonetic_spec}
        3. For orthographic similarity (visually similar spellings), implement: {orthographic_spec}
        4. IMPORTANT: Approximately {rule_percentage}% of the variations should follow these rule-based transformations: {rule_template}
        
        IMPORTANT FORMATTING REQUIREMENTS:
        1. The query MUST use {{name}} as a placeholder for the target name
        2. Use exactly one {{name}} placeholder in the query
        3. Format as a natural language request that explicitly states all requirements
        4. Include both the similarity requirements AND the rule-based transformation requirements in the query
        
        Example format: "Generate {variation_count} variations of the name {{name}}, ensuring phonetic similarity: {phonetic_spec}, and orthographic similarity: {orthographic_spec}, and also include {rule_percentage}% of variations that follow: {rule_template}"
        """
        # Add a clarifying sentence at the beginning to make it clear this is the seed name
        clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "  
        simple_template = f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}, ensuring phonetic similarity: {phonetic_spec}, and orthographic similarity: {orthographic_spec}, and also include {rule_percentage}% of variations that follow: {rule_template}"
        bt.logging.warning(f"Simple template: {simple_template}")
        
        # Get the list of models to try: primary + fallbacks
        primary_model = model_name
        fallback_models = getattr(self.config.neuron, 'ollama_fallback_models', [])
        models_to_try = [primary_model] + fallback_models

        # Get the list of timeouts to try
        primary_timeout = self.config.neuron.ollama_request_timeout
        fallback_timeouts = getattr(self.config.neuron, 'ollama_fallback_timeouts', [])
        timeouts_to_try = [primary_timeout] + fallback_timeouts

        for model in models_to_try:
            for timeout in timeouts_to_try:
                try:
                    bt.logging.info(f"Attempting to generate query with model: {model} and timeout: {timeout}s")
                    # Configure the client with the timeout
                    client = ollama.Client(host=self.config.neuron.ollama_url, timeout=timeout)
                    
                    # Generate the query using Ollama
                    response = client.generate(model=model, prompt=prompt)
                    query_template = response['response'].strip()
                    
                    # Validate the generated template
                    is_valid, error_msg = self.validate_query_template(query_template)
                    if not is_valid:
                        bt.logging.warning(f"LLM '{model}' generated invalid template: {error_msg}. Trying next model/timeout.")
                        continue  # Try next timeout or model

                    bt.logging.info(f"Successfully generated query with model: {model} and timeout: {timeout}s")
                    return query_template, labels, model, timeout

                except Exception as e:
                    bt.logging.warning(f"Failed to generate query with model: {model} and timeout: {timeout}s. Error: {e}")
                    if "timed out" in str(e).lower():
                        bt.logging.warning("Timeout occurred. Trying next timeout or model.")
                        continue # Move to next timeout
                    else:
                        # For other errors, we can break from the inner loop and try the next model
                        bt.logging.warning(f"An unexpected error occurred. Trying next model.")
                        break # break from timeout loop, and try next model.
            
        bt.logging.error("All models and timeouts failed. Falling back to a simple template.")
        return simple_template, labels, None, None
    
    async def build_queries(self) -> Tuple[List[str], str, Dict[str, Any], str, int]:
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
            query_template, query_labels, successful_model, successful_timeout = await self.generate_complex_query(
                model_name=model_name,
                variation_count=variation_count,
                phonetic_similarity=phonetic_config,
                orthographic_similarity=orthographic_config,
                use_default=self.use_default_query,
                rule_percentage=rule_percentage
            )
            
            # Generate test names using Faker
            fake = Faker()
            
            # Create a list to store the generated names
            seed_names = []
            
            # Ensure seed_names config exists
            if not hasattr(self.config, 'seed_names') or self.config.seed_names is None:
                bt.logging.warning("seed_names config not found, creating it now")
                self.config.seed_names = bt.config()
                self.config.seed_names.sample_size = 15
            
            # Ensure sample_size exists and has a valid value. The default is 15, matching config.py.
            sample_size = getattr(self.config.seed_names, 'sample_size', 15)
            if sample_size is None:
                sample_size = 15
                
            bt.logging.info(f"Using name variation sample size: {sample_size}")
            
            # Generate names with random mix of single and full names
            while len(seed_names) < sample_size:
                # Randomly decide whether to generate a single name or full name
                is_full_name = random.choice([True, False])
                
                if is_full_name:
                    # Generate full name
                    first_name = fake.first_name().lower()
                    last_name = fake.last_name().lower()
                    name = f"{first_name} {last_name}"
                    if (name not in seed_names and 
                        3 <= len(first_name) <= 20 and 
                        3 <= len(last_name) <= 20):
                        seed_names.append(name)
                        bt.logging.info(f"Generated full name: {name}")
                else:
                    # Generate single name
                    name = fake.first_name().lower()
                    if name not in seed_names and 3 <= len(name) <= 20:
                        seed_names.append(name)
                        bt.logging.info(f"Generated single name: {name}")
            
            bt.logging.info(f"Generated {len(seed_names)} test names: {seed_names}")
            bt.logging.info(f"Query template: {query_template}")
            bt.logging.info(f"Query labels: {query_labels}")
            return seed_names, query_template, query_labels, successful_model, successful_timeout
            
        except Exception as e:
            bt.logging.error(f"Error building queries: {str(e)}")
            
            # Fallback to simple defaults
            variation_count = DEFAULT_VARIATION_COUNT
            phonetic_config = {"Medium": 0.5}
            orthographic_config = {"Medium": 0.5}
            
            # Generate rule-based template and metadata for fallback
            rule_template, rule_metadata = get_rule_template_and_metadata(rule_percentage)
            
            # Add clarifying sentence to fallback template
            clarifying_prefix = "The following name is the seed name to generate variations for: {name}. "
            query_template = f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}, ensuring phonetic similarity: {phonetic_config}, and orthographic similarity: {orthographic_config}, and also include {rule_percentage}% of variations that follow: {rule_template}."
            
            # Validate the fallback template
            is_valid, error_msg = self.validate_query_template(query_template)
            if not is_valid:
                bt.logging.error(f"Fallback template validation failed: {error_msg}")
                # Use an absolutely basic template as last resort with clarifying sentence
                query_template = f"{clarifying_prefix}Generate {variation_count} variations of the name {{name}}. {rule_template}"
            
            query_labels = {
                "variation_count": variation_count,
                "phonetic_similarity": phonetic_config,
                "orthographic_similarity": orthographic_config,
                "rule_based": rule_metadata
            }
            
            # Generate fallback names with mix of single and full names
            fake = Faker()
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
            
            bt.logging.info(f"Using fallback: {len(seed_names)} test names")
            bt.logging.info(f"Query template: {query_template}")
            bt.logging.info(f"Query labels: {query_labels}")
            return seed_names, query_template, query_labels, None, None
