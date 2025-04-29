import random
import bittensor as bt
import ollama
from typing import Dict, Any, Tuple, List
import os

# Make sure this import is outside any function or conditional blocks
from faker import Faker  # Ensure this is always imported

# Constants for query generation
SIMILARITY_LEVELS = ["Light", "Medium", "Far"]
DEFAULT_VARIATION_COUNT = 10
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

        bt.logging.info(f"#########################################use_default_query: {self.use_default_query}#########################################")
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
        
        name_placeholders = query_template.count("{name}")
        
        if name_placeholders == 0:
            return False, "Query template missing {name} placeholder"
        elif name_placeholders > 1:
            return False, f"Query template contains multiple {name} placeholders ({name_placeholders})"
        
        # Check for proper formatting of the placeholder
        if "{name}" not in query_template:
            return False, "Query template contains malformed name placeholder"
        
        return True, "Query template is valid"
    
    async def generate_complex_query(
        self,
        model_name: str,
        variation_count: int = 10,
        phonetic_similarity: Dict[str, float] = None,
        orthographic_similarity: Dict[str, float] = None,
        use_default: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a query template based on specified parameters"""
        # Default similarity preferences if none provided
        if phonetic_similarity is None:
            phonetic_similarity = {"Medium": 1.0}
        if orthographic_similarity is None:
            orthographic_similarity = {"Medium": 1.0}
        
        # Create the labels dictionary from the parameters
        labels = {
            "variation_count": variation_count,
            "phonetic_similarity": phonetic_similarity,
            "orthographic_similarity": orthographic_similarity
        }
        
        # If use_default flag is True, skip LLM and use default template
        if use_default:
            bt.logging.info("Using default query template (skipping complex query generation)")
            default_template = f"Give me 10 comma separated alternative spellings of the name {{name}}. Include 5 of them should sound similar to the original name and 5 should be orthographically similar. Provide only the names."
            labels = {
                "variation_count": 10,
                "phonetic_similarity": {"Medium": 0.5},
                "orthographic_similarity": {"Medium": 0.5}
            }
            return default_template, labels
        
        # Format the similarity specifications for the prompt
        phonetic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in phonetic_similarity.items()])
        orthographic_spec = ", ".join([f"{int(pct*100)}% {level}" for level, pct in orthographic_similarity.items()])
        
        bt.logging.info(f"Generating query with: {variation_count} variations, " +
                    f"phonetic similarity: {phonetic_spec}, " +
                    f"orthographic similarity: {orthographic_spec}")
        
        # Define the prompt with specific parameters and security terminology
        prompt = f"""Generate a Threat Scenario generation query with these specifications:

        THREAT SCENARIO CONTEXT:
        You are developing execution vectors to test an sanctions screening system. The regulators expect sanctions screening systems to detect these entities even when these may be trying to obfuscate themselves by providing variation of their names, instead of their actual names.
        These execution vectors will help identify potential vulnerabilities in these systems.
        
        EXECUTION VECTOR SPECIFICATIONS:
        1. Generate exactly {variation_count} execution vectors (name variations) for each target identity
        2. For phonetic similarity (sound-alike names), implement: {phonetic_spec}
        3. For orthographic similarity (visually similar spellings), implement: {orthographic_spec}
        
        IMPORTANT FORMATTING REQUIREMENTS:
        1. The query MUST use {{name}} as a placeholder for the target name
        2. Use exactly one {{name}} placeholder in the query
        3. Format as a natural language request that explicitly states all requirements
        
        Example format: "Generate {variation_count} variations of the name {{name}}, ensuring..."
        """

        try:
            # Generate the query using Ollama
            response = ollama.generate(model=model_name, prompt=prompt)
            query_template = response['response'].strip()
            
            # Validate the generated template
            is_valid, error_msg = self.validate_query_template(query_template)
            if not is_valid:
                bt.logging.warning(f"LLM generated invalid template: {error_msg}")
                bt.logging.warning("Falling back to default template")
                # Fall back to a simple, valid template
                query_template = f"Generate {variation_count} comma-separated alternative spellings of the name {{name}}. Include a mix of phonetically similar and orthographically similar variations. Provide only the names."
                # Validate the fallback template
                is_valid, error_msg = self.validate_query_template(query_template)
                if not is_valid:
                    raise ValueError(f"Fallback template validation failed: {error_msg}")
            
            bt.logging.info(f"##########QQQQQQQQQQQ#####################Generated query template: {query_template}#########################################")
            bt.logging.info(f"##########QQQQQQQQQQQ#####################Generated query labels: {labels}#########################################")
            return query_template, labels
            
        except Exception as e:
            bt.logging.error(f"Error generating complex query: {str(e)}")
            # Fallback to a simple query template and default labels
            simple_template = f"Give me {variation_count} comma separated alternative spellings of the name {{name}}. Include a mix of phonetically similar and orthographically similar variations. Provide only the names."
            # Validate the fallback template
            is_valid, error_msg = self.validate_query_template(simple_template)
            if not is_valid:
                raise ValueError(f"Fallback template validation failed: {error_msg}")
            return simple_template, labels
    
    async def build_queries(self) -> Tuple[List[str], str, Dict[str, Any]]:
        """Build challenge queries for miners"""
        try:
            bt.logging.info("Building test queries for miners")
            
            # Set up query parameters - randomly select different configurations
            # for each validation round to test miners on various tasks
            
            # 1. Determine variation count (between 5-15)
            variation_count = random.randint(5, 15)
            
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
            
            if self.use_default_query:
                bt.logging.info("Using default query template")
                variation_count = 10
                phonetic_config = {"Medium": 0.5}
                orthographic_config = {"Medium": 0.5}
                
            # Generate a complex query template
            model_name = getattr(self.config.neuron, 'ollama_model_name', "llama3.1:latest")
            query_template, query_labels = await self.generate_complex_query(
                model_name=model_name,
                variation_count=variation_count,
                phonetic_similarity=phonetic_config,
                orthographic_similarity=orthographic_config,
                use_default=self.use_default_query
            )
            
            # Generate test names using Faker
            fake = Faker()
            
            # Create a list to store the generated names
            seed_names = []
            
            # Ensure name_variation config exists
            if not hasattr(self.config, 'name_variation') or self.config.name_variation is None:
                bt.logging.warning("name_variation config not found, creating it now")
                self.config.name_variation = bt.config()
                self.config.name_variation.sample_size = 5
            
            # Ensure sample_size exists and has a valid value
            sample_size = getattr(self.config.name_variation, 'sample_size', 5)
            if sample_size is None:
                sample_size = 5
                
            bt.logging.info(f"Using name variation sample size: {sample_size}")
            
            # Define supported locales for name generation
            #locales = ['en_US', 'es_ES', 'fr_FR', 'de_DE', 'it_IT', 'pt_BR', 'ru_RU', 'ar_SA', 'ja_JP', 'zh_CN']
            
            # Generate the required number of unique names
            while len(seed_names) < sample_size:
                # Randomly select a locale
                #locale = random.choice(locales)
                locale = 'en_US'
                locale_fake = Faker(locale)
                
                # Generate both first and last names
                first_name = locale_fake.first_name().lower()
                last_name = locale_fake.last_name().lower()
                
                # Create full name as a string
                full_name = f"{first_name} {last_name}"
                
                # Ensure the name is unique and meets length requirements
                if (full_name not in seed_names and 
                    3 <= len(first_name) <= 20 and 
                    3 <= len(last_name) <= 20):
                    seed_names.append(full_name)
                    bt.logging.info(f"Generated name from {locale}: {full_name}")
            
            bt.logging.info(f"#########################################Generated {len(seed_names)} test names: {seed_names}#########################################")
            bt.logging.info(f"#########################################Query template: {query_template}#########################################")
            bt.logging.info(f"#########################################Query labels: {query_labels}#########################################")
            return seed_names, query_template, query_labels
            
        except Exception as e:
            bt.logging.error(f"Error building queries: {str(e)}")
            
            # Fallback to simple defaults
            variation_count = 10
            phonetic_config = {"Medium": 0.5}
            orthographic_config = {"Medium": 0.5}
            
            query_template = f"Give me {variation_count} comma separated alternative spellings of the name {{name}}. Include 5 phonetically similar and 5 orthographically similar variations. Provide only the names."
            
            # Validate the fallback template
            is_valid, error_msg = self.validate_query_template(query_template)
            if not is_valid:
                bt.logging.error(f"Fallback template validation failed: {error_msg}")
                # Use an absolutely basic template as last resort
                query_template = f"Generate {variation_count} variations of the name {{name}}."
            
            query_labels = {
                "variation_count": variation_count,
                "phonetic_similarity": phonetic_config,
                "orthographic_similarity": orthographic_config
            }
            
            # Generate simple test names (first names only)
            fake = Faker()
            seed_names = [f"{fake.first_name().lower()} {fake.last_name().lower()}" for _ in range(5)]
            
            bt.logging.info(f"#########################################Using fallback: {len(seed_names)} test names#########################################")
            bt.logging.info(f"#########################################Query template: {query_template}#########################################")
            bt.logging.info(f"#########################################Query labels: {query_labels}#########################################")
            return seed_names, query_template, query_labels
