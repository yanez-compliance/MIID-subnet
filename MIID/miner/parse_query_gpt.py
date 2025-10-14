"""
Safe GPT Query Parser - Avoids gRPC conflicts with multiprocessing

This module provides a safe way to use OpenAI's GPT-4.1 API without gRPC warnings
when used in multiprocessing environments.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional
import logging
import requests
import threading
from typing import Set
logger = logging.getLogger(__name__)

# -----------------------------
# LLM-BASED PARSING (GPT-4.1)
# -----------------------------

def _load_parse_examples() -> List[Dict[str, str]]:
    """Load parse examples from file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "correct_parse_examples.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _build_llm_prompt(query_text: str) -> str:
    """Construct a concise instruction for the LLM to output strict JSON, with few-shot examples from recent errors."""
    
    id_list_only = "\n".join(f"- {rule_id}" for rule_id in RULE_DESCRIPTIONS.keys())
    rules_descriptions_json = json.dumps(RULE_DESCRIPTIONS, indent=4)

    examples = _load_parse_examples()
    examples_text = ""
    if examples:
        blocks: List[str] = []
        for idx, ex in enumerate(examples, start=1):
            blocks.append(
                "Example {}:\nINPUT:\n{}\nOUTPUT:\n{}\n".format(
                    idx, ex["query_template"], ex["query_label"]
                )
            )
        examples_text = (
            "Past failures and correct outputs (follow these patterns):\n"
            + "\n".join(blocks)
            + "\n"
        )

    return f"""
You are extracting parameters from a natural-language query about generating name variations.

Return ONLY a JSON object with fields:
{{
  "variation_count": <int not 0, valid range: 5-15>,
  "phonetic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
  "orthographic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
  "rule_percentage": <0.1-0.6>
}}

Tips for extracting parameters:
- For variation_count, extract explicit numbers like "generate 5 variations" or "create 10 names", "10 excutor vectors"
- You can use the following 9 patterns and it's appearance weights and to extract the phonetic and orthographic similarity:
    1. Balanced distribution - high weight for balanced testing (0.25)
        {{"Light": 0.3, "Medium": 0.4, "Far": 0.3}}
    2. Focus on Medium similarity - most common real-world scenario (0.20)
        {{"Light": 0.2, "Medium": 0.6, "Far": 0.2}}
    3. Focus on Far similarity - important for edge cases (0.15)
        {{"Light": 0.1, "Medium": 0.3, "Far": 0.6}}
    4. Light-Medium mix - moderate weight (0.12)
        {{"Light": 0.5, "Medium": 0.5}}
    5. Medium-Far mix - moderate weight (0.10)
        ({{"Light": 0.1, "Medium": 0.5, "Far": 0.4}}
    6. Only Medium similarity - common case, default for non-description, visually similar (0.08)
        ({{"Medium": 1.0}}
    7. High Light but not 100% - reduced frequency, (0.05)
        ({{"Light": 0.7, "Medium": 0.3}}
    8. Only Far similarity - edge case, significant/major/heavy (0.03)
        ({{"Far": 1.0}}
    9. Only Light similarity - reduced frequency
        ({{"Light": 1.0}}
- For rule_percentage, extract from "Approximately 43% of the [total] variations"
- When there is SQL statement context, variation_count can be extracted from "HAVING COUNT(*) = 13" or the first value of the last occurance of "LIMIT 13 * 30", rule_percentage can be extracted from "HAVING COUNT(*) > 31",

Input query:
{query_text}

Output: JSON only, no extra text.

{examples_text}
"""

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Find the last JSON object in the text and parse it."""
    try:
        candidates = re.findall(r"\{[\s\S]*\}", text)
        if not candidates:
            return None
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    except Exception:
        return None
    return None

def _normalize_llm_result(obj: Dict[str, Any], query_text: str) -> Dict[str, Any]:
    """Normalize the LLM JSON into the expected dictionary structure."""
    
    result: Dict[str, Any] = {
        "variation_count": None,
        "phonetic_similarity": {},
        "orthographic_similarity": {},
        "rule_percentage": None,
        "selected_rules": [],
    }

    # Variation count
    vc = obj.get("variation_count")
    if not vc:
        vc = 5
    elif vc > 15:
        vc = 10
    result["variation_count"] = int(vc)

    def normalize_distribution(dist: Any) -> Dict[str, float]:
        if not isinstance(dist, dict):
            return {}
        values = {k.capitalize(): float(v) for k, v in dist.items() if k and isinstance(v, (int, float))}
        # Keep only recognized levels
        values = {k: v for k, v in values.items() if k in {"Light", "Medium", "Far"}}
        # If any value > 1, assume percentages and normalize by total to 0..1
        if any(v > 1.0 for v in values.values()):
            total = sum(values.values())
            if total > 0:
                values = {k: round(v / total, 6) for k, v in values.items()}
        return values

    result["phonetic_similarity"] = normalize_distribution(obj.get("phonetic_similarity", {}))
    result["orthographic_similarity"] = normalize_distribution(obj.get("orthographic_similarity", {}))

    # selected_rules are extracted using regex
    
    rp = obj.get("rule_percentage")
    if isinstance(rp, (int, float)):
        if rp == 0.0:
            result["rule_percentage"] = 0.3
        else:
            result["rule_percentage"] = rp
    # warn with 0.09, 0.64, ...
    result["orthographic_similarity"]["Light"] = round(result.get("orthographic_similarity", {"Light": 0.0}).get("Light", 0.0), 1)
    result["orthographic_similarity"]["Medium"] = round(result.get("orthographic_similarity", {"Medium": 0.0}).get("Medium", 0.0), 1)
    result["orthographic_similarity"]["Far"] = round(result.get("orthographic_similarity", {"Far": 0.0}).get("Far", 0.0), 1)
    result["phonetic_similarity"]["Light"] = round(result.get("phonetic_similarity", {"Light": 0.0}).get("Light", 0.0), 1)
    result["phonetic_similarity"]["Medium"] = round(result.get("phonetic_similarity", {"Medium": 0.0}).get("Medium", 0.0), 1)
    result["phonetic_similarity"]["Far"] = round(result.get("phonetic_similarity", {"Far": 0.0}).get("Far", 0.0), 1)

    if result["phonetic_similarity"]["Light"] + result["phonetic_similarity"]["Medium"] + result["phonetic_similarity"]["Far"] < 1.0:
        if result["phonetic_similarity"]["Light"] == 0.2:
            result["phonetic_similarity"]["Medium"] = 0.6
            result["phonetic_similarity"]["Far"] = 0.2

        elif result["phonetic_similarity"]["Light"] == 0.1:
            if result["phonetic_similarity"]["Medium"] == 0.5:
                result["phonetic_similarity"]["Far"] = 0.4
            elif result["phonetic_similarity"]["Medium"] == 0.6:
                result["phonetic_similarity"]["Far"] = 0.3
            else:
                result["phonetic_similarity"]["Far"] = 0.4
                result["phonetic_similarity"]["Medium"] = 0.5

        elif result["phonetic_similarity"]["Light"] == 0.3:
            result["phonetic_similarity"]["Medium"] = 0.4
            result["phonetic_similarity"]["Far"] = 0.3

        elif result["phonetic_similarity"]["Light"] == 0.5:
            result["phonetic_similarity"]["Medium"] = 0.5
            result["phonetic_similarity"]["Far"] = 0.0

        elif result["phonetic_similarity"]["Light"] == 0.7:
            result["phonetic_similarity"]["Medium"] = 0.3
            result["phonetic_similarity"]["Far"] = 0.0

        elif result["phonetic_similarity"]["Medium"] == 0.4:
            result["phonetic_similarity"]["Light"] = 0.3
            result["phonetic_similarity"]["Far"] = 0.3

        elif result["phonetic_similarity"]["Far"] == 0.6:
            result["phonetic_similarity"]["Medium"] = 0.3
            result["phonetic_similarity"]["Far"] = 0.1

        elif result["phonetic_similarity"]["Far"] == 0.4:
            result["phonetic_similarity"]["Medium"] = 0.5
            result["phonetic_similarity"]["Far"] = 0.1

        elif result["phonetic_similarity"]["Light"] + result["phonetic_similarity"]["Medium"] + result["phonetic_similarity"]["Far"]  == 0.0:
            result["phonetic_similarity"]["Light"] = 0.3
            result["phonetic_similarity"]["Medium"] = 0.4
            result["phonetic_similarity"]["Far"] = 0.3

        elif result["phonetic_similarity"]["Light"] + result["phonetic_similarity"]["Medium"] + result["phonetic_similarity"]["Far"]  < 1.0:
            result["phonetic_similarity"]["Light"] = 0.3
            result["phonetic_similarity"]["Medium"] = 0.4
            result["phonetic_similarity"]["Far"] = 0.3

    if result["orthographic_similarity"]["Light"] + result["orthographic_similarity"]["Medium"] + result["orthographic_similarity"]["Far"] < 1.0:
        if result["orthographic_similarity"]["Light"] == 0.2:
            result["orthographic_similarity"]["Medium"] = 0.6
            result["orthographic_similarity"]["Far"] = 0.2

        elif result["orthographic_similarity"]["Light"] == 0.1:
            if result["orthographic_similarity"]["Medium"] == 0.5:
                result["orthographic_similarity"]["Far"] = 0.4
            elif result["orthographic_similarity"]["Medium"] == 0.6:
                result["orthographic_similarity"]["Far"] = 0.3
            else:
                result["orthographic_similarity"]["Far"] = 0.4
                result["orthographic_similarity"]["Medium"] = 0.5

        elif result["orthographic_similarity"]["Light"] == 0.3:
            result["orthographic_similarity"]["Medium"] = 0.4
            result["orthographic_similarity"]["Far"] = 0.3

        elif result["orthographic_similarity"]["Light"] == 0.5:
            result["orthographic_similarity"]["Medium"] = 0.5
            result["orthographic_similarity"]["Far"] = 0.0

        elif result["orthographic_similarity"]["Light"] == 0.7:
            result["orthographic_similarity"]["Medium"] = 0.3
            result["orthographic_similarity"]["Far"] = 0.0

        elif result["orthographic_similarity"]["Medium"] == 0.4:
            result["orthographic_similarity"]["Light"] = 0.3
            result["orthographic_similarity"]["Far"] = 0.3

        elif result["orthographic_similarity"]["Far"] == 0.6:
            result["orthographic_similarity"]["Medium"] = 0.3
            result["orthographic_similarity"]["Far"] = 0.1

        elif result["orthographic_similarity"]["Far"] == 0.4:
            result["orthographic_similarity"]["Medium"] = 0.5
            result["orthographic_similarity"]["Far"] = 0.1

        elif result["orthographic_similarity"]["Light"] + result["orthographic_similarity"]["Medium"] + result["orthographic_similarity"]["Far"]  == 0.0:
            result["orthographic_similarity"]["Light"] = 0.3
            result["orthographic_similarity"]["Medium"] = 0.4
            result["orthographic_similarity"]["Far"] = 0.3

        elif result["orthographic_similarity"]["Light"] + result["orthographic_similarity"]["Medium"] + result["orthographic_similarity"]["Far"]  < 1.0:
            result["orthographic_similarity"]["Light"] = 0.3
            result["orthographic_similarity"]["Medium"] = 0.4
            result["orthographic_similarity"]["Far"] = 0.3
        
    return result


def _extract_rules_from_query(query_text: str, rule_descriptions: Dict[str, str]) -> List[str]:
    """Extract rule IDs from query text by matching rule descriptions with flexible matching."""
    import re
    from itertools import permutations
    
    # Stop words to remove
    stop_words = {
        'a', 'an', 'last', 'first', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'name', 'target', "one", "or", "more", "is" ,
        "identity", "'s", "within", "entire", "it", "and", "full", "except", "those", "list", "above", "aa", "oo","e.g."
    }
    
    def normalize_text(text: str) -> str:
        """Normalize text by removing braces, converting to lowercase, and cleaning whitespace."""
        # Remove braces and normalize
        # Replace words ending with -ed with just 'e'
        text = text.lower().strip()

        # text = remove_level_0_brackets(text)
        # text = re.sub(r'\(([a-z ]+)\)', lambda m: '' if len(m.group(1).split()) < 5 else m.group(0), text) # remove (abc efg hij lmn opq)
        # text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\breplace (.*?) with a null character\b', r'remove \1', text) # Replace name's one character with a null character
        text = re.sub(r'\blisted\b', r'list', text)
        text = re.sub(r'\b(\w+)ed\b', r'\1e', text)
        text = re.sub(r'\b(\w+)s\b', r'\1', text)
        text = re.sub(r'\breplace {name}\\\'s vowel\b', 'replace random vowel', text)
        text = re.sub(r'\band replace it\b', '', text)
        
        text = re.sub(r'\bfirst name initial\b', 'firstnameinitial', text)
        
        text = re.sub(r'\badditionally, (.*?)swap adjacent letter\b', r'additionally, \1swap random adjacent letter', text)
        text = re.sub(r'\badditionally, (.*?)swap two adjacent letter\b', r'additionally, \1swap random adjacent letter', text)
        text = re.sub(r'\brule-base transformation(.*?)swap two adjacent letter\b', r'rule-base transformation\1swap random adjacent letter', text)
        text = re.sub(r'\bswap two random adjacent\b', r'swap random adjacent letter', text)
        # text = re.sub(r'\bswap two adjacent\b', '', text)
        # text = re.sub(r'\btwo adjacent\b', 'random adjacent', text)
        text = re.sub(r'\n', ' newline ', text)
        
        text = re.sub(r'\badditionally, (.*?)replace space with underscore\b', r'additionally, \1remove all space', text)
        # text = re.sub(r'\(e\.g\. remov{name}, ', 'remove', text) # remove (e.g.remov... -> x '('
        # text = re.sub(r'\badditionally, (.*?)\([^)]*\)', r'additionally, \1', text)
        
        text = re.sub(r'\bone or more name part\b', 'name part', text)
        text = re.sub(r'\bone or more random\b', 'random', text)
        text = re.sub(r'\bone or more\b', 'random', text)
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\babbreviate a title\b', 'add a title', text)
        text = re.sub(r'\binclude\b', 'add', text)
        text = re.sub(r'\bappend\b', 'add', text)
        text = re.sub(r'\badding\b', 'add', text)
        text = re.sub(r'\badd or remove suffix\b', 'add title suffix', text)
        
        # Remove all \(*\) patterns
        return text
    

    def extract_meaningful_words(text: str) -> List[str]:
        """Extract meaningful words by removing stop words and punctuation."""
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|,', text.lower())
        # Remove stop words
        meaningful_words = [word for word in words if word not in stop_words and (len(word) > 1 or word == ",")]
        return meaningful_words
    
    def generate_all_variants(words: List[str]) -> List[str]:
        """Generate all permutations of meaningful words as contiguous phrases."""
        if not words:
            return []
        # Limit permutations to avoid explosion
        if len(words) > 6:
            return []
        return [' '.join(p) for p in permutations(words)]
 
    matched_values: Set[str] = set()
    
    for rule_id, description in rule_descriptions.items():
        if rule_id in query_text:
            matched_values.add(rule_id)
    
    normalized_query = normalize_text(query_text)
    query_meaningful_tokens = extract_meaningful_words(normalized_query)
    query_meaningful_str = ' '.join(query_meaningful_tokens)
    
    # Process each rule description
    for rule_id, description in rule_descriptions.items():
        normalized_desc = normalize_text(description)
        meaningful_words = extract_meaningful_words(normalized_desc)
        if not meaningful_words:
            continue
        
        # Generate all permutations and require a contiguous phrase match in the meaningful stream
        for variant in generate_all_variants(meaningful_words):
            if variant and variant in query_meaningful_str:
                matched_values.add(rule_id)
                break
    
    # Return at most 3 rules
    return list(matched_values)[:3]


def _get_api_keys() -> List[(str, str)]:
    """Get API keys from environment variables"""
    api_keys = []

    try:
        keys_path = os.path.join(os.path.dirname(__file__), 'keys.json')
        if os.path.exists(keys_path):
            with open(keys_path, 'r') as f:
                keys_data = json.load(f)
                for k  in keys_data:
                    api_keys.append((k["api_type"], k["key"]))
    except Exception:
        pass
    if not api_keys:
        default_key = ("openai", "--REMOVED--")
        api_keys.append(default_key)
    return api_keys


# updated for search
RULE_DESCRIPTIONS = {
    "replace_spaces_with_random_special_characters": "Replace space with special character",
    "replace_double_letters_with_single_letter": "Replace double letter with single letter",
    "replace_random_vowel_with_random_vowel": "Replace random vowel with different one",
    "replace_random_consonant_with_random_consonant": "Replace random consonant with different one",
    #"replace_random_special_character_with_random_special_character": "Replace special characters with different ones",
    "swap_adjacent_consonants": "Swap adjacent consonant",
    "swap_adjacent_syllables": "Swap adjacent syllable",
    "swap_random_letter": "Swap random adjacent letter",
    "delete_random_letter": "Delete random letter",
    "remove_random_vowel": "Remove random vowel",
    "remove_random_consonant": "Remove random consonant",
    #"remove_random_special_character": "Remove a random special character",
    #"remove_title": "Remove title (Mr., Dr., etc.)",
    "remove_all_spaces": "Remove all space",
    "duplicate_random_letter_as_double_letter": "Duplicate random letter",
    "insert_random_letter": "Insert random letter",
    "add_random_leading_title": "Add title prefix",
    "add_random_trailing_title": "Add title suffix",
    "initial_only_first_name": "Use firstnameinitial with last name",
    "shorten_name_to_initials": "Convert name to initial",
    "shorten_name_to_abbreviations": "Abbreviate name part",
    "name_parts_permutations": "Reorder name part"
}


async def convert_names_to_english(names):
    prompt = (
        "Instruction:\n"
        "You are a multilingual name conversion assistant. Convert each name in the input list from a non-Latin script (e.g., Chinese, Korean, Japanese, Arabic, Cyrillic, Hindi, Thai, etc.) into its English equivalent.\n\n"
        "Rules:\n"
        "If a widely accepted English equivalent exists, use that (e.g. \"محمد\" → \"Mohammed\", \"Иван\" → \"Ivan\").\n"
        "If no direct equivalent exists, use a phonetic transliteration (e.g. \"张伟\" → \"Zhang Wei\").\n"
        "For East Asian names, switch to Western name order (family name last).\n"
        "Maintain capitalization (first letter of each name capitalized).\n"
        "Return only the list of English names in the same order as input.\n"
        "Do not include explanations — output only the list.\n\n"
        f"Input:\n{names}\n\nOutput:"
    )
    # Call OpenRouter-compatible Chat Completions API via AsyncOpenAI client
    try:
        from openai import AsyncOpenAI
        import os
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "MIID NVGen Service"),
            },
        )
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        output_text = response.choices[0].message.content.strip()
    except Exception:
        # Fallback: try legacy openai.ChatCompletion if present
        try:
            import openai as _openai
            output = await _openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
            )
            output_text = output.choices[0].message.content.strip()
        except Exception:
            output_text = str(names)
    # Extract the list from the response
    import ast
    try:
        english_names = ast.literal_eval(output_text)
    except Exception:
        # fallback: try to extract list from string
        import re
        match = re.search(r"\[(.*?)\]", output_text, re.DOTALL)
        if match:
            items = match.group(1)
            english_names = [item.strip().strip('"').strip("'") for item in items.split(",")]
        else:
            english_names = names  # fallback to original if parsing fails
    return english_names


async def query_parser(query_text: str, max_retries: int = 3) -> Dict[str, Any]:
    api_keys = _get_api_keys()
    client = None
    for api_type, key in api_keys:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1" if api_type == "openrouter"
                else "https://api.openai.com/v1",
                api_key=key,
            )
            break
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            continue
    if not client:
        # Fallback to default values if no client available
        return {
            "variation_count": 10,
            "phonetic_similarity": {"Light": 0.3, "Medium": 0.4, "Far": 0.3},
            "orthographic_similarity": {"Light": 0.3, "Medium": 0.4, "Far": 0.3},
            "rule_percentage": 30,
            "selected_rules": [
                "remove_random_consonant",
                "replace_random_vowel_with_random_vowel",
                "replace_spaces_with_random_special_characters"
            ]
        }
    
    prompt = _build_llm_prompt(query_text)
    
    for attempt in range(max_retries):
        try:
            # Use OpenAI Chat Completions API (v1.0.0+)
            response = await client.chat.completions.create(
                model="gpt-4.1",  # Using GPT-4.1 as requested
                messages=[
                    {"role": "system", "content": "You are a deterministic parameter extractor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                stream=False,
                presence_penalty=0,
                frequency_penalty=1.0,
                max_tokens=1024
            )
            
            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                obj = _extract_json_from_text(content)
                if obj:
                    llm_parsed = _normalize_llm_result(obj, query_text)
                    llm_parsed["selected_rules"] = _extract_rules_from_query(query_text, RULE_DESCRIPTIONS)
                    return llm_parsed
            # Add delay between retries
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    # Return default values if all attempts failed
    return {
        "variation_count": 10,
        "phonetic_similarity": {"Light": 0.3, "Medium": 0.4, "Far": 0.3},
        "orthographic_similarity": {"Light": 0.3, "Medium": 0.4, "Far": 0.3},
        "rule_percentage": 0.0,
        "selected_rules": [],
    }

__all__ = [
    "query_parser",
]