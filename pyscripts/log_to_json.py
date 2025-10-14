import re
import json
import asyncio
from dataclasses import dataclass
from this import d
import jellyfish
import os
# ---------------- Phonetic class P0..P7 ----------------
@dataclass(frozen=True)
class SeedCodes:
    sdx: str
    met: str
    nys: str

def seed_codes(name: str) -> SeedCodes:
    return SeedCodes(
        sdx=jellyfish.soundex(name),
        met=jellyfish.metaphone(name),
        nys=jellyfish.nysiis(name),
    )

def phon_class(s: SeedCodes, v: str) -> int:
    se = jellyfish.soundex(v)   == s.sdx
    me = jellyfish.metaphone(v) == s.met
    ne = jellyfish.nysiis(v)    == s.nys
    # pack bits: s(4) + m(2) + n(1)
    idx = (4 if se else 0) + (2 if me else 0) + (1 if ne else 0)
    table = {0:7, 4:6, 2:5, 1:4, 3:3, 5:2, 6:1, 7:0}
    return table[idx]


def parse_log_file(log_file_path: str) -> dict:
    """
    Parse a log file with the format:
    names: {array of names}
    Template: {template text}
    Response({some numbers}): {dictionary in the form of name: [array of variants]}
    
    Returns:
        dict: Parsed data containing names, template, and response
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract names array
    names_match = re.search(r'names:\s*(\[.*?\])', content, re.DOTALL)
    if not names_match:
        raise ValueError("Could not find names array in log file")
    
    try:
        names = json.loads(names_match.group(1))
    except json.JSONDecodeError as e:
        # Try to fix common issues with the names array format
        names_str = names_match.group(1)
        names_str = names_str.replace("'", '"')  # Replace single quotes with double quotes
        names = json.loads(names_str)
    
    # Extract template
    
    template_match = re.search(r'Template:\s*(.*?)(?=Response\(|\Z)', content, re.DOTALL)
    if not template_match:
        raise ValueError("Could not find template in log file")
    
    template = template_match.group(1).strip()
    # Remove quotes if the template is wrapped in quotes
    if template.startswith('"') and template.endswith('"'):
        template = template[1:-1]
    
    # Extract response dictionary
    response_match = re.search(r'Response\(\d+\):\s*(\{.*\})', content, re.DOTALL)
    if not response_match:
        raise ValueError("Could not find response dictionary in log file")
    
    try:
        response_dict = json.loads(response_match.group(1))
    except json.JSONDecodeError:
        # Try to fix common issues with the response format
        response_str = response_match.group(1)
        response_str = response_str.replace("'", '"')  # Replace single quotes with double quotes
        response_dict = json.loads(response_str)
    
    return {
        'names': names,
        'template': template,
        'response': response_dict
    }


def test_parse_log_file(log_file_path):
    """Test parsing of the log file format"""
    # Test with the provided example log file
    
    
    try:
        parsed_data = parse_log_file(log_file_path)
        
        # Verify the structure
        assert 'names' in parsed_data
        assert 'template' in parsed_data
        assert 'response' in parsed_data
        
        # Verify names is a list
        assert isinstance(parsed_data['names'], list)
        assert len(parsed_data['names']) > 0
        
        # Verify template is a string
        assert isinstance(parsed_data['template'], str)
        assert len(parsed_data['template']) > 0
        
        # Verify response is a dictionary
        assert isinstance(parsed_data['response'], dict)
        
        # Verify that each name in names has corresponding variants in response
        for name in parsed_data['names']:
            assert name in parsed_data['response']
            assert isinstance(parsed_data['response'][name], list)
        
        print("Log file parsing test passed!")
        print(f"Found {len(parsed_data['names'])} names")
        print(f"Template length: {len(parsed_data['template'])} characters")
        print(f"Response contains {len(parsed_data['response'])} name entries")
        
        return parsed_data['names'], parsed_data['template'], parsed_data['response']
        
    except Exception as e:
        print(f"Log file parsing test failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    logdir = sys.argv[1] if len(sys.argv) > 1 else "/work/54/tasks_fvs"
    for logfile in os.listdir(logdir):
        if not logfile.endswith(".log"):
            continue
        logfile = os.path.join(logdir, logfile)
        print(f"Processing {logfile}")
        try:
            names, template, name_results1 = test_parse_log_file(logfile)
        except Exception as e:
            print(f"Error parsing {logfile}: {e}")
            continue
        from MIID.miner.parse_query_gemini import query_parser
        from MIID.miner.generate_variations import generate_variations_using_params
        query_params = asyncio.run(query_parser(template))
        # name_results, metrics = asyncio.run(generate_variations(names, template))

        total_count=int(query_params.get("variation_count") or 0)
        rule_percentage=float(query_params.get("rule_percentage") or 0.0)
        selected_rules=list(query_params.get("selected_rules") or [])
        phonetic_similarity=query_params.get("phonetic_similarity")
        orthographic_similarity=query_params.get("orthographic_similarity")
        selected_rules=query_params.get("selected_rules")
        rule_based = {"selected_rules": selected_rules, "rule_percentage": rule_percentage * 100}
        from MIID.validator.reward import get_name_variation_rewards
        try:
            from typing import SimpleNamespace
        except Exception as e:
            from types import SimpleNamespace

        rewards, detailed_metrics = get_name_variation_rewards(
            None,
            seed_names=names,
            responses=[SimpleNamespace(variations=name_results1)],
            uids=[0],
            variation_count=total_count,
            phonetic_similarity=phonetic_similarity,
            orthographic_similarity=orthographic_similarity,
            rule_based=rule_based
        )

        task = {
            'template': template,
            'query_params': query_params,
            'names': names,
            "results": {
                "5FvsYWZq6rwURnkscKYZfLmH7Emn7YacvGvX62XiX5WWgnGr": {
                    "total_reward": rewards[0],
                    "variations": name_results1,
                    "metrics": detailed_metrics[0]
                }
            }
        }
        # Replace extension with .json
        if not logfile.endswith('.json'):
            base_name = logfile.rsplit('.', 1)[0] if '.' in logfile else logfile
            output_file = f"{base_name}.json"
        else:
            output_file = logfile
        os.rename(logfile, output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=4)
        
        