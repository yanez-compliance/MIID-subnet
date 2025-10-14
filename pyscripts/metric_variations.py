import re
import json
import asyncio
from dataclasses import dataclass
from this import d
import jellyfish
import os
from types import SimpleNamespace
from MIID.validator.reward import get_name_variation_rewards


if __name__ == "__main__":
    import sys
    task_file = sys.argv[1] if len(sys.argv) > 1 else "task.json"
    with open(task_file, "r", encoding="utf-8") as f:
        task = json.load(f)
    if "template" not in task:
        print(f"No query template found, skipping")
        sys.exit(0)
    else:
        query_template = task['template']
    if "names" not in task:
        print(f"No names found, skipping")
        sys.exit(0)
    else:
        names = task['names']
    if "query_params" not in task:
        print(f"No parsed query found, parsing query template")
        from MIID.miner.parse_query_gemini import query_parser
        query_params = query_parser(query_template)
        task['query_params'] = query_params
    else:
        query_params = task['query_params']
    total_count=int(query_params.get("variation_count") or 0)
    rule_percentage=float(query_params.get("rule_percentage") or 0.0)
    phonetic_similarity=query_params.get("phonetic_similarity")
    orthographic_similarity=query_params.get("orthographic_similarity")
    selected_rules=query_params.get("selected_rules")
    rule_based = {"selected_rules": selected_rules, "rule_percentage": rule_percentage * 100}

    if "results" not in task:
        task["results"] = {}
    if "me" not in task["results"]:
        task["results"]["metrics"] = {}
        print(f"No results found, generating variations and calculating metrics")
        from MIID.miner.generate_variations import generate_variations_using_params
        name_results, metrics = asyncio.run(generate_variations_using_params(names, query_template))
        task["results"]["me"] = {
            {
                "variations": name_results,
                "metrics": metrics
            }
        }
    else:
        results = task["results"]
        for coldkey, result in results.items():
            if "metrics" not in result:
                if "variations" not in result:
                    print(f"No variations found for {coldkey}, skipping")
                    continue
                else:
                    print(f"No metrics found for {coldkey}, calculating metrics")
                    result["variations"] = list(result["metrics"].keys())
                    print(f"No metrics found for {coldkey}, calculating metrics")
                    variations = result["variations"]
                    rewards, detailed_metrics = get_name_variation_rewards(
                        None,
                        seed_names=names,
                        responses=[SimpleNamespace(variations=variations)],
                        uids=[0],
                        variation_count=total_count,
                        phonetic_similarity=phonetic_similarity,
                        orthographic_similarity=orthographic_similarity,
                        rule_based=rule_based
                    )
                    result["metrics"] = detailed_metrics
    # Replace extension with .json
    if not task_file.endswith('.json'):
        base_name = task_file.rsplit('.', 1)[0] if '.' in task_file else task_file
        output_file = f"{base_name}.json"
    else:
        output_file = task_file
    os.rename(task_file, output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=4)
