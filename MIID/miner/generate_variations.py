import os
import json
import asyncio
import random

from MIID.miner.generate_name_variations import generate_name_variations
from MIID.miner.parse_query_gemini import query_parser, query_parser_sync


def run_single_generation(args):
    """Helper function for multiprocessing that generates variations for a single name"""
    i, name, total_count, rule_percentage, selected_rules, phonetic_similarity, orthographic_similarity, run_dir, timeout = args
    
    return generate_name_variations(
        original_name=name,
        total_count=total_count,
        rule_percentage=rule_percentage,
        selected_rules=selected_rules,
        phonetic_similarity=phonetic_similarity,
        orthographic_similarity=orthographic_similarity,
        logfile=os.path.join(run_dir, f"{i}_{name}.log") if run_dir is not None else None,
        timeout=timeout
    )


def generate_variations_using_params(names: list, query_params: dict, run_dir: str = None, timeout: int = 100) -> list:
    total_count=int(query_params.get("variation_count") or 0)
    rule_percentage=float(query_params.get("rule_percentage") or 0.0)
    selected_rules=list(query_params.get("selected_rules") or [])
    phonetic_similarity=query_params.get("phonetic_similarity")
    orthographic_similarity=query_params.get("orthographic_similarity")
    # Clamp and compute counts
    if total_count < 0:
        total_count = 0
    if rule_percentage < 0.0:
        rule_percentage = 0.0
    if rule_percentage > 1.0:
        rule_percentage = 1.0

    tasks = []
    import concurrent.futures
    
    task_args = []
    for i, name in enumerate(names):
        args = (i, name, total_count, rule_percentage, selected_rules, 
                phonetic_similarity, orthographic_similarity, run_dir, timeout)
        task_args.append(args)

    # Execute in a separate process to avoid blocking
    max_cpus = os.cpu_count()
    max_workers = max(1, min(int(max_cpus/8), len(names)))
    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
        name_results = {}
        metrics = {}
        for name, metrics, rule_count, cand_minrequired_rule_varset, additional_rule_varset, cand_nonrule_varset in executor.map(run_single_generation, task_args):
            from MIID.miner.generate_name_variations import get_sample_from_cands
            rng = random.Random(hash(name))
            name_results[name] = get_sample_from_cands(cand_minrequired_rule_varset, additional_rule_varset, cand_nonrule_varset, rule_count, rng)
            metrics[name] = metrics
        return name_results, metrics

def generate_variations_from_template(names: list, query_template: str, run_dir: str = None) -> list:
    # parse query and save to query.json
    query_file = None
    metric_file = None
    if run_dir is not None:
        query_file = os.path.join(run_dir, "query.json")
        metric_file = os.path.join(run_dir, "metric.json")
    else:
        query_file = "query.json"
        metric_file = "metric.json"
        
    query_params = query_parser_sync(query_template, max_retries=1)
    
    name_results, metrics = generate_variations_using_params(names, query_params, run_dir)
    if query_file:
        output = {}
        output["names"] = names
        output["query_params"] = query_params
        with open(query_file, "w") as f:
            json.dump(output, f, indent=4)
    return name_results, metrics

if __name__ == "__main__":
    generate_variations_using_params()