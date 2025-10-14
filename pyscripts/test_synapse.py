import os
import bittensor as bt
import asyncio
from MIID.validator.reward import get_name_variation_rewards
from MIID.protocol import IdentitySynapse
import json
import sys

async def test_identity_synapse():
    # Initialize bittensor objects
    import argparse
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    parser.add_argument("--query_file", type=str, default=os.path.join(os.path.dirname(__file__), "hard_tasks/17bf13.json"))
    args = parser.parse_args()
    subtensor = bt.subtensor(network="finney")
    metagraph = subtensor.metagraph(54)  # Using netuid 91 as in original test
    wallet = bt.wallet(name="test", hotkey="miner")
    query_file = args.query_file
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    template = query_data['query_template']
    names = query_data['names']
    query_params = None
    if 'query_params' not in query_data:
        from MIID.miner.parse_query_gemini import query_parser
        query_params = await query_parser(template)
        query_data['query_params'] = query_params
    else:
        query_params = query_data['query_params']
    # test_uid = 37 # 5H1jrSmC49vbTbXe8s68xBxHN6djqWQmpvEa8vTLCpfrUfJt
    # test_uid = 246 # 5CysjkSsSS3D8w5Ap61URozFPwBUFPjH7q7wEtqAEXJ5eEW9
    # test_uid = 163 # 5HmoxRai5fq9xjRQnH1Nz8nkgC7K26gxH5AmbVS9GY2GFUX4 
    # test_uid = 14 # 5EWQ
    # test_uid = 138 # 5DHQ
    # test_uid = 41 # 5Hqjr
    test_uid = 128 # 5Ct2
    coldkey = metagraph.axons[test_uid].coldkey
    try:
        async with bt.dendrite(wallet=wallet) as dendrite:
            # Create the synapse with sample data
            synapse = IdentitySynapse(
                names=names,
                query_template=template,
                timeout=720.0
            )

            # Test with a specific validator (using UID 101 as in original test)
            # test_uid = 19 # 5FvsYWZq6rwURnkscKYZfLmH7Emn7YacvGvX62XiX5WWgnGr
            axon = metagraph.axons[test_uid]
            
            bt.logging.info(f"Testing with validator UID={test_uid}, Hotkey={axon.hotkey}")
            
            
            # synapse.dendrite.hotkey = "5Ejk5HeFxruA61fSYE1pzupPf8893Fjq9EUDZxRKjSYG9oD2" # testnet
            synapse.dendrite.hotkey = "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54"
            # Send the query
            bt.logging.info(f"Sending query to validator UID={test_uid}, axon={axon}")
            response = await dendrite(
                axons=[axon],
                synapse=synapse,
                deserialize=False,  # We want the deserialized response
                timeout=720,  # Increased timeout for better reliability
            )
            # Process the response
            if response and len(response) > 0:
                if response[0].axon.status_code == 200:
                    print (response[0].variations)
                    # pass
                try:
                    rule_based = {
                        "selected_rules": query_params.get("selected_rules"),
                        "rule_percentage": query_params.get("rule_percentage") * 100
                    }
                    rewards, detailed_metrics = get_name_variation_rewards(
                        None,
                        seed_names=names,
                        responses=response,
                        uids=[test_uid],
                        variation_count=query_params.get("variation_count"),
                        phonetic_similarity=query_params.get("phonetic_similarity"),
                        orthographic_similarity=query_params.get("orthographic_similarity"),
                        rule_based=rule_based,
                    )
                except Exception as e:
                    bt.logging.error(f"Error during testing: {e}")
                query_data['results'] = {
                    coldkey: {
                        "total_reward": rewards[0],
                        "variations": response[0].variations,
                        "metrics": detailed_metrics[0]
                    }
                }
                from MIID.miner.nvgen_service import make_key
                template_hash = make_key(names, template)
                bt.logging.info(f"Received variations: {response[0].variations}")
                workdir = os.path.dirname(os.path.abspath(__file__)) + f"/tasks/{template_hash}/{test_uid}"
                os.makedirs(workdir, exist_ok=True)
                with open(f"{workdir}/query.json", "w", encoding="utf-8") as f:
                    json.dump(query_data, f, indent=4)
                with open(f"{workdir}/variants.json", "w", encoding="utf-8") as f:
                    json.dump(response[0].variations, f, indent=4)
                metrics = detailed_metrics[0]
                count_matrix = {}
                phonetic_score = {}
                orthographic_score = {}
                nonrule_count = {}
                for name in metrics['name_metrics']:
                    name_metrics = metrics['name_metrics'][name]
                    
                    for sub_name in [f"first_name", "last_name"]:
                        sub_count_matrix = [[0 for _ in range(8)] for _ in range(4)]
                        sub_name_data = name_metrics.get(sub_name, [])
                        if sub_name_data:
                            variation_scores = sub_name_data['metrics'].get('variations', [])
                            for variation in variation_scores:
                                from MIID.miner.pool_generator import orth_level, orth_sim, phon_class, seed_codes
                                
                                o_level = orth_level(orth_sim(name.split(" ")[0] if sub_name == "first_name" else name.split(" ")[1],variation['variation']))
                                p_level = phon_class(seed_codes(name.split(" ")[0] if sub_name == "first_name" else name.split(" ")[1]), variation['variation'])
                                sub_count_matrix[o_level][p_level] += 1
                        count_matrix[name + " - " + sub_name] = sub_count_matrix
                        phonetic_score[name + " - " + sub_name] = sub_name_data['metrics']['similarity']['phonetic']
                        orthographic_score[name + " - " + sub_name] = sub_name_data['metrics']['similarity']['orthographic']
                        nonrule_count[name + " - " + sub_name] = sub_name_data['metrics']['count']['actual']
                output = ""
                for subname in count_matrix:
                    mat = count_matrix[subname]
                    count = nonrule_count[subname]
                    phonetic = phonetic_score[subname]
                    orthographic = orthographic_score[subname]
                    output += f"{subname}\n - Count: {count}\n - Phonetic: {phonetic:.2f}\n - Orthographic: {orthographic:.2f}\n"
                    from MIID.miner.utils import _mat_str84
                    output += _mat_str84(mat)
                with open(f"workdir_count_matrix.txt", "w") as f:
                    f.write(output)
                print(output)
                # bt.logging.info(f"Saved variations: {workdir}/query.json")
            else:
                bt.logging.error("No response received")

    except Exception as e:
        bt.logging.error(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(test_identity_synapse())
    sys.exit(0)
