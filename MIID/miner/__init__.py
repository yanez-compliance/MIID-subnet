import sys
import json

from MIID.miner.generate_variations import generate_variations_using_params
from MIID.miner.parse_query_gemini import query_parser


if __name__ == "__main__":
    query_file = "default_query.json" if len(sys.argv) == 1 else sys.argv[2]
    names = []
    query_template = ""
    with open(query_file, "r") as f:
        query_data = json.load(f)
        names = query_data["names"]
        query_template = query_data["query_template"]
    variations = generate_variations_using_params(names, query_template)
    print(variations)

__all__ = [
    "generate_variations_using_params",
    "query_parser",
]
