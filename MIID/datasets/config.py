import os
from typing import List

# Server configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Data directory configuration
DATA_DIR = "data"

# List of allowed hotkeys that can upload data
# This should be populated with the actual hotkeys of validators
# Next: Optionally, query the chain using Bittensor APIs to get active validators dynamically.
ALLOWED_HOTKEYS: List[str] = [
    # Add your validator hotkeys here
    "5FhCUBvS49UDogEMDukPEqP2JLo3FRM4MkuFPAm8ZPum3Dg6",  # Example validator hotkey
    # Add more validator hotkeys as needed
]

# Hugging Face configuration
HF_REPO_ID = os.getenv("HF_REPO_ID", "username/my-dataset")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset") 