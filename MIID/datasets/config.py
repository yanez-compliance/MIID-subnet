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
    #"5FhCUBvS49UDogEMDukPEqP2JLo3FRM4MkuFPAm8ZPum3Dg6",  # validator 1 test net
    #"5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt",   # validator 2 test net
    #"5CviHjwSckCQbsGMHrdan22XB9L6maeqKfmD2eTYGSyxncg9",  # validator 3 test net
    "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr",  # validator MIID owner
    "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs", #yuma validator
    "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v", #RT21 validator
    "5F2CsUDVbRbVMXTh9fAzF9GacjVX7UapvRxidrxe7z8BYckQ", #Rizzo validator
    "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", #OTF validator
    "5CPR71gqPyvBT449xpezgZiLpxFaabXNLmnfcQdDw2t3BwqC", # 5CPR validator

]

# Hugging Face configuration
HF_REPO_ID = os.getenv("HF_REPO_ID", "username/my-dataset")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset") 