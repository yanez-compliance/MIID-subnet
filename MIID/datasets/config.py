import os
from typing import List

# Server configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Data directory configuration
DATA_DIR = "/data/MIID_data/"

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
    
    #"5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v", #RT21 validator>>june 14
    #"5F2CsUDVbRbVMXTh9fAzF9GacjVX7UapvRxidrxe7z8BYckQ", #Rizzo validator>>june 14
    #"5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", #OTF validator>>> june14
    #"5CPR71gqPyvBT449xpezgZiLpxFaabXNLmnfcQdDw2t3BwqC", # 5CPR validator
    "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54", #RT21 new HK
    #"5GduQSUxNJ4E3ReCDoPDtoHHgeoE4yHmnnLpUXBz9DAwmHWV", #Rizzo new HK
    #"5G3wMP3g3d775hauwmAZioYFVZYnvw6eY46wkFy8hEWD5KP3", #OTF new HK >>> june 23
    "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN",  # Tensora
    "5HbUFHW4XVhbQvMbSy7WDjvhHb62nuYgP1XBsmmz9E2E2K6p", #OTF new HK
    "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54", #Rizzo new HK June30th
    "5EZpnZASr7E13pjbQJVtk2JVQsnB1E9juEw2VuNxDyo54MUV", #MUV validator
    "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u", #Tao.bot
]

# Hugging Face configuration
HF_REPO_ID = os.getenv("HF_REPO_ID", "username/my-dataset")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")

# =============================================================================
# Reputation System Configuration (Phase 3 - Cycle 2)
# =============================================================================

# Reputation snapshot file path (loaded by Flask at startup)
REPUTATION_SNAPSHOT_PATH = os.path.join(DATA_DIR, "reputation_snapshot.json")

# Directory for storing reward allocations extracted from upload_data requests
REWARDS_DIR = os.path.join(DATA_DIR, "rewards")

# Reputation-weighted reward allocation weights
# KAV = Known Attack Vector (online quality from validator evaluation)
# UAV = Unknown Attack Vector (reputation-based from manual validation)
KAV_WEIGHT = 0.20  # 20% allocated to online quality (Q)
UAV_WEIGHT = 0.80  # 80% allocated to reputation-based rewards

# Note: burn_fraction is configured via --neuron.burn_fraction (default 0.75 for Cycle 2)

# Burn UID (hardcoded in existing codebase)
BURN_UID = 59

# Tier multipliers for reputation weighting
# Higher tiers get bonus multipliers on their UAV portion
TIER_MULTIPLIERS = {
    "Diamond": 1.15,
    "Gold": 1.10,
    "Silver": 1.05,
    "Bronze": 1.02,
    "Neutral": 1.00,
    "Watch": 0.90,
}

# Tier boundaries (rep_score -> tier) from reputation-policy-v1.md
# Used for reference and tier determination
TIER_BOUNDARIES = {
    "Diamond": (50.0, 9999.0),   # rep_score >= 50.0
    "Gold": (10.0, 49.999),      # rep_score 10.0 - 49.999
    "Silver": (2.0, 9.999),      # rep_score 2.0 - 9.999
    "Bronze": (1.001, 1.999),    # rep_score > 1.0 - 1.999
    "Neutral": (0.70, 1.00),     # rep_score 0.70 - 1.00
    "Watch": (0.10, 0.699),      # rep_score 0.10 - 0.699
}

# Normalization ranges per tier (rep_min, rep_max, norm_min, norm_max)
# Maps raw rep_score (0.10 - 9999.0) to reward-friendly range (0.5 - 2.0)
# This prevents Diamond miners from dominating emissions
NORM_RANGES = {
    "Watch": (0.10, 0.699, 0.50, 0.70),
    "Neutral": (0.70, 1.00, 0.70, 1.00),
    "Bronze": (1.00, 1.999, 1.00, 1.20),
    "Silver": (2.00, 9.999, 1.20, 1.50),
    "Gold": (10.0, 49.99, 1.50, 1.80),
    "Diamond": (50.0, 9999.0, 1.80, 2.00),
}