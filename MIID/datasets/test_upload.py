import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import requests
from MIID.utils.sign_message import sign_message
from MIID.utils.misc import upload_data
from substrateinterface import Keypair
import bittensor

# === CONFIGURATION ===
HOTKEY = "5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt"  # Replace with your test hotkey
MIID_SERVER = "http://52.44.186.20:5000/upload_data"  # MIID server endpoint
MESSAGE = {"test_data": "hello from auto-signed client"}

# === STEP 1: Load or generate a wallet ===
# Load wallet using bittensor
wallet = bittensor.Wallet(name='validator', hotkey='v')
print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@Hotkey: {wallet.hotkey}@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# === STEP 2: Create test results payload ===
# Create a results structure similar to what forward.py generates
test_results = {
    "timestamp": "2025-08-05_20-01-23",
    "seed_names": ["John Doe", "Jane Smith"],
    "query_template": "Generate variations for {name}",
    "query_labels": {
        "variation_count": 5,
        "phonetic_similarity": True,
        "orthographic_similarity": True
    },
    "responses": {
        "123": {
            "uid": 123,
            "hotkey": HOTKEY,
            "variations": {
                "John Doe": ["Jon Doe", "John Doh", "J Doe"],
                "Jane Smith": ["J Smith", "Jane S", "J. Smith"]
            }
        }
    },
    "rewards": {
        "123": 0.85
    },
    "test_message": MESSAGE
}

# === STEP 3: Sign the message ===
message_to_sign = f"Hotkey: {wallet.hotkey} \n timestamp: {test_results['timestamp']} \n query_template: {test_results['query_template']} \n query_labels: {test_results['query_labels']}"
signed_contents = sign_message(wallet, message_to_sign, output_file=None)
test_results["signature"] = signed_contents

# === STEP 4: Upload to external endpoint (following forward.py pattern) ===
upload_success = False
# If for some reason uploading the data fails, we should just log it and continue. Server might go down but should not be a unique point of failure for the subnet
try:
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@Uploading data to: {MIID_SERVER}@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    upload_success = upload_data(MIID_SERVER, wallet.hotkey, test_results) 
    if upload_success:
        print("Data uploaded successfully to external server")
    else:
        print("Failed to upload data to external server")
except Exception as e:
    print(f"Uploading data failed: {str(e)}")
    upload_success = False

# === STEP 5: Print final status ===
print(f"Upload success: {upload_success}")
if not upload_success:
    print("You might want to reach out to the MIID team to add your hotkey to the allowlist.")