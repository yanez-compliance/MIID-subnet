import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import json
import requests
from MIID.utils.sign_message import sign_message
from substrateinterface import Keypair
import bittensor

# === CONFIGURATION ===
HOTKEY = "5FhCUBvS49"  # Replace with your test hotkey
SERVER_URL = "http://20.83.176.136:5000"  # Replace with your server IP
MESSAGE = {"test_data": "hello from auto-signed client"}

# === STEP 1: Load or generate a wallet ===
# === Load wallet using bittensor ===
#wallet = bittensor.wallet(name='validator', hotkey='validator_default')
#wallet.coldkey = wallet.coldkey
#wallet.hotkey = wallet.hotkey

# === STEP 2: Sign the JSON payload ===
payload_json_str = json.dumps({"results": MESSAGE})
signature_output = sign_message('validator', payload_json_str)

# === STEP 3: Build final payload ===
signed_payload = {
    "results": MESSAGE,
    "signature": signature_output
}

# === STEP 4: POST the payload to Flask server ===
endpoint = f"{SERVER_URL}/upload_data/{HOTKEY}"
response = requests.post(endpoint, json=signed_payload)

# === STEP 5: Print response ===
print("Status Code:", response.status_code)
#print("Response JSON:", response.json())