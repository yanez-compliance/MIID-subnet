## NOTE - TO BE RUN IN BACKEND BY YANEZ, NOT EVEN NEEDED TO BE PUBLIC ON THE SUBNET

"""
upload_hf.py

A simple script to upload all files in the local `data/` directory to a Hugging Face repository.

Requirements:
    pip install huggingface_hub

Steps:
    1. Create (or use an existing) repository on huggingface.co (for example, a dataset).
    2. Generate a Personal Access Token (PAT) with "write" permissions from your HF account settings.
    3. Plug in your HF token, repo ID, and optionally adjust the code for your use-case.
    4. Run this script: python upload_hf.py
"""

import os
import sys
from huggingface_hub import HfApi, HfFolder

# --------------------------
# USER-SPECIFIC CONFIG
# --------------------------
# REPO_ID: "username/my-cool-dataset" or "username/my-model"
REPO_ID = "username/my-cool-dataset"
# TOKEN: Your Hugging Face access token (with write permissions).
HF_TOKEN = "hf_xxx_your_access_token"
# If you want to treat the repo as a dataset, set this to "dataset". Otherwise "model" or "space".
REPO_TYPE = "dataset"  

# The local directory containing files to upload
DATA_DIR = "data"

def upload_all_files_to_hf(data_dir: str, repo_id: str, hf_token: str, repo_type: str = "dataset"):
    """
    Upload all files from `data_dir` to a Hugging Face repository using `huggingface_hub`.

    Args:
        data_dir (str): The local directory containing files to upload.
        repo_id (str): The target HF repository, e.g., "username/my-dataset".
        hf_token (str): A valid Hugging Face access token with write permissions.
        repo_type (str): The type of HF repository. Common values: "dataset" or "model".
    """
    api = HfApi()

    # Optional: you could also do:
    # HfFolder.save_token(hf_token)
    # so that your token is cached locally. But we'll just pass it directly here.

    # If you haven't created the repo yet, uncomment the below line to create it.
    # Note: This will fail if the repo already exists; remove or handle exceptions if so.
    #
    # api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            # Construct the full local path
            local_path = os.path.join(root, filename)

            # We'll keep the same relative path in the HF repo as local
            # e.g., if it's data/subfolder/file.json, it will go to subfolder/file.json in HF
            path_in_repo = os.path.relpath(local_path, data_dir)

            # Upload file
            print(f"Uploading {local_path} to {repo_id} (as {path_in_repo})")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=hf_token,
                commit_message=f"Add {filename}",
            )

    print("✅ All files successfully uploaded to Hugging Face!")

if __name__ == "__main__":
    # Basic checks
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' does not exist. Please create it or update the DATA_DIR path.")
        sys.exit(1)

    # Upload everything
    upload_all_files_to_hf(
        data_dir=DATA_DIR,
        repo_id=REPO_ID,
        hf_token=HF_TOKEN,
        repo_type=REPO_TYPE
    )
