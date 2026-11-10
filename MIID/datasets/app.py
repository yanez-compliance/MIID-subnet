"""
MIID datasets Flask API.

Gunicorn:
  gunicorn MIID.datasets.app:app --bind 127.0.0.1:5000 --workers 4

Routes:
  POST /upload_data/<hotkey>   — validator result upload + reputation decay
  POST /images/<hotkey>        — base images (image blueprint)
  POST /image/<hotkey>         — random batch image (image blueprint)
  POST /voice/<hotkey>         — reference WAV (voice blueprint)
"""

import json
import os
import secrets
import sys

from datetime import datetime
from flask import Flask, request, jsonify

from MIID.datasets.config import (
    HOST,
    PORT,
    DEBUG,
    DATA_DIR,
    ALLOWED_HOTKEYS,
)
from MIID.datasets.signature_auth import verify_request_signature
from MIID.datasets.reputation import (
    load_reputation_snapshot,
    process_reward_allocation,
    validate_reward_allocations,
)
from MIID.datasets.routes.images import image_bp
from MIID.datasets.routes.voice import voice_bp

sys.path.insert(0, "/home/ubuntu/YanezMIIDManage")
from db_machine import get_snapshot, reward_allocation

# Use fake calls for testing (reads from JSON file instead of database)
# from MIID.datasets.fake_calls import get_snapshot, reward_allocation

# Load snapshot on module import (Flask startup)
load_reputation_snapshot()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
os.makedirs(DATA_DIR, exist_ok=True)

app.register_blueprint(image_bp)
app.register_blueprint(voice_bp)


@app.route("/upload_data/<hotkey>", methods=["POST"])
def upload_data(hotkey):
    if hotkey not in ALLOWED_HOTKEYS:
        return jsonify({"error": "Unauthorized hotkey"}), 403

    err_resp, err_status = verify_request_signature(hotkey)
    if err_resp is not None:
        return err_resp, err_status

    data = request.get_json()
    try:
        allocations = validate_reward_allocations(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    hotkey_folder = os.path.join(DATA_DIR, hotkey)
    os.makedirs(hotkey_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_hex = secrets.token_hex(4)
    final_filename = f"{hotkey}.{timestamp}.{random_hex}.json"
    filepath = os.path.join(hotkey_folder, final_filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    try:
        rep = process_reward_allocation(allocations, get_snapshot, reward_allocation)
    except Exception:
        app.logger.exception("Could not process reputation batch")
        return jsonify({"error": "Reputation batch was not acknowledged"}), 503

    return (
        jsonify(
            {
                "message": "Data received and verified successfully",
                "filename": final_filename,
                "allocation_committed": True,
                "rep_snapshot_version": rep["snapshot_version"],
                "generated_at": rep["generated_at"],
                "rep_cache": rep["rep_cache"],
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
