# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Face Variation Miner Module

This module implements a Bittensor miner that generates face image variations
using a generative model (via FLUX). The miner receives image variation requests
from validators containing a base face image and variation parameters, generates
the requested variations, encrypts them with drand timelock, uploads to S3, and
returns S3 references back to the validator.

The miner pipeline:
1. Receive ImageRequest (base image + VariationRequest list)
2. Generate face variations using FLUX (pose, lighting, expression, background, screen_replay)
3. Validate face identity is preserved (AdaFace similarity check)
4. Encrypt each variation with drand timelock
5. Upload encrypted images to S3
6. Return S3Submission references to the validator
"""

import hashlib
import json
import time
import typing
import io
import gc
import base64
import bittensor as bt
import os
from typing import List, Optional
from PIL import Image

from bittensor.core.errors import NotVerifiedException

# Protocol
from MIID.protocol import IdentitySynapse, S3Submission, ScreenReplayUAV

# Base miner class
from MIID.base.miner import BaseMinerNeuron

# screen_replay.json lives under MIID/miner/real_image_miner_guide/. Miners fill
# it in (or run the helper submit_real_photo.py) to queue a real screen-replay
# submission. See MIID/miner/real_image_miner_guide/README.md.
SCREEN_REPLAY_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "MIID", "miner", "real_image_miner_guide", "screen_replay.json",
)

# Holds extra captures submitted (via submit_real_photo.py) while a previous
# one was still pending — see queue_existing_pending_capture() there. Drained
# oldest-first, one per successful screen-replay submission, so a miner can
# queue up several captures back-to-back without any of them being dropped.
SCREEN_REPLAY_QUEUE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "MIID", "miner", "real_image_miner_guide", "queue",
)

# Today's and tomorrow's IOTD, written each time a validator sends them so
# miners can display the files for screen-replay captures.
IOTD_SEEDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "MIID", "miner", "real_image_miner_guide", "seeds",
)


def _free_gpu_memory(stage: str = "") -> None:
    """Release inter-request GPU memory and log resident VRAM."""
    try:
        gc.collect()
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            try:
                _torch.cuda.ipc_collect()
            except Exception:
                pass
            free_b, total_b = _torch.cuda.mem_get_info(0)
            reserved_b = _torch.cuda.memory_reserved(0)
            allocated_b = _torch.cuda.memory_allocated(0)
            gib = 1024 ** 3
            bt.logging.info(
                f"GPU mem [{stage}]: "
                f"free={free_b / gib:.2f} GiB / total={total_b / gib:.2f} GiB, "
                f"torch_reserved={reserved_b / gib:.2f} GiB, "
                f"torch_allocated={allocated_b / gib:.2f} GiB"
            )
    except Exception as _e:
        bt.logging.debug(f"_free_gpu_memory({stage}) failed: {_e}")


# Phase 4 imports (optional — miner still registers without these)
try:
    from MIID.miner.image_generator import decode_base_image, generate_variations, validate_face_variation
    from MIID.miner.drand_encrypt import encrypt_image_for_drand, is_timelock_available
    from MIID.miner.s3_upload import upload_to_s3
    PHASE4_AVAILABLE = True
except ImportError as _phase4_err:
    PHASE4_AVAILABLE = False


class Miner(BaseMinerNeuron):
    """
    Face Variation Miner Neuron.

    Receives image variation requests from validators and responds with
    encrypted image variations uploaded to S3.

    Configuration:
    - output_path: Directory for saving intermediate results (default: logging_dir/mining_results)
    """

    WHITELISTED_VALIDATORS = {
        "5Dvgtk1bqLycAyyc2VFKvAqmpvoQf5oWsD4qnu6vqbdWSL54": "RoundTable21",
        "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr": "Yanez",
        "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs": "Yuma",
        "5HbUFHW4XVhbQvMbSy7WDjvhHb62nuYgP1XBsmmz9E2E2K6p": "OpenTensor",
        "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54": "Rizzo",
        "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN": "Tensora",
        "5GMqiKcdq5WtHA4XaioRD29FL2UtJ8CW1MVQtYHyFsqzrrmM": "Kraken",
        "5GuPvuyKBJAWQbEGAkMbfRpG5qDqqhML8uDVSWoFjqcKKvDU": "Testnet_omar",
        "5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt": "Testnet_asem",
    }

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.output_path = os.path.join(self.config.logging.logging_dir, "mining_results")
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")

        self.axon.verify_fns[IdentitySynapse.__name__] = self._verify_validator_request

        if PHASE4_AVAILABLE:
            bt.logging.info("Phase 4 image generation: ENABLED")

            forced_model = os.environ.get("MIID_MODEL", "").strip() or "(unset -> random base model)"
            random_flag = os.environ.get("MIID_MODEL_RANDOM", "1").strip()
            inference_steps = os.environ.get("MIID_INFERENCE_STEPS", "20").strip()
            guidance_scale = os.environ.get("MIID_GUIDANCE_SCALE", "3.5").strip()
            flux_device = os.environ.get("FLUX_DEVICE", "(auto)").strip()
            enable_offload = os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "(default)").strip()
            seq_offload = os.environ.get("MIID_SEQUENTIAL_CPU_OFFLOAD", "1").strip()
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(unset)").strip()
            bt.logging.info(
                f"Phase 4 env: MIID_MODEL={forced_model} MIID_MODEL_RANDOM={random_flag} "
                f"MIID_INFERENCE_STEPS={inference_steps} MIID_GUIDANCE_SCALE={guidance_scale} "
                f"FLUX_DEVICE={flux_device} MIID_ENABLE_CPU_OFFLOAD={enable_offload} "
                f"MIID_SEQUENTIAL_CPU_OFFLOAD={seq_offload} PYTORCH_CUDA_ALLOC_CONF={alloc_conf}"
            )

            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _props = _torch.cuda.get_device_properties(0)
                    bt.logging.info(
                        f"CUDA device: {_props.name} "
                        f"({_props.total_memory / 1024 ** 3:.2f} GiB total)"
                    )
                else:
                    bt.logging.info("CUDA device: (not available - using CPU/MPS)")
            except Exception as _e:
                bt.logging.debug(f"Could not query CUDA device info: {_e}")

            if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
                bt.logging.warning(
                    "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_TOKEN in your "
                    'environment, e.g. export HF_TOKEN="hf_..."'
                )
        else:
            bt.logging.warning(
                "Phase 4 image generation: DISABLED (missing packages). "
                "Install with: pip install -r requirements-miner.txt  "
                "See docs/miner.md for the full setup."
            )

    async def _verify_validator_request(self, synapse: IdentitySynapse) -> None:
        """
        Rejects any RPC not cryptographically proven to come from a whitelisted validator.
        Raises NotVerifiedException (→ 401) if anything is missing or incorrect.
        """
        if synapse.dendrite is None:
            msg = "Rejecting request: missing dendrite terminal."
            bt.logging.warning(msg)
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey    = synapse.dendrite.hotkey
        nonce     = synapse.dendrite.nonce
        uuid      = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        if hotkey not in self.WHITELISTED_VALIDATORS:
            msg = f"Rejecting request: validator hotkey not in WHITELISTED_VALIDATORS: {hotkey}"
            bt.logging.warning(msg)
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        bt.logging.info(f"Verifying message: {message}")

        try:
            await self.axon.default_verify(synapse)
        except NotVerifiedException as e:
            bt.logging.warning(f"default_verify failed for whitelisted hotkey {hotkey}: {e}")
            raise

        bt.logging.info(f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})")

    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        """
        Process an image variation request.

        Generates face image variations, encrypts with drand timelock,
        uploads to S3, and returns S3 submission references.

        Args:
            synapse: IdentitySynapse containing image_request

        Returns:
            The synapse with s3_submissions populated
        """
        run_id = int(time.time())
        timeout = getattr(synapse, 'timeout', 120.0)
        start_time = time.time()
        bt.logging.info(f"Starting run {run_id}, timeout={timeout:.1f}s")

        if synapse.image_request is None:
            bt.logging.warning("Received synapse with no image_request; returning empty response.")
            synapse.s3_submissions = []
            return synapse

        req = synapse.image_request
        today_label = req.daily_seed_filename or "(none)"
        if req.daily_seed_date:
            today_label = f"{today_label} [{req.daily_seed_date} UTC]"
        tomorrow_label = req.tomorrow_seed_filename or "(none)"
        if req.tomorrow_seed_date:
            tomorrow_label = f"{tomorrow_label} [{req.tomorrow_seed_date} UTC]"
        bt.logging.info(
            f"Received 3 images: "
            f"IMAGE 1 (face variations)='{req.image_filename}' | "
            f"IMAGE 2 (today IOTD)='{today_label}' | "
            f"IMAGE 3 (tomorrow IOTD)='{tomorrow_label}'"
        )
        self._persist_iotd_seeds(req)

        bt.logging.info("Processing image variation request")

        if not PHASE4_AVAILABLE:
            bt.logging.warning(
                "Phase 4: Received image request but packages are not installed. "
                "Install with: pip install -r requirements-miner.txt"
            )
            synapse.s3_submissions = []
            return synapse

        try:
            s3_submissions = self.process_image_request(synapse)
            bt.logging.info(f"Phase 4: Generated {len(s3_submissions)} S3 submissions")
        except Exception as e:
            bt.logging.error(f"Phase 4: Failed to process image request: {e}")
            s3_submissions = []

        # Try to attach a real screen-replay submission from screen_replay.json
        sr_sub = self._try_screen_replay_submission(req)
        if sr_sub is not None:
            s3_submissions.append(sr_sub)

        synapse.s3_submissions = s3_submissions

        total_time = time.time() - start_time
        bt.logging.info(f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed.")

        return synapse

    def _persist_iotd_seeds(self, image_request) -> None:
        """Write today's and tomorrow's IOTD to disk for screen-replay captures.

        Files land in MIID/miner/real_image_miner_guide/seeds/ as
        today_<filename> and tomorrow_<filename>, plus a seeds.json index
        that submit_real_photo.py reads.
        """
        slots = [
            ("today", image_request.daily_seed_image, image_request.daily_seed_filename, image_request.daily_seed_date),
            ("tomorrow", image_request.tomorrow_seed_image, image_request.tomorrow_seed_filename, image_request.tomorrow_seed_date),
        ]
        if not any(b64 for _, b64, _, _ in slots):
            return

        os.makedirs(IOTD_SEEDS_DIR, exist_ok=True)
        meta = {}
        for slot, b64, filename, seed_date in slots:
            if not b64 or not filename:
                continue
            try:
                raw = base64.b64decode(b64)
            except Exception as e:
                bt.logging.warning(f"Could not decode {slot} IOTD: {e}")
                continue
            ext = os.path.splitext(filename)[1] or ".png"
            saved_name = f"{slot}_{filename}" if not filename.startswith(f"{slot}_") else filename
            if os.path.splitext(saved_name)[1] == "":
                saved_name = f"{saved_name}{ext}"
            path = os.path.join(IOTD_SEEDS_DIR, saved_name)
            try:
                with open(path, "wb") as f:
                    f.write(raw)
            except Exception as e:
                bt.logging.warning(f"Could not save {slot} IOTD to {path}: {e}")
                continue
            meta[slot] = {
                "filename": filename,
                "seed_date": seed_date,
                "path": path,
            }
            bt.logging.info(
                f"Saved {slot} IOTD: {filename} ({len(raw)} bytes) -> {path}"
            )

        # Drop stale files only for slots we just rewrote (don't wipe tomorrow
        # just because this validator only sent today).
        keep = {os.path.basename(info["path"]) for info in meta.values()}
        written_slots = set(meta.keys())
        try:
            for name in os.listdir(IOTD_SEEDS_DIR):
                if name in ("seeds.json", ".gitkeep") or name in keep:
                    continue
                slot_prefix = name.split("_", 1)[0]
                if slot_prefix in written_slots:
                    try:
                        os.remove(os.path.join(IOTD_SEEDS_DIR, name))
                    except Exception:
                        pass
        except Exception:
            pass

        meta_path = os.path.join(IOTD_SEEDS_DIR, "seeds.json")
        existing = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(meta)
        try:
            with open(meta_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            bt.logging.warning(f"Could not write seeds.json: {e}")

    def is_valid_image_bytes(self, image_bytes: bytes) -> bool:
        """Validate whether raw bytes represent a valid image."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
            return True
        except Exception:
            return False

    def is_valid_video_bytes(self, video_bytes: bytes, suffix: str = "") -> bool:
        """Lightweight check that bytes look like a short screen-replay video."""
        if not video_bytes or len(video_bytes) < 32:
            return False
        head = video_bytes[:64]
        ext = (suffix or "").lower()
        if ext in (".mp4", ".mov", ".m4v"):
            return b"ftyp" in head
        if ext == ".webm":
            return head.startswith(b"\x1a\x45\xdf\xa3")
        return b"ftyp" in head or head.startswith(b"\x1a\x45\xdf\xa3")

    def process_image_request(self, synapse: IdentitySynapse) -> List[S3Submission]:
        """
        Process an image variation request end-to-end.

        Generates image variations via FLUX, validates face identity,
        encrypts with drand timelock, uploads to S3, and returns
        S3 submission objects.

        Args:
            synapse: IdentitySynapse with image_request

        Returns:
            List of S3Submission objects
        """
        image_request = synapse.image_request
        if not image_request:
            return []

        _free_gpu_memory("before_request")

        try:
            bt.logging.info(f"Phase 4: Decoding base image: {image_request.image_filename}")
            base_image = decode_base_image(image_request.base_image)

            seed_image_name = image_request.image_filename
            for ext in ('.png', '.jpg', '.jpeg'):
                if seed_image_name.endswith(ext):
                    seed_image_name = seed_image_name[:-len(ext)]
                    break

            bt.logging.info(
                f"Phase 4: Generating {image_request.requested_variations} variations "
                f"(from validator: {[f'{v.type}({v.intensity})' for v in image_request.variation_requests]})"
            )
            variations = generate_variations(
                base_image,
                image_request.variation_requests
            )

            s3_submissions = []
            target_round = image_request.target_drand_round
            challenge_id = image_request.challenge_id or "sandbox_test"

            # Generate path_signature once per challenge (prevents path hijacking)
            path_message = f"{challenge_id}:{self.wallet.hotkey.ss58_address}"
            path_signature = self.wallet.hotkey.sign(path_message.encode()).hex()[:16]
            bt.logging.debug(f"Phase 4: Generated path_signature: {path_signature}")

            for var in variations:
                try:
                    if not self.is_valid_image_bytes(var["image_bytes"]):
                        bt.logging.warning(
                            f"Phase 4: Skipping invalid/corrupt image for {var['variation_type']}"
                        )
                        continue

                    if not validate_face_variation(var, base_image, min_similarity=0.4):
                        bt.logging.warning(
                            f"Phase 4: Skipping {var['variation_type']} — face identity not preserved"
                        )
                        continue

                    message = f"challenge:{challenge_id}:hash:{var['image_hash']}"
                    signature = self.wallet.hotkey.sign(message.encode()).hex()

                    if is_timelock_available():
                        encrypted_data = encrypt_image_for_drand(var["image_bytes"], target_round)
                        if encrypted_data is None:
                            bt.logging.warning(f"Phase 4: Encryption failed for {var['variation_type']}")
                            continue
                    else:
                        bt.logging.warning("Phase 4: Timelock not available, using raw bytes (SANDBOX ONLY)")
                        encrypted_data = var["image_bytes"]

                    s3_key = upload_to_s3(
                        encrypted_data=encrypted_data,
                        miner_hotkey=self.wallet.hotkey.ss58_address,
                        signature=signature,
                        image_hash=var["image_hash"],
                        target_round=target_round,
                        challenge_id=challenge_id,
                        variation_type=var["variation_type"],
                        path_signature=path_signature,
                        seed_image_name=seed_image_name,
                    )

                    if s3_key:
                        s3_submissions.append(S3Submission(
                            s3_key=s3_key,
                            image_hash=var["image_hash"],
                            signature=signature,
                            variation_type=var["variation_type"],
                            path_signature=path_signature,
                        ))
                        bt.logging.debug(f"Phase 4: Created submission for {var['variation_type']}")

                except Exception as e:
                    bt.logging.error(f"Phase 4: Error processing variation {var['variation_type']}: {e}")
                    continue

            bt.logging.info(f"Phase 4: Successfully created {len(s3_submissions)} S3 submissions")
            return s3_submissions

        except Exception as e:
            bt.logging.error(f"Phase 4: Error in process_image_request: {e}")
            return []
        finally:
            try:
                del base_image
            except Exception:
                pass
            try:
                del variations
            except Exception:
                pass
            _free_gpu_memory("after_request")

    def _try_screen_replay_submission(self, image_request) -> Optional[S3Submission]:
        """Submit a real screen-replay capture if screen_replay.json says it's ready.

        Reads screen_replay.json (MIID/miner/real_image_miner_guide/). Miners
        (or the helper script submit_real_photo.py in that same folder) fill
        in TWO media paths (face close-up photo/video + environment still of
        the same capture) + metadata (including capture_variant) and flip
        "ready" to true. This runs on every validator query — since the axon
        is always listening, this is effectively a background check — so as
        soon as "ready" is true the next query submits both as one
        screen_replay submission. After a
        successful upload the whole file is reset back to its blank/not-ready
        state so it won't accidentally re-submit the exact same capture again
        (that would be a duplicate). There's no limit on how many *different*
        captures a miner can queue up and submit over time — just never the
        same one twice.

        If "ready" is false (the normal/default state), this is a no-op —
        real screen-replay submissions are optional and not expected every
        round.
        """
        if not os.path.exists(SCREEN_REPLAY_JSON):
            return None

        try:
            with open(SCREEN_REPLAY_JSON, "r") as f:
                data = json.load(f)
        except Exception as e:
            bt.logging.warning(f"screen_replay.json: could not read: {e}")
            return None

        if not bool(data.get("ready", False)):
            return None  # nothing queued — normal, expected most rounds

        photo_path = data.get("photo_path", "").strip()
        photo_path_2 = data.get("photo_path_2", "").strip()
        if not photo_path or not os.path.exists(photo_path):
            bt.logging.warning(
                f"screen_replay.json: ready=true but photo_path is missing/invalid "
                f"('{photo_path}'). Expected the FACE CLOSE-UP path. "
                f"Leaving ready=true and will retry next round."
            )
            return None
        if not photo_path_2 or not os.path.exists(photo_path_2):
            bt.logging.warning(
                f"screen_replay.json: ready=true but photo_path_2 (environment shot) is "
                f"missing/invalid ('{photo_path_2}'). A screen-replay submission needs "
                f"TWO files of the same capture (face close-up + environment). "
                f"Leaving ready=true and will retry next round."
            )
            return None

        # Validator now sends today's and tomorrow's IOTD. Prefer the filename
        # the miner recorded for this capture; fall back to today's IOTD.
        chosen_seed_image = (data.get("seed_image") or "").strip() or (image_request.daily_seed_filename or "")
        if not chosen_seed_image:
            bt.logging.warning(
                "screen_replay.json: ready=true but 'seed_image' is missing — record "
                "which IOTD you photographed (today or tomorrow; re-run "
                "submit_real_photo.py with --seed-image or answer the prompt). "
                "Leaving ready=true and will retry next round."
            )
            return None

        capture_variant = (data.get("capture_variant") or "seed_unchanged").strip()
        primary_media = (data.get("primary_media") or "photo").strip().lower()
        # Prefer capture_variant over stale primary_media if they disagree.
        # Include legacy names so already-queued captures still upload as video.
        video_variants = {
            "seed_video_blinking",
            "seed_video_smiling",
            "seed_video_smile_and_blink",
            "synthetic_video_blinking",
            "synthetic_video_smiling",
            "synthetic_video_smile_and_blink",
            "synthetic_video_expression",
        }
        if capture_variant in video_variants:
            primary_media = "video"
        primary_is_video = primary_media == "video"

        try:
            photo_bytes = open(photo_path, "rb").read()
            photo_bytes_2 = open(photo_path_2, "rb").read()
        except Exception as e:
            bt.logging.warning(f"screen_replay.json: cannot read media: {e}")
            return None

        if primary_is_video:
            suffix = os.path.splitext(photo_path)[1]
            if not self.is_valid_video_bytes(photo_bytes, suffix=suffix):
                bt.logging.warning(
                    f"screen_replay.json: '{photo_path}' is not a valid video "
                    f"(variant={capture_variant})"
                )
                return None
        else:
            if not self.is_valid_image_bytes(photo_bytes):
                bt.logging.warning(f"screen_replay.json: '{photo_path}' is not a valid image")
                return None

        if not self.is_valid_image_bytes(photo_bytes_2):
            bt.logging.warning(
                f"screen_replay.json: '{photo_path_2}' is not a valid environment image"
            )
            return None

        try:
            challenge_id   = image_request.challenge_id or "sandbox_test"
            image_hash     = hashlib.sha256(photo_bytes).hexdigest()
            image_hash_2   = hashlib.sha256(photo_bytes_2).hexdigest()

            if image_hash == image_hash_2:
                bt.logging.warning(
                    "screen_replay.json: both files hash identically (same file "
                    "submitted twice) — need a FACE CLOSE-UP and a distinct "
                    "ENVIRONMENT shot. Rejecting locally; take a second, distinct capture."
                )
                return None

            message        = f"challenge:{challenge_id}:hash:{image_hash}"
            signature      = self.wallet.hotkey.sign(message.encode()).hex()
            message_2      = f"challenge:{challenge_id}:hash:{image_hash_2}"
            signature_2    = self.wallet.hotkey.sign(message_2.encode()).hex()
            path_message   = f"{challenge_id}:{self.wallet.hotkey.ss58_address}"
            path_signature = self.wallet.hotkey.sign(path_message.encode()).hex()[:16]

            if is_timelock_available():
                encrypted_data = encrypt_image_for_drand(photo_bytes, image_request.target_drand_round)
                encrypted_data_2 = encrypt_image_for_drand(photo_bytes_2, image_request.target_drand_round)
                if encrypted_data is None or encrypted_data_2 is None:
                    bt.logging.warning("screen_replay.json: timelock encryption failed")
                    return None
            else:
                encrypted_data = photo_bytes
                encrypted_data_2 = photo_bytes_2

            seed_name = chosen_seed_image.rsplit(".", 1)[0]
            primary_ext = os.path.splitext(photo_path)[1] or (".mp4" if primary_is_video else ".png")
            env_ext = os.path.splitext(photo_path_2)[1] or ".png"

            s3_key = upload_to_s3(
                encrypted_data=encrypted_data,
                miner_hotkey=self.wallet.hotkey.ss58_address,
                signature=signature,
                image_hash=image_hash,
                target_round=image_request.target_drand_round,
                challenge_id=challenge_id,
                variation_type="screen_replay",
                path_signature=path_signature,
                seed_image_name=seed_name,
                source_ext=primary_ext,
            )
            if not s3_key:
                bt.logging.warning("screen_replay.json: S3 upload failed (face close-up)")
                return None

            s3_key_2 = upload_to_s3(
                encrypted_data=encrypted_data_2,
                miner_hotkey=self.wallet.hotkey.ss58_address,
                signature=signature_2,
                image_hash=image_hash_2,
                target_round=image_request.target_drand_round,
                challenge_id=challenge_id,
                variation_type="screen_replay_angle2",
                path_signature=path_signature,
                seed_image_name=seed_name,
                source_ext=env_ext,
            )
            if not s3_key_2:
                bt.logging.warning("screen_replay.json: S3 upload failed (environment shot)")
                return None

            uav = ScreenReplayUAV(
                seed_image=chosen_seed_image,
                date=data.get("date", ""),
                camera_used=data.get("camera_used", ""),
                device_photographed=data.get("device_photographed", "phone"),
                capture_variant=capture_variant,
                moire_pixel_grid=bool(data.get("moire_pixel_grid", False)),
                screen_glare_hotspots=bool(data.get("screen_glare_hotspots", False)),
                perspective_keystone_distortion=bool(data.get("perspective_keystone_distortion", False)),
                gamma_contrast_shift=bool(data.get("gamma_contrast_shift", False)),
                edge_crop_cues=bool(data.get("edge_crop_cues", False)),
            )

            # Reset to a blank/not-ready state so we don't re-submit this exact
            # capture again (that would be a duplicate) — miners are free to
            # queue up a brand-new capture right away for the next submission.
            with open(SCREEN_REPLAY_JSON, "w") as f:
                json.dump({
                    "ready": False,
                    "photo_path": "",
                    "photo_path_2": "",
                    "seed_image": "",
                    "date": "",
                    "camera_used": "",
                    "device_photographed": "",
                    "capture_variant": "",
                    "primary_media": "photo",
                    "moire_pixel_grid": False,
                    "screen_glare_hotspots": False,
                    "perspective_keystone_distortion": False,
                    "gamma_contrast_shift": False,
                    "edge_crop_cues": False,
                }, f, indent=2)

            # The active slot just freed up — pull in the next queued capture
            # (if any) so it goes out on a later validator query instead of
            # sitting untouched. Oldest first.
            self._promote_next_queued_screen_replay()

            media_note = "video + environment" if primary_is_video else "face + environment"
            bt.logging.info(
                f"Screen-replay submitted ({media_note}, variant={capture_variant}): "
                f"{s3_key} + {s3_key_2}"
            )
            return S3Submission(
                s3_key=s3_key,
                image_hash=image_hash,
                signature=signature,
                variation_type="screen_replay",
                path_signature=path_signature,
                s3_key_angle2=s3_key_2,
                image_hash_angle2=image_hash_2,
                signature_angle2=signature_2,
                screen_replay_uav=uav,
            )

        except Exception as e:
            bt.logging.error(f"screen_replay.json: submission failed: {e}")
            return None

    def _promote_next_queued_screen_replay(self) -> None:
        """Promote the oldest queued capture (if any) into the active slot.

        submit_real_photo.py queues extra captures in SCREEN_REPLAY_QUEUE_DIR
        when screen_replay.json is already occupied (ready=true) — see
        queue_existing_pending_capture() there. Called right after resetting
        screen_replay.json back to blank following a successful submission,
        so a miner can queue up several captures back-to-back and have each
        one sent out automatically, one per subsequent validator query,
        without any of them being dropped or overwritten.
        """
        if not os.path.isdir(SCREEN_REPLAY_QUEUE_DIR):
            return

        try:
            queued_files = sorted(
                f for f in os.listdir(SCREEN_REPLAY_QUEUE_DIR)
                if f.endswith(".json")
            )
        except Exception as e:
            bt.logging.warning(f"screen_replay queue: could not list {SCREEN_REPLAY_QUEUE_DIR}: {e}")
            return

        if not queued_files:
            return

        # Filenames are timestamp-based (queued_YYYYMMDDTHHMMSSffffff.json),
        # so a plain sort gives FIFO order — oldest capture goes out first.
        next_file = os.path.join(SCREEN_REPLAY_QUEUE_DIR, queued_files[0])
        try:
            with open(next_file, "r") as f:
                queued_data = json.load(f)
        except Exception as e:
            bt.logging.warning(f"screen_replay queue: could not read {next_file}: {e}, discarding it")
            try:
                os.remove(next_file)
            except Exception:
                pass
            return

        try:
            with open(SCREEN_REPLAY_JSON, "w") as f:
                json.dump(queued_data, f, indent=2)
            os.remove(next_file)
            remaining = len(queued_files) - 1
            bt.logging.info(
                f"Screen-replay queue: promoted {queued_files[0]} to active slot "
                f"({remaining} still queued)."
            )
        except Exception as e:
            bt.logging.warning(f"screen_replay queue: failed to promote {next_file}: {e}")

    async def blacklist(self, synapse: IdentitySynapse) -> typing.Tuple[bool, str]:
        """Blacklist requests from non-whitelisted validators."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        if synapse.dendrite.hotkey not in self.WHITELISTED_VALIDATORS:
            hk = synapse.dendrite.hotkey
            msg = f"Blacklisting request (hotkey not in WHITELISTED_VALIDATORS): {hk}"
            bt.logging.warning(msg)
            return True, "Unrecognized hotkey"

        bt.logging.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: IdentitySynapse) -> float:
        """Priority derived from the validator's stake (higher stake → higher priority)."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"----------------------------------Face Variation Miner running... {time.time()}")
            time.sleep(30)
