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

import time
import typing
import io
import gc
import bittensor as bt
import os
from typing import List, Optional
from PIL import Image

from bittensor.core.errors import NotVerifiedException

# Protocol
from MIID.protocol import IdentitySynapse, S3Submission

# Base miner class
from MIID.base.miner import BaseMinerNeuron


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
        "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54": "RoundTable21",
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
            synapse.s3_submissions = s3_submissions
            bt.logging.info(f"Phase 4: Generated {len(s3_submissions)} S3 submissions")
        except Exception as e:
            bt.logging.error(f"Phase 4: Failed to process image request: {e}")
            synapse.s3_submissions = []

        total_time = time.time() - start_time
        bt.logging.info(f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed.")

        return synapse

    def is_valid_image_bytes(self, image_bytes: bytes) -> bool:
        """Validate whether raw bytes represent a valid image."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
            return True
        except Exception:
            return False

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
