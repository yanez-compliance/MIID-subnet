#!/usr/bin/env python3
"""Generate one image with FLUX.2 Klein (same pipeline path as MIID miner default)."""

import os

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from _cuda_place import place_diffusers_pipeline

# Exact string the miner builds from VariationRequest.description + .detail
# (see MIID/miner/generate_variations._get_prompt_from_request).
# background_edit / medium + religious_head_covering, matching get_random_background_variation()
# when intensity is medium and accessory is religious_head_covering.
MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

MODEL_NAME = "flux_klein"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
INTENSITY = "medium"
# Flux2KleinPipeline has no ``strength``; match miner: intensity -> guidance multiplier.
_INTENSITY_GUIDANCE_MULT = {"light": 0.92, "medium": 1.0, "far": 1.12}
NUM_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))
GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))


def _device() -> str:
    explicit = os.environ.get("FLUX_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(dev: str) -> torch.dtype:
    if dev.startswith("cuda"):
        return torch.bfloat16
    if dev == "mps":
        return torch.float16
    return torch.float32


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_TOKEN for Hugging Face model access.")

    here = os.path.dirname(os.path.abspath(__file__))
    seed_path = os.path.join(here, "seed_images", f"{SEED_ID}.png")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_NAME}_{SEED_ID}_{BACKGROUND_CHANGE}.png")

    dev = _device()
    dtype = _dtype(dev)
    base = Image.open(seed_path).convert("RGB")

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )
    # Image-conditioned Klein runs VAE encode on the seed; full .to(cuda) often OOMs on ~16GB.
    # Disable offload with MIID_ENABLE_CPU_OFFLOAD=0 for large GPUs.
    place_diffusers_pipeline(pipe, dev, default_offload_on_cuda=True)

    guidance = GUIDANCE * _INTENSITY_GUIDANCE_MULT.get(INTENSITY, 1.0)
    out = pipe(
        prompt=MINER_PROMPT,
        image=[base],
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance,
    )
    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
