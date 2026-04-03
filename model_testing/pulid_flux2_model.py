#!/usr/bin/env python3
"""FLUX.2 family image with the miner prompt.

MIID does not yet load Fayens PuLID-FLUX2 weights on top of Klein; this script uses the same
FLUX.2 Klein img2img path as the subnet miner (black-forest-labs/FLUX.2-klein-4B) so you get
a FLUX.2 output with filename prefix pulid_flux2 for side-by-side experiments.
"""

import os

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

MODEL_NAME = "pulid_flux2"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
INTENSITY = "medium"
STRENGTH = {"light": 0.35, "medium": 0.55, "far": 0.75}[INTENSITY]
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
    pipe = pipe.to(dev)

    out = pipe(
        prompt=MINER_PROMPT,
        image=[base],
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        strength=STRENGTH,
    )
    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
