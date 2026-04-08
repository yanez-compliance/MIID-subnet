#!/usr/bin/env python3
"""FLUX.2 Klein (PuLID-FLUX2 compatible) image variation — pipeline loader and standalone test.

Model: black-forest-labs/FLUX.2-klein-4B
Type:  Image-conditioned generation via FLUX.2 Klein
VRAM:  ~8-16 GB recommended
Note:  PuLID-FLUX2 adapter weights (Fayens/Pulid-Flux2) can be loaded on top
       of this pipeline for stronger identity preservation.  Currently uses the
       base FLUX.2 Klein path.

Reusable API (imported by generate_variations.py):
    load_pipeline(device, token)  -> pipeline
    generate(pipe, image, prompt) -> PIL Image

Standalone test:
    python pulid_flux2_model.py [path/to/seed_image.png]
"""

import os
import sys
from typing import Optional

import torch
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))
INTENSITY_GUIDANCE_MULT = {"light": 0.92, "medium": 1.0, "far": 1.12}


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


def load_pipeline(
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load and return a ready-to-use Flux2KleinPipeline."""
    from diffusers import Flux2KleinPipeline

    dev = device or _device()
    dtype = _dtype(dev)
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, token=tok, low_cpu_mem_usage=True,
    )

    try:
        from ._cuda_place import place_diffusers_pipeline
    except ImportError:
        from _cuda_place import place_diffusers_pipeline

    place_diffusers_pipeline(pipe, dev, default_offload_on_cuda=True)
    return pipe


def generate(
    pipe,
    image: Image.Image,
    prompt: str,
    intensity: str = "medium",
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Image.Image:
    """Generate a single variation and return the PIL Image."""
    mult = INTENSITY_GUIDANCE_MULT.get(intensity, 1.0)
    guidance = (guidance_scale or DEFAULT_GUIDANCE) * mult
    steps = num_steps or DEFAULT_STEPS
    out = pipe(
        prompt=prompt,
        image=[image],
        num_inference_steps=steps,
        guidance_scale=guidance,
    )
    return out.images[0]


# ── Testing ──────────────────────────────────────────────────────────────────

TEST_PROMPT = (
    "Same person, same identity, Change background environment while keeping "
    "subject unchanged. Add religious head covering, Change environment type "
    "(office to outdoor, solid color to gradient). Additionally, include: "
    "Religious head covering (hijab, turban, kippah, taqiyah, etc.) "
    "appropriate to subject. Preserve face identity."
)


def main() -> None:
    """Run a test generation.

    Usage: python pulid_flux2_model.py [path/to/seed_image.png]
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_TOKEN for Hugging Face model access.")

    seed_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "seed_images", "475c5c38e38b_m_doc.png",
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pulid_flux2_test.png")

    base = Image.open(seed_path).convert("RGB")
    pipe = load_pipeline(token=token)
    result = generate(pipe, base, TEST_PROMPT)
    result.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
