#!/usr/bin/env python3
"""Generate one image with FLUX.1 Kontext (same as MIID miner when MIID_MODEL=flux_kontext)."""

# Accelerate hooks + torch dynamo can spike VRAM on first transformer forward; disable by default.
import os as _os

if _os.environ.get("MIID_DISABLE_TORCH_DYNAMO", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "",
):
    _os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import gc
import os

import torch
from diffusers import FluxKontextPipeline
from PIL import Image

from _cuda_place import place_diffusers_pipeline

MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

MODEL_NAME = "flux_kontext"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
INTENSITY = "medium"
# FluxKontextPipeline has no ``strength``; match miner FLUX_INTENSITY_GUIDANCE_MULT.
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
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    base = Image.open(seed_path).convert("RGB")

    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
    )
    # Kontext: sequential offload first — model offload still OOMs on ~16GB during transformer fwd.
    place_diffusers_pipeline(
        pipe, dev, default_offload_on_cuda=True, prefer_sequential_offload=True
    )
    if dev == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    guidance = GUIDANCE * _INTENSITY_GUIDANCE_MULT.get(INTENSITY, 1.0)
    kw: dict = {}
    _h = os.environ.get("MIID_KONTEXT_HEIGHT", "").strip()
    _w = os.environ.get("MIID_KONTEXT_WIDTH", "").strip()
    if _h.isdigit() and _w.isdigit():
        kw["height"], kw["width"] = int(_h), int(_w)
    out = pipe(
        prompt=MINER_PROMPT,
        image=base,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance,
        **kw,
    )
    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
