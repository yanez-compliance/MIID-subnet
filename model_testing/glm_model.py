#!/usr/bin/env python3
"""Generate one image with GLM-Image (image-to-image, same prompt path as other model_testing scripts)."""

import os

import torch
from PIL import Image

try:
    from diffusers import GlmImagePipeline
except ImportError as exc:
    raise SystemExit(
        "GlmImagePipeline not found in your diffusers install.\n"
        "GLM-Image requires a recent diffusers with glm_image support, e.g.:\n"
        "  pip install git+https://github.com/huggingface/diffusers"
    ) from exc

from _cuda_place import place_diffusers_pipeline

# Exact string the miner builds from VariationRequest.description + .detail
# (see MIID/miner/generate_variations._get_prompt_from_request).
MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

MODEL_NAME = "glm_image"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
MODEL_ID = "zai-org/GLM-Image"
INTENSITY = "medium"
_INTENSITY_GUIDANCE_MULT = {"light": 0.92, "medium": 1.0, "far": 1.12}
# GLM-Image docs use 30 steps / 1.5 guidance as defaults; override with env like other scripts.
NUM_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "30"))
GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "1.5"))


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


def _glm_hw(base: Image.Image) -> tuple[int, int]:
    """Height/width must be multiples of 32 (see diffusers GLM-Image docs)."""
    env_h = os.environ.get("GLM_IMAGE_HEIGHT", "").strip()
    env_w = os.environ.get("GLM_IMAGE_WIDTH", "").strip()
    if env_h.isdigit() and env_w.isdigit():
        h, w = int(env_h), int(env_w)
    else:
        w, h = base.size
    h = max(32, (h // 32) * 32)
    w = max(32, (w // 32) * 32)
    return h, w


def _generator(dev: str):
    if dev == "cpu":
        return torch.Generator(device="cpu").manual_seed(0)
    return torch.Generator(device=dev).manual_seed(0)


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
    height, width = _glm_hw(base)

    pipe = GlmImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )
    place_diffusers_pipeline(pipe, dev, default_offload_on_cuda=True)

    guidance = GUIDANCE * _INTENSITY_GUIDANCE_MULT.get(INTENSITY, 1.0)
    out = pipe(
        prompt=MINER_PROMPT,
        image=[base],
        height=height,
        width=width,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance,
        generator=_generator(dev),
    )
    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
