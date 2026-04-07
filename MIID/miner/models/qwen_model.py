#!/usr/bin/env python3
"""Qwen-Image-Edit-2511 image variation — pipeline loader and standalone test.

Model: Qwen/Qwen-Image-Edit-2511
Type:  Instruction-based image editing
VRAM:  Very large model; CPU offload enabled by default on CUDA

Reusable API (imported by generate_variations.py):
    load_pipeline(device, token)  -> pipeline
    generate(pipe, image, prompt) -> PIL Image

Standalone test:
    python qwen_model.py [path/to/seed_image.png]
"""

import os
import sys
from typing import Optional

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "40"))
TRUE_CFG_SCALE = float(os.environ.get("MIID_TRUE_CFG_SCALE", "4.0"))
DEFAULT_GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "1.0"))


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


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def load_pipeline(
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load and return a ready-to-use QwenImageEditPlusPipeline."""
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise SystemExit(
            "QwenImageEditPlusPipeline not found in your diffusers install.\n"
            "Install latest diffusers from source:\n"
            "  pip install git+https://github.com/huggingface/diffusers"
        ) from exc

    dev = device or _device()
    dtype = _dtype(dev)
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, token=tok, low_cpu_mem_usage=True,
    )

    use_offload = dev.startswith("cuda") and _truthy_env("MIID_ENABLE_CPU_OFFLOAD", "1")
    if use_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)

    pipe.enable_attention_slicing()
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
    pipe.set_progress_bar_config(disable=None)

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
    steps = num_steps or DEFAULT_STEPS
    guidance = guidance_scale or DEFAULT_GUIDANCE
    out = pipe(
        image=[image],
        prompt=prompt,
        generator=torch.manual_seed(0),
        true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=" ",
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=1,
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

    Usage: python qwen_model.py [path/to/seed_image.png]
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
    out_path = os.path.join(out_dir, "qwen_test.png")

    base = Image.open(seed_path).convert("RGB")
    pipe = load_pipeline(token=token)
    result = generate(pipe, base, TEST_PROMPT)
    result.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
