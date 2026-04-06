#!/usr/bin/env python3
"""Generate one image with GLM-Image (image-to-image, same prompt path as other model_testing scripts).

GLM-Image is ~36GiB weights in bf16; a *full* ``pipe.to("cuda")`` needs roughly 40GiB+ VRAM.
On smaller GPUs, use sequential CPU offload (Diffusers ``model_cpu_offload_seq``), not a full-device load.

Env:
  GLM_PLACEMENT=auto|full|sequential|cpu   (default: auto)
  GLM_MIN_FULL_GPU_GIB=40                 (auto: use sequential if CUDA total VRAM is below this)
  FLUX_DEVICE=cuda|mps|cpu                (device selection; default auto)
  MIID_INFERENCE_STEPS, MIID_GUIDANCE_SCALE, GLM_IMAGE_HEIGHT, GLM_IMAGE_WIDTH — same as before.
"""

from __future__ import annotations

import gc
import os
import sys

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


def _cuda_total_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _placement(dev: str) -> str:
    """auto: sequential on CUDA when VRAM < GLM_MIN_FULL_GPU_GIB; else full GPU load (or CPU)."""
    raw = os.environ.get("GLM_PLACEMENT", "auto").strip().lower()
    if raw in ("full", "sequential", "cpu"):
        return raw
    if raw != "auto":
        print(f"glm_model: unknown GLM_PLACEMENT={raw!r}, using auto", file=sys.stderr)
    if dev == "cpu":
        return "cpu"
    if dev == "mps":
        return "full"
    if dev.startswith("cuda"):
        try:
            min_gib = float(os.environ.get("GLM_MIN_FULL_GPU_GIB", "40"))
        except ValueError:
            min_gib = 40.0
        total = _cuda_total_gib()
        if total is not None and total < min_gib:
            return "sequential"
        return "full"
    return "full"


def _apply_placement(pipe: GlmImagePipeline, dev: str, placement: str) -> None:
    if placement == "cpu":
        pipe.to("cpu")
        return
    if placement == "full":
        try:
            pipe.to(dev)
        except torch.OutOfMemoryError as exc:
            raise SystemExit(
                "glm_model: CUDA OOM while moving the full pipeline to GPU. "
                "GLM-Image needs roughly 40GiB+ VRAM for GLM_PLACEMENT=full.\n"
                "Use sequential offload (default on smaller GPUs): GLM_PLACEMENT=sequential\n"
                "Or run on CPU if you have enough system RAM: GLM_PLACEMENT=cpu FLUX_DEVICE=cpu"
            ) from exc
        return
    if placement == "sequential":
        if not dev.startswith("cuda"):
            pipe.to(dev)
            return
        # Keeps weights mostly on CPU and moves submodules to GPU in order (see pipeline
        # ``model_cpu_offload_seq``). Required when the full checkpoint does not fit in VRAM.
        pipe.enable_sequential_cpu_offload()
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        return
    raise AssertionError(f"unexpected placement {placement!r}")


def _generator_device(dev: str, placement: str) -> str:
    # Noise / scheduler tensors follow the pipeline execution device (CUDA for sequential offload).
    if placement == "sequential" and dev.startswith("cuda"):
        return "cuda"
    return dev


def _generator(gen_dev: str) -> torch.Generator:
    return torch.Generator(device=gen_dev).manual_seed(0)


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
    placement = _placement(dev)
    if placement == "cpu" and dev != "cpu":
        print(
            "glm_model: GLM_PLACEMENT=cpu — loading on CPU (slow; needs large system RAM).",
            file=sys.stderr,
        )
        dev = "cpu"

    dtype = _dtype(dev)
    base = Image.open(seed_path).convert("RGB")
    height, width = _glm_hw(base)

    if dev.startswith("cuda") and placement == "sequential":
        total = _cuda_total_gib()
        print(
            f"glm_model: GLM_PLACEMENT=sequential (~{total:.1f} GiB GPU — full checkpoint does not fit).",
            file=sys.stderr,
        )

    pipe = GlmImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )
    _apply_placement(pipe, dev, placement)
    gen_dev = _generator_device(dev, placement)

    guidance = GUIDANCE * _INTENSITY_GUIDANCE_MULT.get(INTENSITY, 1.0)
    try:
        out = pipe(
            prompt=MINER_PROMPT,
            image=[base],
            height=height,
            width=width,
            num_inference_steps=NUM_STEPS,
            guidance_scale=guidance,
            generator=_generator(gen_dev),
        )
    except RuntimeError as exc:
        err = str(exc).lower()
        if "meta tensor" in err or "out of memory" in err:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise SystemExit(
                "glm_model: inference failed (often transformers/diffusers + sequential offload).\n"
                "Try: pip install -U transformers accelerate diffusers\n"
                "Or a GPU with ~40GiB+ VRAM: GLM_PLACEMENT=full\n"
                "Or CPU with enough RAM: GLM_PLACEMENT=cpu FLUX_DEVICE=cpu"
            ) from exc
        raise
    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
