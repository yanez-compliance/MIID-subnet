#!/usr/bin/env python3
"""Generate one image with Qwen-Image-Edit-2511 for local testing."""

import os
import sys
import gc
from importlib.util import find_spec

import torch
from PIL import Image

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError as exc:
    raise SystemExit(
        "QwenImageEditPlusPipeline not found in your diffusers install.\n"
        "Install latest diffusers from source:\n"
        "  pip install git+https://github.com/huggingface/diffusers"
    ) from exc

MODEL_NAME = "qwen_image_edit_2511"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# Keep the same miner-like prompt used in other model_testing scripts.
MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

NUM_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "40"))
TRUE_CFG_SCALE = float(os.environ.get("MIID_TRUE_CFG_SCALE", "4.0"))
# Qwen docs use guidance_scale=1.0 for this pipeline family.
GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "1.0"))


def _check_runtime_deps() -> None:
    if find_spec("torchvision") is None:
        raise SystemExit(
            "Missing dependency: torchvision.\n"
            "Qwen-Image-Edit-2511 loads a Transformers video processor that requires torchvision.\n"
            "Install in this env, then rerun:\n"
            "  pip install torchvision"
        )


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


def _run_generation(pipe: QwenImageEditPlusPipeline, base: Image.Image, steps: int):
    return pipe(
        image=[base],
        prompt=MINER_PROMPT,
        generator=torch.manual_seed(0),
        true_cfg_scale=TRUE_CFG_SCALE,
        negative_prompt=" ",
        num_inference_steps=steps,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=1,
    )


def main() -> None:
    _check_runtime_deps()

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

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )
    # This model is very large; full .to("cuda") can OOM on ~16GB VRAM.
    # Default to CPU offload on CUDA unless disabled.
    use_offload = dev.startswith("cuda") and _truthy_env("MIID_ENABLE_CPU_OFFLOAD", "1")
    if use_offload:
        # Sequential offload has lower VRAM peaks than model offload.
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(dev)

    # Extra memory savings.
    pipe.enable_attention_slicing()
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
    pipe.set_progress_bar_config(disable=None)

    try:
        out = _run_generation(pipe, base, NUM_STEPS)
    except torch.OutOfMemoryError as exc:
        if not dev.startswith("cuda"):
            raise
        print(
            "qwen_model: CUDA OOM during inference. Retrying on CPU (slower, high RAM usage).",
            file=sys.stderr,
        )
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cpu_steps = int(os.environ.get("MIID_CPU_INFERENCE_STEPS", str(min(NUM_STEPS, 20))))
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            token=token,
            low_cpu_mem_usage=True,
        )
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        pipe.set_progress_bar_config(disable=None)
        try:
            out = _run_generation(pipe, base, cpu_steps)
        except Exception:
            raise exc

    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
