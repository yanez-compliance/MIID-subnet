#!/usr/bin/env python3
"""Generate one image with PhotoMaker + RealVisXL (same as MIID miner when MIID_MODEL=photomaker)."""

import os
import sys
import types

import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image

from _cuda_place import place_diffusers_pipeline

MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

MODEL_NAME = "photomaker"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
PHOTOMAKER_ADAPTER = "TencentARC/PhotoMaker"
SDXL_BASE = "SG161222/RealVisXL_V4.0"
PHOTOMAKER_TRIGGER = "img"
INTENSITY = "medium"
# MIID/miner/generate_variations.PHOTOMAKER_INTENSITY_TO_GUIDANCE
GUIDANCE = {"light": 7.5, "medium": 5.0, "far": 3.0}[INTENSITY]
NUM_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))


def _ensure_insightface_stubs() -> None:
    if "insightface" in sys.modules:
        return
    try:
        import insightface  # noqa: F401
        return
    except ImportError:
        pass

    def _stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
        if "." in name:
            parent_name, _, child_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, mod)
        return mod

    _stub("insightface")
    _stub("insightface.app")
    _stub("insightface.utils")
    _stub("insightface.utils.face_align")
    _stub("insightface.data")
    sys.modules["insightface.app"].FaceAnalysis = type("FaceAnalysis", (), {})  # type: ignore[attr-defined]
    sys.modules["insightface.data"].get_image = lambda *a, **kw: None  # type: ignore[attr-defined]


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

    _ensure_insightface_stubs()
    try:
        from photomaker import PhotoMakerStableDiffusionXLPipeline  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            "PhotoMaker: wrong or missing package. PyPI 'photomaker' is not Tencent PhotoMaker.\n"
            "  pip uninstall photomaker -y\n"
            "  pip install git+https://github.com/TencentARC/PhotoMaker.git\n"
        ) from e

    here = os.path.dirname(os.path.abspath(__file__))
    seed_path = os.path.join(here, "seed_images", f"{SEED_ID}.png")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_NAME}_{SEED_ID}_{BACKGROUND_CHANGE}.png")

    dev = _device()
    if dev.startswith("cuda"):
        dtype = torch.bfloat16
    elif dev == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    variant = "fp16" if dtype != torch.float32 else None

    base = Image.open(seed_path).convert("RGB")

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE,
        torch_dtype=dtype,
        use_safetensors=True,
        variant=variant,
        token=token,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_photomaker_adapter(
        PHOTOMAKER_ADAPTER,
        weight_name="photomaker-v1.bin",
        trigger_word=PHOTOMAKER_TRIGGER,
        strict=False,
    )
    pipe.fuse_lora()
    place_diffusers_pipeline(pipe, dev, default_offload_on_cuda=True)

    prompt = MINER_PROMPT
    if PHOTOMAKER_TRIGGER not in prompt.split():
        prompt = f"{PHOTOMAKER_TRIGGER} {prompt}"

    out = pipe(
        prompt=prompt,
        input_id_images=[base],
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
    )
    out.images[0].save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
