#!/usr/bin/env python3
"""PuLID identity-preserving image variation — pipeline loader and standalone test.

Primary:  Nunchaku PuLIDFluxPipeline (CUDA + nunchaku package required)
Fallback: FLUX.1 Kontext with identity-preserving prompt

Reusable API (imported by generate_variations.py):
    load_pipeline(device, token)  -> pipeline
    generate(pipe, image, prompt) -> PIL Image

Standalone test:
    python pulid_model.py [path/to/seed_image.png]
"""

import gc
import os
import sys
import warnings
from typing import Optional

import torch
from PIL import Image

DEFAULT_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))
INTENSITY_GUIDANCE_MULT = {"light": 0.92, "medium": 1.0, "far": 1.12}

_loaded_backend: Optional[str] = None


def _device() -> str:
    explicit = os.environ.get("FLUX_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _try_load_nunchaku(token: Optional[str] = None):
    """Attempt to load PuLID via Nunchaku. Returns pipeline or None."""
    if not torch.cuda.is_available():
        return None
    try:
        from types import MethodType

        from nunchaku.models.pulid.pulid_forward import pulid_forward
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
        from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
        from nunchaku.utils import get_precision
    except ImportError:
        return None

    try:
        precision = get_precision()
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
            token=token or None,
        )
        pipeline = PuLIDFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            token=token,
        ).to("cuda")
        pipeline.transformer.forward = MethodType(pulid_forward, pipeline.transformer)
        return pipeline
    except Exception as exc:
        warnings.warn(f"Nunchaku PuLID failed ({exc}); will try fallback.", stacklevel=2)
        return None


def _load_kontext_fallback(
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load FLUX.1 Kontext as a fallback when Nunchaku is unavailable."""
    from diffusers import FluxKontextPipeline

    dev = device or _device()
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
        token=token,
    )

    try:
        from ._cuda_place import place_diffusers_pipeline
    except ImportError:
        from _cuda_place import place_diffusers_pipeline

    place_diffusers_pipeline(
        pipe, dev, default_offload_on_cuda=True, prefer_sequential_offload=True,
    )
    if dev == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return pipe


def load_pipeline(
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load PuLID pipeline (Nunchaku if available, else FLUX Kontext fallback).

    Sets module-level ``_loaded_backend`` to ``"nunchaku"`` or ``"kontext"``.
    """
    global _loaded_backend
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    pipe = _try_load_nunchaku(tok)
    if pipe is not None:
        _loaded_backend = "nunchaku"
        return pipe

    print(
        "pulid_model: Nunchaku PuLID unavailable; using FLUX.1 Kontext fallback "
        "(install nunchaku + CUDA for true PuLID).",
        file=sys.stderr,
    )
    _loaded_backend = "kontext"
    return _load_kontext_fallback(device, tok)


def generate(
    pipe,
    image: Image.Image,
    prompt: str,
    intensity: str = "medium",
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Image.Image:
    """Generate a variation using whichever PuLID backend was loaded."""
    mult = INTENSITY_GUIDANCE_MULT.get(intensity, 1.0)
    guidance = (guidance_scale or DEFAULT_GUIDANCE) * mult
    steps = num_steps or DEFAULT_STEPS

    if _loaded_backend == "nunchaku":
        out = pipe(
            prompt,
            id_image=image,
            id_weight=1,
            num_inference_steps=min(steps, 28),
            guidance_scale=guidance,
        )
    else:
        kw = {}
        _h = os.environ.get("MIID_KONTEXT_HEIGHT", "").strip()
        _w = os.environ.get("MIID_KONTEXT_WIDTH", "").strip()
        if _h.isdigit() and _w.isdigit():
            kw["height"], kw["width"] = int(_h), int(_w)
        out = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance,
            **kw,
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

    Usage: python pulid_model.py [path/to/seed_image.png]
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
    out_path = os.path.join(out_dir, "pulid_test.png")

    base = Image.open(seed_path).convert("RGB")
    pipe = load_pipeline(token=token)
    result = generate(pipe, base, TEST_PROMPT)
    result.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
