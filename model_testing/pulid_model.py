#!/usr/bin/env python3
"""PuLID-style identity edit: uses Nunchaku PuLIDFluxPipeline on CUDA when available.

If Nunchaku/CUDA is unavailable, falls back to FLUX.1 Kontext img2img with the same miner
prompt so you still get an output file for comparison.
"""

import os
import sys
import warnings

import torch
from PIL import Image

MINER_PROMPT = (
    "Same person, same identity, Change background environment while keeping subject unchanged. "
    "Add religious head covering, Change environment type (office to outdoor, solid color to gradient). "
    "Additionally, include: Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject. "
    "Preserve face identity."
)

MODEL_NAME = "pulid"
SEED_ID = "475c5c38e38b_m_doc"
BACKGROUND_CHANGE = "medium_background_edit_religious_head_covering"
INTENSITY = "medium"
STRENGTH = {"light": 0.35, "medium": 0.55, "far": 0.75}[INTENSITY]
NUM_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))
GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))


def _run_nunchaku_pulid(base: Image.Image, out_path: str, token: str) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from types import MethodType

        from nunchaku.models.pulid.pulid_forward import pulid_forward
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
        from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
        from nunchaku.utils import get_precision
    except ImportError:
        return False

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

        image = pipeline(
            MINER_PROMPT,
            id_image=base,
            id_weight=1,
            num_inference_steps=min(NUM_STEPS, 28),
            guidance_scale=GUIDANCE,
        ).images[0]
        image.save(out_path)
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Nunchaku PuLID failed ({exc}); will try fallback.", stacklevel=1)
        return False


def _run_kontext_fallback(base: Image.Image, out_path: str, token: str) -> None:
    from diffusers import FluxKontextPipeline

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
        token=token,
    )
    pipe = pipe.to(dev)
    out = pipe(
        prompt=MINER_PROMPT,
        image=base,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        strength=STRENGTH,
    )
    out.images[0].save(out_path)


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_TOKEN for Hugging Face model access.")

    here = os.path.dirname(os.path.abspath(__file__))
    seed_path = os.path.join(here, "seed_images", f"{SEED_ID}.png")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_NAME}_{SEED_ID}_{BACKGROUND_CHANGE}.png")

    base = Image.open(seed_path).convert("RGB")

    if _run_nunchaku_pulid(base, out_path, token):
        print(out_path, file=sys.stdout)
        return

    print(
        "pulid_model: Nunchaku PuLID unavailable or failed; using FLUX.1 Kontext img2img fallback "
        "(install nunchaku + CUDA for true PuLID).",
        file=sys.stderr,
    )
    _run_kontext_fallback(base, out_path, token)
    print(out_path, file=sys.stdout)


if __name__ == "__main__":
    main()
