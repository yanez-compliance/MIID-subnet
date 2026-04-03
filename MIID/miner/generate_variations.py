# The MIT License (MIT)
# Copyright © 2025 YANEZ
# MIID miner: Multi-model image variation generation.

"""
This file makes the miner generate image variations.

What this file does:
1. Keeps a list of image models the miner can use.
2. Picks one of the active models at random.
3. Loads that model.
4. Uses the model to make the requested image variations.
5. Keeps the same return format so the rest of the miner still works.

The 3 active models in this file:
1. ``flux_klein``
   - Model: ``black-forest-labs/FLUX.2-klein-4B``
   - Why keep it: fastest and already close to the old setup.
2. ``flux_kontext``
   - Model: ``black-forest-labs/FLUX.1-Kontext-dev``
   - Why add it: better text-guided image editing quality.
3. ``photomaker``
   - Model: ``TencentARC/PhotoMaker`` on top of ``SG161222/RealVisXL_V4.0``
   - Why add it: strong identity preservation from a single face image,
     uses a LoRA adapter so it keeps the face consistent.

How model choice works:
1. If ``MIID_MODEL`` is set, this file uses that exact model.
2. If ``MIID_MODEL`` is not set, this file defaults to ``flux_klein`` (smallest reliable
   path for typical miner GPUs; avoids broken PhotoMaker imports and random picks
   of very large checkpoints).
3. Set ``MIID_MODEL_RANDOM=1`` to restore the old behavior (random choice).
4. The chosen model is kept in memory for the session.

How intensity works:
- ``light`` keeps the new image closer to the original.
- ``medium`` makes a balanced edit.
- ``far`` allows a bigger change.

Simple setup steps:
1. Create a Hugging Face token.
2. Put it in your environment:
   ``export HF_TOKEN="hf_..."``
3. Install the packages for the model you want to use.

Packages for ``flux_klein`` and ``flux_kontext``:
- ``pip install diffusers transformers accelerate``

Packages for ``photomaker``:
- ``pip install diffusers transformers accelerate photomaker``
- Note: the photomaker package tries to import insightface at startup.
  If you do NOT have insightface installed, this file creates harmless
  stubs so the import still works.  You do NOT need to install insightface
  just for PhotoMaker.

Helpful environment variables:
- ``MIID_MODEL``: force one model, for example ``flux_klein``
- ``FLUX_DEVICE``: choose ``cuda``, ``mps``, or ``cpu`` (if unset, uses ``cuda`` when
  available, else ``mps``, else ``cpu`` — **not** forcing ``cpu`` on GPU hosts)
- ``MIID_MODEL_RANDOM``: set to ``1`` to randomly pick among ``AVAILABLE_MODELS``
- ``MIID_INFERENCE_STEPS``: change number of generation steps
- ``MIID_GUIDANCE_SCALE``: change prompt strength
- ``HF_TOKEN``: Hugging Face access token
"""

import os
import sys
import types
import random
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


def _resolve_device() -> str:
    """Prefer GPU when available; defaulting to CPU on CUDA hosts OOMs 16GB RAM loaders."""
    explicit = os.environ.get("FLUX_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _resolve_device()


def _flux_klein_torch_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        return torch.bfloat16
    if DEVICE == "mps":
        return torch.float16
    return torch.float32

NUM_INFERENCE_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))

GUIDANCE_SCALE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))

DEFAULT_INTENSITY = "medium"

# This maps the validator's requested intensity to model edit strength.
# Lower value = safer edit, closer to the original face.
# Higher value = stronger edit, but more risk of changing identity too much.
INTENSITY_TO_STRENGTH: Dict[str, float] = {
    "light":  0.35,
    "medium": 0.55,
    "far":    0.75,
}

# PhotoMaker does not use img2img strength.  It preserves identity through its
# adapter.  Instead we control variation amount with guidance_scale:
#   higher guidance = follows "preserve identity" prompt more strictly
#   lower guidance  = allows more creative deviation
PHOTOMAKER_INTENSITY_TO_GUIDANCE: Dict[str, float] = {
    "light":  7.5,
    "medium": 5.0,
    "far":    3.0,
}

# PhotoMaker needs this word somewhere in the prompt so the adapter activates.
PHOTOMAKER_TRIGGER = "img"

# =============================================================================
# List of models that are active right now.
# Default is flux_klein unless MIID_MODEL or MIID_MODEL_RANDOM=1 (see _select_model).
# =============================================================================

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "flux_klein": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "type": "flux",
        "params": "4B",
        "license": "FLUX.1 dev non-commercial (commercial available from BFL)",
        "notes": (
            "Original default. Lightweight, fast inference. "
            "Good baseline for GPU-constrained setups."
        ),
    },
    "flux_kontext": {
        "model_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "type": "flux_kontext",
        "params": "12B",
        "license": "FLUX.1 dev non-commercial (commercial $999/mo from BFL)",
        "notes": (
            "Context-aware editing model from Black Forest Labs. Best text-guided "
            "editing quality with minimal visual drift across successive edits. "
            "Requires ≥24 GB VRAM on GPU."
        ),
    },
    "photomaker": {
        "model_id": "TencentARC/PhotoMaker",
        "sdxl_base": "SG161222/RealVisXL_V4.0",
        "type": "photomaker",
        "params": "~6.6B + LoRA adapter",
        "license": "Apache-2.0",
        "notes": (
            "Identity-preserving generation using Stacked ID Embedding. "
            "Loads a LoRA adapter on top of RealVisXL (SDXL). Keeps the same "
            "face across generated images. Requires: pip install photomaker"
        ),
    },
}

# ---------------------------------------------------------------------------
# Suggested future improvements
# ---------------------------------------------------------------------------
# These 2 models are NOT active yet. They are listed here so you can see
# what to add next and follow the same steps as the active models above.
#
# How to add one of these models:
# 1. Pick one suggested model below.
# 2. Add a new entry for it inside AVAILABLE_MODELS (copy one of the
#    existing entries and change the values).
# 3. Create a loader function named like: _load_<model_name>()
#    This function downloads and returns the ready-to-use pipeline.
# 4. Create a generation function named like: _generate_with_<model_name>()
#    This function takes (pipe, base_image, prompt, intensity) and returns
#    a PIL Image.
# 5. Register the loader in _MODEL_LOADERS (the dict near line ~300).
# 6. Register the generator in _GENERATORS (the dict near line ~470).
# 7. Install the extra packages that model needs (listed below).
# 8. Test by setting MIID_MODEL to that model's key name, for example:
#    export MIID_MODEL="pulid"
#
# ── Suggested model 1: "pulid" ──
# - Hugging Face: https://huggingface.co/guozinan/PuLID
# - What it is: PuLID (Pure and Lightning ID Customization).
# - Good for: very high identity fidelity without changing background,
#   lighting, or style of the base model's output.
# - Works with: FLUX (PuLID-FLUX) or SDXL (pulid_v1.1).
# - How it works: extracts face embedding via InsightFace, captures visual
#   features via EVA-CLIP, injects identity tokens into the model.
# - Extra packages: pip install insightface onnxruntime
# - License: check the repo (NeurIPS 2024 paper by ByteDance).
#
# ── Suggested model 2: "pulid_flux2" ──
# - Hugging Face: https://huggingface.co/Fayens/Pulid-Flux2
# - What it is: PuLID weights trained specifically for FLUX.2 (klein and dev).
# - Good for: strong identity preservation with reduced artifacts,
#   directly compatible with the FLUX.2 klein model already in this file.
# - Available weights:
#   pulid_flux2_klein_v1.safetensors
#   pulid_flux2_klein_v2.safetensors
#   (dev variants also available)
# - How it works: same InsightFace + EVA-CLIP approach as PuLID, but the
#   weights are native to the FLUX.2 transformer blocks.
# - Extra packages: pip install insightface onnxruntime
#   EVA-CLIP downloads automatically (~800 MB on first run).
# - Recommended strength: 1.0 (normal) or 1.4 (best balance).
# - License: check the repo.
# ---------------------------------------------------------------------------

# =============================================================================
# Module state
# =============================================================================

_selected_model: Optional[str] = None
_cached_pipeline: Any = None
_cached_model_key: Optional[str] = None

# =============================================================================
# Model selection
# =============================================================================


def _select_model() -> str:
    """Choose the model to use for this session."""
    global _selected_model
    if _selected_model is not None:
        return _selected_model

    forced = os.environ.get("MIID_MODEL", "").strip().lower()
    random_flag = os.environ.get("MIID_MODEL_RANDOM", "").strip().lower() in (
        "1", "true", "yes",
    )

    if forced and forced in AVAILABLE_MODELS:
        _selected_model = forced
        logger.info("Model forced via MIID_MODEL env var: %s", _selected_model)
    elif random_flag:
        _selected_model = random.choice(list(AVAILABLE_MODELS.keys()))
        logger.info("Randomly selected model (MIID_MODEL_RANDOM=1): %s", _selected_model)
    else:
        # Default: single stable path for small RAM/VRAM hosts (e.g. 16GB RAM).
        # Avoids photomaker import failures and accidental flux_kontext selection.
        _selected_model = "flux_klein"
        logger.info(
            "Using default model: %s (override with MIID_MODEL=..., or MIID_MODEL_RANDOM=1)",
            _selected_model,
        )

    cfg = AVAILABLE_MODELS[_selected_model]
    logger.info(
        "  model_id=%s  params=%s  license=%s",
        cfg["model_id"], cfg["params"], cfg["license"],
    )
    return _selected_model


def get_selected_model_info() -> Dict[str, Any]:
    """Return config dict for the currently selected model (after first call)."""
    key = _select_model()
    return {"key": key, **AVAILABLE_MODELS[key]}


# =============================================================================
# HF token helper
# =============================================================================


def _get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token.  Set HF_TOKEN or HUGGINGFACE_TOKEN "
            "in your environment, e.g.\n  export HF_TOKEN=\"hf_...\""
        )
    return token


# =============================================================================
# Loaders for each model.
# These are lazy-loaded, which means the code only imports what it needs.
# That helps avoid loading extra packages for models that were not selected.
# =============================================================================


def _load_flux_klein() -> Any:
    from diffusers import Flux2KleinPipeline

    token = _get_hf_token()
    dtype = _flux_klein_torch_dtype()
    pipe = Flux2KleinPipeline.from_pretrained(
        AVAILABLE_MODELS["flux_klein"]["model_id"],
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )
    if os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "").strip().lower() in ("1", "true", "yes"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception as exc:
            logger.warning("MIID_ENABLE_CPU_OFFLOAD set but enable_model_cpu_offload failed: %s", exc)
            pipe = pipe.to(DEVICE)
    else:
        pipe = pipe.to(DEVICE)
    return pipe


def _load_flux_kontext() -> Any:
    from diffusers import FluxKontextPipeline

    token = _get_hf_token()
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    pipe = FluxKontextPipeline.from_pretrained(
        AVAILABLE_MODELS["flux_kontext"]["model_id"],
        torch_dtype=dtype,
        token=token,
    )
    # Kontext ~12B: full .to(cuda) often OOMs on 16GB GPUs. Default to CPU offload on CUDA
    # unless explicitly disabled (MIID_ENABLE_CPU_OFFLOAD=0).
    offload = os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "1").strip().lower() not in (
        "0", "false", "no",
    )
    if DEVICE.startswith("cuda") and offload:
        try:
            pipe.enable_model_cpu_offload()
            return pipe
        except Exception as exc:
            logger.warning(
                "Flux Kontext: enable_model_cpu_offload failed (%s); using full GPU load",
                exc,
            )
    return pipe.to(DEVICE)


def _ensure_insightface_stubs() -> None:
    """The ``photomaker`` package imports ``insightface`` at the top of its
    files even when it is not actually needed for generation.  If insightface
    is not installed this would crash the import.

    This function creates small empty stand-in modules so the import
    succeeds without installing insightface.  If insightface IS installed
    it does nothing.
    """
    if "insightface" in sys.modules:
        return
    try:
        import insightface  # type: ignore[import-untyped]  # noqa: F401
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


def _load_photomaker() -> Any:
    from diffusers import EulerDiscreteScheduler

    _ensure_insightface_stubs()
    try:
        from photomaker import PhotoMakerStableDiffusionXLPipeline  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "PhotoMaker import failed. The PyPI package 'photomaker' is not Tencent PhotoMaker.\n"
            "  pip uninstall photomaker -y\n"
            "  pip install git+https://github.com/TencentARC/PhotoMaker.git\n"
        ) from e

    token = _get_hf_token()
    config = AVAILABLE_MODELS["photomaker"]

    if DEVICE.startswith("cuda"):
        dtype = torch.bfloat16
    elif DEVICE == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    variant = "fp16" if dtype != torch.float32 else None

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        config["sdxl_base"],
        torch_dtype=dtype,
        use_safetensors=True,
        variant=variant,
        token=token,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_photomaker_adapter(
        config["model_id"],
        weight_name="photomaker-v1.bin",
        trigger_word=PHOTOMAKER_TRIGGER,
        strict=False,
    )
    pipe.fuse_lora()
    if DEVICE.startswith("cuda") and os.environ.get(
        "MIID_ENABLE_CPU_OFFLOAD", ""
    ).strip().lower() in ("1", "true", "yes"):
        try:
            pipe.enable_model_cpu_offload()
            return pipe
        except Exception as exc:
            logger.warning("PhotoMaker: CPU offload failed (%s); using .to(cuda)", exc)
    return pipe.to(DEVICE)


_MODEL_LOADERS: Dict[str, Any] = {
    "flux_klein":   _load_flux_klein,
    "flux_kontext":  _load_flux_kontext,
    "photomaker":   _load_photomaker,
}


def _get_pipeline(model_key: str) -> Any:
    """Load the selected model pipeline and reuse it.

    If loading fails, we fall back to ``flux_klein`` so the miner can still run.
    """
    global _cached_pipeline, _cached_model_key, _selected_model

    if _cached_pipeline is not None and _cached_model_key == model_key:
        return _cached_pipeline

    loader = _MODEL_LOADERS.get(model_key)
    if loader is None:
        raise ValueError(
            f"No loader for model '{model_key}'.  "
            f"Available: {list(_MODEL_LOADERS.keys())}"
        )

    logger.info(
        "Loading pipeline: %s (%s) …",
        model_key, AVAILABLE_MODELS[model_key]["model_id"],
    )

    try:
        _cached_pipeline = loader()
    except Exception as exc:
        if model_key == "flux_klein":
            raise
        logger.warning(
            "Failed to load %s: %s — falling back to flux_klein", model_key, exc,
        )
        _cached_pipeline = _load_flux_klein()
        _cached_model_key = "flux_klein"
        _selected_model = "flux_klein"
        return _cached_pipeline

    _cached_model_key = model_key
    logger.info("Pipeline ready: %s", model_key)
    return _cached_pipeline


# =============================================================================
# Helper functions for reading the request and building the prompt.
# =============================================================================


def _get_type_and_intensity(req: Any) -> Tuple[str, str]:
    """Extract .type and .intensity from a VariationRequest-like object or dict."""
    var_type = getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else None)
    intensity = getattr(req, "intensity", None) or (req.get("intensity") if isinstance(req, dict) else None)
    if not var_type:
        raise ValueError("variation_requests entry missing 'type'")
    if intensity not in ("light", "medium", "far"):
        intensity = DEFAULT_INTENSITY
    return (var_type, intensity)


def _get_prompt_from_request(req: Any, var_type: str, intensity: str) -> str:
    """Build generation prompt from protocol fields (description + detail)."""
    description = getattr(req, "description", None) or (req.get("description") if isinstance(req, dict) else None) or ""
    detail = getattr(req, "detail", None) or (req.get("detail") if isinstance(req, dict) else None) or ""
    parts = [p.strip() for p in (description, detail) if p and p.strip()]
    if parts:
        return f"Same person, same identity, {', '.join(parts)}. Preserve face identity."
    return f"Same person, same identity, {var_type} variation ({intensity} intensity). Preserve face identity."


# =============================================================================
# Each model has its own generation function below.
# This makes it easy to swap models or add new ones later.
# =============================================================================


def _generate_with_flux(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    """Generate a variation using FLUX.2 Klein (image-conditioned generation).

    Current ``Flux2KleinPipeline`` does not accept ``strength``; the reference
    image is encoded as conditioning. Approximate intensity via guidance scale.
    """
    mult = {"light": 0.92, "medium": 1.0, "far": 1.12}.get(intensity, 1.0)
    guidance = float(GUIDANCE_SCALE) * mult
    out = pipe(
        prompt=prompt,
        image=[base_image],
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=guidance,
    )
    return out.images[0]


def _generate_with_flux_kontext(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    """Generate a variation using FLUX.1-Kontext (instruction-based editing)."""
    strength = INTENSITY_TO_STRENGTH.get(intensity, 0.55)
    out = pipe(
        prompt=prompt,
        image=base_image,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        strength=strength,
    )
    return out.images[0]


def _generate_with_photomaker(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    """Generate a variation using PhotoMaker (identity-conditioned SDXL).

    PhotoMaker preserves identity through its adapter, not through img2img
    strength.  We control how much the output can deviate via guidance_scale:
    higher guidance follows the "preserve identity" prompt more strictly,
    lower guidance gives more creative freedom.
    """
    if PHOTOMAKER_TRIGGER not in prompt.split():
        prompt = f"{PHOTOMAKER_TRIGGER} {prompt}"

    guidance = PHOTOMAKER_INTENSITY_TO_GUIDANCE.get(intensity, 5.0)

    out = pipe(
        prompt=prompt,
        input_id_images=[base_image],
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=guidance,
    )
    return out.images[0]


# This connects each model type name to the function that generates images.
_GENERATORS: Dict[str, Any] = {
    "flux":          _generate_with_flux,
    "flux_kontext":  _generate_with_flux_kontext,
    "photomaker":    _generate_with_photomaker,
}


# =============================================================================
# Public API
# =============================================================================


def generate_variations(
    base_image: Image.Image,
    variation_requests: List[Any],
) -> List[Dict[str, Any]]:
    """Generate image variations from a base face image.

    Step by step:
    1. Pick the model.
    2. Load the model.
    3. Read the request type and intensity.
    4. Build the prompt.
    5. Generate the image.
    6. Return results in the same format the miner already expects.

    Args:
        base_image: PIL Image of the base face.
        variation_requests: List of validator requests; each has ``.type`` and
            ``.intensity`` (e.g. ``VariationRequest`` or dict).

    Returns:
        List of dicts with ``"image"`` (PIL Image) and ``"variation_type"`` (str).
    """
    if not variation_requests:
        return []

    model_key = _select_model()
    pipe = _get_pipeline(model_key)
    model_type = AVAILABLE_MODELS[model_key]["type"]
    generator = _GENERATORS.get(model_type)

    if generator is None:
        raise RuntimeError(
            f"No generation handler for model type '{model_type}'.  "
            f"Registered types: {list(_GENERATORS.keys())}"
        )

    logger.info(
        "Generating %d variation(s) with %s (%s)",
        len(variation_requests), model_key, AVAILABLE_MODELS[model_key]["model_id"],
    )

    results: List[Dict[str, Any]] = []

    for req in variation_requests:
        var_type, intensity = _get_type_and_intensity(req)
        prompt = _get_prompt_from_request(req, var_type, intensity)

        try:
            gen_image = generator(pipe, base_image, prompt, intensity)
        except Exception as e:
            raise RuntimeError(
                f"Variation failed for {var_type}({intensity}) "
                f"with {model_key}: {e}"
            ) from e

        results.append({
            "image": gen_image,
            "variation_type": var_type,
        })

    return results
