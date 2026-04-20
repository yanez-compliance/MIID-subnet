# The MIT License (MIT)
# Copyright © 2025 YANEZ
# MIID miner: Multi-model image variation generation.

"""
This file makes the miner generate image variations.

What this file does:
1. Keeps a list of image models the miner can use.
2. Picks one of the active models at random (or uses the forced model).
3. Loads that model (importing from MIID.miner.models.*).
4. Uses the model to make the requested image variations.
5. Keeps the same return format so the rest of the miner still works.

=============================================================================
Base models (default pool)
=============================================================================

1. ``flux_klein``
   - Model: ``black-forest-labs/FLUX.2-klein-4B``
   - Why: fastest and lightest, good baseline for GPU-constrained setups.

2. ``pulid``
   - Model: PuLID via Nunchaku (falls back to FLUX.1 Kontext)
   - Why: very high identity fidelity without changing background/lighting.
   - Extra packages: ``pip install nunchaku`` (CUDA required for true PuLID).

3. ``pulid_flux2``
   - Model: ``black-forest-labs/FLUX.2-klein-4B`` (PuLID-FLUX2 adapter compatible)
   - Why: strong identity preservation with the FLUX.2 Klein backbone,
     compatible with Fayens PuLID-FLUX2 adapter weights.

=============================================================================
Recommended alternatives (easy to set up)
=============================================================================

4. ``flux_kontext``
   - Model: ``black-forest-labs/FLUX.1-Kontext-dev``
   - Why: best text-guided editing quality with minimal visual drift.
   - Requires ≥24 GB VRAM on GPU.

5. ``qwen``
   - Model: ``Qwen/Qwen-Image-Edit-2511``
   - Why: strong instruction-following image editor from Qwen.
   - Install latest diffusers from source for this model.

=============================================================================
Paid / API model recommendations (not integrated — for future work)
=============================================================================

These models offer excellent quality via paid APIs.  They are not yet
integrated into this file but are recommended for miners who want top-tier
output and are willing to use API credits:

- **Soul**        — high-quality identity-preserving generation
- **Grok Imagination** — xAI's image generation API
- **Seedream**    — ByteDance's SeedreamDiT image model
- **Nonobana**    — advanced image editing model
- **Nonobana2**   — next-gen Nonobana with improved identity preservation

=============================================================================
How model choice works
=============================================================================

1. If ``MIID_MODEL`` is set, this file uses that exact model.
2. If ``MIID_MODEL`` is not set, this file defaults to ``flux_klein``.
3. Set ``MIID_MODEL_RANDOM=1`` to randomly pick from the 3 base models.
4. The chosen model is kept in memory for the session.

=============================================================================
How intensity works
=============================================================================

- ``light`` keeps the new image closer to the original.
- ``medium`` makes a balanced edit.
- ``far`` allows a bigger change.

=============================================================================
Simple setup steps
=============================================================================

1. Create a Hugging Face token.
2. Put it in your environment:
   ``export HF_TOKEN="hf_..."``
3. Install the packages for the model you want to use.

Packages for ``flux_klein`` and ``pulid_flux2``:
- ``pip install diffusers transformers accelerate``

Packages for ``pulid``:
- ``pip install diffusers transformers accelerate``
- For true PuLID: ``pip install nunchaku`` (requires CUDA)
- Without nunchaku, falls back to FLUX.1 Kontext automatically.

Packages for ``flux_kontext``:
- ``pip install diffusers transformers accelerate``

Packages for ``qwen``:
- ``pip install git+https://github.com/huggingface/diffusers``
- ``pip install transformers accelerate torchvision``

=============================================================================
Helpful environment variables
=============================================================================

- ``MIID_MODEL``: force one model, for example ``flux_klein``
- ``FLUX_DEVICE``: choose ``cuda``, ``mps``, or ``cpu``
- ``MIID_MODEL_RANDOM``: set to ``1`` to randomly pick among base models
- ``MIID_INFERENCE_STEPS``: change number of generation steps
- ``MIID_GUIDANCE_SCALE``: change prompt strength
- ``HF_TOKEN``: Hugging Face access token

=============================================================================
Testing individual models
=============================================================================

Each model can be tested standalone from MIID/miner/models/:
    python MIID/miner/models/flux_klein_model.py [seed_image.png]
    python MIID/miner/models/pulid_model.py [seed_image.png]
    python MIID/miner/models/pulid_flux2_model.py [seed_image.png]
    python MIID/miner/models/flux_kontext_model.py [seed_image.png]
    python MIID/miner/models/qwen_model.py [seed_image.png]

This lets you see the output for a given prompt and check if the model works
on your hardware before running the full miner.
"""

import os
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
    """Prefer GPU when available."""
    explicit = os.environ.get("FLUX_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _resolve_device()

NUM_INFERENCE_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))

GUIDANCE_SCALE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))

DEFAULT_INTENSITY = "medium"

INTENSITY_TO_STRENGTH: Dict[str, float] = {
    "light":  0.35,
    "medium": 0.55,
    "far":    0.75,
}

# =============================================================================
# Available models
# =============================================================================

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # ── Base models (default pool for random selection) ──
    "flux_klein": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "type": "flux_klein",
        "params": "4B",
        "license": "FLUX.1 dev non-commercial (commercial available from BFL)",
        "base": True,
        "notes": (
            "Original default. Lightweight, fast inference. "
            "Good baseline for GPU-constrained setups."
        ),
    },
    "pulid": {
        "model_id": "guozinan/PuLID (Nunchaku) / FLUX.1-Kontext-dev (fallback)",
        "type": "pulid",
        "params": "~12B",
        "license": "NeurIPS 2024 (ByteDance) / FLUX dev non-commercial",
        "base": True,
        "notes": (
            "PuLID: Pure and Lightning ID Customization. Very high identity "
            "fidelity. Uses Nunchaku PuLIDFluxPipeline on CUDA; falls back to "
            "FLUX.1 Kontext when Nunchaku is unavailable."
        ),
    },
    "pulid_flux2": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "type": "pulid_flux2",
        "params": "4B",
        "license": "FLUX.1 dev non-commercial",
        "base": True,
        "notes": (
            "FLUX.2 Klein backbone compatible with Fayens PuLID-FLUX2 adapter "
            "weights (pulid_flux2_klein_v1/v2.safetensors). Strong identity "
            "preservation with reduced artifacts."
        ),
    },
    # ── Recommended alternatives ──
    "flux_kontext": {
        "model_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "type": "flux_kontext",
        "params": "12B",
        "license": "FLUX.1 dev non-commercial (commercial $999/mo from BFL)",
        "base": False,
        "notes": (
            "Context-aware editing model from Black Forest Labs. Best text-guided "
            "editing quality with minimal visual drift. Requires ≥24 GB VRAM."
        ),
    },
    "qwen": {
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "type": "qwen",
        "params": "~14B",
        "license": "Check Qwen model card",
        "base": False,
        "notes": (
            "Qwen's instruction-based image editor. Strong prompt following. "
            "Requires latest diffusers from source and torchvision."
        ),
    },
}

BASE_MODELS = [k for k, v in AVAILABLE_MODELS.items() if v.get("base")]

# =============================================================================
# Module state
# =============================================================================

_cached_pipeline: Any = None
_cached_model_key: Optional[str] = None
_selected_model_key: Optional[str] = None

# =============================================================================
# Model selection
# =============================================================================


def _select_model() -> str:
    """Choose the model for this generation round."""

    forced = os.environ.get("MIID_MODEL", "").strip().lower()
    # Default behavior: randomly pick among base models each call.
    random_flag = os.environ.get("MIID_MODEL_RANDOM", "1").strip().lower() in (
        "1", "true", "yes",
    )

    if forced and forced in AVAILABLE_MODELS:
        selected_model = forced
        logger.info("Model forced via MIID_MODEL env var: %s", selected_model)
    elif random_flag:
        selected_model = random.choice(BASE_MODELS)
        logger.info("Randomly selected model for this round: %s", selected_model)
    else:
        selected_model = "flux_klein"
        logger.info(
            "Using default model: %s (override with MIID_MODEL=..., or MIID_MODEL_RANDOM=1)",
            selected_model,
        )

    cfg = AVAILABLE_MODELS[selected_model]
    logger.info(
        "  model_id=%s  params=%s  license=%s",
        cfg["model_id"], cfg["params"], cfg["license"],
    )
    return selected_model


def _get_selected_model_key() -> str:
    """Return the process-level selected model key, choosing once on first use."""
    global _selected_model_key
    if _selected_model_key is None:
        _selected_model_key = _select_model()
    return _selected_model_key


def get_selected_model_info() -> Dict[str, Any]:
    """Return config dict for the process-level selected model."""
    key = _get_selected_model_key()
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
# Loaders — each imports from the corresponding model module.
# Lazy imports keep startup fast and avoid loading unneeded dependencies.
# =============================================================================


def _load_flux_klein() -> Any:
    from .models.flux_klein_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_pulid() -> Any:
    from .models.pulid_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_pulid_flux2() -> Any:
    from .models.pulid_flux2_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_flux_kontext() -> Any:
    from .models.flux_kontext_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_qwen() -> Any:
    from .models.qwen_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


_MODEL_LOADERS: Dict[str, Any] = {
    "flux_klein":   _load_flux_klein,
    "pulid":        _load_pulid,
    "pulid_flux2":  _load_pulid_flux2,
    "flux_kontext": _load_flux_kontext,
    "qwen":         _load_qwen,
}


def _get_pipeline(model_key: str) -> Any:
    """Load the selected model pipeline and reuse it.

    If loading fails, we fall back to ``flux_klein`` so the miner can still run.
    """
    global _cached_pipeline, _cached_model_key

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
# Generators — each imports from the corresponding model module.
# =============================================================================


def _generate_with_flux_klein(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    from .models.flux_klein_model import generate
    return generate(
        pipe, base_image, prompt,
        intensity=intensity,
        num_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )


def _generate_with_pulid(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    from .models.pulid_model import generate
    return generate(
        pipe, base_image, prompt,
        intensity=intensity,
        num_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )


def _generate_with_pulid_flux2(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    from .models.pulid_flux2_model import generate
    return generate(
        pipe, base_image, prompt,
        intensity=intensity,
        num_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )


def _generate_with_flux_kontext(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    from .models.flux_kontext_model import generate
    return generate(
        pipe, base_image, prompt,
        intensity=intensity,
        num_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )


def _generate_with_qwen(
    pipe: Any, base_image: Image.Image, prompt: str, intensity: str,
) -> Image.Image:
    from .models.qwen_model import generate
    return generate(
        pipe, base_image, prompt,
        intensity=intensity,
        num_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )


_GENERATORS: Dict[str, Any] = {
    "flux_klein":   _generate_with_flux_klein,
    "pulid":        _generate_with_pulid,
    "pulid_flux2":  _generate_with_pulid_flux2,
    "flux_kontext": _generate_with_flux_kontext,
    "qwen":         _generate_with_qwen,
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

    model_key = _get_selected_model_key()
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
