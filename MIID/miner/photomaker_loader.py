# MIID/miner/photomaker_loader.py
#
# Tencent PhotoMaker's photomaker/__init__.py imports controlnet / insightface helpers and
# often fails on modern diffusers or minimal installs. Loading photomaker.pipeline directly
# after seeding the package namespace avoids that __init__ entirely.

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def get_photomaker_sdxl_pipeline_class():  # type: ignore[no-untyped-def]
    """Return ``PhotoMakerStableDiffusionXLPipeline`` from the Tencent git package.

    Tries a normal import first; if that fails, loads ``model`` + ``model_v2`` + ``pipeline``
    without executing ``photomaker/__init__.py``.
    """
    try:
        from photomaker import PhotoMakerStableDiffusionXLPipeline as _Cls  # type: ignore[import-untyped]

        if hasattr(_Cls, "load_photomaker_adapter"):
            return _Cls
    except Exception:
        pass

    spec = importlib.util.find_spec("photomaker")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "No 'photomaker' package found.\n"
            "  pip uninstall photomaker -y\n"
            "  pip install 'git+https://github.com/TencentARC/PhotoMaker.git'"
        ) from None

    root = Path(list(spec.submodule_search_locations)[0])
    if not (root / "pipeline.py").is_file() or not (root / "model.py").is_file():
        raise ImportError(
            f"photomaker at {root} is not the Tencent PhotoMaker layout (missing model.py/pipeline.py). "
            "Uninstall PyPI 'photomaker' 1.x and install the git fork (see MIID docs / requirements-miner.txt)."
        ) from None

    for key in list(sys.modules):
        if key == "photomaker" or key.startswith("photomaker."):
            del sys.modules[key]

    pkg = types.ModuleType("photomaker")
    pkg.__path__ = [str(root)]
    sys.modules["photomaker"] = pkg

    m = importlib.import_module("photomaker.model")
    pkg.PhotoMakerIDEncoder = m.PhotoMakerIDEncoder
    m2 = importlib.import_module("photomaker.model_v2")
    pkg.PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken = m2.PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
    pipe_mod = importlib.import_module("photomaker.pipeline")
    cls = pipe_mod.PhotoMakerStableDiffusionXLPipeline
    if not hasattr(cls, "load_photomaker_adapter"):
        raise ImportError("Loaded class is not PhotoMakerStableDiffusionXLPipeline")
    return cls
