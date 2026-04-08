"""Shared CUDA placement for model_testing scripts (CPU offload vs full GPU)."""

from __future__ import annotations

import os
import sys
from typing import Any


def place_diffusers_pipeline(
    pipe: Any,
    dev: str,
    *,
    default_offload_on_cuda: bool,
    prefer_sequential_offload: bool = False,
) -> None:
    """Move pipeline to device, or enable CPU offload on CUDA when requested.

    - If ``default_offload_on_cuda`` is True and ``dev == "cuda"``, offload is used unless
      ``MIID_ENABLE_CPU_OFFLOAD`` is ``0`` / ``false`` / ``no``.
    - If ``default_offload_on_cuda`` is False, offload is used only when
      ``MIID_ENABLE_CPU_OFFLOAD`` is ``1`` / ``true`` / ``yes``.
    - For huge models (e.g. FLUX Kontext), ``prefer_sequential_offload=True`` tries
      ``enable_sequential_cpu_offload()`` first (lower peak VRAM than model offload on ~16GB).
      Disable with ``MIID_SEQUENTIAL_CPU_OFFLOAD=0``.
    """
    if dev != "cuda":
        pipe.to(dev)
        return

    env = os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "").strip().lower()
    if default_offload_on_cuda:
        want_offload = env not in ("0", "false", "no")
    else:
        want_offload = env in ("1", "true", "yes")

    if not want_offload:
        pipe.to(dev)
        return

    seq_ok = os.environ.get("MIID_SEQUENTIAL_CPU_OFFLOAD", "1").strip().lower() not in (
        "0", "false", "no",
    )
    if prefer_sequential_offload and seq_ok:
        try:
            pipe.enable_sequential_cpu_offload()
            return
        except Exception as exc:  # noqa: BLE001
            print(
                f"enable_sequential_cpu_offload failed ({exc}); trying model CPU offload",
                file=sys.stderr,
            )

    try:
        pipe.enable_model_cpu_offload()
        return
    except Exception as exc:  # noqa: BLE001
        print(f"enable_model_cpu_offload failed ({exc}); falling back to pipe.to(cuda)", file=sys.stderr)
    pipe.to(dev)
