"""Shared CUDA placement for model_testing scripts (CPU offload vs full GPU)."""

from __future__ import annotations

import os
import sys
from typing import Any


def place_diffusers_pipeline(pipe: Any, dev: str, *, default_offload_on_cuda: bool) -> None:
    """Move pipeline to device, or enable model CPU offload on CUDA when requested.

    - If ``default_offload_on_cuda`` is True and ``dev == "cuda"``, offload is used unless
      ``MIID_ENABLE_CPU_OFFLOAD`` is ``0`` / ``false`` / ``no``.
    - If ``default_offload_on_cuda`` is False, offload is used only when
      ``MIID_ENABLE_CPU_OFFLOAD`` is ``1`` / ``true`` / ``yes``.
    """
    if dev != "cuda":
        pipe.to(dev)
        return

    env = os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "").strip().lower()
    if default_offload_on_cuda:
        want_offload = env not in ("0", "false", "no")
    else:
        want_offload = env in ("1", "true", "yes")

    if want_offload:
        try:
            pipe.enable_model_cpu_offload()
            return
        except Exception as exc:  # noqa: BLE001
            print(f"enable_model_cpu_offload failed ({exc}); falling back to pipe.to(cuda)", file=sys.stderr)
    pipe.to(dev)
