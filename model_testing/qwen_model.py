#!/usr/bin/env python3
"""Run a smaller Qwen2.5-VL image QA test on a local seed image."""

import os

import torch
from transformers import pipeline

MODEL_NAME = "qwen25_vl_7b_instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SEED_ID = "475c5c38e38b_m_doc"
QUESTION = os.environ.get("QWEN_VL_QUESTION", "What is in this image?")
MAX_NEW_TOKENS = int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "128"))


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


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_TOKEN for Hugging Face model access.")

    here = os.path.dirname(os.path.abspath(__file__))
    seed_path = os.path.abspath(os.path.join(here, "seed_images", f"{SEED_ID}.png"))
    if not os.path.isfile(seed_path):
        raise SystemExit(f"Seed image not found: {seed_path}")

    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_NAME}_{SEED_ID}.txt")

    dev = _device()
    dtype = _dtype(dev)
    device_map = "auto" if dev.startswith("cuda") else dev

    pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        dtype=dtype,
        device_map=device_map,
        token=token,
    )

    # load_image() accepts http(s) URLs, a plain filesystem path, or base64 — not file:// URLs.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": seed_path},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    out = pipe(text=messages, max_new_tokens=MAX_NEW_TOKENS)
    text = str(out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(out_path)


if __name__ == "__main__":
    main()
