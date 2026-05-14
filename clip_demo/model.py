"""CLIP model loading via open_clip."""

from __future__ import annotations

import warnings
from typing import Callable

import open_clip
import torch

SUPPORTED_MODELS: dict[str, list[str]] = {
    "ViT-B-32": ["openai", "laion2b_s34b_b79k"],
    "ViT-B-16": ["openai", "laion2b_s34b_b88k"],
    "ViT-L-14": ["openai", "laion2b_s32b_b82k"],
}


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
        return "cpu"
    return requested


def load_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
) -> tuple[torch.nn.Module, Callable, Callable]:
    """Load an open_clip model, its image preprocessing transform, and tokenizer.

    Returns a tuple `(model, preprocess, tokenizer)`.

    Raises `ValueError` if the (model_name, pretrained) pair is not in
    `SUPPORTED_MODELS`. The model is moved to `device` and set to eval mode.
    """
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(
            f"Unknown model_name {model_name!r}. Supported: {supported}"
        )
    valid_tags = SUPPORTED_MODELS[model_name]
    if pretrained not in valid_tags:
        raise ValueError(
            f"Unknown pretrained tag {pretrained!r} for {model_name}. "
            f"Supported: {', '.join(valid_tags)}"
        )

    device = resolve_device(device)
    # OpenAI's pretrained checkpoints were trained with QuickGELU; force it on
    # to silence the activation-mismatch warning and match the original numerics.
    force_quick_gelu = pretrained == "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        force_quick_gelu=force_quick_gelu,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
