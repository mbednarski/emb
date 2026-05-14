"""Encode images and text into L2-normalized CLIP embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image


@torch.inference_mode()
def encode_images(
    model: torch.nn.Module,
    preprocess: Callable,
    paths: Iterable[Path],
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode images at `paths` into a `[N, D]` L2-normalized tensor on `device`."""
    paths = list(paths)
    batches: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in chunk]
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        batches.append(feats)
    return torch.cat(batches, dim=0)


@torch.inference_mode()
def encode_texts(
    model: torch.nn.Module,
    tokenizer: Callable,
    prompts: list[str],
    device: str,
) -> torch.Tensor:
    """Encode text `prompts` into a `[M, D]` L2-normalized tensor on `device`."""
    tokens = tokenizer(prompts).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats
