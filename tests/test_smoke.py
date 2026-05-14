"""Smoke test: load smallest model on CPU and verify CLIP behaves sensibly.

Runs offline against cached images if available, otherwise downloads first.
"""

from __future__ import annotations

from pathlib import Path

import torch

from clip_demo import (
    encode_images,
    encode_texts,
    fetch_sample_images,
    load_model,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def test_clip_matches_image_to_correct_caption():
    paths = fetch_sample_images(CACHE)
    assert len(paths) >= 2, "expected at least 2 cached sample images"

    # Pick two visually-distinct images to make the test unambiguous.
    by_name = {p.name: p for p in paths}
    cat = by_name.get("tabby_cat.jpg")
    car = by_name.get("sports_car.jpg")
    assert cat is not None and car is not None, "missing required sample images"
    image_paths = [cat, car]

    model, preprocess, tokenizer = load_model("ViT-B-32", "openai", device="cpu")
    img_emb = encode_images(model, preprocess, image_paths, device="cpu")
    txt_emb = encode_texts(
        model,
        tokenizer,
        ["a photo of a cat", "a photo of a car"],
        device="cpu",
    )

    # Shape and normalization invariants
    assert img_emb.shape[0] == 2
    assert txt_emb.shape[0] == 2
    assert img_emb.shape[1] == txt_emb.shape[1]
    assert torch.allclose(img_emb.norm(dim=-1), torch.ones(2), atol=1e-3)
    assert torch.allclose(txt_emb.norm(dim=-1), torch.ones(2), atol=1e-3)

    # The "correct" caption should beat the wrong one for each image.
    sims = (img_emb @ txt_emb.T).cpu()
    # Row 0 = cat image, col 0 = cat caption -> sims[0,0] > sims[0,1]
    assert sims[0, 0] > sims[0, 1], f"cat image preferred wrong caption: {sims[0]}"
    # Row 1 = car image, col 1 = car caption -> sims[1,1] > sims[1,0]
    assert sims[1, 1] > sims[1, 0], f"car image preferred wrong caption: {sims[1]}"
