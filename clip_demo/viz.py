"""Plotting helpers for the CLIP demo notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from sklearn.decomposition import PCA


def _short(text: str, n: int = 28) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _load_thumb(path: Path, size: int = 96) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return np.asarray(img)


def draw_clip_diagram() -> Figure:
    """Render a stylized diagram of CLIP's dual-encoder architecture."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Image branch
    ax.add_patch(patches.FancyBboxPatch(
        (0.3, 3.4), 1.6, 0.9, boxstyle="round,pad=0.05",
        facecolor="#d6e9ff", edgecolor="#1f5fb0", linewidth=1.5,
    ))
    ax.text(1.1, 3.85, "Image\n(pixels)", ha="center", va="center", fontsize=10)

    ax.add_patch(patches.FancyBboxPatch(
        (2.4, 3.3), 2.0, 1.1, boxstyle="round,pad=0.05",
        facecolor="#1f5fb0", edgecolor="#0e3a73", linewidth=1.5,
    ))
    ax.text(3.4, 3.85, "Image Encoder\n(ViT)", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

    # Text branch
    ax.add_patch(patches.FancyBboxPatch(
        (0.3, 0.7), 1.6, 0.9, boxstyle="round,pad=0.05",
        facecolor="#ffe1d6", edgecolor="#b04a1f", linewidth=1.5,
    ))
    ax.text(1.1, 1.15, "Text\n(tokens)", ha="center", va="center", fontsize=10)

    ax.add_patch(patches.FancyBboxPatch(
        (2.4, 0.6), 2.0, 1.1, boxstyle="round,pad=0.05",
        facecolor="#b04a1f", edgecolor="#6b2c10", linewidth=1.5,
    ))
    ax.text(3.4, 1.15, "Text Encoder\n(Transformer)", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

    # Arrows into shared space
    ax.annotate("", xy=(5.4, 3.0), xytext=(4.4, 3.85),
                arrowprops=dict(arrowstyle="->", color="#1f5fb0", lw=2))
    ax.annotate("", xy=(5.4, 2.0), xytext=(4.4, 1.15),
                arrowprops=dict(arrowstyle="->", color="#b04a1f", lw=2))

    # Shared embedding space
    ax.add_patch(patches.FancyBboxPatch(
        (5.4, 1.5), 2.6, 2.0, boxstyle="round,pad=0.1",
        facecolor="#f4f0e6", edgecolor="#666", linewidth=1.5, linestyle="--",
    ))
    ax.text(6.7, 3.2, "Shared Embedding Space", ha="center", va="center",
            fontsize=10, fontweight="bold")
    # Scatter a few "embeddings" inside
    rng = np.random.default_rng(0)
    pts_img = rng.uniform([5.6, 1.7], [7.8, 3.0], size=(6, 2))
    pts_txt = pts_img + rng.normal(0, 0.06, size=pts_img.shape)
    ax.scatter(pts_img[:, 0], pts_img[:, 1], c="#1f5fb0", s=40, marker="o", zorder=3)
    ax.scatter(pts_txt[:, 0], pts_txt[:, 1], c="#b04a1f", s=40, marker="^", zorder=3)
    for a, b in zip(pts_img, pts_txt):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#888", lw=0.6, zorder=2)

    # Cosine similarity output
    ax.annotate("", xy=(9.5, 2.5), xytext=(8.0, 2.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.text(9.7, 2.5, "cosine\nsimilarity", ha="left", va="center", fontsize=10)

    ax.set_title("CLIP: dual encoders trained contrastively in a shared space",
                 fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def plot_image_grid(
    paths: list[Path], labels: list[str] | None = None, cols: int = 5
) -> Figure:
    n = len(paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(_load_thumb(paths[i], size=200))
            if labels:
                ax.set_title(_short(labels[i], 22), fontsize=8)
    fig.tight_layout()
    return fig


def plot_similarity_matrix(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    image_paths: list[Path],
    prompts: list[str],
    title: str = "Image ↔ text cosine similarity",
) -> Figure:
    """Heatmap of image×prompt cosine similarities with image thumbnails as row labels."""
    sims = (img_emb @ txt_emb.T).detach().float().cpu().numpy()
    n_imgs, n_txt = sims.shape

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n_txt + 4), max(4, 0.55 * n_imgs + 2)))
    im = ax.imshow(sims, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_txt))
    ax.set_xticklabels([_short(p, 30) for p in prompts], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_imgs))
    ax.set_yticklabels([p.stem.replace("_", " ") for p in image_paths], fontsize=9)
    for i in range(n_imgs):
        for j in range(n_txt):
            ax.text(j, i, f"{sims[i, j]:.2f}", ha="center", va="center",
                    color="white" if sims[i, j] < sims.max() * 0.7 else "black",
                    fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="cosine similarity")
    fig.tight_layout()
    return fig


def plot_zeroshot_predictions(
    img_emb: torch.Tensor,
    label_txt_emb: torch.Tensor,
    image_paths: list[Path],
    labels: list[str],
    top_k: int = 3,
) -> Figure:
    """Show each image with a small horizontal bar chart of its top-k label probabilities."""
    # CLIP-style softmax with a typical temperature of 100 over cosine sims
    logits = 100.0 * (img_emb @ label_txt_emb.T)
    probs = logits.softmax(dim=-1).detach().float().cpu().numpy()

    n = len(image_paths)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.5, rows * 2.6),
                             gridspec_kw={"width_ratios": [1, 1.6] * cols})
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax_img = axes[i // cols, (i % cols) * 2]
        ax_bar = axes[i // cols, (i % cols) * 2 + 1]
        ax_img.axis("off")
        ax_bar.axis("off") if i >= n else None
        if i >= n:
            continue
        ax_img.imshow(_load_thumb(image_paths[i], size=220))
        order = np.argsort(probs[i])[::-1][:top_k]
        names = [_short(labels[k], 22) for k in order]
        vals = probs[i, order]
        ax_bar.barh(range(top_k)[::-1], vals, color="#1f5fb0")
        ax_bar.set_yticks(range(top_k)[::-1])
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("probability", fontsize=8)
        ax_bar.tick_params(axis="x", labelsize=7)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)

    fig.suptitle("Zero-shot classification (CLIP softmax over candidate labels)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_topk_retrieval(
    query: str,
    query_emb: torch.Tensor,  # [1, D]
    img_emb: torch.Tensor,    # [N, D]
    image_paths: list[Path],
    k: int = 4,
) -> Figure:
    """Show the top-k matching images for a single text query."""
    sims = (query_emb @ img_emb.T).squeeze(0).detach().float().cpu().numpy()
    order = np.argsort(sims)[::-1][:k]
    fig, axes = plt.subplots(1, k, figsize=(k * 2.6, 3.2))
    if k == 1:
        axes = [axes]
    for rank, idx in enumerate(order):
        ax = axes[rank]
        ax.imshow(_load_thumb(image_paths[idx], size=220))
        ax.set_title(f"#{rank + 1}  sim={sims[idx]:.3f}\n{image_paths[idx].stem}",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle(f'Top-{k} for query: "{query}"', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_embedding_2d(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    image_labels: list[str],
    text_labels: list[str],
    pair_lines: bool = True,
) -> Figure:
    """PCA-project image+text embeddings to 2D and scatter them in one space.

    If `pair_lines=True` and `image_labels` and `text_labels` are the same
    length, draws a thin line connecting matching pairs to make the
    co-location of related image and text embeddings legible.
    """
    img = img_emb.detach().float().cpu().numpy()
    txt = txt_emb.detach().float().cpu().numpy()
    stacked = np.vstack([img, txt])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(stacked)
    img_xy = coords[: len(img)]
    txt_xy = coords[len(img) :]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(img_xy[:, 0], img_xy[:, 1], c="#1f5fb0", s=90, marker="o",
               edgecolor="white", linewidth=1.2, label="image", zorder=3)
    ax.scatter(txt_xy[:, 0], txt_xy[:, 1], c="#b04a1f", s=90, marker="^",
               edgecolor="white", linewidth=1.2, label="text", zorder=3)

    if pair_lines and len(image_labels) == len(text_labels):
        for (ix, iy), (tx, ty) in zip(img_xy, txt_xy):
            ax.plot([ix, tx], [iy, ty], color="#999", lw=0.7, zorder=2)

    for (x, y), lbl in zip(img_xy, image_labels):
        ax.annotate(_short(lbl, 24), (x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color="#1f5fb0")
    for (x, y), lbl in zip(txt_xy, text_labels):
        ax.annotate(_short(lbl, 24), (x, y), xytext=(6, -10),
                    textcoords="offset points", fontsize=8, color="#b04a1f")

    ax.set_title("CLIP shared embedding space (PCA → 2D)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
