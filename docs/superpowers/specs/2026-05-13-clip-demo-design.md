# CLIP Image↔Text Matching Demo — Design

**Date:** 2026-05-13
**Status:** Approved (approach B + decisions captured below)

## Goal

A presentation-quality Jupyter notebook that demonstrates how CLIP places
images and text in a shared embedding space and matches them via cosine
similarity. The notebook is the *demonstration*; a small Python module
holds the reusable plumbing.

## Audience

Technical viewers seeing CLIP for the first time or as a refresher.
The notebook includes a detailed conceptual intro with an inline diagram.

## Decisions (from brainstorming)

| Question | Choice |
|---|---|
| Format | Jupyter notebook |
| Architecture | Notebook + supporting module (approach B) |
| Demos included | Similarity matrix, zero-shot classification, text→image retrieval, embedding-space visualization |
| Image source | Bundled auto-download with override via local folder |
| Model picker | Top-of-notebook config cell with multiple choices |
| CLIP library | `open_clip_torch` |
| Intro depth | Detailed, with a diagram |

## Project layout

```
emb/
├── pyproject.toml          # uv project, CUDA-enabled torch
├── README.md               # Quick start
├── clip_demo/
│   ├── __init__.py
│   ├── model.py            # load_model
│   ├── data.py             # fetch_sample_images, load_images_from_folder
│   ├── encode.py           # encode_images, encode_texts
│   └── viz.py              # plot_* helpers + draw_clip_diagram
├── images/                 # user-droppable (gitkeep)
├── cache/                  # auto-downloaded samples (gitignored)
├── notebooks/
│   └── clip_demo.ipynb     # the demonstration
└── tests/
    └── test_smoke.py
```

## Module contracts

```python
# clip_demo/model.py
SUPPORTED_MODELS: dict[str, list[str]]  # model_name -> list of valid pretrained tags
def load_model(model_name: str, pretrained: str, device: str) -> tuple[Model, Preprocess, Tokenizer]

# clip_demo/data.py
SAMPLE_IMAGES: list[tuple[str, str, str]]  # (filename, url, sha256)
def fetch_sample_images(cache_dir: Path) -> list[Path]
def load_images_from_folder(folder: Path) -> list[Path]

# clip_demo/encode.py
def encode_images(model, preprocess, paths, device, batch_size=8) -> torch.Tensor  # [N, D], L2-normalized
def encode_texts(model, tokenizer, prompts, device) -> torch.Tensor                # [M, D], L2-normalized

# clip_demo/viz.py
def draw_clip_diagram() -> matplotlib.figure.Figure
def plot_similarity_matrix(img_emb, txt_emb, image_paths, prompts) -> Figure
def plot_zeroshot_predictions(img_emb, label_txt_emb, image_paths, labels, top_k=3) -> Figure
def plot_topk_retrieval(query, query_emb, img_emb, image_paths, k=4) -> Figure
def plot_embedding_2d(img_emb, txt_emb, image_labels, text_labels) -> Figure
```

All encoders return L2-normalized tensors so cosine similarity collapses to
`img_emb @ txt_emb.T`.

## Notebook sections

1. **Title + intro markdown** — what CLIP is, dual encoders, contrastive
   training, the shared embedding space. Inline diagram from
   `viz.draw_clip_diagram`.
2. **Config cell** — `MODEL_NAME`, `PRETRAINED`, `IMAGE_DIR`, `DEVICE`.
   Model picker exposes the supported list as a comment.
3. **Load model** — print model name, parameter count, device.
4. **Load images** — fetch samples if `IMAGE_DIR` empty, else load from
   folder. Display thumbnail grid.
5. **Demo 1 — similarity matrix.** Caption-style prompts paired with the
   images; heatmap of cosine similarity + softmax-over-text probabilities.
6. **Demo 2 — zero-shot classification.** Candidate label set with prompt
   template `"a photo of a {label}"`; per-image top-1 prediction with bar
   chart of top-k scores.
7. **Demo 3 — text→image retrieval.** Run a few canned queries
   (`"a cute dog"`, `"food on a plate"`, etc.) and show top-k matches with
   scores; an optional `ipywidgets.Text` cell for interactive querying.
8. **Demo 4 — embedding-space visualization.** Stack image + text
   embeddings, PCA to 2D, scatter with distinct markers, connect each
   image to its matching caption to show co-clustering.
9. **Wrap-up markdown** — recap, suggested next experiments
   (try a different model, add your own images, swap the prompt template).

## Dependencies

`torch`, `torchvision` (CUDA 12.1 index for GPU), `open_clip_torch`,
`pillow`, `numpy`, `matplotlib`, `scikit-learn`, `ipywidgets`,
`jupyterlab`, `requests`, `tqdm`.

Dev: `pytest`.

## Sample images

~10 diverse public-domain images pinned by URL + SHA-256 in
`clip_demo/data.py`. Sourced from Wikimedia Commons / unsplash-style CDNs
that allow direct hotlinks. Cached under `cache/` after first download.

## Error handling

Boundary only:

- Unknown `MODEL_NAME`/`PRETRAINED` → raise with the supported list.
- Image folder empty → fall back to sample fetch with a printed note.
- `device="cuda"` requested but unavailable → fall back to CPU with a
  printed warning.

No defensive try/except inside encoders.

## Testing

One smoke test (`tests/test_smoke.py`):

- Load smallest model on CPU.
- Encode 2 cached sample images and 2 prompts.
- Assert tensor shapes match `[N, D]` / `[M, D]`.
- Assert L2 norms ≈ 1.
- Assert the "correct" caption for an image beats a deliberately-wrong
  caption (logical sanity, not a benchmark).

Runnable via `uv run pytest`. No CI.

## Out of scope

Training/fine-tuning, Gradio UI, web deployment, multi-GPU, video,
multi-language prompts, custom prompt engineering tooling.
