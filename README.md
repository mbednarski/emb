# CLIP image ↔ text matching demo

A small Jupyter-notebook demonstration of OpenAI's CLIP model: how it
places images and text in a *shared* embedding space, and how cosine
similarity in that space drives zero-shot classification, retrieval, and
similarity scoring.

Built with [uv](https://docs.astral.sh/uv/), PyTorch (CUDA), and
[`open_clip_torch`](https://github.com/mlfoundations/open_clip).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbednarski/emb/blob/main/notebooks/clip_demo.ipynb)

## Run in Colab (zero setup)

Click the badge above. The first cell auto-detects Colab, clones this
repo, and installs the one missing dependency (`open-clip-torch`) — the
rest (torch, torchvision, numpy, matplotlib, sklearn, ipywidgets) is
already pre-installed on Colab. Then *Runtime → Run all*.

**Switch to a GPU runtime first** (*Runtime → Change runtime type →
T4 GPU*); CPU works but ViT-B-32 image encoding crawls.

## What's inside

- `notebooks/clip_demo.ipynb` — the demonstration: intro + diagram,
  4 demos (similarity matrix, zero-shot classification, text→image
  retrieval, embedding-space PCA), and a wrap-up. Ships *already
  executed* — open it and the figures/probabilities are there.
- `notebooks/clip_demo.html` — the same notebook exported to a
  standalone HTML page, easy to share without Jupyter.
- `clip_demo/` — small reusable module: model loading, image fetching,
  encoding helpers, plotting helpers.
- `tests/test_smoke.py` — fast sanity test (loads ViT-B-32 on CPU,
  encodes 2 images + 2 captions, checks shapes / L2-norms / that CLIP
  prefers the right caption per image).
- `docs/superpowers/specs/2026-05-13-clip-demo-design.md` — design notes.

## Run locally

Requires Python 3.11+ and `uv`. A CUDA GPU is recommended (the
`pyproject.toml` pins the CUDA 12.1 wheels of `torch`/`torchvision`);
CPU works too, just slower.

```powershell
uv sync                                # one-time, ~5-10 min for the CUDA torch wheel
uv run pytest tests/                   # smoke test (downloads sample images on first run)
uv run jupyter lab notebooks/clip_demo.ipynb
```

On the first notebook run, ~10 sample images (~5–10 MB total) are
downloaded into `./cache/` from Wikimedia Commons. Subsequent runs are
offline.

## Using your own images

Drop `.jpg` / `.png` / `.webp` files into `./images/` and re-run the
notebook. If `./images/` is non-empty, the bundled samples are skipped
and your files are used instead.

## Swapping models

The `MODEL_NAME` / `PRETRAINED` cell near the top of the notebook
exposes the supported combinations:

| `MODEL_NAME` | `PRETRAINED` tags                       |
|--------------|------------------------------------------|
| `ViT-B-32`   | `openai`, `laion2b_s34b_b79k`           |
| `ViT-B-16`   | `openai`, `laion2b_s34b_b88k`           |
| `ViT-L-14`   | `openai`, `laion2b_s32b_b82k`           |

ViT-B-32 is the smallest (151M params) and works well for a live demo
on a consumer GPU. ViT-L-14 is sharper but ~3× larger.

## Layout

```
emb/
├── pyproject.toml
├── README.md
├── clip_demo/
│   ├── __init__.py
│   ├── model.py
│   ├── data.py
│   ├── encode.py
│   └── viz.py
├── notebooks/
│   └── clip_demo.ipynb
├── tests/
│   └── test_smoke.py
├── images/                # user-droppable
├── cache/                 # auto-downloaded samples (gitignored)
└── docs/superpowers/specs/2026-05-13-clip-demo-design.md
```
