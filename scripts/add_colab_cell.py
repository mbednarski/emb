"""Inserts a Colab bootstrap cell into notebooks/clip_demo.ipynb (idempotent).

Run once after editing; safe to re-run.
"""
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "clip_demo.ipynb"
MARKER = "# --- Colab bootstrap"

BOOTSTRAP_SRC = """# --- Colab bootstrap (no-op when running locally) ---
# When this notebook is opened in Google Colab, clone the repo so the
# `clip_demo` package is importable, install the one dependency Colab
# does not ship with, and enable widget rendering for the interactive
# retrieval cell. On a local Jupyter kernel this block is skipped.
import sys

if "google.colab" in sys.modules:
    import os
    if not os.path.exists("/content/emb"):
        !git clone -q https://github.com/mbednarski/emb.git /content/emb
    %cd /content/emb
    !pip install -q open-clip-torch
    from google.colab import output as _colab_output
    _colab_output.enable_custom_widget_manager()
    print("Colab bootstrap complete - using GPU runtime is strongly recommended.")
"""


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code" and MARKER in cell.source:
            cell.source = BOOTSTRAP_SRC
            cell.outputs = []
            cell.execution_count = None
            print("Updated existing bootstrap cell.")
            break
    else:
        new_cell = nbformat.v4.new_code_cell(source=BOOTSTRAP_SRC)
        # Clear default execution metadata so it diffs cleanly.
        new_cell.metadata = {}
        nb.cells.insert(1, new_cell)
        print("Inserted bootstrap cell at index 1.")

    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
