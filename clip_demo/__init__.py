"""CLIP image<->text matching demo helpers."""

from clip_demo.data import (
    SAMPLE_IMAGES,
    fetch_sample_images,
    load_images_from_folder,
)
from clip_demo.encode import encode_images, encode_texts
from clip_demo.model import SUPPORTED_MODELS, load_model
from clip_demo.viz import (
    draw_clip_diagram,
    plot_embedding_2d,
    plot_similarity_matrix,
    plot_topk_retrieval,
    plot_zeroshot_predictions,
)

__all__ = [
    "SAMPLE_IMAGES",
    "SUPPORTED_MODELS",
    "draw_clip_diagram",
    "encode_images",
    "encode_texts",
    "fetch_sample_images",
    "load_images_from_folder",
    "load_model",
    "plot_embedding_2d",
    "plot_similarity_matrix",
    "plot_topk_retrieval",
    "plot_zeroshot_predictions",
]
