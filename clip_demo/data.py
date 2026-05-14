"""Sample-image registry and folder loaders."""

from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Wikimedia's Special:FilePath redirects to the current URL of a file,
# so we can pin filenames without computing the SHA-hash-based path.
_WM_BASE = "https://commons.wikimedia.org/wiki/Special:FilePath/"

# (local_filename, label, wikimedia_filename)
# Verified to resolve via Special:FilePath. Chosen for visual diversity so
# CLIP's behavior is easy to read.
SAMPLE_IMAGES: list[tuple[str, str, str]] = [
    ("golden_retriever.jpg", "a golden retriever dog",
     _WM_BASE + "Golden_Retriever_Carlos_(10581910556).jpg"),
    ("tabby_cat.jpg", "a tabby cat",
     _WM_BASE + "Cat_November_2010-1a.jpg"),
    ("sports_car.jpg", "a sports car",
     _WM_BASE + "2018_Bugatti_Chiron.jpg"),
    ("pizza.jpg", "a pizza on a plate",
     _WM_BASE + "Eq_it-na_pizza-margherita_sep2005_sml.jpg"),
    ("mountain.jpg", "snowy mountains",
     _WM_BASE + "Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"),
    ("beach.jpg", "a tropical beach",
     _WM_BASE + "Maldives.jpg"),
    ("coffee.jpg", "a cup of coffee",
     _WM_BASE + "A_small_cup_of_coffee.JPG"),
    ("eiffel_tower.jpg", "the Eiffel Tower in Paris",
     _WM_BASE + "Tour_Eiffel_Wikimedia_Commons.jpg"),
    ("apple.jpg", "a red apple",
     _WM_BASE + "Red_Apple.jpg"),
    ("books.jpg", "a stack of books",
     _WM_BASE + "Books_HD_(8314929977).jpg"),
]

USER_AGENT = "clip-demo/0.1 (educational; https://example.org/clip-demo) python-requests"
MAX_CACHE_WIDTH = 800  # downsize on cache to keep disk usage modest
INTER_REQUEST_DELAY_S = 0.6  # be polite to Wikimedia, avoid 429s


def _download_one(url: str, dest: Path) -> bool:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    if img.width > MAX_CACHE_WIDTH:
        ratio = MAX_CACHE_WIDTH / img.width
        new_size = (MAX_CACHE_WIDTH, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    tmp = dest.with_suffix(dest.suffix + ".part")
    img.save(tmp, format="JPEG", quality=90)
    tmp.replace(dest)
    return True


def fetch_sample_images(cache_dir: Path) -> list[Path]:
    """Download the sample image set into `cache_dir`, returning local paths.

    Already-cached files are skipped. Failed downloads emit a warning and are
    omitted from the returned list. Images are downsized to `MAX_CACHE_WIDTH`
    on the way in to keep the cache modest.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (name, url)
        for name, _label, url in SAMPLE_IMAGES
        if not (cache_dir / name).exists()
    ]
    if pending:
        for i, (name, url) in enumerate(tqdm(pending, desc="Downloading sample images")):
            try:
                _download_one(url, cache_dir / name)
            except Exception as exc:  # network failure / 404 / 429 — keep going
                warnings.warn(f"Failed to fetch {name}: {exc}", stacklevel=2)
            if i < len(pending) - 1:
                time.sleep(INTER_REQUEST_DELAY_S)

    paths: list[Path] = []
    for name, _label, _url in SAMPLE_IMAGES:
        candidate = cache_dir / name
        if candidate.exists():
            paths.append(candidate)
    return paths


def load_images_from_folder(folder: Path) -> list[Path]:
    """Return image paths under `folder`, filtered by extension, sorted."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def labels_for_sample_paths(paths: list[Path]) -> list[str]:
    """Look up the registered short labels for sample paths.

    Falls back to the filename stem for any path not in `SAMPLE_IMAGES`
    (e.g., user-supplied images).
    """
    by_name = {name: label for name, label, _ in SAMPLE_IMAGES}
    return [by_name.get(p.name, p.stem.replace("_", " ")) for p in paths]
