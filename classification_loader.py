"""Load a prepared classification crop as a normalized CHW float32 array."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_normalized_image(path: str | Path, normalization: dict) -> np.ndarray:
    """Return RGB pixels normalized with train-set mean/std, shape (3, 224, 224)."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    return ((pixels - mean) / std).transpose(2, 0, 1)
