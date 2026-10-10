"""Nạp ảnh phân loại và áp dụng mean/std tính từ tập train."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed_classification"


def load_normalization():
    with (DATA_DIR / "normalization.json").open(encoding="utf-8") as file:
        return json.load(file)


def make_dataset(split, augment=False, normalization=None):
    stats = normalization if normalization is not None else load_normalization()
    steps = []
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ]
    return datasets.ImageFolder(DATA_DIR / split, transform=transforms.Compose(steps))


def load_normalized_image(path, normalization=None):
    """Trả về ảnh float32 dạng (3, 224, 224) để dùng riêng lẻ."""
    stats = normalization or load_normalization()
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((pixels - mean) / std).transpose(2, 0, 1)
