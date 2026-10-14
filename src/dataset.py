"""Nạp ảnh phân loại và áp dụng mean/std tính từ tập train."""

import json
from pathlib import Path

from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_classification"


def load_normalization():
    with (DATA_DIR / "normalization.json").open(encoding="utf-8") as file:
        return json.load(file)


def make_dataset(split, augment=False, normalization=None):
    stats = normalization if normalization is not None else load_normalization()
    steps = []
    if augment and split == "train":
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ]
    return datasets.ImageFolder(DATA_DIR / split, transform=transforms.Compose(steps))
