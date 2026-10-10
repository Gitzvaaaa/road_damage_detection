"""Chia ảnh clean của từng nước thành train/val/test và tính mean/std từ tập train."""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COUNTRIES = ["Japan", "Czech"]
OUTPUT_DIR = DATA_DIR / "processed_classification"
SEED = 42


def split_images(images):
    # Một ảnh gốc có thể tạo nhiều crop. Giữ chúng trong cùng một tập.
    source_images = [path.name.split("_obj")[0] + ".jpg" for path in images]
    train_idx, remaining_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED).split(
            images, groups=source_images
        )
    )
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED).split(
            remaining_idx, groups=[source_images[i] for i in remaining_idx]
        )
    )

    splits = [""] * len(images)
    for name, indices in (
        ("train", train_idx),
        ("val", remaining_idx[val_idx]),
        ("test", remaining_idx[test_idx]),
    ):
        for index in indices:
            splits[index] = name
    return source_images, splits


def main():
    # Chia riêng từng nước để train/val/test đều có ảnh của mọi nước theo cùng tỉ lệ.
    samples = []
    for country in COUNTRIES:
        clean_dir = DATA_DIR / "train" / country / "processed_raw/clean"
        images = sorted(clean_dir.glob("*/*.jpg"))
        if not images:
            raise FileNotFoundError(f"Không có ảnh trong {clean_dir}")
        source_images, splits = split_images(images)
        samples += [(country, *sample) for sample in zip(images, source_images, splits)]
    classes = sorted({path.parent.name for _, path, _, _ in samples})

    # Thư mục này chỉ chứa kết quả do script tạo, nên chạy lại sẽ tạo mới.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ("train", "val", "test"):
        for class_name in classes:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True)

    counts = {split: Counter() for split in ("train", "val", "test")}
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    with (OUTPUT_DIR / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["country", "split", "class_name", "source_image", "source_path", "output_path"])

        for country, path, source_image, split in samples:
            class_name = path.parent.name
            relative_output = Path(split) / class_name / path.name
            shutil.copy2(path, OUTPUT_DIR / relative_output)
            writer.writerow([
                country, split, class_name, source_image,
                path.relative_to(ROOT).as_posix(), relative_output.as_posix(),
            ])
            counts[split][class_name] += 1

            if split == "train":
                with Image.open(path) as image:
                    pixels = np.asarray(image, dtype=np.float64) / 255.0
                channel_sum += pixels.sum(axis=(0, 1))
                channel_square_sum += np.square(pixels).sum(axis=(0, 1))
                pixel_count += pixels.shape[0] * pixels.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_square_sum / pixel_count - mean**2)
    (OUTPUT_DIR / "normalization.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "class_to_idx.json").write_text(
        json.dumps({name: index for index, name in enumerate(classes)}, indent=2),
        encoding="utf-8",
    )

    for split, class_counts in counts.items():
        print(f"{split}: {sum(class_counts.values())} images - {dict(class_counts)}")
    print(f"mean: {mean.tolist()}")
    print(f"std:  {std.tolist()}")


if __name__ == "__main__":
    main()
