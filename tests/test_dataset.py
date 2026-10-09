"""Kiểm tra nạp dữ liệu mới và tách augmentation khỏi validation/test."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import dataset


class DatasetTests(unittest.TestCase):
    def test_new_layout_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "data/processed_classification"
            stats = {"mean": [0.3, 0.4, 0.5], "std": [0.1, 0.2, 0.3]}
            pixels = np.random.default_rng(42).integers(
                0, 256, (224, 224, 3), dtype=np.uint8
            )
            classes = ["D00", "D10", "D20", "D40"]
            for split in ("train", "val", "test"):
                for name in classes:
                    folder = data_dir / split / name
                    folder.mkdir(parents=True)
                    Image.fromarray(pixels).save(folder / "sample.png")
            (data_dir / "normalization.json").write_text(json.dumps(stats))

            with patch.object(dataset, "DATA_DIR", data_dir):
                for split in ("val", "test"):
                    data = dataset.make_dataset(split)
                    self.assertEqual(data.class_to_idx, dict(zip(classes, range(4))))
                    image, label = data[0]
                    self.assertEqual(label, 0)
                    torch.testing.assert_close(image, data[0][0], rtol=0, atol=0)
                    expected = dataset.load_normalized_image(data.samples[0][0])
                    np.testing.assert_allclose(image.numpy(), expected)
                    with self.assertRaisesRegex(ValueError, "train"):
                        dataset.make_dataset(split, augment=True)

                train = dataset.make_dataset("train", augment=True)
                source = Path(train.samples[0][0])
                original = source.read_bytes()
                with torch.random.fork_rng():
                    torch.manual_seed(42)
                    first, label = train[0]
                    torch.manual_seed(42)
                    torch.testing.assert_close(first, train[0][0], rtol=0, atol=0)
                    second, _ = train[0]
                self.assertEqual(label, 0)
                self.assertEqual(tuple(first.shape), (3, 224, 224))
                self.assertEqual(first.dtype, torch.float32)
                self.assertTrue(torch.isfinite(first).all())
                self.assertFalse(torch.equal(first, second))
                self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
