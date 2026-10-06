"""Kiểm tra checkpoint, chọn epoch tốt nhất và luồng train/evaluate trên dữ liệu nhỏ."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from models import create_model
from models.complex_cnn import ConvBlock
import train


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILES = {"best_model.pth", "summary.json", "history.csv", "training_curves.png"}
EVAL_FILES = TRAIN_FILES | {"classification_report.csv", "confusion_matrix.png"}


class ComplexCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_gradients_and_checkpoint_round_trip(self):
        model = create_model("complex_cnn", 4)
        blocks = [m for m in model.modules() if isinstance(m, ConvBlock)]
        self.assertEqual(len(blocks), 4)
        images = torch.randn(2, 3, 224, 224)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = [block.layers[0].weight.detach().clone() for block in blocks]
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 4))
        loss = nn.CrossEntropyLoss()(output, torch.tensor([0, 3]))
        loss.backward()
        for block in blocks:
            gradient = block.layers[0].weight.grad
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)
        optimizer.step()
        for old, block in zip(before, blocks):
            self.assertFalse(torch.equal(old, block.layers[0].weight))
        model.eval()
        with tempfile.TemporaryDirectory() as temp, torch.no_grad():
            path = Path(temp) / "model.pth"
            torch.save(model.state_dict(), path)
            restored = create_model("complex_cnn", 4).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)

    def test_early_stopping_threshold(self):
        stopping = train.EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual([stopping.update(v) for v in (1.0, 0.995, 0.98, 0.99, 1.0)],
                         [False, False, False, False, True])
        cumulative = train.EarlyStopping(patience=3, min_delta=0.001)
        self.assertFalse(any(cumulative.update(v) for v in (0.52, 0.5198, 0.5196, 0.5188)))
        self.assertEqual(cumulative.bad_epochs, 0)
        disabled = train.EarlyStopping(patience=0)
        self.assertFalse(any(disabled.update(1.0) for _ in range(20)))

    def test_checkpoint_selection_scheduler_and_stopping(self):
        data = torch.utils.data.TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
        data.classes = ["D00", "D10"]
        # Epoch 3 improves loss by less than min_delta but must still be saved in loss mode.
        losses, accuracies = [1.0, 0.8, 0.7998, 0.7999], [0.5, 0.5, 0.25, 0.25]
        for monitor, expected_epoch in (("val_acc", 2), ("val_loss", 3)):
            with self.subTest(monitor=monitor), tempfile.TemporaryDirectory() as temp:
                workspace = Path(temp)
                (workspace / "configs").mkdir()
                config = {"epochs": 8, "batch_size": 2, "learning_rate": 0.01, "seed": 42,
                          "checkpoint_monitor": monitor,
                          "lr_scheduler": {"factor": 0.5, "patience": 0, "min_lr": 0.00001},
                          "early_stopping": {"patience": 2, "min_delta": 0.001}}
                (workspace / "configs/simple_cnn.json").write_text(json.dumps(config))
                model, states = nn.Linear(3, 2), []

                def validation(*args):
                    index = len(states)
                    states.append({k: v.clone() for k, v in model.state_dict().items()})
                    return losses[index], accuracies[index]

                with (patch.object(train, "ROOT", workspace),
                      patch.object(train, "make_dataset", return_value=data) as load_data,
                      patch.object(train, "load_normalization", return_value={"mean": [0.5]*3, "std": [0.25]*3}),
                      patch.object(train, "create_model", return_value=model),
                      patch.object(train, "evaluate_loss_accuracy", side_effect=validation),
                      patch.object(train, "plot_history"),
                      patch.object(sys, "argv", ["train.py", "--device", "cpu"])):
                    train.main()
                run = workspace / "runs/simple_cnn/run_001"
                summary = json.loads((run / "summary.json").read_text())
                self.assertEqual(summary["best_epoch"], expected_epoch)
                self.assertEqual(summary["completed_epochs"], 4)
                self.assertTrue(summary["stopped_early"])
                self.assertEqual(summary["stop_reason"], "validation_loss_plateau")
                self.assertAlmostEqual(summary["final_learning_rate"], 0.0025)
                self.assertEqual([c.args[0] for c in load_data.call_args_list], ["train", "val"])
                self.assertNotIn("test_acc", summary)
                saved = torch.load(run / "best_model.pth", weights_only=True)
                for name in saved:
                    torch.testing.assert_close(saved[name], states[expected_epoch - 1][name])
                with (run / "history.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([float(r["learning_rate"]) for r in rows], [0.01, 0.01, 0.01, 0.005])

    def test_train_evaluate_cli_and_saved_normalization(self):
        with tempfile.TemporaryDirectory(prefix="road_damage_pipeline_") as temp:
            workspace = Path(temp)
            for name in ("dataset.py", "train.py", "evaluate.py", "plot_results.py"):
                shutil.copy2(ROOT / name, workspace / name)
            shutil.copytree(ROOT / "models", workspace / "models", ignore=shutil.ignore_patterns("__pycache__"))
            (workspace / "configs").mkdir()
            data = workspace / "data/train/Japan/processed_classification"
            classes = ["D00", "D10", "D20", "D40"]
            rng = np.random.default_rng(42)
            for split, count in (("train", 2), ("val", 1), ("test", 1)):
                for name in classes:
                    folder = data / split / name
                    folder.mkdir(parents=True)
                    for index in range(count):
                        Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(folder / f"{index}.jpg")
            normalization = {"mean": [0.5]*3, "std": [0.25]*3}
            stats_path = data / "normalization.json"
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(workspace / "mpl")}

            def run_cli(*args, success=True):
                result = subprocess.run([sys.executable, *args], cwd=workspace, env=env,
                                        capture_output=True, text=True, encoding="utf-8",
                                        errors="replace", timeout=180)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            for model_name in ("simple_cnn", "complex_cnn"):
                with self.subTest(model=model_name):
                    stats_path.write_text(json.dumps(normalization))
                    config = {"epochs": 1, "batch_size": 4, "learning_rate": 0.001, "seed": 42}
                    (workspace / f"configs/{model_name}.json").write_text(json.dumps(config))
                    run_cli("train.py", model_name, "--device", "cpu")
                    run = workspace / "runs" / model_name / "run_001"
                    self.assertEqual({p.name for p in run.iterdir()}, TRAIN_FILES)
                    summary = json.loads((run / "summary.json").read_text())
                    self.assertEqual(summary["normalization"], normalization)
                    self.assertEqual(summary["completed_epochs"], 1)
                    self.assertEqual(summary["status"], "trained")
                    self.assertNotIn("test_acc", summary)
                    run_cli("evaluate.py", str(run), "--device", "cpu")
                    self.assertEqual({p.name for p in run.iterdir()}, EVAL_FILES)
                    for path in run.iterdir():
                        self.assertGreater(path.stat().st_size, 0)
                    evaluated = json.loads((run / "summary.json").read_text())
                    self.assertEqual(evaluated["split_sizes"], {"train": 8, "val": 4, "test": 4})
                    with (run / "classification_report.csv").open() as file:
                        report = {r["class"]: r for r in csv.DictReader(file)}
                    self.assertAlmostEqual(float(report["macro avg"]["f1-score"]), evaluated["test_macro_f1"])
                    self.assertAlmostEqual(float(report["weighted avg"]["recall"]), evaluated["test_acc"])
                    self.assertEqual(sum(int(float(report[c]["support"])) for c in classes), 4)
                    stats_path.write_text(json.dumps({"mean": [0]*3, "std": [1]*3}))
                    run_cli("evaluate.py", str(run), "--device", "cpu")
                    restored = json.loads((run / "summary.json").read_text())
                    for key in ("test_acc", "test_loss", "test_macro_f1"):
                        self.assertEqual(restored[key], evaluated[key])
                    restored["classes"] = list(reversed(classes))
                    (run / "summary.json").write_text(json.dumps(restored))
                    error = run_cli("evaluate.py", str(run), "--device", "cpu", success=False)
                    self.assertIn("ValueError", error.stderr)
                    self.assertEqual({p.name for p in run.iterdir()}, EVAL_FILES)


if __name__ == "__main__":
    unittest.main()
