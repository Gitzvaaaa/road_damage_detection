"""Kiểm thử model và CLI train/evaluate bằng dữ liệu tổng hợp, không cần data thật."""

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
from train import EarlyStopping


ROOT = Path(__file__).resolve().parents[1]


class ComplexCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_blocks_receive_gradients_and_update(self):
        torch.manual_seed(42)
        model = create_model("complex_cnn", 4)
        blocks = [layer for layer in model.modules() if isinstance(layer, ConvBlock)]
        self.assertEqual(len(blocks), 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = [block.layers[0].weight.detach().clone() for block in blocks]
        logits = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2, 4))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 3]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for block in blocks:
            gradient = block.layers[0].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)
        optimizer.step()
        for old_weight, block in zip(before, blocks):
            self.assertFalse(torch.equal(old_weight, block.layers[0].weight))

    def test_checkpoint_round_trip_and_eval_mode(self):
        model = create_model("complex_cnn", 5).eval()
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = model(images)
            torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory(prefix="road_damage_checkpoint_") as temp:
            path = Path(temp) / "weights.pth"
            torch.save(model.state_dict(), path)
            restored = create_model("complex_cnn", 5).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                actual = restored(images)
        self.assertEqual(tuple(actual.shape), (1, 5))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_class_count(self):
        with self.assertRaises(ValueError):
            create_model("complex_cnn", 1)

    def test_early_stopping_resets_after_meaningful_improvement(self):
        stopping = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(stopping.update(1.0))
        self.assertFalse(stopping.update(0.995))
        self.assertFalse(stopping.update(0.98))
        self.assertFalse(stopping.update(0.99))
        self.assertTrue(stopping.update(1.0))
        disabled = EarlyStopping(patience=0)
        self.assertFalse(any(disabled.update(1.0) for _ in range(20)))
        cumulative = EarlyStopping(patience=3, min_delta=0.001)
        for loss in (0.5200, 0.5198, 0.5196, 0.5188):
            self.assertFalse(cumulative.update(loss))
        self.assertEqual(cumulative.bad_epochs, 0)

    def test_plateau_reduces_lr_stops_and_keeps_best_checkpoint_without_test(self):
        import train
        from sklearn.metrics import classification_report
        from torch.utils.data import TensorDataset

        data = TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
        data.classes = ["D00", "D10"]
        data.class_to_idx = {name: i for i, name in enumerate(data.classes)}
        model = nn.Linear(3, 2)
        report = classification_report([0, 1], [0, 0], labels=[0, 1],
                                       target_names=data.classes, output_dict=True, zero_division=0)
        epoch_states = []

        def validation(*args):
            epoch_states.append({name: value.clone() for name, value in model.state_dict().items()})
            return 1.0 + (len(epoch_states) - 1) * 0.1, 0.5, report

        with tempfile.TemporaryDirectory(prefix="road_damage_stopping_") as temp:
            workspace = Path(temp)
            (workspace / "configs").mkdir()
            config = {"epochs": 8, "batch_size": 2, "learning_rate": 0.01, "seed": 42,
                      "lr_scheduler": {"factor": 0.5, "patience": 0, "min_lr": 0.00001},
                      "early_stopping": {"patience": 2, "min_delta": 0.0001}}
            (workspace / "configs/simple_cnn.json").write_text(json.dumps(config))
            with (patch.object(train, "ROOT", workspace),
                  patch.object(train, "make_dataset", return_value=data),
                  patch.object(train, "load_normalization", return_value={"mean": [0.5]*3, "std": [0.25]*3}),
                  patch.object(train, "create_model", return_value=model),
                  patch.object(train, "evaluate_loss_accuracy", side_effect=validation),
                  patch.object(train.subprocess, "run") as subprocess_run,
                  patch.object(sys, "argv", ["train.py", "--device", "cpu", "--skip-test"])):
                train.main()
            run = workspace / "runs/simple_cnn/run_001"
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["completed_epochs"], 3)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertTrue(summary["stopped_early"])
            self.assertEqual(summary["stop_reason"], "validation_loss_plateau")
            self.assertEqual(summary["status"], "trained")
            self.assertNotIn("test_acc", summary)
            self.assertFalse(summary["test_evaluation_requested"])
            self.assertAlmostEqual(summary["final_learning_rate"], 0.0025)
            with (run / "history.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([float(row["learning_rate"]) for row in rows], [0.01, 0.01, 0.005])
            best = torch.load(run / "best_model.pth", weights_only=True)
            for name in best:
                torch.testing.assert_close(best[name], epoch_states[0][name])
            self.assertFalse(torch.equal(best["weight"], epoch_states[-1]["weight"]))
            self.assertEqual(json.loads((run / "validation_report.json").read_text()), report)
            subprocess_run.assert_called_once()
            self.assertEqual(Path(subprocess_run.call_args.args[0][1]).name, "plot_results.py")

    def test_loss_checkpoint_keeps_small_gain_despite_lower_accuracy(self):
        import train
        from sklearn.metrics import classification_report
        from torch.utils.data import TensorDataset

        actual = [0, 1, 0, 1]
        data = TensorDataset(torch.randn(4, 3), torch.tensor(actual))
        data.classes = ["D00", "D10"]
        data.class_to_idx = {name: i for i, name in enumerate(data.classes)}
        model = nn.Linear(3, 2)
        losses = [1.0, 0.8, 0.7998, 0.7999]
        predictions = [actual, [0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0]]
        reports = [classification_report(actual, pred, labels=[0, 1],
                    target_names=data.classes, output_dict=True, zero_division=0) for pred in predictions]
        epoch_states = []

        def validation(*args):
            index = len(epoch_states)
            epoch_states.append({name: value.clone() for name, value in model.state_dict().items()})
            return losses[index], reports[index]["accuracy"], reports[index]

        with tempfile.TemporaryDirectory(prefix="road_damage_loss_checkpoint_") as temp:
            workspace = Path(temp)
            (workspace / "configs").mkdir()
            config = {"epochs": 8, "batch_size": 2, "learning_rate": 0.01, "seed": 42,
                      "checkpoint_monitor": "val_loss", "evaluate_test": False,
                      "early_stopping": {"patience": 2, "min_delta": 0.001}}
            (workspace / "configs/simple_cnn.json").write_text(json.dumps(config))
            with (patch.object(train, "ROOT", workspace),
                  patch.object(train, "make_dataset", return_value=data) as load_data,
                  patch.object(train, "load_normalization", return_value={"mean": [0.5]*3, "std": [0.25]*3}),
                  patch.object(train, "create_model", return_value=model),
                  patch.object(train, "evaluate_loss_accuracy", side_effect=validation),
                  patch.object(train.subprocess, "run") as subprocess_run,
                  patch.object(sys, "argv", ["train.py", "--device", "cpu"])):
                train.main()
            run = workspace / "runs/simple_cnn/run_001"
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["completed_epochs"], 4)
            self.assertEqual(summary["best_epoch"], 3)
            self.assertEqual(summary["best_val_loss"], 0.7998)
            self.assertEqual(summary["best_val_acc"], 0.25)
            self.assertEqual(summary["checkpoint_monitor"], "val_loss")
            self.assertEqual(summary["early_stopping_bad_epochs"], 2)
            self.assertEqual(summary["early_stopping_reference_loss"], 0.8)
            self.assertTrue(summary["stopped_early"])
            self.assertEqual(summary["status"], "trained")
            self.assertFalse(summary["test_evaluation_requested"])
            self.assertNotIn("test_acc", summary)
            self.assertEqual([call.args[0] for call in load_data.call_args_list], ["train", "val"])
            saved = torch.load(run / "best_model.pth", weights_only=True)
            for name in saved:
                torch.testing.assert_close(saved[name], epoch_states[2][name])
            self.assertFalse(torch.equal(saved["weight"], epoch_states[0]["weight"]))
            self.assertEqual(json.loads((run / "validation_report.json").read_text()), reports[2])
            subprocess_run.assert_called_once()
            self.assertEqual(Path(subprocess_run.call_args.args[0][1]).name, "plot_results.py")

    def test_train_evaluate_cli_for_simple_and_complex(self):
        # Chạy CLI thật, bao gồm subprocess evaluate/plot, trong project tạm.
        with tempfile.TemporaryDirectory(prefix="road_damage_pipeline_") as temp:
            workspace = Path(temp)
            for filename in ("dataset.py", "train.py", "evaluate.py", "plot_results.py"):
                shutil.copy2(ROOT / filename, workspace / filename)
            for folder in ("models", "configs"):
                (workspace / folder).mkdir()
                for path in (ROOT / folder).glob("*.*"):
                    if path.suffix in (".py", ".json"):
                        shutil.copy2(path, workspace / folder / path.name)
            data = workspace / "data/train/Japan/processed_classification"
            classes = ["D00", "D10", "D20", "D40"]
            rng = np.random.default_rng(42)
            for split, count in (("train", 2), ("val", 1), ("test", 1)):
                for class_name in classes:
                    folder = data / split / class_name
                    folder.mkdir(parents=True)
                    for index in range(count):
                        pixels = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(folder / f"sample_{index}.jpg")
            normalization = {"mean": [0.5] * 3, "std": [0.25] * 3}
            (data / "normalization.json").write_text(json.dumps(normalization), encoding="utf-8")
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(workspace / "mpl")}
            for model_name in ("simple_cnn", "complex_cnn"):
                with self.subTest(model=model_name):
                    result = subprocess.run(
                        [sys.executable, str(workspace / "train.py"), model_name,
                         "--epochs", "1", "--batch-size", "4", "--evaluate-test"],
                        cwd=workspace, env=env, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=180,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    run = workspace / "runs" / model_name / "run_001"
                    for filename in ("best_model.pth", "architecture.txt", "history.csv",
                                     "training_curves.png", "confusion_matrix.csv",
                                     "confusion_matrix.png", "classification_report.json",
                                     "classification_report.csv", "classification_report.txt",
                                     "validation_report.json", "validation_report.csv"):
                        self.assertGreater((run / filename).stat().st_size, 0)
                    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["status"], "evaluated")
                    self.assertEqual(summary["device"], "cpu")
                    self.assertFalse(summary["runtime"]["amp"])
                    self.assertFalse(summary["runtime"]["pin_memory"])
                    self.assertEqual(summary["completed_epochs"], 1)
                    self.assertEqual(summary["best_epoch"], 1)
                    self.assertFalse(summary["stopped_early"])
                    validation_report = json.loads((run / "validation_report.json").read_text())
                    self.assertEqual(summary["best_val_macro_f1"], validation_report["macro avg"]["f1-score"])
                    self.assertEqual(sum(validation_report[c]["support"] for c in classes), 4)
                    self.assertEqual(summary["normalization"], normalization)
                    self.assertEqual(summary["split_sizes"], {"train": 8, "val": 4, "test": 4})
                    with (run / "confusion_matrix.csv").open(newline="", encoding="utf-8") as file:
                        rows = list(csv.reader(file))
                    self.assertEqual(rows[0][1:], classes)
                    matrix = np.array([[int(value) for value in row[1:]] for row in rows[1:]])
                    self.assertEqual(matrix.sum(), 4)
                    self.assertAlmostEqual(summary["test_acc"], np.trace(matrix) / matrix.sum())
                    report = json.loads((run / "classification_report.json").read_text())
                    self.assertEqual(summary["test_macro_f1"], report["macro avg"]["f1-score"])
                    for class_name in classes:
                        self.assertEqual(report[class_name]["support"], 1)
                        self.assertTrue(0 <= report[class_name]["f1-score"] <= 1)
                    # Nạp lại checkpoint bằng CLI đánh giá và dùng normalization của run.
                    (data / "normalization.json").write_text(
                        json.dumps({"mean": [0.0] * 3, "std": [1.0] * 3}), encoding="utf-8",
                    )
                    evaluated = subprocess.run(
                        [sys.executable, str(workspace / "evaluate.py"), str(run),
                         "--device", "cpu", "--num-workers", "1"],
                        cwd=workspace, env=env, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=120,
                    )
                    self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
                    restored = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                    for key in ("test_acc", "test_loss", "test_macro_f1"):
                        self.assertEqual(restored[key], summary[key])
                    (data / "normalization.json").write_text(json.dumps(normalization), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
