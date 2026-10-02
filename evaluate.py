"""Đánh giá checkpoint tốt nhất và tạo confusion matrix trên tập test."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn

from dataset import make_dataset
from models import create_model
from train import (add_runtime_arguments, configure_runtime, make_loader,
                   move_batch, print_runtime, select_device)


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        runs = sorted(
            path for path in (ROOT / "runs/simple_cnn").glob("run_*")
            if (path / "best_model.pth").is_file()
        )
        if not runs:
            raise FileNotFoundError("Chưa có run simple_cnn nào chứa best_model.pth.")
        run_dir = runs[-1]

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_runtime = summary.get("runtime", {})
    device = select_device(args.device)
    runtime = configure_runtime(
        device, args.num_workers,
        amp=saved_runtime.get("amp", False) and not args.no_amp,
        deterministic=args.deterministic or saved_runtime.get("deterministic", False),
    )
    print_runtime(runtime)
    test_data = make_dataset("test", normalization=summary.get("normalization"))
    if test_data.classes != summary["classes"]:
        raise ValueError("Thứ tự lớp của tập test không khớp với checkpoint.")
    batch_size = summary.get("config", {}).get("batch_size", summary.get("batch_size", 32))
    test_loader = make_loader(test_data, batch_size, runtime)
    model = create_model(summary["model"], len(test_data.classes)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    matrix = [[0] * len(test_data.classes) for _ in test_data.classes]
    total_loss = torch.zeros((), device=device)
    actual_labels = []
    predicted_labels = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = move_batch(images, labels, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=runtime["amp"]):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            total_loss += loss * labels.size(0)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            actual_batch = labels.cpu().tolist()
            actual_labels.extend(actual_batch)
            predicted_labels.extend(predictions)
            for actual, predicted in zip(actual_batch, predictions):
                matrix[actual][predicted] += 1

    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted", *test_data.classes])
        for class_name, row in zip(test_data.classes, matrix):
            writer.writerow([class_name, *row])

    correct = sum(matrix[i][i] for i in range(len(matrix)))
    summary["test_loss"] = total_loss.item() / len(test_data)
    summary["evaluation_runtime"] = runtime
    summary["test_acc"] = correct / len(test_data)
    report_args = {
        "labels": list(range(len(test_data.classes))),
        "target_names": test_data.classes,
        "zero_division": 0,
    }
    report = classification_report(actual_labels, predicted_labels, output_dict=True, **report_args)
    (run_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    report_text = classification_report(actual_labels, predicted_labels, digits=4, **report_args)
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (run_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name in [*test_data.classes, "macro avg", "weighted avg"]:
            writer.writerow([name, *[report[name][key] for key in
                                    ("precision", "recall", "f1-score", "support")]])
    summary["test_macro_precision"] = report["macro avg"]["precision"]
    summary["test_macro_recall"] = report["macro avg"]["recall"]
    summary["test_macro_f1"] = report["macro avg"]["f1-score"]
    summary["test_weighted_f1"] = report["weighted avg"]["f1-score"]
    summary.setdefault("split_sizes", {})["test"] = len(test_data)
    if summary.get("status") in ("trained", "evaluated"):
        summary["status"] = "evaluated"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
    print(f"Test accuracy: {summary['test_acc']:.2%} | Test loss: {summary['test_loss']:.4f}")
    print(report_text)
    print(f"Saved confusion matrix: {run_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
