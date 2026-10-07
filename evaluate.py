"""Đánh giá checkpoint trên test, lưu một báo cáo CSV và hình confusion matrix."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_dataset
from models import create_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a road damage checkpoint")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device == "cuda" and summary.get("runtime", {}).get("amp", False)
    data = make_dataset("test", normalization=summary.get("normalization"))
    if data.classes != summary["classes"]:
        raise ValueError("Thứ tự lớp của test không khớp với checkpoint.")
    batch_size = summary.get("config", {}).get("batch_size", summary.get("batch_size", 32))
    loader = DataLoader(data, batch_size=batch_size)
    model = create_model(summary["model"], len(data.classes)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = torch.zeros((), device=device)
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            total_loss += loss * labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())

    indices = list(range(len(data.classes)))
    report = classification_report(actual, predicted, labels=indices,
                                   target_names=data.classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=indices)
    with (run_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name in [*data.classes, "macro avg", "weighted avg"]:
            writer.writerow([name, *[report[name][key] for key in
                                    ("precision", "recall", "f1-score", "support")]])
    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted", *data.classes])
        for class_name, row in zip(data.classes, matrix):
            writer.writerow([class_name, *row])
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "plot_results.py"),
                    str(run_dir)], check=True)
    summary.update(test_loss=total_loss.item() / len(data), test_acc=float(matrix.trace() / matrix.sum()),
                   test_macro_f1=report["macro avg"]["f1-score"], status="evaluated",
                   test_macro_precision=report["macro avg"]["precision"],
                   test_macro_recall=report["macro avg"]["recall"],
                   test_weighted_f1=report["weighted avg"]["f1-score"],
                   evaluation_runtime={"device": device, "amp": amp, "num_workers": 0})
    summary.setdefault("split_sizes", {})["test"] = len(data)
    summary.pop("test_results_status", None)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Test accuracy: {summary['test_acc']:.2%} | Macro-F1: {summary['test_macro_f1']:.2%}")
    print(f"Saved report and confusion matrix: {run_dir}")


if __name__ == "__main__":
    main()
