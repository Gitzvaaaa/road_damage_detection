"""Đánh giá checkpoint tốt nhất và tạo confusion matrix trên tập test."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_dataset
from models import create_model


ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
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
    test_data = make_dataset("test")
    if test_data.classes != summary["classes"]:
        raise ValueError("Thứ tự lớp của tập test không khớp với checkpoint.")
    batch_size = summary.get("config", {}).get("batch_size", summary.get("batch_size", 32))
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(summary["model"], len(test_data.classes)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    matrix = [[0] * len(test_data.classes) for _ in test_data.classes]
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * labels.size(0)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            for actual, predicted in zip(labels.cpu().tolist(), predictions):
                matrix[actual][predicted] += 1

    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted", *test_data.classes])
        for class_name, row in zip(test_data.classes, matrix):
            writer.writerow([class_name, *row])

    correct = sum(matrix[i][i] for i in range(len(matrix)))
    summary["test_loss"] = total_loss / len(test_data)
    summary["test_acc"] = correct / len(test_data)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
    print(f"Test accuracy: {summary['test_acc']:.2%} | Test loss: {summary['test_loss']:.4f}")
    print(f"Saved confusion matrix: {run_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
