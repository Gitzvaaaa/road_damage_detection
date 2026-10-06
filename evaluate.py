"""Đánh giá checkpoint tốt nhất và tạo confusion matrix trên tập test."""

import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint and create the test confusion matrix.")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--device", choices=["cpu", "cuda", "directml"], default=None)
    parser.add_argument("--batch-size", type=int, help="Override batch size for this machine.")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0.")
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
    test_data = make_dataset(
        "test", normalization=summary.get("normalization"),
        image_size=summary.get("image_size"),
    )
    if test_data.classes != summary["classes"]:
        raise ValueError("Thứ tự lớp của tập test không khớp với checkpoint.")
    batch_size = summary.get("config", {}).get("batch_size", summary.get("batch_size", 32))
    batch_size = args.batch_size or batch_size
    test_loader = DataLoader(test_data, batch_size=batch_size)

    backend = args.device
    if backend is None:
        backend = "cuda" if torch.cuda.is_available() else "cpu"
        if backend == "cpu":
            try:
                import torch_directml
                if torch_directml.is_available():
                    backend = "directml"
            except (ImportError, OSError, RuntimeError):
                pass
    if backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA không khả dụng. Kiểm tra driver và environment-cuda.yml.")
    if backend == "directml":
        try:
            import torch_directml
            if not torch_directml.is_available():
                raise RuntimeError("Không tìm thấy GPU DirectML.")
            device = torch_directml.device()
        except (ImportError, OSError, RuntimeError) as error:
            parser.error(f"DirectML không khả dụng: {error}. Kiểm tra environment-directml.yml.")
    else:
        device = torch.device(backend)
    print(f"Backend: {backend} | Device: {device}", flush=True)
    if summary.get("backbone") == "MobileNetV2":
        # Giữ khả năng đánh giá các checkpoint MobileNetV2 đã huấn luyện trước đây.
        from torchvision.models import mobilenet_v2
        base_model = mobilenet_v2(weights=None)
        model = nn.Sequential()
        model.add_module("features", base_model.features)
        model.add_module("classifier", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, len(test_data.classes)),
        ))
    else:
        model = create_model(
            summary["model"], len(test_data.classes), pretrained=False,
            fine_tune_blocks=summary.get("config", {}).get("fine_tune_blocks", 3),
            # Các run ResNet50 cũ dùng head 256 neuron.
            hidden_dim=summary.get("config", {}).get("hidden_dim", 256),
            dropout=summary.get("config", {}).get("dropout", 0.5),
        )
    model = model.to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True))
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
    summary["evaluation_backend"] = backend
    summary["evaluation_device"] = str(device)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
    print(f"Test accuracy: {summary['test_acc']:.2%} | Test loss: {summary['test_loss']:.4f}")
    print(f"Saved confusion matrix: {run_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
