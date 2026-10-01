"""Huấn luyện model và lưu mỗi lần chạy vào một thư mục run riêng."""

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


def next_run_dir(model_name):
    model_dir = ROOT / "runs" / model_name
    numbers = [
        int(path.name[4:])
        for path in model_dir.glob("run_*")
        if path.is_dir() and path.name[4:].isdigit()
    ]
    return model_dir / f"run_{max(numbers, default=0) + 1:03d}"


def evaluate_loss_accuracy(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "simple_cnn"
    config_path = ROOT / "configs" / f"{model_name}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = make_dataset("train", augment=True)
    val_data = make_dataset("val")
    if train_data.classes != val_data.classes:
        raise ValueError("Tên hoặc thứ tự lớp của train và val không khớp.")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])

    model = create_model(model_name, len(train_data.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    run_dir = next_run_dir(model_name)
    run_dir.mkdir(parents=True)
    summary = {
        "model": model_name,
        "run": run_dir.name,
        "config": config,
        "classes": train_data.classes,
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Device: {device} | Run: {run_dir} | Classes: {train_data.classes}", flush=True)

    best_val_acc = -1
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            train_loss = 0
            train_correct = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            val_loss, val_acc = evaluate_loss_accuracy(model, val_loader, loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / len(train_data),
                "train_acc": train_correct / len(train_data),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            writer.writerow(row)
            file.flush()
            subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
            print(
                f"Epoch {epoch}/{config['epochs']} | "
                f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.2%} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.2%}",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                summary["best_epoch"] = epoch
                summary["best_val_acc"] = val_acc
                torch.save(model.state_dict(), run_dir / "best_model.pth")
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    subprocess.run([sys.executable, str(ROOT / "evaluate.py"), str(run_dir)], check=True)
    print(f"Saved results: {run_dir}")


if __name__ == "__main__":
    main()
