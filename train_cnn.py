"""Huấn luyện CNN và lưu biểu đồ train/validation để xem overfit."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/train/Japan/processed_classification"
RUN_DIR = ROOT / "runs/cnn_with_metrics"
MODEL_PATH = RUN_DIR / "best_model.pth"
HISTORY_PATH = RUN_DIR / "history.csv"
SUMMARY_PATH = RUN_DIR / "summary.json"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.network(images)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with (DATA_DIR / "normalization.json").open(encoding="utf-8") as file:
        stats = json.load(file)

    # ToTensor đưa pixel về [0, 1]; Normalize dùng mean/std tính từ train.
    normalize = transforms.Normalize(stats["mean"], stats["std"])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_data = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_data = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = SimpleCNN(num_classes=len(train_data.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "classes": train_data.classes,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Device: {device} | Classes: {train_data.classes}")
    best_val_acc = -1
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
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

            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / len(train_data),
                "train_acc": train_correct / len(train_data),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            writer.writerow(row)
            file.flush()
            subprocess.run([sys.executable, str(ROOT / "plot_history.py")], check=True)
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.2%} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.2%}",
                flush=True,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                summary["best_epoch"] = epoch
                torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    summary.update(best_val_acc=best_val_acc, test_loss=test_loss, test_acc=test_acc)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best val acc: {best_val_acc:.2%} | Test acc: {test_acc:.2%}")
    print(f"Saved results: {RUN_DIR}")


if __name__ == "__main__":
    main()
