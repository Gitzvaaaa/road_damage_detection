"""Vẽ đường học và confusion matrix từ các file CSV của một run."""

import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
conda_bin = ROOT / ".conda/Library/bin"
if os.name == "nt" and conda_bin.is_dir():
    os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ["PATH"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(run_dir):
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="Train (augmented)")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [float(row["train_acc"]) * 100 for row in rows], label="Train (augmented)")
    axes[1].plot(epochs, [float(row["val_acc"]) * 100 for row in rows], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(run_dir):
    with (run_dir / "confusion_matrix.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    classes = rows[0][1:]
    matrix = [[int(value) for value in row[1:]] for row in rows[1:]]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of images")
    ax.set_xticks(range(len(classes)), labels=classes)
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Confusion matrix (test set)")
    threshold = max(max(row) for row in matrix) / 2
    for row in range(len(classes)):
        for col in range(len(classes)):
            color = "white" if matrix[row][col] > threshold else "black"
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        runs = sorted(
            path for path in (ROOT / "runs/simple_cnn").glob("run_*")
            if (path / "history.csv").is_file()
        )
        if not runs:
            raise FileNotFoundError("Chưa có run simple_cnn nào chứa history.csv.")
        run_dir = runs[-1]
    if (run_dir / "history.csv").is_file():
        plot_history(run_dir)
    if (run_dir / "confusion_matrix.csv").is_file():
        plot_confusion(run_dir)
    print(f"Saved plots in {run_dir}")


if __name__ == "__main__":
    main()
