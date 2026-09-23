"""Vẽ lại đường train/validation từ runs/cnn_with_metrics/history.csv."""

import csv
import os
from pathlib import Path

# Với môi trường .conda trên Windows, Matplotlib cần các DLL trong Library/bin.
ROOT = Path(__file__).resolve().parent
conda_bin = ROOT / ".conda/Library/bin"
if os.name == "nt" and conda_bin.is_dir():
    os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ["PATH"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR = ROOT / "runs/cnn_with_metrics"
with (RUN_DIR / "history.csv").open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

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
fig.savefig(RUN_DIR / "training_curves.png", dpi=150)
plt.close(fig)
