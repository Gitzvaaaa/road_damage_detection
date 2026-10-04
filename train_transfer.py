"""Train fc trước, sau đó fine-tune layer4 của ResNet18."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_dataset
from models.transfer_model import build_model, NORMALIZATION
from plot_results import plot_history
from train import next_run_dir


ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, loss_fn, device, optimizer=None, fine_tune=False):
    # Giữ BatchNorm của các block đóng băng ở chế độ eval.
    model.eval()
    if optimizer is not None:
        model.fc.train()
        if fine_tune:
            model.layer4.train()

    total_loss = 0.0
    actual, predicted = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        actual.extend(labels.cpu().tolist())
        predicted.extend(outputs.argmax(dim=1).detach().cpu().tolist())

    accuracy = float(np.mean(np.array(actual) == np.array(predicted)))
    f1 = f1_score(actual, predicted, labels=list(range(model.fc.out_features)),
                  average="macro", zero_division=0)
    return total_loss / len(loader.dataset), accuracy, float(f1)


def main(device=None):
    config = json.loads((ROOT / "configs/transfer_model.json").read_text(encoding="utf-8"))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_data = make_dataset("train", augment=True, normalization=NORMALIZATION)
    val_data = make_dataset("val", normalization=NORMALIZATION)
    if train_data.classes != val_data.classes:
        raise ValueError("Các lớp trong train và val không khớp.")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])

    model = build_model(len(train_data.classes), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    run_dir = next_run_dir("transfer_model")
    run_dir.mkdir(parents=True)
    summary = {
        "model": "transfer_model", "backbone": "resnet18", "config": config,
        "classes": train_data.classes, "normalization": NORMALIZATION,
        "split_sizes": {"train": len(train_data), "val": len(val_data)},
        "checkpoint_monitor": "val_macro_f1",
    }
    best_f1, best_loss = -1.0, float("inf")
    epoch = 0
    print(f"Device: {device} | Run: {run_dir}", flush=True)

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "stage", "train_loss", "train_acc",
                         "val_loss", "val_acc", "val_macro_f1"])
        for stage, epochs in (("head", config["head_epochs"]),
                              ("finetune", config["finetune_epochs"])):
            if epochs == 0:
                continue
            fine_tune = stage == "finetune"
            if fine_tune:
                model.load_state_dict(torch.load(run_dir / "best_model.pth",
                                                map_location=device, weights_only=True))
                for param in model.layer4.parameters():
                    param.requires_grad = True
                params = [
                    {"params": model.fc.parameters(), "lr": config["finetune_lr"]},
                    {"params": model.layer4.parameters(), "lr": config["backbone_lr"]},
                ]
            else:
                params = [{"params": model.fc.parameters(), "lr": config["learning_rate"]}]
            optimizer = torch.optim.Adam(params, weight_decay=config["weight_decay"])

            for step in range(1, epochs + 1):
                epoch += 1
                train_loss, train_acc, _ = run_epoch(
                    model, train_loader, loss_fn, device, optimizer, fine_tune)
                val_loss, val_acc, val_f1 = run_epoch(model, val_loader, loss_fn, device)
                writer.writerow([epoch, stage, train_loss, train_acc, val_loss, val_acc, val_f1])
                file.flush()
                print(f"{stage} {step}/{epochs} | Train acc: {train_acc:.2%} | "
                      f"Val acc: {val_acc:.2%} | Val F1: {val_f1:.2%}", flush=True)

                if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss):
                    best_f1, best_loss = val_f1, val_loss
                    torch.save(model.state_dict(), run_dir / "best_model.pth")
                    summary.update(best_epoch=epoch, best_stage=stage, best_val_loss=val_loss,
                                   best_val_acc=val_acc, best_val_macro_f1=val_f1)

    summary.update(completed_epochs=epoch, status="trained")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_history(run_dir)
    print(f"Saved: {run_dir} | Best epoch: {summary['best_epoch']}")


if __name__ == "__main__":
    main()
