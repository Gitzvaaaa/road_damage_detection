import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import load_normalization, make_dataset
from models import MODEL_NAMES, create_model, model_normalization


ROOT = Path(__file__).resolve().parent


def pick_device(requested):
    if requested == "directml":
        import torch_directml
        return "directml", torch_directml.device()
    if requested is not None:
        return requested, torch.device(requested)
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    # Tự chọn thiết bị: dùng CPU nếu DirectML không khả dụng.
    try:
        import torch_directml
        if torch_directml.is_available():
            return "directml", torch_directml.device()
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu", torch.device("cpu")


def next_run_dir(model_name):
    model_dir = ROOT / "runs" / model_name
    numbers = [int(p.name[4:]) for p in model_dir.glob("run_*")
               if p.is_dir() and p.name[4:].isdigit()]
    return model_dir / f"run_{max(numbers, default=0) + 1:03d}"


class EarlyStopping:
    def __init__(self, patience=0, min_delta=0.0001):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.bad_epochs = float("inf"), 0

    def update(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.bad_epochs = val_loss, 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience


def make_scheduler(optimizer, config):
    if not config.get("lr_scheduler"):
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", threshold_mode="abs", **config["lr_scheduler"])


def finetune(model, checkpoint_path, config, backend):
    # Fine-tune từ head tốt nhất, không phải epoch head cuối cùng.
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.set_finetune(True)
    return torch.optim.Adam([
        {"params": [parameter for parameter in model.features.parameters() if parameter.requires_grad],
         "lr": config["finetune_learning_rate"]},
        {"params": model.classifier.parameters(), "lr": config["finetune_head_learning_rate"]},
    ], weight_decay=config.get("weight_decay", 0.0),
       foreach=False if backend == "directml" else None)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp=False):
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * labels.size(0)
        correct += (outputs.detach().argmax(dim=1) == labels).sum()
    return total_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def evaluate_loss_accuracy(model, loader, loss_fn, device, amp=False):
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            total_loss += loss * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum()
    return total_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train a road damage classifier")
    parser.add_argument("model", nargs="?", default="simple_cnn", choices=MODEL_NAMES)
    parser.add_argument("--device", choices=["cpu", "cuda", "directml"], default=None)
    parser.add_argument("--seed", type=int, default=None, help="Ghi đè random seed")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of data loading workers (default: 2 on CUDA, 0 on CPU)")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / f"{args.model}.json").read_text(encoding="utf-8"))
    transfer = args.model == "transfer_model"
    if transfer:
        config["epochs"] = config["head_epochs"] + config["finetune_epochs"]
    if args.seed is not None:
        config["seed"] = args.seed
    monitor = config.get("checkpoint_monitor", "val_acc")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    backend, device = pick_device(args.device)
    amp = backend == "cuda" and torch.cuda.get_device_capability()[0] >= 6
    if backend == "cuda":
        torch.backends.cudnn.benchmark = True

    num_workers = args.num_workers if args.num_workers is not None else (2 if backend == "cuda" else 0)
    pin_memory = backend == "cuda"
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    normalization = model_normalization(args.model) or load_normalization()
    train_data = make_dataset("train", augment=True, normalization=normalization)
    val_data = make_dataset("val", normalization=normalization)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    model = create_model(args.model, len(train_data.classes), config).to(device)
    # Label smoothing chỉ dùng khi train; validation loss luôn là cross-entropy thường.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    val_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    parameters = model.classifier.parameters() if transfer else model.parameters()
    optimizer = torch.optim.Adam(
        parameters, lr=config["head_learning_rate"] if transfer else config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
        foreach=False if backend == "directml" else None)
    stopping = EarlyStopping(**({} if transfer else config.get("early_stopping", {})))
    scheduler = None if transfer else make_scheduler(optimizer, config)

    run_dir = next_run_dir(args.model)
    run_dir.mkdir(parents=True)
    summary = {
        "model": args.model, "run": run_dir.name, "config": config,
        "classes": train_data.classes, "normalization": normalization,
        "split_sizes": {"train": len(train_data), "val": len(val_data)},
        "runtime": {
            "device": str(device),
            "amp": amp,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "cudnn_benchmark": backend == "cuda",
        },
        "checkpoint_monitor": monitor, "status": "training",
    }
    if transfer:
        from models.transfer_model import ARCHITECTURE, REPO_ID, REVISION
        summary.update(architecture=ARCHITECTURE, finetune_blocks=[7, 8, 9],
                       pretrained={"repo_id": REPO_ID, "revision": REVISION, "filename": "best.pt"})
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Device: {device} | AMP: {amp} | Workers: {num_workers} | Run: {run_dir}", flush=True)
    best_score = (float("-inf"), float("-inf"))
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            + (["stage", "head_learning_rate"] if transfer else []))
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            if transfer and epoch == config["head_epochs"] + 1:
                optimizer = finetune(model, run_dir / "best_model.pth", config, backend)
                summary.update(finetune_start_epoch=epoch, head_best_epoch=summary["best_epoch"])
            stage = "head" if transfer and epoch <= config["head_epochs"] else "finetune"
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device, amp)
            val_loss, val_acc = evaluate_loss_accuracy(model, val_loader, val_loss_fn, device, amp)
            row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc, learning_rate=learning_rate)
            if transfer:
                row.update(stage=stage, head_learning_rate=optimizer.param_groups[-1]["lr"])
            writer.writerow(row)
            file.flush()
            print(f"Epoch {epoch}/{config['epochs']} | train loss {train_loss:.4f}, acc {train_acc:.2%} | "
                  f"val loss {val_loss:.4f}, acc {val_acc:.2%} | lr {learning_rate:.2g}", flush=True)
            # Phần tử thứ hai quyết định khi chỉ số chính bằng nhau.
            score = (val_acc, -val_loss) if monitor == "val_acc" else (-val_loss, val_acc)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), run_dir / "best_model.pth")
                summary.update(best_epoch=epoch, best_val_acc=val_acc, best_val_loss=val_loss)
                if transfer:
                    summary["best_stage"] = stage
            if scheduler is not None:
                scheduler.step(val_loss)
            should_stop = stopping.update(val_loss)
            summary.update(completed_epochs=epoch, stopped_early=should_stop and epoch < config["epochs"],
                           stop_reason="validation_loss_plateau" if should_stop else "max_epochs",
                           final_learning_rate=optimizer.param_groups[0]["lr"])
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if should_stop:
                break

    summary["status"] = "trained"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
    print(f"Saved: {run_dir} | Best epoch: {summary['best_epoch']} | {summary['stop_reason']}")


if __name__ == "__main__":
    main()
