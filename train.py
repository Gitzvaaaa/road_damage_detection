"""Huấn luyện model và lưu mỗi lần chạy vào một thư mục run riêng."""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import load_normalization, make_dataset
from models import create_model


ROOT = Path(__file__).resolve().parent


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return number


def add_runtime_arguments(parser):
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=nonnegative_int, default=None,
                        help="Default: up to 4 data workers on GPU, 0 on CPU.")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Use only one GPU, even when multiple GPUs are visible.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic cuDNN kernels instead of speed autotuning.")


def select_device(requested="auto"):
    if requested == "cpu":
        return "cpu"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        # Availability alone does not guarantee that the installed CUDA kernels work.
        probe = torch.ones((2, 2), device="cuda")
        probe = probe @ probe
        torch.cuda.synchronize()
    except (RuntimeError, AssertionError) as error:
        if requested == "cuda":
            raise RuntimeError(f"Requested CUDA cannot be used: {error}") from error
        warnings.warn(f"Using CPU: {error}", RuntimeWarning, stacklevel=2)
        return "cpu"
    return "cuda"


def configure_runtime(device, num_workers=None, amp=True, deterministic=False):
    cuda = device == "cuda"
    if num_workers is None:
        num_workers = min(4, max(1, (os.cpu_count() or 1) // 2)) if cuda else 0
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    # FP16 is useful on Kaggle's T4/P100 as well as newer NVIDIA GPUs.
    amp = cuda and amp and torch.cuda.get_device_capability()[0] >= 6
    torch.backends.cudnn.benchmark = cuda and not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name() if cuda else "CPU",
        "cuda_version": torch.version.cuda,
        "amp": amp,
        "num_workers": num_workers,
        "pin_memory": cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic": deterministic,
    }


def make_loader(dataset, batch_size, settings, shuffle=False):
    workers = settings["num_workers"]
    options = {"prefetch_factor": 2} if workers else {}
    if workers:
        # Spawn avoids forking a CUDA-initialized process on Linux/Kaggle.
        options["multiprocessing_context"] = "spawn"
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=settings["pin_memory"],
        persistent_workers=workers > 0, **options,
    )


def move_batch(images, labels, device):
    return (images.to(device, non_blocking=device == "cuda"),
            labels.to(device, non_blocking=device == "cuda"))


def print_runtime(settings):
    print(f"Device: {settings['device']} ({settings['device_name']}) | "
          f"AMP: {settings['amp']} | Workers: {settings['num_workers']} | "
          f"cuDNN benchmark: {settings['cudnn_benchmark']}", flush=True)


def next_run_dir(model_name):
    model_dir = ROOT / "runs" / model_name
    numbers = [
        int(path.name[4:])
        for path in model_dir.glob("run_*")
        if path.is_dir() and path.name[4:].isdigit()
    ]
    return model_dir / f"run_{max(numbers, default=0) + 1:03d}"


class EarlyStopping:
    """Stop after consecutive epochs without a meaningful validation-loss gain."""

    def __init__(self, patience=0, min_delta=0.0001):
        if patience < 0 or min_delta < 0:
            raise ValueError("Early-stopping patience and min_delta must be >= 0.")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def update(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp=False):
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    for images, labels in loader:
        images, labels = move_batch(images, labels, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * labels.size(0)
        correct += (outputs.detach().argmax(dim=1) == labels).sum()
    return total_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def evaluate_loss_accuracy(model, loader, loss_fn, device, amp=False, class_names=None):
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    actual_batches, predicted_batches = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = move_batch(images, labels, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            total_loss += loss * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum()
            if class_names is not None:
                actual_batches.append(labels)
                predicted_batches.append(predicted)
    metrics = (total_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset))
    if class_names is None:
        return metrics
    report = classification_report(
        torch.cat(actual_batches).cpu().tolist(), torch.cat(predicted_batches).cpu().tolist(),
        labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    return *metrics, report


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Giá trị phải là số nguyên dương.")
    return number


def main():
    parser = argparse.ArgumentParser(description="Train a road damage classifier and save a separate run.")
    parser.add_argument("model", nargs="?", default="simple_cnn",
                        choices=["simple_cnn", "complex_cnn", "transfer_model"])
    parser.add_argument("--epochs", type=positive_int, help="Override the epoch limit from config.")
    parser.add_argument("--batch-size", type=positive_int, help="Override the global batch size from config.")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--skip-test", dest="evaluate_test", action="store_false",
                            help="Tune on validation only; evaluate the test set separately later.")
    test_group.add_argument("--evaluate-test", dest="evaluate_test", action="store_true",
                            help="Evaluate the test set after training the chosen configuration.")
    parser.set_defaults(evaluate_test=None)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    model_name = args.model
    config_path = ROOT / "configs" / f"{model_name}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    evaluate_test = bool(args.evaluate_test)
    config.pop("evaluate_test", None)
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("epochs và batch_size phải >= 1.")
    checkpoint_monitor = config.get("checkpoint_monitor", "val_acc")
    if checkpoint_monitor not in ("val_loss", "val_acc"):
        raise ValueError("checkpoint_monitor must be val_loss or val_acc.")
    checkpoint_selection = (
        "lowest validation loss; highest validation accuracy breaks ties"
        if checkpoint_monitor == "val_loss"
        else "highest validation accuracy; lowest validation loss breaks ties"
    )
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = select_device(args.device)
    runtime = configure_runtime(device, args.num_workers, not args.no_amp, args.deterministic)
    print_runtime(runtime)
    print(f"Max epochs: {config['epochs']} | Checkpoint: {checkpoint_monitor} | "
          f"Evaluate test: {evaluate_test}", flush=True)
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    train_data = make_dataset("train", augment=True)
    val_data = make_dataset("val")
    if train_data.classes != val_data.classes:
        raise ValueError("Tên hoặc thứ tự lớp của train và val không khớp.")
    train_loader = make_loader(train_data, config["batch_size"], runtime, shuffle=True)
    val_loader = make_loader(val_data, config["batch_size"], runtime)

    model = create_model(model_name, len(train_data.classes)).to(device)
    gpu_count = torch.cuda.device_count() if device == "cuda" else 0
    used_gpus = 1 if gpu_count and args.single_gpu else gpu_count
    if used_gpus > 1:
        if config["batch_size"] < used_gpus:
            raise ValueError("Batch size must be at least the number of GPUs.")
        model = nn.DataParallel(model)
    runtime["gpu_count"] = gpu_count
    runtime["used_gpus"] = used_gpus
    runtime["parallelism"] = "DataParallel" if used_gpus > 1 else "single_device"
    print(f"GPUs used: {used_gpus} | Global batch size: {config['batch_size']}", flush=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["amp"])
    early_stopping = EarlyStopping(**config.get("early_stopping", {}))
    scheduler_config = config.get("lr_scheduler")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", threshold=early_stopping.min_delta,
            threshold_mode="abs", **scheduler_config,
        ) if scheduler_config else None
    )

    run_dir = next_run_dir(model_name)
    run_dir.mkdir(parents=True)
    summary = {
        "model": model_name,
        "run": run_dir.name,
        "config": config,
        "classes": train_data.classes,
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam",
        "device": device,
        "runtime": runtime,
        "torch_version": str(torch.__version__),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "class_to_idx": {name: i for i, name in enumerate(train_data.classes)},
        "normalization": load_normalization(),
        "split_sizes": {"train": len(train_data), "val": len(val_data)},
        "checkpoint_monitor": checkpoint_monitor,
        "checkpoint_selection": checkpoint_selection,
        "convergence_monitor": "validation loss",
        "scheduler": "ReduceLROnPlateau" if scheduler else None,
        "test_evaluation_requested": evaluate_test,
        "stopped_early": False,
        "status": "training",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Device: {device} | Run: {run_dir} | Classes: {train_data.classes}", flush=True)

    best_val_acc = -1
    best_val_loss = float("inf")
    training_start = time.perf_counter()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_macro_f1", "learning_rate", "epoch_seconds", "train_images_per_second",
        ])
        writer.writeheader()

        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device, runtime["amp"],
            )
            train_seconds = time.perf_counter() - epoch_start
            val_loss, val_acc, val_report = evaluate_loss_accuracy(
                model, val_loader, loss_fn, device, runtime["amp"], train_data.classes,
            )
            if not all(math.isfinite(value) for value in (train_loss, val_loss)):
                raise FloatingPointError("Non-finite loss; stop before selecting a checkpoint.")
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_report["macro avg"]["f1-score"],
                "learning_rate": learning_rate,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_images_per_second": len(train_data) / train_seconds,
            }
            writer.writerow(row)
            file.flush()
            print(
                f"Epoch {epoch}/{config['epochs']} | "
                f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.2%} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.2%} | "
                f"val macro-F1 {row['val_macro_f1']:.2%} | lr {learning_rate:.2g} | "
                f"{row['epoch_seconds']:.1f}s | {row['train_images_per_second']:.1f} train images/s",
                flush=True,
            )
            if checkpoint_monitor == "val_loss":
                improved = val_loss < best_val_loss or (val_loss == best_val_loss and val_acc > best_val_acc)
            else:
                improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
            # Save the true best checkpoint even for gains smaller than early-stopping min_delta.
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                summary["best_epoch"] = epoch
                summary["best_val_acc"] = val_acc
                summary["best_val_loss"] = val_loss
                summary["best_val_macro_f1"] = row["val_macro_f1"]
                torch.save((model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                           run_dir / "best_model.pth")
                with (run_dir / "validation_report.csv").open("w", newline="", encoding="utf-8") as report_file:
                    report_writer = csv.writer(report_file)
                    report_writer.writerow(["class", "precision", "recall", "f1-score", "support"])
                    for name in [*train_data.classes, "macro avg", "weighted avg"]:
                        report_writer.writerow([name, *[val_report[name][key] for key in
                                                       ("precision", "recall", "f1-score", "support")]])
            if scheduler is not None:
                scheduler.step(val_loss)
                if optimizer.param_groups[0]["lr"] < learning_rate:
                    print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']:.2g}", flush=True)
            should_stop = early_stopping.update(val_loss)
            summary["completed_epochs"] = epoch
            summary["final_learning_rate"] = optimizer.param_groups[0]["lr"]
            summary["early_stopping_bad_epochs"] = early_stopping.bad_epochs
            summary["early_stopping_reference_loss"] = early_stopping.best_loss
            summary["stopped_early"] = should_stop and epoch < config["epochs"]
            summary["stop_reason"] = "validation_loss_plateau" if should_stop else "max_epochs"
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if should_stop:
                print(f"Early stopping: validation loss did not improve for "
                      f"{early_stopping.patience} epochs. Best checkpoint: epoch {summary['best_epoch']}.",
                      flush=True)
                break

    summary["training_seconds"] = time.perf_counter() - training_start
    summary["status"] = "trained"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training finished: {summary['stop_reason']} | "
          f"Completed: {summary['completed_epochs']}/{config['epochs']} epochs | "
          f"Selected epoch: {summary['best_epoch']} | "
          f"Validation loss: {summary['best_val_loss']:.6f}", flush=True)
    # Release training memory/workers before the evaluation subprocess starts.
    del model, optimizer, scaler, scheduler, train_loader, val_loader
    if device == "cuda":
        torch.cuda.empty_cache()
    if not evaluate_test:
        subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
    else:
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), str(run_dir),
                        "--device", device], check=True)
    print(f"Saved results: {run_dir}")


if __name__ == "__main__":
    main()