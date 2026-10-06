"""Huấn luyện model và lưu mỗi lần chạy vào một thư mục run riêng."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="simple_cnn",
                        choices=["simple_cnn", "complex_cnn", "transfer_model"])
    parser.add_argument("--device", choices=["cpu", "cuda", "directml"], default=None)
    parser.add_argument("--batch-size", type=int, help="Override batch size for this machine.")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0.")
    model_name = args.model
    config_path = ROOT / "configs" / f"{model_name}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    torch.manual_seed(config["seed"])
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

    dataset_options = {}
    if model_name == "transfer_model":
        from models.transfer_model import IMG_SIZE, NORMALIZATION
        # Lưu cả giá trị mặc định để evaluate dựng đúng head khi config thiếu khóa.
        config.setdefault("fine_tune_blocks", 3)
        config.setdefault("hidden_dim", 128)
        config.setdefault("dropout", 0.5)
        config.setdefault("classifier_learning_rate", 0.0003)
        dataset_options = {"normalization": NORMALIZATION, "image_size": IMG_SIZE}
    train_data = make_dataset("train", augment=True, **dataset_options)
    val_data = make_dataset("val", **dataset_options)
    if train_data.classes != val_data.classes:
        raise ValueError("Tên hoặc thứ tự lớp của train và val không khớp.")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])

    model = create_model(
        model_name, len(train_data.classes),
        fine_tune_blocks=config.get("fine_tune_blocks", 3),
        hidden_dim=config.get("hidden_dim", 128), dropout=config.get("dropout", 0.5),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # Loss trong CSV vẫn là CE thường để so sánh train/val/test cùng thang đo.
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if model_name == "transfer_model":
        # Classifier mới học nhanh hơn backbone pretrained.
        parameters = [
            {"params": [p for p in model.features.parameters() if p.requires_grad]},
            {"params": model.classifier.parameters(), "lr": config["classifier_learning_rate"]},
        ]
    optimizer_class = torch.optim.AdamW if model_name == "transfer_model" else torch.optim.Adam
    optimizer = optimizer_class(
        parameters, lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0), foreach=False,
    )
    scheduler = None
    if model_name == "transfer_model":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1,
        )

    run_dir = next_run_dir(model_name)
    run_dir.mkdir(parents=True)
    summary = {
        "model": model_name,
        "run": run_dir.name,
        "config": config,
        "classes": train_data.classes,
        "loss_function": "CrossEntropyLoss",
        "optimizer": optimizer_class.__name__,
        "device": str(device),
        "backend": backend,
        "checkpoint_metric": "val_loss" if model_name == "transfer_model" else "val_acc",
    }
    if model_name == "transfer_model":
        summary.update({
            "backbone": "ResNet50",
            "pretrained_weights": "IMAGENET1K_V2",
            "fine_tune_blocks": config.get("fine_tune_blocks", 3),
            "fine_tune_convolutions": [
                f"features.{name}" for name, layer in model.features.named_modules()
                if isinstance(layer, nn.Conv2d) and layer.weight.requires_grad
            ],
            "classifier": str(model.classifier),
            "scheduler": "ReduceLROnPlateau(val_loss, factor=0.5, patience=1)",
            "train_objective": f"CrossEntropyLoss(label_smoothing={config.get('label_smoothing', 0.0)})",
            "augmentation": "Affine(15 degrees, translate=0.1), ColorJitter(0.2, 0.2, 0.1), HorizontalFlip",
            "normalization": NORMALIZATION,
            "image_size": IMG_SIZE,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
        })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Device: {device} | Run: {run_dir} | Classes: {train_data.classes}", flush=True)

    best_val_acc = -1
    best_val_loss = float("inf")
    early_stop_loss = float("inf")
    stale_epochs = 0
    patience = config.get("early_stopping_patience", 0)
    min_delta = config.get("early_stopping_min_delta", 0.0)
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
                loss = train_loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss_fn(outputs.detach(), labels).item() * labels.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            val_loss, val_acc = evaluate_loss_accuracy(model, val_loader, loss_fn, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / len(train_data),
                "train_acc": train_correct / len(train_data),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            writer.writerow(row)
            file.flush()
            print(
                f"Epoch {epoch}/{config['epochs']} | "
                f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.2%} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.2%}",
                flush=True,
            )
            improved = val_loss < best_val_loss if model_name == "transfer_model" else val_acc > best_val_acc
            if improved:
                summary["best_epoch"] = epoch
                summary["best_val_loss"] = val_loss
                # Accuracy tại checkpoint được chọn; xem checkpoint_metric trong summary.
                summary["best_val_acc"] = val_acc
                torch.save(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    run_dir / "best_model.pth",
                )
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            if val_loss < early_stop_loss - min_delta:
                early_stop_loss, stale_epochs = val_loss, 0
            else:
                stale_epochs += 1
            stop = patience > 0 and stale_epochs >= patience
            summary.update(epochs_completed=epoch, stopped_early=stop, max_val_acc=best_val_acc)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)
            if stop:
                print(f"Early stopping: no sufficient validation loss improvement for {patience} epochs.", flush=True)
                break

    subprocess.run([
        sys.executable, str(ROOT / "evaluate.py"), str(run_dir), "--device", backend,
    ], check=True)
    print(f"Saved results: {run_dir}")


if __name__ == "__main__":
    main()
