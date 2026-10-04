"""Train ResNet18 cho 4 loại hư hỏng mặt đường."""

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


def main(device=None):
    # 1. Đọc cấu hình
    with open(ROOT / "configs/transfer_model.json", encoding="utf-8") as file:
        config = json.load(file)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    head_epochs = config["head_epochs"]
    epochs = head_epochs + config["finetune_epochs"]

    # 2. Đọc dữ liệu và chia thành từng batch
    train_data = make_dataset("train", augment=True, normalization=NORMALIZATION)
    val_data = make_dataset("val", normalization=NORMALIZATION)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])

    # 3. Tạo ResNet18, ban đầu chỉ train fc
    model = build_model(len(train_data.classes), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])

    run_dir = next_run_dir("transfer_model")
    run_dir.mkdir(parents=True)
    summary = {
        "model": "transfer_model", "backbone": "resnet18", "config": config,
        "classes": train_data.classes, "normalization": NORMALIZATION,
        "split_sizes": {"train": len(train_data), "val": len(val_data)},
        "checkpoint_monitor": "val_macro_f1",
    }
    history = []
    best_f1 = -1.0
    best_loss = float("inf")
    stage = "head"
    print("Thiết bị huấn luyện:", device)
    print("Bắt đầu huấn luyện lớp fc.")

    for epoch in range(1, epochs + 1):
        # Kết thúc train fc, nạp model tốt nhất và mở layer4 để học tiếp
        if epoch == head_epochs + 1:
            model.load_state_dict(torch.load(run_dir / "best_model.pth",
                                            map_location=device, weights_only=True))
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.fc.parameters(), "lr": config["finetune_lr"]},
                {"params": model.layer4.parameters(), "lr": config["backbone_lr"]},
            ], weight_decay=config["weight_decay"])
            stage = "finetune"
            print("Bắt đầu fine-tune layer4 và fc.")

        # 4. Train: giữ BatchNorm của các block đóng băng ở chế độ eval
        model.eval()
        model.fc.train()
        if epoch > head_epochs:
            model.layer4.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_data)
        train_acc = train_correct / len(train_data)

        # 5. Validation: chỉ dự đoán, không cập nhật trọng số
        model.eval()
        val_loss = 0.0
        val_correct = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())

        val_loss = val_loss / len(val_data)
        val_acc = val_correct / len(val_data)
        val_f1 = f1_score(y_true, y_pred, labels=list(range(len(train_data.classes))),
                          average="macro", zero_division=0)
        history.append([epoch, stage, train_loss, train_acc, val_loss, val_acc, val_f1])
        print("Epoch", epoch, "trong", epochs)
        print("Độ chính xác trên tập train:", train_acc)
        print("Độ chính xác trên tập validation:", val_acc)
        print("Macro F1 trên tập validation:", val_f1)

        # 6. Giữ checkpoint có F1 cao nhất; F1 bằng nhau thì chọn loss thấp hơn
        if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss):
            best_f1 = val_f1
            best_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            summary.update(best_epoch=epoch, best_stage=stage, best_val_loss=val_loss,
                           best_val_acc=val_acc, best_val_macro_f1=float(val_f1))

    # 7. Lưu kết quả và vẽ đường học
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "stage", "train_loss", "train_acc",
                         "val_loss", "val_acc", "val_macro_f1"])
        writer.writerows(history)
    summary.update(completed_epochs=epochs, status="trained")
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    plot_history(run_dir)
    print("Huấn luyện xong. Model tốt nhất ở epoch", summary["best_epoch"])
    print("Kết quả được lưu tại:", run_dir)


if __name__ == "__main__":
    main()
