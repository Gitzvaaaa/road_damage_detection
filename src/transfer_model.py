"""Transfer learning: 10 block đầu của YOLOv8s pretrained trên RDD2022 + classifier mới.

Train 2 giai đoạn: (1) chỉ train classifier, (2) fine-tune thêm 3 block cuối của backbone.
Run: python src/transfer_model.py
Lần chạy đầu tải trọng số pretrained từ Hugging Face nên cần internet.
"""

import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from dataset import make_dataset


# Hyperparameter
HEAD_EPOCHS = 3        # giai đoạn 1: đóng băng backbone, chỉ train phần classifier
FINETUNE_EPOCHS = 30   # giai đoạn 2: mở thêm block 7-9 của backbone để train cùng classifier
BATCH_SIZE = 16
HIDDEN_DIM = 128
DROPOUT = 0.5
HEAD_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.00001
FINETUNE_HEAD_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
LABEL_SMOOTHING = 0.05
SEED = 42
# Để None sẽ tự chọn thiết bị: CUDA, DirectML, CPU
DEVICE = None
SAVE_PATH = Path(__file__).resolve().parents[1] / "saved_models" / "transfer_model.pth"

# Trọng số YOLOv8s đã được train sẵn trên bộ RDD2022, tải về từ Hugging Face
REPO_ID = "dronefreak/rdd2022-yolov8s"
REVISION = "06d1dbb6cb1b30f54a5f32f9fd9ee29695e1b106"
# YOLO nhận ảnh trong khoảng 0-1 nên ở đây không chuẩn hóa, mean 0 và std 1 nghĩa là giữ nguyên ảnh
NORMALIZATION = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}


class TransferModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        if pretrained:
            weights = hf_hub_download(REPO_ID, "best.pt", revision=REVISION)
            base = YOLO(weights).model.cpu().float()
        else:
            # Chỉ dựng khung model, không tải trọng số pretrained vì sau đó sẽ nạp trọng số đã train
            base = DetectionModel("yolov8s.yaml", nc=4, verbose=False)

        self.features = nn.Sequential(*list(base.model.children())[:10])
        channels = self.features[-1].cv2.conv.out_channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.set_finetune(False)

    def set_finetune(self, enabled):
        self.finetuning = enabled
        self.features.requires_grad_(False)
        self.features[7:].requires_grad_(enabled)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Block nào đang đóng băng thì để ở chế độ eval, không thì BatchNorm vẫn cập nhật mean/var lúc train
        for index, block in enumerate(self.features):
            if index < 7 or not self.finetuning:
                block.eval()
        return self

    def forward(self, x):
        return self.classifier(self.features(x))

    def fit(self, train_loader, val_loader, device, backend, amp):
        """Train 2 giai đoạn, lưu trọng số vào SAVE_PATH mỗi khi val accuracy tốt hơn."""
        # Label smoothing chỉ dùng lúc train, loss của val/test trong evaluate vẫn là cross-entropy thường
        loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        foreach = False if backend == "directml" else None
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=HEAD_LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY, foreach=foreach)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        epochs = HEAD_EPOCHS + FINETUNE_EPOCHS
        best_epoch, best_val_acc, best_val_loss = 0, float("-inf"), float("inf")
        for epoch in range(1, epochs + 1):
            if epoch == HEAD_EPOCHS + 1:
                # Sang giai đoạn 2 thì fine-tune tiếp từ head tốt nhất, chưa chắc là head ở epoch cuối
                print(f"Fine-tuning from best head epoch {best_epoch}", flush=True)
                self.load_state_dict(torch.load(SAVE_PATH, map_location="cpu", weights_only=True))
                self.set_finetune(True)
                optimizer = torch.optim.Adam([
                    {"params": [parameter for parameter in self.features.parameters()
                                if parameter.requires_grad],
                     "lr": FINETUNE_LEARNING_RATE},
                    {"params": self.classifier.parameters(), "lr": FINETUNE_HEAD_LEARNING_RATE},
                ], weight_decay=WEIGHT_DECAY, foreach=foreach)
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_one_epoch(
                train_loader, loss_fn, optimizer, scaler, device, amp)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, device, amp)
            print(f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f}, acc {train_acc:.2%} | "
                  f"val loss {val_loss:.4f}, acc {val_acc:.2%} | lr {learning_rate:.2g}", flush=True)
            # Lưu lại epoch có val accuracy cao hơn, nếu accuracy bằng nhau thì xét tới val loss
            if (val_acc, -val_loss) > (best_val_acc, -best_val_loss):
                best_epoch, best_val_acc, best_val_loss = epoch, val_acc, val_loss
                torch.save(self.state_dict(), SAVE_PATH)
        return best_epoch, best_val_loss, best_val_acc

    def train_one_epoch(self, loader, loss_fn, optimizer, scaler, device, amp):
        self.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                outputs = self(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * labels.size(0)
            correct += (outputs.detach().argmax(dim=1) == labels).sum()
        return total_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)

    def evaluate(self, loader, device, amp):
        """Trả về loss, accuracy, nhãn thật và nhãn dự đoán trên loader."""
        self.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = torch.zeros((), device=device)
        actual, predicted = [], []
        # Dùng no_grad chứ không dùng inference_mode, vì BatchNorm trên DirectML cần no_grad
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                    outputs = self(images)
                    loss = loss_fn(outputs, labels)
                total_loss += loss * labels.size(0)
                actual.extend(labels.cpu().tolist())
                predicted.extend(outputs.argmax(dim=1).cpu().tolist())
        correct = sum(a == p for a, p in zip(actual, predicted))
        return total_loss.item() / len(loader.dataset), correct / len(loader.dataset), actual, predicted


def pick_device(requested):
    if requested == "directml":
        import torch_directml
        return "directml", torch_directml.device()
    if requested is not None:
        return requested, torch.device(requested)
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    # Không có CUDA sẽ thử DirectML, máy không cài hoặc không dùng được sẽ sử dụng CPU
    try:
        import torch_directml
        if torch_directml.is_available():
            return "directml", torch_directml.device()
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu", torch.device("cpu")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    backend, device = pick_device(DEVICE)
    amp = backend == "cuda" and torch.cuda.get_device_capability()[0] >= 6
    if backend == "cuda":
        torch.backends.cudnn.benchmark = True
    num_workers = 2 if backend == "cuda" else 0
    print(f"Device: {device} | AMP: {amp} | Workers: {num_workers}", flush=True)

    # Load dữ liệu: tập train có augmentation, ảnh để nguyên trong khoảng 0-1, không chuẩn hóa
    train_data = make_dataset("train", augment=True, normalization=NORMALIZATION)
    val_data = make_dataset("val", normalization=NORMALIZATION)
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=num_workers,
                         pin_memory=backend == "cuda", persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, **loader_kwargs)

    # Khởi tạo model rồi train, nếu có epoch tốt hơn thì model được lưu vào SAVE_PATH
    model = TransferModel(len(train_data.classes)).to(device)
    SAVE_PATH.parent.mkdir(exist_ok=True)
    best_epoch, val_loss, val_acc = model.fit(train_loader, val_loader, device, backend, amp)

    # In kết quả trên tập validation, tập test sẽ chạy: python src/predict.py transfer_model
    print(f"\nBest epoch: {best_epoch} | val loss {val_loss:.4f}, acc {val_acc:.2%}")
    print(f"Saved best model: {SAVE_PATH}")
