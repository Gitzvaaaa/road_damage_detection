"""CNN cơ bản với ba lớp convolution.

Chạy từ thư mục gốc của project: python src/simple_cnn.py
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_dataset


# Hyperparameter, muốn thử cấu hình khác thì sửa trực tiếp ở đây
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42
# Để None thì tự chọn thiết bị: có CUDA thì dùng CUDA, không thì DirectML, cuối cùng mới là CPU.
# Muốn ép dùng một thiết bị cụ thể thì ghi "cpu", "cuda" hoặc "directml".
DEVICE = None
SAVE_PATH = Path(__file__).resolve().parents[1] / "saved_models" / "simple_cnn.pth"


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

    def fit(self, train_loader, val_loader, device, backend, amp):
        """Train EPOCHS epoch, lưu trọng số vào SAVE_PATH mỗi khi val accuracy tốt hơn."""
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE,
                                     foreach=False if backend == "directml" else None)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best_epoch, best_val_acc, best_val_loss = 0, float("-inf"), float("inf")
        for epoch in range(1, EPOCHS + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_one_epoch(
                train_loader, loss_fn, optimizer, scaler, device, amp)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, device, amp)
            print(f"Epoch {epoch}/{EPOCHS} | train loss {train_loss:.4f}, acc {train_acc:.2%} | "
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
    # Không có CUDA thì thử DirectML, máy không cài hoặc không dùng được thì quay về CPU
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

    # Load dữ liệu: tập train có augmentation, cả train và val đều chuẩn hóa bằng mean/std của tập train
    train_data = make_dataset("train", augment=True)
    val_data = make_dataset("val")
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=num_workers,
                         pin_memory=backend == "cuda", persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, **loader_kwargs)

    # Khởi tạo model rồi train, cứ có epoch tốt hơn là model được lưu luôn vào SAVE_PATH
    model = SimpleCNN(len(train_data.classes)).to(device)
    SAVE_PATH.parent.mkdir(exist_ok=True)
    best_epoch, val_loss, val_acc = model.fit(train_loader, val_loader, device, backend, amp)

    # In kết quả trên tập validation, còn tập test thì chạy: python src/predict.py simple_cnn
    print(f"\nBest epoch: {best_epoch} | val loss {val_loss:.4f}, acc {val_acc:.2%}")
    print(f"Saved best model: {SAVE_PATH}")
