"""CNN cơ bản với ba lớp convolution.

Run: python src/simple_cnn.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_dataset


# Hyperparameter
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
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

    def fit(self, train_loader, val_loader):
        """Train EPOCHS epoch, lưu trọng số vào SAVE_PATH mỗi khi val accuracy tốt hơn."""
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        best_epoch, best_val_acc, best_val_loss = 0, float("-inf"), float("inf")
        for epoch in range(1, EPOCHS + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_one_epoch(
                train_loader, loss_fn, optimizer)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            print(f"Epoch {epoch}/{EPOCHS} | train loss {train_loss:.4f}, acc {train_acc:.2%} | "
                  f"val loss {val_loss:.4f}, acc {val_acc:.2%} | lr {learning_rate:.2g}", flush=True)
            # Lưu lại epoch có val accuracy cao hơn, nếu accuracy bằng nhau thì xét tới val loss
            if (val_acc, -val_loss) > (best_val_acc, -best_val_loss):
                best_epoch, best_val_acc, best_val_loss = epoch, val_acc, val_loss
                torch.save(self.state_dict(), SAVE_PATH)
        return best_epoch, best_val_loss, best_val_acc

    def train_one_epoch(self, loader, loss_fn, optimizer):
        self.train()
        total_loss = torch.zeros(())
        correct = torch.zeros((), dtype=torch.long)
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = self(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * labels.size(0)
            correct += (outputs.detach().argmax(dim=1) == labels).sum()
        return total_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)

    def evaluate(self, loader):
        """Trả về loss, accuracy, nhãn thật và nhãn dự đoán trên loader."""
        self.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = torch.zeros(())
        actual, predicted = [], []
        with torch.no_grad():
            for images, labels in loader:
                outputs = self(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss * labels.size(0)
                actual.extend(labels.cpu().tolist())
                predicted.extend(outputs.argmax(dim=1).cpu().tolist())
        correct = sum(a == p for a, p in zip(actual, predicted))
        return total_loss.item() / len(loader.dataset), correct / len(loader.dataset), actual, predicted

if __name__ == "__main__":
    # Load dữ liệu: tập train có augmentation, cả train và val đều chuẩn hóa bằng mean/std của tập train
    train_data = make_dataset("train", augment=True)
    val_data = make_dataset("val")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Khởi tạo model rồi train, nếu có epoch tốt hơn thì model được lưu vào SAVE_PATH
    model = SimpleCNN(len(train_data.classes))
    SAVE_PATH.parent.mkdir(exist_ok=True)
    best_epoch, val_loss, val_acc = model.fit(train_loader, val_loader)

    # In kết quả trên tập validation, tập test sẽ chạy: python src/predict.py simple_cnn
    print(f"\nBest epoch: {best_epoch} | val loss {val_loss:.4f}, acc {val_acc:.2%}")
    print(f"Saved best model: {SAVE_PATH}")
