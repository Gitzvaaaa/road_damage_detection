"""Complex CNN: 4 block Conv-BatchNorm-ReLU, có dropout, giảm learning rate và early stopping.

Run: python src/complex_cnn.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_dataset


# Hyperparameter
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
# Khi val loss không giảm nữa thì giảm learning rate (dùng ReduceLROnPlateau)
LR_FACTOR = 0.5
LR_PATIENCE = 3
LR_THRESHOLD = 0.0001
MIN_LR = 0.000001
# Early stopping: val loss sau 10 epoch không giảm thì dừng train sớm
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0001
SAVE_PATH = Path(__file__).resolve().parents[1] / "saved_models" / "complex_cnn.pth"


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, images):
        return self.layers(images)


class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(3, 16, dropout=0.10),
            ConvBlock(16, 32, dropout=0.15),
            ConvBlock(32, 64, dropout=0.20),
            ConvBlock(64, 128, dropout=0.25),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, images):
        return self.network(images)

    def fit(self, train_loader, val_loader):
        """Train tối đa epoch, lưu trọng số vào SAVE_PATH mỗi khi val accuracy tốt hơn."""
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
            threshold=LR_THRESHOLD, threshold_mode="abs", min_lr=MIN_LR)
        best_epoch, best_val_acc, best_val_loss = 0, float("-inf"), float("inf")
        lowest_val_loss, bad_epochs = float("inf"), 0  # hai biến này để theo dõi early stopping
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
            scheduler.step(val_loss)
            # Đếm số epoch liên tiếp mà val loss không giảm được quá MIN_DELTA, đủ PATIENCE epoch thì dừng
            if val_loss < lowest_val_loss - EARLY_STOPPING_MIN_DELTA:
                lowest_val_loss, bad_epochs = val_loss, 0
            else:
                bad_epochs += 1
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}: val loss has not improved "
                      f"for {EARLY_STOPPING_PATIENCE} epochs")
                break
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
    model = ComplexCNN(len(train_data.classes))
    SAVE_PATH.parent.mkdir(exist_ok=True)
    best_epoch, val_loss, val_acc = model.fit(train_loader, val_loader)

    # In kết quả trên tập validation, tập test sẽ chạy: python src/predict.py complex_cnn
    print(f"\nBest epoch: {best_epoch} | val loss {val_loss:.4f}, acc {val_acc:.2%}")
    print(f"Saved best model: {SAVE_PATH}")
