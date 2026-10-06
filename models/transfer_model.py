"""ResNet50 pretrained và fine-tuning các residual block cuối cho road damage."""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


IMG_SIZE = 224
NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class TransferModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, fine_tune_blocks=3, hidden_dim=128, dropout=0.5):
        super().__init__()
        if type(fine_tune_blocks) is not int or not 0 <= fine_tune_blocks <= 16:
            raise ValueError("fine_tune_blocks phải là số nguyên từ 0 đến 16.")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim phải là số nguyên dương.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout phải nằm trong khoảng [0, 1).")

        # 1. Load ResNet50 và freeze toàn bộ backbone.
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = resnet50(weights=weights)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.features.requires_grad_(False)

        # 2. Head nhỏ hơn để giảm số tham số cần học trên dữ liệu road damage.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # 3. Mở N block cuối trong 16 block; mặc định toàn bộ layer4 (3 block).
        # Mỗi block có 3 convolution chính và có thể có nhánh shortcut.
        blocks = [block for stage in list(self.features.children())[-4:] for block in stage]
        for block in blocks[len(blocks) - fine_tune_blocks:]:
            block.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Giữ running mean/variance pretrained ổn định trong các block mở khóa.
        for layer in self.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, images):
        # Trả logits vì CrossEntropyLoss đã bao gồm log-softmax.
        return self.classifier(self.features(images))


def build_model(num_classes, pretrained=True, fine_tune_blocks=3, hidden_dim=128, dropout=0.5):
    return TransferModel(num_classes, pretrained, fine_tune_blocks, hidden_dim, dropout)
