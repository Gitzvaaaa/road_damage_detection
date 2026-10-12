import torch
from torch import nn
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


IMG_SIZE = 224
NORMALIZATION = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}


class TransferModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, fine_tune_blocks=3, dropout=0.5):
        super().__init__()
        weights = "yolov8m.yaml"
        if pretrained:
            weights = hf_hub_download("dronefreak/rdd2022-yolov8m", "best.pt")
        base = YOLO(weights).model.cpu().float()

        self.features = nn.Sequential(*list(base.model.children())[:10])
        self.features.requires_grad_(False)
        self.features.eval()
        self.fine_tune_blocks = fine_tune_blocks
        with torch.no_grad():
            channels = self.features(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def unfreeze(self, num_blocks=None):
        num_blocks = self.fine_tune_blocks if num_blocks is None else num_blocks
        blocks = list(self.features.children())
        self.features.requires_grad_(False)
        for block in blocks[len(blocks) - num_blocks:]:
            block.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Giữ thống kê BatchNorm pretrained trong cả hai giai đoạn.
        for layer in self.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_classes, pretrained=True, fine_tune_blocks=3, dropout=0.5):
    return TransferModel(num_classes, pretrained, fine_tune_blocks, dropout)
