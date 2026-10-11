import os
from pathlib import Path

from torch import nn


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "yolov8s_rdd2022_classifier"
REPO_ID = "dronefreak/rdd2022-yolov8s"
REVISION = "06d1dbb6cb1b30f54a5f32f9fd9ee29695e1b106"
NORMALIZATION = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}


class TransferModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, hidden_dim=128, dropout=0.5):
        super().__init__()
        config_dir = ROOT / "runs/.ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        if pretrained:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(REPO_ID, "best.pt", revision=REVISION,
                                      cache_dir=ROOT / "runs/.pretrained")
            base = YOLO(weights).model.cpu().float()
        else:
            # Dựng offline để nạp checkpoint phân loại đã train.
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
        # Block đóng băng phải giữ cả thống kê BatchNorm.
        for index, block in enumerate(self.features):
            if index < 7 or not self.finetuning:
                block.eval()
        return self

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_classes, pretrained=True, hidden_dim=128, dropout=0.5):
    return TransferModel(num_classes, pretrained, hidden_dim, dropout)
