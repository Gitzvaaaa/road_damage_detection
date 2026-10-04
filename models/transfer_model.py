"""ResNet18 pretrained, thay lớp cuối để phân loại hư hỏng mặt đường."""

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def build_model(num_classes, pretrained=False):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    return model
