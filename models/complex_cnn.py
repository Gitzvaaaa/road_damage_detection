"""CNN tự thiết kế với bốn block, huấn luyện từ đầu cho ảnh road damage."""

from torch import nn


class ConvBlock(nn.Module):
    """Hai Conv-BatchNorm-ReLU, sau đó giảm kích thước và regularization."""

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, images):
        return self.layers(images)


class ComplexCNN(nn.Module):
    """8 convolution + 2 fully connected; đầu ra là logits, không softmax."""

    def __init__(self, num_classes):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes phải lớn hơn hoặc bằng 2.")
        self.features = nn.Sequential(
            ConvBlock(3, 16, dropout=0.10),
            ConvBlock(16, 32, dropout=0.15),
            ConvBlock(32, 64, dropout=0.20),
            ConvBlock(64, 128, dropout=0.25),
        )
        # Giữ lưới 4x4 để bộ phân loại còn thông tin vị trí/hướng vết nứt.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(num_classes):
    return ComplexCNN(num_classes)
