"""Các kiến trúc dùng cho bài toán phân loại hư hỏng đường."""

from .complex_cnn import ComplexCNN
from .simple_cnn import SimpleCNN


MODEL_NAMES = ["simple_cnn", "complex_cnn", "transfer_model"]


def create_model(model_name, num_classes, config=None, pretrained=True):
    """config là cấu hình của run; pretrained=False khi sắp nạp checkpoint đã train."""
    if model_name == "transfer_model":
        from .transfer_model import TransferModel
        config = config or {}
        return TransferModel(num_classes, pretrained=pretrained,
                             hidden_dim=config.get("hidden_dim", 128),
                             dropout=config.get("dropout", 0.5))
    return {"simple_cnn": SimpleCNN, "complex_cnn": ComplexCNN}[model_name](num_classes)


def model_normalization(model_name):
    """Mean/std riêng của model; None nghĩa là dùng mean/std tính từ tập train."""
    if model_name == "transfer_model":
        from .transfer_model import NORMALIZATION
        return NORMALIZATION
    return None
