"""Các kiến trúc dùng cho bài toán phân loại hư hỏng đường."""

from .simple_cnn import SimpleCNN


MODEL_NAMES = ["simple_cnn", "complex_cnn", "transfer_model"]


def create_model(model_name, num_classes, config=None, pretrained=True):
    """config là cấu hình của run; pretrained=False khi sắp nạp checkpoint đã train."""
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name == "complex_cnn":
        from .complex_cnn import build_model
        return build_model(num_classes)
    if model_name == "transfer_model":
        from .transfer_model import build_model
        config = config or {}
        return build_model(num_classes, pretrained=pretrained,
                           fine_tune_blocks=config.get("fine_tune_blocks", 3),
                           dropout=config.get("dropout", 0.5))
    raise ValueError(f"Không có model: {model_name}")


def model_normalization(model_name):
    """Mean/std riêng của model; None nghĩa là dùng mean/std tính từ tập train."""
    if model_name == "transfer_model":
        from .transfer_model import NORMALIZATION
        return NORMALIZATION
    return None
