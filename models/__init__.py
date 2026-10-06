"""Các kiến trúc dùng cho bài toán phân loại hư hỏng đường."""

from .simple_cnn import SimpleCNN


def create_model(model_name, num_classes, pretrained=True, fine_tune_blocks=3, hidden_dim=128, dropout=0.5):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name == "complex_cnn":
        from .complex_cnn import build_model
        return build_model(num_classes)
    if model_name == "transfer_model":
        from .transfer_model import build_model
        return build_model(num_classes, pretrained, fine_tune_blocks, hidden_dim, dropout)
    raise ValueError(f"Không có model: {model_name}")
