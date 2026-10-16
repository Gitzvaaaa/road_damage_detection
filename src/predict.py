"""Đánh giá model đã train (saved_models/<model>.pth) trên tập test.

Run: python src/predict.py simple_cnn
"""

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import make_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test set")
    parser.add_argument("model", choices=["simple_cnn", "complex_cnn", "transfer_model"])
    args = parser.parse_args()

    # Load tập test (tiền xử lý giống lúc train, nhưng không có augmentation) và dựng model cần đánh giá
    if args.model == "simple_cnn":
        import simple_cnn as module
        test_data = make_dataset("test")
        model = module.SimpleCNN(len(test_data.classes))
    elif args.model == "complex_cnn":
        import complex_cnn as module
        test_data = make_dataset("test")
        model = module.ComplexCNN(len(test_data.classes))
    else:
        import transfer_model as module
        test_data = make_dataset("test", normalization=module.NORMALIZATION)
        # Model dựng từ backbone pretrained, trọng số này sẽ bị thay bằng trọng số đã train ở code bên dưới
        model = module.TransferModel(len(test_data.classes))

    # Nạp trọng số đã train trong saved_models/ vào model
    model.load_state_dict(torch.load(module.SAVE_PATH, map_location="cpu", weights_only=True))
    print(f"Model: {module.SAVE_PATH}", flush=True)

    # Cho model dự đoán toàn bộ tập test
    test_loader = DataLoader(test_data, batch_size=module.BATCH_SIZE)
    test_loss, test_acc, actual, predicted = model.evaluate(test_loader)

    # In kết quả ra màn hình
    classes = test_data.classes
    labels = list(range(len(classes)))
    print(f"\nTest: loss {test_loss:.4f}, acc {test_acc:.2%}\n")
    print(classification_report(actual, predicted, labels=labels, target_names=classes,
                                digits=4, zero_division=0))
    print("Confusion matrix (rows: actual, columns: predicted)")
    print("      " + "".join(f"{name:>6}" for name in classes))
    for name, row in zip(classes, confusion_matrix(actual, predicted, labels=labels)):
        print(f"{name:>6}" + "".join(f"{count:>6}" for count in row))
