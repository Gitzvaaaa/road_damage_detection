# Road damage classification

Dự án dùng chung dữ liệu Japan cho ba hướng mô hình: simple CNN, complex CNN và transfer learning. Hiện `simple_cnn` đã hoạt động; hai mô hình còn lại là chỗ trống để xây dựng sau.

## Cấu trúc

```text
road_damage/
├── data/train/Japan/
│   ├── processed_raw/clean/
│   └── processed_classification/
│       ├── train/  val/  test/
│       ├── normalization.json
│       └── class_to_idx.json
├── models/                # Định nghĩa từng kiến trúc
├── configs/               # Epoch, batch size, learning rate, seed
├── dataset.py             # Nạp ảnh và chuẩn hóa
├── prepare_classification.py
├── train.py
├── evaluate.py
├── plot_results.py
└── runs/<model>/run_001/, run_002/, ...
```

`data_processing.ipynb` là notebook tạo crop gốc. Thư mục `data/` và `runs/` không được đưa vào Git vì chứa ảnh và checkpoint lớn.

## Chuẩn bị dữ liệu

Dữ liệu đã được chuẩn bị tại `data/train/Japan/processed_classification/`. Chỉ chạy lại lệnh sau khi muốn tạo lại train/val/test từ `processed_raw/clean`; script sẽ thay thế thư mục dữ liệu đã xử lý:

```powershell
& .\.conda\python.exe .\prepare_classification.py
```

Các crop từ cùng một ảnh gốc ở cùng một tập. Tỷ lệ chia theo ảnh gốc là 70/15/15. Mean/std trong `normalization.json` chỉ tính từ train.

## Huấn luyện

```powershell
& .\.conda\python.exe .\train.py simple_cnn
```

Tham số nằm trong `configs/simple_cnn.json`. Mỗi lần chạy tạo một thư mục mới `runs/simple_cnn/run_###/`, không ghi đè run trước. Trong mỗi run có `best_model.pth`, `history.csv`, `training_curves.png`, `confusion_matrix.csv`, `confusion_matrix.png` và `summary.json`. Sau khi huấn luyện, `train.py` tự đánh giá trên test và tạo confusion matrix.

Chạy `train.py complex_cnn` hoặc `train.py transfer_model` sau khi bạn đã xây dựng kiến trúc tương ứng trong `models/`. Hai file model này hiện sẽ báo `NotImplementedError`.

## Đánh giá hoặc vẽ lại run có sẵn

Mặc định hai lệnh sau dùng run mới nhất của simple CNN:

```powershell
& .\.conda\python.exe .\evaluate.py
& .\.conda\python.exe .\plot_results.py
```

Để chọn run cụ thể:

```powershell
& .\.conda\python.exe .\evaluate.py .\runs\simple_cnn\run_001
& .\.conda\python.exe .\plot_results.py .\runs\simple_cnn\run_001
```

Trong confusion matrix, hàng là lớp thật và cột là lớp dự đoán. Đường train dùng ảnh lật ngang ngẫu nhiên, còn val/test không dùng augmentation.
