# Road damage classification

Phân loại ảnh hư hỏng mặt đường (Japan và Czech) thành bốn lớp: `D00`, `D10`, `D20`, `D40`.
Dự án so sánh ba mô hình dùng chung một luồng train/evaluate: `simple_cnn`,
`complex_cnn` và `transfer_model`.

## Môi trường

```powershell
conda env create -f environment.yml
conda activate road_damage_env
```

Chạy mọi lệnh từ thư mục gốc dự án. Nếu dùng môi trường cục bộ `.conda/`, thay `python`
bằng `.\.conda\Scripts\python.exe`. Chỉ `transfer_model` cần `ultralytics` và
`huggingface_hub`.

## Dữ liệu

Dữ liệu đã xử lý nằm ở `data/processed_classification/`, gồm các crop 224×224:

| Tập | Japan | Czech | Tổng |
|---|---:|---:|---:|
| train | 7.725 | 1.002 | 8.727 |
| val | 1.628 | 214 | 1.842 |
| test | 1.693 | 215 | 1.908 |

Dữ liệu được tạo qua hai bước, không cần chạy lại để train:

1. `crop_images.ipynb` cắt crop từ ảnh và annotation trong `data/<nước>/`.
2. `prepare_classification.py` chia train/val/test theo tỉ lệ 70/15/15 cho từng nước và
   tính mean/std từ tập train. Các crop của cùng một ảnh gốc nằm trong cùng một tập.

`EDA.ipynb` chứa phần khám phá dữ liệu.

## Ba mô hình

Kiến trúc nằm trong `models/`, tham số huấn luyện nằm trong `configs/<model>.json`.

| Mô hình | Kiến trúc | Cấu hình train |
|---|---|---|
| `simple_cnn` | 3 lớp Conv–ReLU–MaxPool, rồi 2 lớp Linear | 10 epoch, batch 32, Adam learning rate 0.001 |
| `complex_cnn` | 4 block (2 Conv–BatchNorm–ReLU, MaxPool, Dropout), rồi 2 lớp Linear có dropout | Tối đa 100 epoch, batch 32, Adam learning rate 0.001, weight decay 0.0001; giảm learning rate và dừng sớm theo validation loss |
| `transfer_model` | 10 block đầu của YOLOv8s pretrained trên RDD2022, thêm classifier mới | 3 epoch chỉ train classifier, rồi 15 epoch fine-tune thêm 3 block cuối của backbone; batch 16, label smoothing 0.05 |

Cả ba dùng chung augmentation khi train (lật ngang, affine, color jitter, perspective,
blur) và giữ checkpoint có validation accuracy cao nhất. `simple_cnn` và `complex_cnn`
chuẩn hóa ảnh bằng mean/std của tập train; `transfer_model` dùng ảnh trong khoảng 0–1.

## Huấn luyện

```powershell
python train.py simple_cnn
python train.py complex_cnn
python train.py transfer_model
```

| Cờ | Ý nghĩa |
|---|---|
| `--device cpu\|cuda` | Mặc định dùng CUDA nếu có |
| `--seed N` | Ghi đè `seed` trong config |
| `--num-workers N` | Số tiến trình nạp dữ liệu; mặc định 2 trên CUDA, 0 trên CPU |

Mỗi lần chạy tạo một thư mục mới `runs/<model>/run_##/` (`run_01`, `run_02`, ...), không
ghi đè run cũ. Lần train `transfer_model` đầu tiên tải trọng số pretrained từ Hugging
Face về `runs/.pretrained/` nên cần mạng.

## Đánh giá

```powershell
python evaluate.py runs/complex_cnn/run_01
```

Thay đường dẫn bằng run cần đánh giá. Lệnh dùng chung cho cả ba mô hình vì tên mô hình
và cấu hình được đọc từ `summary.json` của run; có thể thêm `--device cpu|cuda|directml`.
Chỉ đánh giá trên test sau khi đã chốt mô hình.

Sau khi train và đánh giá, mỗi run có 6 file:

| File | Nội dung |
|---|---|
| `best_model.pth` | Trọng số tại epoch được chọn |
| `summary.json` | Cấu hình, epoch được chọn và các metric validation/test |
| `history.csv`, `training_curves.png` | Loss và accuracy train/validation theo epoch |
| `classification_report.csv` | Precision, recall, F1 từng lớp trên test |
| `confusion_matrix.png` | Confusion matrix trên test |

Các hình do `plot_results.py` vẽ; `train.py` và `evaluate.py` tự gọi file này.

## Kết quả

| Mô hình | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| `simple_cnn` | chưa cập nhật | chưa cập nhật | chưa cập nhật |
| `complex_cnn` | 85,47 ± 0,69% | 84,75 ± 1,21% | 85,28 ± 1,12% |
| `transfer_model` | chưa cập nhật | chưa cập nhật | chưa cập nhật |

Số của `complex_cnn` là trung bình ± độ lệch chuẩn của ba seed (42, 123, 999); các run
nằm trong `runs/complex_cnn/run_01` đến `run_03`.

`runs/`, `data/` và `.conda/` không được đưa lên Git nên cần chia sẻ riêng.
