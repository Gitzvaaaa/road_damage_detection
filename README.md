# Road damage classification

Phân loại ảnh hư hỏng mặt đường (Japan và Czech) thành bốn lớp: `D00`, `D10`, `D20`, `D40`.
Ba mô hình dùng chung một luồng train/evaluate: `simple_cnn`, `complex_cnn`, `transfer_model`.
Kiến trúc nằm trong `models/`, tham số huấn luyện nằm trong `configs/`.

## Môi trường và dữ liệu

Chạy các lệnh từ thư mục gốc dự án. Trên máy hiện tại, môi trường có đủ thư viện cho
`simple_cnn` và `complex_cnn` là `.conda/Scripts/python.exe`:

```powershell
& .\.conda\Scripts\python.exe .\train.py --help
```

Trên máy khác, tạo môi trường bằng `conda env create -f environment.yml`, sau đó
`conda activate road_damage_env` và dùng `python` thay cho đường dẫn trên.
`transfer_model` cần thêm `ultralytics` và `huggingface_hub` (đã có trong `environment.yml`).

Dữ liệu đã xử lý ở `data/processed_classification/`, gồm các thư mục `train/`, `val/`,
`test/`, `normalization.json`, `class_to_idx.json` và `split_manifest.csv`:

| Tập | Japan | Czech | Tổng |
|---|---:|---:|---:|
| train | 7.725 | 1.002 | 8.727 |
| val | 1.628 | 214 | 1.842 |
| test | 1.693 | 215 | 1.908 |

Các bước tạo dữ liệu:

1. `crop_images.ipynb` cắt crop 224×224 từ `data/<nước>/images` và
   `annotations/xmls`, lưu vào `data/<nước>/processed_raw/{clean,ambiguous}`.
   Cell đầu chạy lần lượt mọi nước trong danh sách `countries`.
2. `prepare_classification.py` chia crop `clean` của từng nước theo tỉ lệ 70/15/15
   (seed 42), gộp vào `data/processed_classification/` và tính mean/std từ train.
   Các crop cùng ảnh gốc nằm trong cùng một tập.

Không cần chạy lại các bước này để train. Chạy lại `prepare_classification.py` sẽ thay
thế thư mục `data/processed_classification/`.

## Các file chính

| File | Công việc |
|---|---|
| `models/simple_cnn.py` | 3 lớp Conv–ReLU–pooling, cuối là bộ phân loại |
| `models/complex_cnn.py` | 4 block, mỗi block có 2 Conv–BatchNorm–ReLU, pooling và dropout; cuối là bộ phân loại |
| `models/transfer_model.py` | Backbone YOLOv8m pretrained trên RDD2022, thêm bộ phân loại mới |
| `models/__init__.py` | `create_model` tạo model theo tên; `model_normalization` trả về mean/std riêng của model |
| `dataset.py` | Nạp ảnh, augmentation khi train, chuẩn hóa |
| `train.py` | Train/validation, giảm learning rate, dừng sớm và lưu checkpoint |
| `evaluate.py` | Nạp checkpoint, đánh giá test, xuất báo cáo |
| `plot_results.py` | Vẽ loss/accuracy và confusion matrix |

`train.py`, `evaluate.py` và `plot_results.py` chạy được cho cả ba mô hình.

## Huấn luyện

```powershell
& .\.conda\Scripts\python.exe .\train.py complex_cnn
```

Thay `complex_cnn` bằng `simple_cnn` hoặc `transfer_model`. Lệnh này bắt đầu một lượt
train mới, không tiếp tục checkpoint cũ. Mỗi lượt tạo `runs/<model>/run_##/` với số
kế tiếp (`run_01`, `run_02`, ...), không ghi đè các lượt trước.

| Cờ | Ý nghĩa |
|---|---|
| `--device cpu\|cuda` | Mặc định tự chọn CUDA nếu có; AMP bật trên GPU hỗ trợ |
| `--seed N` | Ghi đè `seed` trong config |
| `--num-workers N` | Số tiến trình nạp dữ liệu; mặc định 2 trên CUDA, 0 trên CPU |

Tham số nằm trong `configs/<model>.json`. Cấu hình hiện tại của `complex_cnn`:

| Tham số | Giá trị |
|---|---|
| Epoch tối đa / batch size | 100 / 32 |
| Adam: learning rate / weight decay | 0.001 / 0.0001 |
| Giảm learning rate | Nhân 0.5 sau 3 epoch không giảm validation loss đủ ngưỡng, tối thiểu 0.000001 |
| Early stopping | Dừng sau 10 epoch không giảm validation loss đủ 0.0001 |
| Chọn checkpoint | Accuracy validation cao nhất; bằng nhau thì lấy loss thấp hơn |
| Seed | 42 |

Scheduler và early stopping theo dõi validation loss. Ngưỡng `min_delta` dùng cho
việc giảm learning rate/dừng sớm, không ngăn lưu checkpoint tốt hơn dù cải thiện nhỏ.
`checkpoint_monitor` có thể là `val_acc` hoặc `val_loss`; cấu hình thực tế của từng run
nằm trong `summary.json` của run đó.

Các khóa tùy chọn trong config, mô hình nào không khai báo thì không dùng:

| Khóa | Ý nghĩa |
|---|---|
| `weight_decay`, `lr_scheduler`, `early_stopping`, `checkpoint_monitor` | Như bảng trên |
| `label_smoothing` | Chỉ áp dụng cho loss khi train; validation loss luôn là cross-entropy thường |
| `classifier_learning_rate` | Learning rate riêng cho `model.classifier`; phần còn lại dùng `learning_rate` |
| `head_epochs`, `fine_tune_epochs`, `final_tune_epochs` | Chỉ cho `transfer_model`: số epoch của ba giai đoạn dưới đây; tổng phải bằng `epochs` |
| `fine_tune_blocks`, `final_tune_blocks`, `dropout` | Chỉ cho `transfer_model`: số block cuối của backbone được mở ở từng giai đoạn và dropout của classifier |

`transfer_model` train theo ba giai đoạn nối tiếp; giai đoạn nào có số epoch bằng 0 thì bỏ qua:

| Giai đoạn | Epoch hiện tại | Backbone được mở | Learning rate classifier | Learning rate backbone |
|---|---|---|---|---|
| `head` | 1–3 | Không | `classifier_learning_rate` | — |
| `fine_tune` | 4–15 | `fine_tune_blocks` block cuối | `fine_tune_classifier_learning_rate` | `learning_rate` |
| `final_tune` | 16–30 | `final_tune_blocks` block cuối | `final_classifier_learning_rate` | `final_learning_rate` cho block đã mở từ trước, `new_blocks_learning_rate` cho block mới mở |

Đầu mỗi giai đoạn, optimizer và scheduler được tạo lại và bộ đếm dừng sớm về 0. Early
stopping dừng cả lượt train chứ không chuyển sang giai đoạn sau. Cột `learning_rate`
trong `history.csv` của `transfer_model` là learning rate của classifier.

`transfer_model` dùng ảnh trong khoảng 0–1, không trừ mean/std của tập train
(`NORMALIZATION` trong `models/transfer_model.py`). Normalization thực tế của mỗi run
được lưu trong `summary.json` và `evaluate.py` dùng lại đúng giá trị đó.

## Kết quả và đánh giá

Sau train có 4 file; sau đánh giá test có tổng cộng 6 file:

| File | Nội dung |
|---|---|
| `best_model.pth` | Trọng số tại epoch được chọn |
| `summary.json` | Cấu hình thực tế, thứ tự lớp, normalization, epoch tốt nhất và metric |
| `history.csv` | Loss, accuracy train/validation và learning rate mỗi epoch |
| `training_curves.png` | Hai biểu đồ loss và accuracy |
| `classification_report.csv` | Precision, recall, F1, support từng lớp và trung bình trên test |
| `confusion_matrix.png` | Confusion matrix trên test: hàng là lớp thật, cột là dự đoán |

Chỉ đánh giá test sau khi chốt mô hình:

```powershell
& .\.conda\Scripts\python.exe .\evaluate.py .\runs\complex_cnn\run_01
```

Thay `run_01` bằng thư mục thực tế. `evaluate.py` không train lại; nó đọc tên mô hình,
cấu hình và normalization từ `summary.json` của run, rồi cập nhật báo cáo test và
summary trong chính thư mục đó. Có thể thêm `--device cpu|cuda|directml` và
`--batch-size N`. `plot_results.py <run_dir>` vẽ lại `training_curves.png` từ
`history.csv`; confusion matrix chỉ được vẽ khi chạy `evaluate.py`.

## Kết quả Complex CNN

Cấu hình ở trên, train trên `data/processed_classification/` (Japan và Czech) với ba
seed: `runs/complex_cnn/run_01` (seed 42), `run_02` (seed 123), `run_03` (seed 999).
Cột cuối là trung bình ± độ lệch chuẩn của ba seed.

| Chỉ số tại checkpoint | Seed 42 | Seed 123 | Seed 999 | Trung bình |
|---|---:|---:|---:|---:|
| Epoch đã chạy / epoch checkpoint | 89 / 85 | 100 / 99 | 100 / 87 | — |
| Validation accuracy | 85,72% | 84,69% | 85,99% | 85,47 ± 0,69% |
| Validation loss | 0,4158 | 0,4514 | 0,4257 | 0,4310 ± 0,0184 |
| Test accuracy | 85,69% | 83,39% | 85,17% | 84,75 ± 1,21% |
| Test loss | 0,4306 | 0,4668 | 0,4450 | 0,4475 ± 0,0182 |
| Test macro-precision | 86,34% | 84,30% | 85,83% | 85,49 ± 1,06% |
| Test macro-recall | 86,05% | 83,81% | 85,46% | 85,10 ± 1,16% |
| Test macro-F1 | 86,19% | 84,02% | 85,63% | 85,28 ± 1,12% |

Trung bình ba seed theo từng lớp trên test (1.908 ảnh):

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---:|---:|---:|---:|
| D00 | 83,61% | 83,76% | 83,68% | 581 |
| D10 | 86,90% | 86,96% | 86,93% | 432 |
| D20 | 80,88% | 82,55% | 81,69% | 550 |
| D40 | 90,56% | 87,15% | 88,82% | 345 |

Test accuracy theo nước: Japan 85,15 ± 1,33% (1.693 ảnh), Czech 81,55 ± 1,07% (215 ảnh).
Nhầm lẫn nhiều nhất là giữa D00 và D20: 10,1% ảnh D00 bị dự đoán là D20 và 9,9% ảnh
D20 bị dự đoán là D00.

`runs/`, `data/`, `.conda/` và checkpoint bị Git bỏ qua: cần sao lưu/chia sẻ riêng.
