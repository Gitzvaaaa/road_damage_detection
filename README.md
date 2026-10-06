# Road damage classification

Phân loại ảnh hư hỏng mặt đường Japan thành bốn lớp: `D00`, `D10`, `D20`, `D40`.
Kiến trúc nằm trong `models/`, tham số huấn luyện nằm trong `configs/`.

## Môi trường và dữ liệu

Chạy các lệnh từ thư mục gốc dự án. Trên máy hiện tại, môi trường có đủ thư viện là
`.conda/Scripts/python.exe` (môi trường venv, không phải conda prefix):

```powershell
.\.conda\Scripts\python.exe train.py --help
```

Trên máy khác, tạo môi trường bằng `conda env create -f environment.yml`, sau đó
`conda activate road_damage_env` và dùng `python` thay cho đường dẫn trên.

Dữ liệu đã xử lý ở `data/train/Japan/processed_classification/`, gồm các thư mục
`train/`, `val/`, `test/` và `normalization.json`. Dữ liệu hiện có gồm 7.657 ảnh
train, 1.711 ảnh validation và 1.668 ảnh test. Các crop cùng ảnh gốc nằm trong cùng
một tập; mean/std được tính từ train.

`data_processing.ipynb` tạo crop; `prepare_classification.py` chia tập và chuẩn hóa.
Không cần chạy lại các bước này để dùng checkpoint đã có. Chạy lại
`prepare_classification.py` sẽ thay thế thư mục dữ liệu phân loại.

## Các file chính

| File | Công việc |
|---|---|
| `models/complex_cnn.py` | 4 block, mỗi block có 2 Conv–BatchNorm–ReLU, pooling và dropout; cuối là bộ phân loại |
| `dataset.py` | Nạp ảnh, lật ngang khi train, chuẩn hóa |
| `train.py` | Train/validation, giảm learning rate, dừng sớm và lưu checkpoint |
| `evaluate.py` | Nạp checkpoint, đánh giá test, xuất báo cáo |
| `plot_results.py` | Vẽ loss/accuracy và confusion matrix |

## Huấn luyện

```powershell
.\.conda\Scripts\python.exe train.py complex_cnn
```

Lệnh này bắt đầu một lượt train mới, không tiếp tục checkpoint cũ.
Mỗi lượt tạo `runs/complex_cnn/run_###/`, không ghi đè các lượt trước.
Simple CNN cũng dùng được qua `train.py simple_cnn`.

Sửa tham số trong `configs/complex_cnn.json` trước khi train:

| Tham số mặc định | Giá trị |
|---|---|
| Epoch tối đa / batch size | 40 / 32 |
| Adam: learning rate / weight decay | 0.001 / 0.0001 |
| Giảm learning rate | Nhân 0.5 sau 4 epoch không giảm validation loss đủ ngưỡng, tối thiểu 0.000001 |
| Early stopping | Dừng sau 10 epoch không giảm validation loss đủ 0.0001 |
| Chọn checkpoint | Accuracy validation cao nhất; bằng nhau thì lấy loss thấp hơn |
| Seed | 42 |

Scheduler và early stopping theo dõi validation loss. Ngưỡng `min_delta` dùng cho
việc giảm learning rate/dừng sớm, không ngăn lưu checkpoint tốt hơn dù cải thiện nhỏ.
`checkpoint_monitor` có thể là `val_acc` hoặc `val_loss`; cấu hình lịch sử từng run
nằm trong `summary.json` của run đó.

Code tự chọn CUDA nếu có, dùng AMP trên GPU hỗ trợ và FP32 trên CPU; nạp dữ liệu
trong tiến trình chính (`num_workers=0`). Có thể ép CPU bằng `--device cpu`.
Các cờ cũ như `--epochs`, `--batch-size`, `--num-workers`, `--no-amp`,
`--deterministic`, `--skip-test`, `--evaluate-test` đã bỏ.
Không còn khóa `evaluate_test` trong config: mọi lượt train chỉ dùng train/validation.

## Kết quả và đánh giá

Sau train có đúng 4 file; sau đánh giá test có tổng cộng 6 file:

| File | Nội dung |
|---|---|
| `best_model.pth` | Trọng số tại epoch được chọn |
| `summary.json` | Cấu hình thực tế, thứ tự lớp, normalization, epoch tốt nhất và metric |
| `history.csv` | Loss, accuracy train/validation và learning rate mỗi epoch |
| `training_curves.png` | Hai biểu đồ loss và accuracy |
| `classification_report.csv` | Precision, recall, F1, support từng lớp và trung bình trên test |
| `confusion_matrix.png` | Test: hàng là lớp thật, cột là dự đoán |

Chỉ đánh giá test sau khi chốt mô hình. Ví dụ với một lượt train mới:

```powershell
.\.conda\Scripts\python.exe evaluate.py runs/complex_cnn/run_001
.\.conda\Scripts\python.exe plot_results.py runs/complex_cnn/run_001
```

Thay `run_001` bằng thư mục thực tế. `evaluate.py` không train lại; nó cập nhật
báo cáo test và summary trong thư mục được truyền vào, dùng normalization đã lưu.
`plot_results.py` vẽ lại đường học từ CSV; với run cũ có `confusion_matrix.csv`,
nó cũng vẽ lại confusion matrix. Run mới lưu confusion matrix trực tiếp thành PNG,
không có CSV trung gian. Các script không xóa file thừa trong run cũ.

## Complex CNN hiện tại: dùng V2, không cần train lại

| Chỉ số tại checkpoint | V1 | V2 | V3 |
|---|---:|---:|---:|
| Epoch tối đa / thực tế | 10 / 10 | 40 / 40 | 100 / 63 |
| Epoch checkpoint | 10 | 40 | 53 |
| Validation loss | 0,72938 | **0,51889** | 0,52224 |
| Validation accuracy | 73,23% | **81,94%** | 81,41% |
| Validation macro-F1 đã lưu | Không có | **82,54%** | 82,04% |

V2 là checkpoint chính đã chọn dựa trên validation. Test V2 đã có:
accuracy **83,51%**, macro-F1 **84,37%**, loss **0,47417**, trên **1.668 ảnh**.

Bộ kết quả gọn để xem và chia sẻ nằm ở **`runs/complex_cnn/v2/`**, gồm 6 file
trong bảng trên. Đây là kết quả V2 đã có, chỉ vẽ lại đường học; không train
hay đánh giá test lại. Summary và lịch sử vẫn ghi đúng lần huấn luyện gốc.
`v1/` và `v3/` mỗi thư mục giữ 4 file: checkpoint, summary, history và đường học.
Các file kiến trúc/báo cáo phụ, bản V2 trùng và thư mục backup đã được dọn.

`runs/cnn_with_metrics/` là kết quả Simple CNN cũ (10 epoch, checkpoint epoch 9,
validation accuracy 71,83%, test accuracy 71,76%), được giữ riêng để so sánh baseline.
Thư mục này không được dùng khi train hay đánh giá Complex CNN.

V1/V2 chọn checkpoint theo accuracy; V3 chọn theo loss và có cấu hình khác.
V3 dừng sớm ở epoch 63, giữ checkpoint epoch 53. Chênh lệch V2–V3 nhỏ, không đủ
để khẳng định train lâu hơn luôn kém hơn. V1/V3 từng được đánh giá test rồi dọn
báo cáo ở phiên bản trước; việc dọn không xóa lịch sử đánh giá đó.

Việc rút gọn giữ nguyên kiến trúc, tiền xử lý và checkpoint. Một lần train mới có
thể cho kết quả khác và chạy chậm hơn bản có nhiều worker; không gán kết quả
lịch sử cho một lần chạy mới.

Việc cần làm tiếp với Complex CNN:
1. Xem `training_curves.png`, `classification_report.csv`, `confusion_matrix.png`
   trong `v2/`.
2. Viết phần thí nghiệm: mô tả kiến trúc, cấu hình V2 trong summary, so sánh validation
   V1–V3, rồi báo cáo test của V2. Giải thích các lớp dễ nhầm từ confusion matrix.
3. Giữ và chia sẻ cả thư mục `v2/` để có trọng số kèm cấu hình và kết quả.
   Không cần train hay chạy test lại chỉ vì đã rút gọn code.

`runs/`, `data/`, `.conda/` và checkpoint bị Git bỏ qua: cần sao lưu/chia sẻ riêng.

## Kiểm thử

```powershell
.\.conda\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm thử gradient, nạp trọng số, chọn checkpoint, scheduler, early stopping và
train/evaluate trên ảnh tổng hợp cho cả Simple CNN và Complex CNN. Dữ liệu thử nằm
trong thư mục tạm, không thay đổi dữ liệu và kết quả thí nghiệm thật.
