# Road damage classification

Phân loại ảnh hư hỏng mặt đường Japan và Czech thành bốn lớp: `D00`, `D10`, `D20`, `D40`.
Kiến trúc nằm trong `models/`, tham số huấn luyện nằm trong `configs/`.

## Môi trường và dữ liệu

Chạy các lệnh từ thư mục gốc dự án. Trên máy hiện tại, môi trường có đủ thư viện là
`.conda/python.exe`:

```powershell
& .\.conda\python.exe .\train.py --help
```

Trên máy khác, tạo môi trường bằng `conda env create -f environment.yml`, sau đó
`conda activate road_damage_env` và dùng `python` thay cho đường dẫn trên.

Dữ liệu đã xử lý ở `data/processed_classification/`, gồm 8.727 ảnh train,
1.842 ảnh validation và 1.908 ảnh test. Tất cả ảnh hiện có kích thước 224 × 224.

```text
data/processed_classification/
├── train/{D00,D10,D20,D40}/
├── val/{D00,D10,D20,D40}/
├── test/{D00,D10,D20,D40}/
├── class_to_idx.json
├── normalization.json
└── split_manifest.csv
```

`dataset.py` đọc mean/std từ `normalization.json`; `ImageFolder` gán chỉ số lớp
theo thứ tự tên thư mục, khớp `class_to_idx.json` hiện tại.
`split_manifest.csv` ghi nguồn ảnh và đường dẫn crop: `output_path` tương đối với
thư mục dữ liệu hiện tại; `source_path` lưu đường dẫn nguồn cũ để truy vết.
Các script train/evaluate dùng trực tiếp dữ liệu đã xử lý này.

## Các file chính

| File | Công việc |
|---|---|
| `models/complex_cnn.py` | 4 block, mỗi block có 2 Conv–BatchNorm–ReLU, pooling và dropout; cuối là bộ phân loại |
| `dataset.py` | Nạp ảnh từ cấu trúc data hiện tại, augmentation khi train, chuẩn hóa |
| `train.py` | Train/validation, giảm learning rate, dừng sớm và lưu checkpoint |
| `evaluate.py` | Nạp checkpoint, đánh giá test, xuất báo cáo |
| `plot_results.py` | Vẽ loss/accuracy và confusion matrix |

## Huấn luyện

```powershell
& .\.conda\python.exe .\train.py complex_cnn
```

Lệnh này bắt đầu một lượt train mới, không tiếp tục checkpoint cũ.
Mỗi lượt tạo `runs/complex_cnn/run_###/`, không ghi đè các lượt trước.
Simple CNN cũng dùng được qua `train.py simple_cnn`.

Augmentation áp dụng online cho tập train của cả ba mô hình, theo thứ tự:

| Phép biến đổi | Tham số |
|---|---|
| `RandomHorizontalFlip` | `p=0.5` |
| `RandomAffine` | `degrees=10`, `translate=(0.05, 0.05)`, `scale=(0.9, 1.1)` |
| `ColorJitter` | `brightness=0.2`, `contrast=0.2`, `saturation=0.1`, `hue=0.02` |
| `RandomPerspective` | `distortion_scale=0.1`, `p=0.2` |
| `GaussianBlur` | `kernel_size=3`, `sigma=(0.1, 1.0)` |

Sau augmentation là `ToTensor` và `Normalize`. Validation/test chỉ chuyển tensor
và chuẩn hóa; ảnh gốc trên đĩa không bị thay đổi. Gaussian blur áp dụng mỗi lần
nạp ảnh train, với sigma ngẫu nhiên trong khoảng đã chỉ định.

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

Code tự chọn CUDA nếu có; `--device cuda` yêu cầu GPU và báo lỗi nếu CUDA không dùng được.
Trên CUDA, mặc định bật AMP khi GPU hỗ trợ, pin memory, nạp trước dữ liệu,
worker chạy bằng spawn và cuDNN benchmark. Số worker tự chọn tối đa 4;
CPU mặc định dùng 0 worker. `--num-workers` ghi đè số worker.
`--no-amp` tắt AMP; `--deterministic` tắt benchmark và bật cuDNN deterministic
(không bảo đảm toàn bộ pipeline tái lập bit-for-bit).

Mặc định chỉ train/validation, kể cả config cũ có `evaluate_test: true`.
Chỉ `--evaluate-test` mới chạy test sau train; `--skip-test` giữ hành vi mặc định.
`--epochs` và `--batch-size` ghi đè config cho riêng lượt chạy; cấu hình thực tế
được lưu vào summary.

### Chạy trên Kaggle T4 x2

Đặt code và `data/processed_classification/` đúng cấu trúc trong
`/kaggle/working/road_damage`, rồi chạy trong Notebook:

```python
%cd /kaggle/working/road_damage
!python -u train.py complex_cnn --device cuda --num-workers 2 --batch-size 64
```

Nếu PyTorch nhận cả hai GPU, code tự dùng `DataParallel`, log ghi `GPUs used: 2`.
Batch size 64 là tổng, khoảng 32 ảnh/GPU. Mặc định trong config vẫn là 32;
batch size mới có thể làm thay đổi kết quả học, cần so sánh validation.
Checkpoint được lưu không có tiền tố `module.`, dùng được với evaluate trên
CPU hoặc một GPU. Đánh giá test chạy trên một GPU và chỉ khi yêu cầu.

Để so sánh một GPU với hai GPU, thêm `--single-gpu` vào cùng lệnh, giữ batch size
và số worker giống nhau. So sánh `epoch_seconds` và `train_images_per_second`
trong history sau epoch đầu (epoch đầu gồm khởi động worker/autotuning).
Hai GPU không bảo đảm nhanh hơn cho CNN nhỏ: có chi phí chia batch/gộp gradient,
và augmentation chạy trên CPU có thể khiến GPU chờ. Thử worker 2 rồi 4;
không đánh giá tốc độ chỉ theo màu biểu đồ hay lượng VRAM đã dùng.
DataParallel tính BatchNorm theo từng phần batch, không đồng bộ thống kê giữa GPU;
kết quả có thể khác chạy một GPU. PyTorch khuyến nghị DDP cho hiệu năng đa GPU
quy mô lớn; ở đây dùng DataParallel để giữ lệnh chạy một tiến trình đơn giản.

Chỉ chạy test sau khi chốt mô hình:

```python
!python -u evaluate.py runs/complex_cnn/run_001 --device cuda
```

Thay tên run theo log. Lưu/tải kết quả trong `runs/` trước khi kết thúc phiên.
`best_model.pth` chỉ chứa trọng số để đánh giá, chưa hỗ trợ resume huấn luyện.

## Kết quả và đánh giá

Sau train có 5 file; sau đánh giá test có tổng cộng 8 file:

| File | Nội dung |
|---|---|
| `best_model.pth` | Trọng số tại epoch được chọn |
| `summary.json` | Cấu hình thực tế, thứ tự lớp, normalization, epoch tốt nhất và metric |
| `history.csv` | Loss, accuracy, validation macro-F1, learning rate, thời gian và ảnh train/giây mỗi epoch |
| `validation_report.csv` | Precision, recall, F1 trên validation tại checkpoint tốt nhất |
| `training_curves.png` | Hai biểu đồ loss và accuracy |
| `classification_report.csv` | Precision, recall, F1, support từng lớp và trung bình trên test |
| `confusion_matrix.csv` | Ma trận số lượng dự đoán để vẽ lại biểu đồ |
| `confusion_matrix.png` | Test: hàng là lớp thật, cột là dự đoán |

Chỉ đánh giá test sau khi chốt mô hình. Ví dụ với một lượt train mới:

```powershell
& .\.conda\python.exe .\evaluate.py .\runs\complex_cnn\run_001
& .\.conda\python.exe .\plot_results.py .\runs\complex_cnn\run_001
```

Thay `run_001` bằng thư mục thực tế. `evaluate.py` không train lại; nó cập nhật
báo cáo test và summary trong thư mục được truyền vào, dùng normalization đã lưu.
`plot_results.py` vẽ lại đường học và confusion matrix từ CSV.
Các script không xóa file thừa trong run cũ.

## Kết quả Complex CNN lịch sử trên dữ liệu Japan cũ

Các kết quả dưới đây dùng bộ dữ liệu và augmentation cũ (chỉ lật ngang).
Chúng không đại diện cho dữ liệu Japan + Czech và augmentation hiện tại.
Để đo kết quả cấu hình mới, cần train một run mới rồi đánh giá test.
Đánh giá lại checkpoint cũ bằng `evaluate.py` sẽ dùng tập test hiện tại và cập nhật
báo cáo trong run đó; dùng run mới nếu muốn giữ nguyên báo cáo lịch sử.

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
trong bảng trên (không có `confusion_matrix.csv`). Đây là kết quả V2 đã có, chỉ vẽ lại đường học; không train
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

Kiến trúc và checkpoint lịch sử được giữ nguyên. Dữ liệu và augmentation của lần
train mới đã thay đổi; không gán kết quả lịch sử cho một lần chạy mới.

Cách đọc lại thí nghiệm Complex CNN lịch sử:
1. Xem `training_curves.png`, `classification_report.csv`, `confusion_matrix.png`
   trong `v2/`.
2. Viết phần thí nghiệm: mô tả kiến trúc, cấu hình V2 trong summary, so sánh validation
   V1–V3, rồi báo cáo test của V2. Giải thích các lớp dễ nhầm từ confusion matrix.
3. Giữ và chia sẻ cả thư mục `v2/` để có trọng số kèm cấu hình và kết quả.
   Không cần train hay chạy test lại chỉ vì đã rút gọn code.

`runs/`, `data/`, `.conda/` và checkpoint bị Git bỏ qua: cần sao lưu/chia sẻ riêng.

## Kiểm thử

```powershell
& .\.conda\python.exe -m unittest discover -s tests -v
```

Kiểm thử cấu trúc dữ liệu, augmentation chỉ trên train, gradient, nạp trọng số,
chọn checkpoint, scheduler, early stopping và
train/evaluate trên ảnh tổng hợp cho cả Simple CNN và Complex CNN. Dữ liệu thử nằm
trong thư mục tạm, không thay đổi dữ liệu và kết quả thí nghiệm thật.
