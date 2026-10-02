# Road damage classification

Phân loại ảnh hư hỏng mặt đường từ dữ liệu Japan thành bốn lớp: `D00`, `D10`, `D20`, `D40`. Dự án gồm simple CNN, complex CNN và transfer learning; `simple_cnn` và `complex_cnn` đã triển khai, `transfer_model` đang chờ hoàn thiện.

## Cấu trúc

```text
road_damage_detection/
├── configs/                  # Cấu hình huấn luyện từng model
├── models/                   # Định nghĩa kiến trúc
├── data/train/Japan/
│   ├── processed_raw/clean/
│   └── processed_classification/
│       ├── train/  val/  test/
│       ├── normalization.json
│       ├── class_to_idx.json
│       └── split_manifest.csv
├── data_processing.ipynb      # Tạo crop từ dữ liệu gốc
├── prepare_classification.py  # Chia tập và tính mean/std
├── dataset.py                # Nạp ảnh, augmentation và chuẩn hóa
├── train.py
├── evaluate.py
├── plot_results.py
├── environment.yml
├── tests/
└── runs/
    └── complex_cnn/
        ├── v1/               # Giới hạn 10 epoch, chạy đủ 10
        ├── v2/               # Giới hạn 40 epoch, chạy đủ 40
        ├── v3/               # Giới hạn 100 epoch, dừng sớm ở 63
        └── run_###/          # Các lượt train mới
```

`data/`, `runs/` và môi trường `.conda/` không được đưa vào Git. Các thành viên cần chuẩn bị dữ liệu riêng hoặc nhận bản dữ liệu đã xử lý của nhóm; checkpoint và log được chia sẻ riêng qua thư mục lưu trữ dùng chung.

## Môi trường

Tạo và kích hoạt môi trường từ thư mục gốc dự án:

```bash
conda env create -f environment.yml
conda activate road_damage_env
```

Nếu đã có môi trường tại `.conda/`, có thể kích hoạt bằng `conda activate ./.conda` thay vì tạo lại. Các lệnh dưới đây chạy từ thư mục gốc dự án, trong môi trường đã kích hoạt.

## Chuẩn bị dữ liệu

Nếu đã có đầy đủ `data/train/Japan/processed_classification/`, có thể chuyển sang huấn luyện. Để tạo lại từ các crop trong `processed_raw/clean/`:

```bash
python prepare_classification.py
```

Lệnh này thay thế thư mục dữ liệu phân loại đã xử lý. Các crop từ cùng một ảnh gốc luôn thuộc cùng một tập, với tỷ lệ chia theo ảnh gốc là 70/15/15. Mean/std chỉ được tính trên tập train. Bản dữ liệu hiện tại gồm 7.657 ảnh train, 1.711 ảnh validation và 1.668 ảnh test.

## Huấn luyện

```bash
python train.py simple_cnn
python train.py complex_cnn
```

Tham số nằm trong `configs/<model>.json`. Mỗi lần chạy tạo thư mục mới `runs/<model>/run_###/`, không ghi đè kết quả cũ. Hiện chưa hỗ trợ tiếp tục huấn luyện từ checkpoint; mỗi lệnh train bắt đầu một lượt mới.

Mặc định tự ưu tiên GPU CUDA và chuyển sang CPU nếu CUDA không khả dụng khi khởi động. Khi dùng GPU, code bật mixed precision, cuDNN autotuning và tối ưu nạp dữ liệu; CPU dùng FP32. Thiết bị và cấu hình thực tế được in ra khi chạy và lưu trong `summary.json`.

Các tùy chọn thường dùng:

```bash
python train.py complex_cnn --epochs 40 --batch-size 32
python train.py complex_cnn --device cpu
python train.py complex_cnn --num-workers 2
python train.py complex_cnn --no-amp
python train.py complex_cnn --deterministic
```

`--deterministic` tắt autotuning và bật chế độ tính toán xác định của cuDNN, có thể giảm tốc độ; không bảo đảm kết quả giống tuyệt đối giữa các môi trường. Mặc định dùng một GPU. Nếu hết bộ nhớ GPU giữa chừng, cần giảm batch size rồi chạy lại; code không tự chuyển sang CPU trong lúc train.

Complex CNN mặc định dùng các tham số huấn luyện của V2:

| Tham số | Giá trị |
|---|---|
| Số epoch tối đa | 40 |
| Batch size | 32 |
| Optimizer | Adam, learning rate ban đầu 0.001 |
| Weight decay | 0.0001 |
| Giảm learning rate | Nhân 0.5 sau 4 epoch không cải thiện đủ; thấp nhất 0.000001 |
| Dừng sớm | Sau 10 epoch không cải thiện validation loss đủ ngưỡng 0.0001 |
| Chọn checkpoint | Validation accuracy cao nhất; nếu bằng nhau, chọn loss thấp hơn |

Scheduler và early stopping cùng theo dõi validation loss với ngưỡng tuyệt đối 0.0001, so sánh với mốc loss tốt đã ghi nhận, không chỉ với epoch ngay trước. Các mức giảm nhỏ tích lũy đủ ngưỡng vẫn đặt lại bộ đếm. Việc chọn checkpoint theo validation accuracy độc lập với tiêu chí dừng theo loss; ngưỡng này không giới hạn việc lưu checkpoint tốt hơn.

Complex CNN mặc định chỉ train/validation. Simple CNN mặc định chọn checkpoint theo validation accuracy và tự đánh giá test sau train. Có thể điều khiển việc đánh giá bằng `--skip-test` hoặc `--evaluate-test`; hai cờ không dùng cùng nhau. Chỉ dùng train/validation để điều chỉnh mô hình, đánh giá test sau khi chốt cấu hình.

## Đánh giá và xem kết quả

Đánh giá checkpoint V2 đã chốt và vẽ lại biểu đồ:

```bash
python evaluate.py runs/complex_cnn/v2
python plot_results.py runs/complex_cnn/v2
```

Thay đường dẫn bằng run cần xem, ví dụ `runs/complex_cnn/v2`. `evaluate.py` nạp `best_model.pth` và đánh giá trên test, không huấn luyện lại. Nếu không truyền đường dẫn, cả hai script chọn run mới nhất của simple CNN. Các tùy chọn thiết bị như `--device cpu` và `--num-workers` cũng dùng được với `evaluate.py`.

Mỗi run lưu các file sau:

| File | Nội dung |
|---|---|
| `best_model.pth` | Trọng số checkpoint được chọn |
| `architecture.txt` | Kiến trúc model |
| `history.csv` | Log loss, accuracy, validation macro-F1 và learning rate từng epoch, tùy phiên bản |
| `training_curves.png` | Biểu đồ quá trình huấn luyện |
| `summary.json` | Cấu hình, thiết bị, thời gian chạy, epoch tốt nhất, lý do dừng và metric |
| `validation_report.json`, `.csv` | Precision, recall, F1 từng lớp tại checkpoint được chọn; có từ v2 |
| `classification_report.json`, `.csv`, `.txt` | Báo cáo test, có sau khi đánh giá |
| `confusion_matrix.csv`, `.png` | Ma trận nhầm lẫn trên test, có sau khi đánh giá |

Trong `summary.json`, `best_val_acc` và `best_val_macro_f1` là metric tại checkpoint đã chọn, không nhất thiết là giá trị cao nhất của toàn bộ lịch sử. Trong confusion matrix, hàng là lớp thật và cột là lớp dự đoán.

Checkpoint, log và báo cáo đầy đủ nằm trong `runs/complex_cnn/v1/`, `v2/`, `v3/`; không cần giữ thêm file ZIP trong dự án. Khi chia sẻ với nhóm, sao lưu cả thư mục kết quả để giữ đủ trọng số và báo cáo.

## Kết quả complex CNN

Các số liệu dưới đây lấy từ báo cáo tại checkpoint đã lưu của từng phiên bản:

| Chỉ số | V1 | V2 | V3 |
|---|---:|---:|---:|
| Số epoch tối đa | 10 | 40 | 100 |
| Số epoch thực tế | 10 | 40 | 63 |
| Epoch của checkpoint | 10 | 40 | 53 |
| Validation loss | 0,72938 | **0,51889** | 0,52224 |
| Validation accuracy | 73,23% | **81,94%** | 81,41% |
| Validation macro-F1 | Không lưu | **82,54%** | 82,04% |

Chọn **checkpoint v2 tại epoch 40** làm kết quả chính của complex CNN, nằm ở `runs/complex_cnn/v2/best_model.pth`. V2 cải thiện rõ so với v1 và có các metric validation nhỉnh hơn checkpoint v3.

Chỉ giữ báo cáo test của **V2**: accuracy **83,51%**, macro-F1 **84,37%**, loss **0,47417** trên **1.668 ảnh**. Các báo cáo nằm trong `runs/complex_cnn/v2/`; không cần đánh giá lại để lấy các số liệu này.

V1 và V3 giữ checkpoint, lịch sử train và các số liệu validation đã có để so sánh thí nghiệm. Báo cáo và metric test của hai phiên bản này đã được xóa; không chạy test thêm cho chúng. Việc dọn kết quả không thay đổi lịch sử đã đánh giá test, được ghi nhận trong `summary.json` bằng `test_results_status: removed_after_evaluation`.

V3 dừng sớm ở epoch 63 sau 10 epoch không cải thiện đủ validation loss; checkpoint tốt nhất ở epoch 53. Thí nghiệm này không cho thấy lợi ích rõ rệt của việc kéo dài huấn luyện. Chốt lượt tối ưu complex CNN tại đây, chưa cần tăng epoch hay sửa kiến trúc chỉ từ các kết quả này.

V1/v2 chọn checkpoint theo validation accuracy; v3 chọn theo validation loss. Các lượt chạy bắt đầu từ đầu và có thay đổi cấu hình scheduler/early stopping, nên không thể kết luận cứ train quá 40 epoch là kém hơn. Chênh lệch v2–v3 nhỏ; kết luận chọn v2 dựa trên validation trong các lượt chạy đã có, không khẳng định v2 luôn tốt hơn qua nhiều lần chạy. Kết quả test V2 dùng để báo cáo model đã chốt, không dùng để mở thêm vòng điều chỉnh model.

Cấu hình mặc định đã trở về các tham số huấn luyện của V2, đồng thời giữ `evaluate_test: false` để không tự đánh giá test sau train. Dùng checkpoint V2 đã lưu không cần train lại. Một lượt train mới không bảo đảm cho kết quả giống tuyệt đối checkpoint cũ. Cấu hình lịch sử của V1, V2 và V3 vẫn được giữ trong `summary.json` tương ứng.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Kiểm thử bao gồm gradient, cập nhật trọng số, nạp checkpoint, giảm learning rate, dừng sớm và pipeline train/evaluate/plot trên dữ liệu tổng hợp. Chúng kiểm chứng hoạt động của code, không đo chất lượng phân loại trên dữ liệu Japan.
