# Road damage classification

Dự án dùng chung dữ liệu Japan cho ba hướng mô hình: simple CNN, complex CNN và transfer learning. Hiện `simple_cnn` và `transfer_model` đã hoạt động; `complex_cnn` là chỗ trống để xây dựng sau.

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

Train và evaluate tự chọn **CUDA → DirectML → CPU** theo khả năng của máy. Chỉ cài một trong các môi trường sau trên mỗi máy:

| Phần cứng | Tạo môi trường | Kích hoạt |
|---|---|---|
| NVIDIA, driver tương thích CUDA 12.1 | `conda env create -f environment-cuda.yml` | `conda activate road_damage_cuda` |
| GPU dùng DirectML trên Windows | `conda env create -f environment-directml.yml` | `conda activate road_damage_directml` |
| CPU | `conda env create -f environment.yml` | `conda activate road_damage_env` |

Các môi trường dùng cùng phiên bản API `torch==2.4.1`, `torchvision==0.19.1`, nhưng cài binary theo backend. CUDA dùng wheel CUDA 12.1 theo [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/#v241). DirectML giữ bộ phiên bản đã kiểm tra trong `environment-directml.yml`. Không cài `torch-directml` vào môi trường CUDA vì dependency của nó có thể thay bản torch đã cài. Máy CUDA/CPU không cần import hoặc cài DirectML.

Sau khi kích hoạt môi trường, mọi thành viên chạy cùng một lệnh:

```powershell
python train.py simple_cnn
python train.py transfer_model
```

Có thể chọn rõ backend và batch size bằng CLI, không cần sửa file chung rồi push lên Git:

```powershell
python train.py transfer_model --device cuda --batch-size 16
python train.py transfer_model --device directml --batch-size 8
python train.py transfer_model --device cpu --batch-size 4
python evaluate.py ./runs/transfer_model/run_004 --device cpu --batch-size 8
```

Khi không truyền `--device`, `main()` trong train/evaluate tự chọn CUDA → DirectML → CPU. Option nhận `cpu`, `cuda` hoặc `directml`; chỉ định backend không khả dụng sẽ báo lỗi. Checkpoint luôn lưu tensor trên CPU nên có thể chuyển giữa các backend; evaluation chọn thiết bị của máy hiện tại, không phụ thuộc backend ghi trong run cũ. Kết quả số học giữa các backend có thể có sai khác nhỏ. Batch size ghi đè được lưu vào summary của run, không sửa JSON cấu hình.

Tham số nằm trong `configs/simple_cnn.json`. Mỗi lần chạy tạo một thư mục mới `runs/simple_cnn/run_###/`, không ghi đè run trước. Trong mỗi run có `best_model.pth`, `history.csv`, `training_curves.png`, `confusion_matrix.csv`, `confusion_matrix.png` và `summary.json`. Sau khi huấn luyện, `train.py` tự đánh giá trên test và tạo confusion matrix.

### Transfer learning

```powershell
python train.py transfer_model
```

`models/transfer_model.py` dùng **ResNet50 pretrained ImageNet (`IMAGENET1K_V2`)** trên PyTorch. ResNet50 được dùng cho các đặc trưng vết nứt và kết cấu mặt đường ([nghiên cứu](https://cronfa.swansea.ac.uk/Record/cronfa63288/Download/63288__27294__29c0c00d035e453286cc6c90575766e7.pdf), [weights Torchvision](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html)). Head hiện tại: global average pooling → Linear(2048, 128) → ReLU → Dropout(0.5) → Linear(128, số lớp). `hidden_dim` và `dropout` chỉnh được trong config. Với dữ liệu hiện tại, số lớp là 4: D00, D10, D20, D40; đầu ra là logits.

Chỉnh `fine_tune_blocks` trong `configs/transfer_model.json` để mở số residual block cuối mong muốn, từ **0 đến 16**. `0` chỉ train classifier; `1` mở block cuối của `layer4`; mặc định **`3` mở toàn bộ `layer4`**; `6` mở thêm 3 block cuối của `layer3`; `16` mở tất cả residual block nhưng vẫn freeze stem. BatchNorm trong block mở khóa học weight/bias nhưng giữ running mean/variance pretrained.

Cấu hình hiện tại cân bằng khả năng thích nghi và kiểm soát overfitting. `run_004` có validation loss tăng về cuối; cấu hình hạn chế hơn của `run_005` chỉ đạt validation accuracy 0.7881 và test accuracy 0.7785. Lần điều chỉnh này chỉ mở lại 3 block cuối, giữ các siêu tham số khác của `run_005` để đo riêng ảnh hưởng của vùng fine-tuning:

- Fine-tune **3 block của layer4**, đóng băng stem và layer1–layer3; giữ head nhỏ **128 neuron**.
- **AdamW**, weight decay `1e-3`; learning rate backbone `1e-5`, classifier `3e-4`.
- Label smoothing `0.05` cho loss dùng để backpropagation. Loss trong CSV và test vẫn là cross-entropy thường, để train/val/test có cùng cách tính.
- Giữ Dropout `0.5`, augmentation hình học nhẹ; thêm ColorJitter cho độ sáng, tương phản và độ bão hòa. Không xoay 90° vì D00/D10 phụ thuộc hướng vết nứt.
- `ReduceLROnPlateau(val_loss, factor=0.5, patience=1)` giảm learning rate khi loss chững lại.
- **Early stopping** sau 4 epoch không cải thiện validation loss đủ `min_delta=0.001`, tối đa 15 epoch, batch size 16. Đặt `early_stopping_patience=0` để tắt.
- **Chọn checkpoint theo validation loss nhỏ nhất**. `best_val_acc` là accuracy tại checkpoint này; `max_val_acc` ghi accuracy cao nhất quan sát được. `checkpoint_metric` phân biệt tiêu chí với các run cũ và simple CNN (vẫn chọn theo accuracy). Checkpoint cập nhật khi loss giảm dù mức giảm nhỏ hơn `min_delta`; ngưỡng này chỉ điều khiển early stopping.

Run ResNet50 cũ `run_004` đạt test accuracy 0.8541 nhưng có dấu hiệu overfitting trên đường loss. Cấu hình hiện tại cần được huấn luyện để xác nhận mức cải thiện; không bảo đảm accuracy hoặc loại bỏ tuyệt đối overfitting. Early stopping giới hạn việc tiếp tục train khi validation loss không cải thiện; file `best_model.pth` giữ checkpoint có validation loss nhỏ nhất, không mặc định lấy epoch cuối. Chỉ dùng validation để chọn cấu hình, sau đó báo cáo test của checkpoint được chọn.

Ảnh transfer learning được resize về 224 × 224 và chuẩn hóa theo ImageNet. Train xoay ±15°, dịch tối đa 10%, lật ngang và ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1); val/test không augmentation. Các CNN khác vẫn dùng mean/std của tập train. Lần train đầu cần tải pretrained weights; evaluate nạp checkpoint mà không tải lại weights.

Kết quả vẫn lưu tại `runs/transfer_model/run_###/`: `best_model.pth`, `history.csv`, `training_curves.png`, `confusion_matrix.csv`, `confusion_matrix.png`, `summary.json`. Summary lưu cấu hình, kiến trúc, backend train/evaluate, số epoch thực tế và trạng thái early stopping. `evaluate.py` đọc được cả checkpoint MobileNetV2 và ResNet50 head 256 cũ; cấu hình mới dùng head 128 được dựng từ config đã lưu. Checkpoint chỉ chứa trọng số/buffer để suy luận, chưa lưu trạng thái optimizer để resume chính xác.

`train.py complex_cnn` vẫn báo `NotImplementedError` cho đến khi xây dựng kiến trúc tương ứng.

## Đánh giá hoặc vẽ lại run có sẵn

Mặc định hai lệnh sau dùng run mới nhất của simple CNN:

```powershell
python evaluate.py
python plot_results.py
```

Để chọn run cụ thể:

```powershell
python evaluate.py .\runs\simple_cnn\run_001
python plot_results.py .\runs\simple_cnn\run_001
```

Trong confusion matrix, hàng là lớp thật và cột là lớp dự đoán. Đường train dùng ảnh lật ngang ngẫu nhiên, còn val/test không dùng augmentation.

## Kiểm tra trước khi push

```powershell
python test_training.py -v
```

Các test chạy trên CPU, không tải pretrained weights và không cần dữ liệu ảnh. Kiểm tra chọn backend (CUDA/DirectML được mô phỏng), số block mở khóa, BatchNorm, dựng head cũ và chọn checkpoint/early stopping. Cần kiểm tra huấn luyện trên GPU thật của từng backend khi thay phiên bản thư viện.
