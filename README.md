# Road damage classification

Classifies road damage images (Japan and Czech) into four classes: `D00`, `D10`, `D20`, `D40`.
The project compares three models that share one train/evaluate pipeline: `simple_cnn`,
`complex_cnn` and `transfer_model`.

## Environment

```powershell
conda env create -f environment.yml
conda activate road_damage_env
```

Run every command from the project root. If you use the local `.conda/` environment, replace
`python` with `.\.conda\Scripts\python.exe`. Only `transfer_model` needs `ultralytics` and
`huggingface_hub`.

## Data

The processed data is in `data/processed_classification/` and consists of 224×224 crops:

| Split | Japan | Czech | Total |
|---|---:|---:|---:|
| train | 7,725 | 1,002 | 8,727 |
| val | 1,628 | 214 | 1,842 |
| test | 1,693 | 215 | 1,908 |

The data is created in two steps, which do not need to be rerun for training:

1. `crop_images.ipynb` cuts crops from the images and annotations in `data/<country>/`.
2. `prepare_classification.py` splits each country into train/val/test at a 70/15/15 ratio and
   computes the mean/std from the training set. Crops from the same source image stay in the
   same split.

`EDA.ipynb` contains the exploratory data analysis.

## The three models

Architectures are in `models/`, training parameters are in `configs/<model>.json`.

| Model | Architecture | Training configuration |
|---|---|---|
| `simple_cnn` | 3 Conv–ReLU–MaxPool layers, then 2 Linear layers | 10 epochs, batch 32, Adam learning rate 0.001 |
| `complex_cnn` | 4 blocks (2 Conv–BatchNorm–ReLU, MaxPool, Dropout), then 2 Linear layers with dropout | Up to 100 epochs, batch 32, Adam learning rate 0.001, weight decay 0.0001; learning rate reduction and early stopping based on validation loss |
| `transfer_model` | First 10 blocks of YOLOv8s pretrained on RDD2022, plus a new classifier | 3 epochs training only the classifier, then 15 epochs fine-tuning the last 3 backbone blocks as well; batch 16, label smoothing 0.05 |

All three use the same training augmentation (horizontal flip, affine, color jitter,
perspective, blur) and keep the checkpoint with the highest validation accuracy. `simple_cnn`
and `complex_cnn` normalize images with the training set mean/std; `transfer_model` uses
images in the 0–1 range.

## Training

```powershell
python train.py simple_cnn
python train.py complex_cnn
python train.py transfer_model
python train.py transfer_model --device directml --num-workers 0
```

| Flag | Meaning |
|---|---|
| `--device cpu\|cuda\|directml` | Defaults to CUDA, then DirectML, then CPU |
| `--seed N` | Overrides `seed` in the config |
| `--num-workers N` | Number of data loading workers; default 2 on CUDA, 0 on DirectML/CPU |

DirectML requires `torch-directml` in a separate Python environment from CUDA.

Each run creates a new folder `runs/<model>/run_###/` (`run_001`, `run_002`, ...) and never
overwrites an earlier run. The first `transfer_model` training downloads the pretrained
weights from Hugging Face into `runs/.pretrained/`, so it needs an internet connection.

## Evaluation

```powershell
python evaluate.py runs/simple_cnn/run_001
python evaluate.py runs/complex_cnn/run_001
python evaluate.py runs/transfer_model/run_001
```

Replace the path with the run to evaluate. The same command works for all three models
because the model name and configuration are read from the run's `summary.json`; you can add
`--device cpu|cuda|directml`. Evaluate on the test set only after the model is final.

After training and evaluation, each run has 6 files:

| File | Content |
|---|---|
| `best_model.pth` | Weights at the selected epoch |
| `summary.json` | Configuration, selected epoch and validation/test metrics |
| `history.csv`, `training_curves.png` | Train/validation loss and accuracy per epoch |
| `classification_report.csv` | Per-class precision, recall and F1 on the test set |
| `confusion_matrix.png` | Confusion matrix on the test set |

## Results

| Model | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| `simple_cnn` | to be updated | to be updated | to be updated |
| `complex_cnn` | 85.99% | 85.32% | 85.78% |
| `transfer_model` | to be updated | to be updated | to be updated |
