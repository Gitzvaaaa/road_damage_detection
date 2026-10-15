# Road damage classification

Classifies road damage images (Japan and Czech) into four classes: `D00`, `D10`, `D20`, `D40`.
The project compares three models: `simple_cnn`, `complex_cnn` and `transfer_model`. Each model
is one standalone file in `src/`.

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

## Project structure

| File | Content |
|---|---|
| `src/dataset.py` | Loads a split as an `ImageFolder`, with training augmentation and normalization (shared by the three models) |
| `src/simple_cnn.py` | `SimpleCNN` |
| `src/complex_cnn.py` | `ComplexCNN` |
| `src/transfer_model.py` | `TransferModel` |
| `src/predict.py` | Evaluates a saved model on the test set |

Each model file contains, in order: the hyperparameters as constants, the model class (architecture
plus `fit`, `train_one_epoch` and `evaluate`), and the training code under
`if __name__ == "__main__":`.

## The three models

| Model | Architecture | Training configuration |
|---|---|---|
| `simple_cnn` | 3 Conv–ReLU–MaxPool layers, then 2 Linear layers | 10 epochs, batch 32, Adam learning rate 0.001 |
| `complex_cnn` | 4 blocks (2 Conv–BatchNorm–ReLU, MaxPool, Dropout), then 2 Linear layers with dropout | Up to 100 epochs, batch 32, Adam learning rate 0.001, weight decay 0.0001; learning rate reduction and early stopping based on validation loss |
| `transfer_model` | First 10 blocks of YOLOv8s pretrained on RDD2022, plus a new classifier | 3 epochs training only the classifier, then 30 epochs fine-tuning the last 3 backbone blocks as well; batch 16, label smoothing 0.05 |

All three use the same training augmentation (horizontal flip, affine, color jitter,
perspective, blur) and keep the weights from the epoch with the highest validation accuracy.
`simple_cnn` and `complex_cnn` normalize images with the training set mean/std;
`transfer_model` uses images in the 0–1 range.

## Training

```powershell
python src/simple_cnn.py
python src/complex_cnn.py
python src/transfer_model.py
```

Each command trains on the train set and prints the train/validation loss and accuracy of every
epoch. Whenever the validation accuracy improves, the weights (`state_dict`) are saved to
`saved_models/<model>.pth`, so the file always holds the best epoch so far. Running a model again
overwrites its file.

Training and evaluation use CPU by default, without device selection or a fixed random seed.

## Evaluation

```powershell
python src/predict.py simple_cnn
python src/predict.py complex_cnn
python src/predict.py transfer_model
```

Loads `saved_models/<model>.pth` and prints the test loss and accuracy, per-class precision,
recall and F1 with macro and weighted averages, and the confusion matrix. Evaluate on the test set
only after the model is final.

The first `transfer_model` run downloads the pretrained weights from Hugging Face into the
Hugging Face cache, so it needs an internet connection.

## Results

| Model | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| `simple_cnn` | to be updated | to be updated | to be updated |
| `complex_cnn` | 85.99% | 85.32% | 85.78% |
| `transfer_model` | to be updated | to be updated | to be updated |
