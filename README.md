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

The processed data is in `data/processed_classification/` and consists of 12,467 crops of
224×224 pixels:

| Split | Japan | Czech | Total |
|---|---:|---:|---:|
| train | 7,657 | 1,002 | 8,659 |
| val | 1,711 | 214 | 1,925 |
| test | 1,668 | 215 | 1,883 |

The data is created in two steps, which do not need to be rerun for training:

1. `crop_images.ipynb` cuts crops from the images and annotations in `data/<country>/`. Its
   second cell then deletes 10 Japan crops whose bounding boxes do not mark road damage.
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
| `simple_cnn` | 3 Conv–ReLU–MaxPool layers, then 2 Linear layers | 50 epochs, batch 32, Adam learning rate 0.001 |
| `complex_cnn` | 4 blocks (2 Conv–BatchNorm–ReLU, MaxPool, Dropout), then 2 Linear layers with dropout | Up to 100 epochs, batch 32, Adam learning rate 0.001, weight decay 0.0001; learning rate reduction and early stopping based on validation loss |
| `transfer_model` | First 10 blocks of YOLOv8s pretrained on RDD2022, plus a new classifier | 3 epochs training only the classifier (Adam learning rate 0.001), then 30 epochs fine-tuning the last 3 backbone blocks as well (learning rate 0.00001 for the backbone, 0.0001 for the classifier); batch 16, weight decay 0.001, label smoothing 0.05 |

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

Training and evaluation do not fix a random seed,
so results differ slightly between runs.

Training and evaluating `transfer_model` both load the pretrained weights from Hugging Face. The
first run downloads them into the Hugging Face cache, so it needs an internet connection.

## Evaluation

```powershell
python src/predict.py simple_cnn
python src/predict.py complex_cnn
python src/predict.py transfer_model
```

Loads `saved_models/<model>.pth` and prints the test loss and accuracy, per-class precision,
recall and F1 with macro and weighted averages, and the confusion matrix. Evaluate on the test set
only after the model is final.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `simple_cnn` | 76.78% | 78.28% | 77.24% | 77.36% |
| `complex_cnn` | 85.32% | 86.00% | 85.64% | 85.78% |
| `transfer_model` | 89.06% | 89.44% | 89.07% | 89.23% |