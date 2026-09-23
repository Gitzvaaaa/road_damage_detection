# Japan road damage classification data

The original cropped images are in `data/train/Japan/processed_raw/clean/<class>/`.
They already have size 224×224. To prepare class folders for classification:

```powershell
.conda\python.exe prepare_classification.py
```

The script creates `data/train/Japan/processed_classification/` with `train/`,
`val/`, and `test/` folders for D00, D10, D20, and D40. The split is 70/15/15
by original image, with seed 42; crop counts can differ slightly from these
percentages. All crops from one original image stay in the same split. Only
existing `clean` images are included; `ambiguous` images are excluded. The
script recreates the output folder each time it runs.

`class_to_idx.json` maps classes to labels, and `split_manifest.csv` records
every crop's split and source image. The images are RGB and 224×224.
`normalization.json` contains per-channel mean and standard deviation computed
**only from train** on pixels scaled to [0, 1]. Use the same values for all
three splits in the training loader:

```python
import json
from classification_loader import load_normalized_image

root = "data/train/Japan/processed_classification"
with open(f"{root}/normalization.json", encoding="utf-8") as handle:
    normalization = json.load(handle)
x = load_normalized_image(f"{root}/train/D00/Japan_000011_obj0_D00.jpg", normalization)
# x is float32 with shape (3, 224, 224)
```

Apply any augmentation to training images only, before normalization. Keep
validation and test images unaugmented.

## Train a small CNN

```powershell
.conda\python.exe train_cnn.py
```

The script uses three convolution layers, batch size 32, and 10 epochs. It
loads the saved mean/std and applies a random horizontal flip to training images.
Each epoch's train/validation loss and accuracy are saved to
`runs/cnn_with_metrics/history.csv`, and the curves are saved to
`runs/cnn_with_metrics/training_curves.png`. The plot updates after every epoch.
Run `.conda\python.exe plot_history.py` to redraw it from the saved CSV.
`summary.json` records the loss function (`CrossEntropyLoss`), optimizer,
hyperparameters, best validation epoch, and test metrics. `best_model.pth`
contains the weights selected by validation accuracy. The earlier
`runs/simple_cnn_japan.pth` is left untouched.
The train curve uses batches with random horizontal flips, so compare its
values with validation as a trend. Rerunning training replaces the files in
`runs/cnn_with_metrics/`.
Change `EPOCHS`, `BATCH_SIZE`, or `LEARNING_RATE` near the top of `train_cnn.py`
to experiment.
