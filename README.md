# Robust CIFAR-10 Classification with 400 Training Images

This project trains a small Inception-style CNN **from scratch** on only 400 CIFAR-10 images
and tests it on a shifted test set (rotated, darker/brighter, occluded, color-changed images).

## What is in this repository

| File | Purpose |
|---|---|
| `notebook.ipynb` | Full executed notebook (Section B, Parts A to F) with all outputs and plots |
| `Summary_Section_B.docx` | One-page summary (final score, final configuration, most useful change) |
| `README.md` | This file |

> Rename `notebook.ipynb` to match your real notebook file name.

## How to reproduce the results

1. Open the notebook in **Kaggle** (GPU T4, Internet ON) or Google Colab (GPU runtime).
2. Libraries used: `torch`, `torchvision`, `numpy`, `matplotlib`, `pandas`, `scikit-learn`. All are preinstalled on Kaggle and Colab.
3. Click **Restart & Run All**. No manual download is needed:
   CIFAR-10 is downloaded with `torchvision` the first time and cached as `cifar10_raw.npz`.
4. Runtime is short (a few minutes). One training run takes only a few seconds on a GPU.

To run locally: `pip install torch torchvision numpy matplotlib pandas scikit-learn jupyter`, then open the notebook.
A GPU is recommended but not required.

## Data (we built everything from CIFAR-10)

No dataset files were provided, so we created them with fixed seeds:

- `train_small`: 500 images from the CIFAR-10 train set, 50 per class (seed 42).
- Split: 400 train (40 per class) and 100 validation (10 per class). There is no overlap.
- `clean_test`: the original 10,000 CIFAR-10 test images (reporting only).
- `shifted_test`: the 10,000 test images after this fixed pipeline (seed 123):
  rotation up to ±25°, brightness and contrast change (0.5 to 1.5), a hidden corruption
  (random channel swap or grayscale, one third of images each, one third unchanged),
  and a black occlusion box covering about 25% of the image.

**Important note:** this shifted set is our own proxy. The instructor's hidden set may differ.

## Model

- Inception-style CNN: stem conv, two Inception blocks (1×1, 1×1→3×3, 1×1→5×5 and pool→1×1 branches),
  global average pooling, Dropout 0.4, one FC layer with 10 outputs.
- Final model has **160,250 trainable parameters** (limit: 1,000,000).
- Normalization uses the mean and std of the 400 training images only.

## Training setup

- AdamW (lr 2e-3, weight decay 5e-4), cosine LR schedule, batch size 64, 60 epochs.
- Best-validation checkpoint (early stopping), BatchNorm, label smoothing 0.1, Dropout after global pooling.
- Seeds 0, 1, 2 for every result (mean ± std). Data split seed is fixed separately.
- Augmentation runs **on the fly on the GPU, on training batches only**. Nothing is pre-generated or saved.
- All hyperparameters were chosen using **validation accuracy only**. Test sets were used for reporting.

## Results (mean ± std over 3 seeds, accuracy in %)

| Configuration | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation (narrow, 27k params) | 48.33 ± 0.94 | 43.00 ± 0.60 | 16.83 ± 0.79 |
| Geometric only | 43.00 ± 1.63 | 39.86 ± 0.38 | 17.98 ± 1.44 |
| Photometric only | 51.00 ± 4.55 | 42.30 ± 1.08 | 19.28 ± 0.17 |
| Geometric + photometric | 42.67 ± 2.05 | 36.42 ± 0.99 | 19.44 ± 0.29 |
| + Cutout | 36.00 ± 0.82 | 34.36 ± 0.49 | 23.18 ± 1.06 |
| + Mixup + label smoothing | 36.00 ± 0.82 | 33.02 ± 1.67 | 21.56 ± 1.73 |
| **Final model** (wide, mild aug, Mixup, smoothing) | 45.67 ± 2.05 | 40.87 ± 1.24 | **24.72 ± 0.64** |

Rows 1 to 6 use the narrow model, so they show the effect of each augmentation.
The final row also widens the model (160k params) and uses milder augmentation, so it is a combined improvement.

### Main findings

- **Baseline failure:** overfitting (97.3% train vs 43.0% clean) plus covariate shift (43.0% clean vs 16.8% shifted).
- **Largest single gain: Cutout** (shifted 19.44% to 23.18%). It matches the 25% occlusion box in the shifted set.
- **Too much augmentation hurt the small 27k model** (train accuracy fell to about 43%). A wider model and milder augmentation fixed this.
- **Mixup + label smoothing** gave no clear gain on the narrow model (the difference is smaller than the std).
- **Hidden corruption (seed 0):** hidden-only accuracy was 31.35% for the final model (clean 39.54%) and 35.73% for the baseline (clean 42.16%).
  Our augmentation did **not** generalize to it. Channel swap hurt more than grayscale.
- **Error analysis (seed 0):** deer (10.0%), airplane (12.5%) and automobile (13.7%) are near chance on the shifted set.
  "Bird" predictions rise from 2.9% on clean images to 23.1% on shifted images.
- **TTA (bonus, 8 views):** TTA (8 views): +0.22 clean, +0.62 shifted (seed 0), about 8x inference time. The gain is small.
- **Occlusion**alone lowers accuracy by only 2 points (39.54% to 37.55%); the final model handles it well.

## Limitations

- Absolute accuracy is low because only 400 training images and a small from-scratch model are allowed.
- The shifted test set was built by us, so the "hidden corruption" is not truly hidden.
- The validation set has only 100 images, so differences under about 4 points can be noise.
- Error analysis and the hidden-corruption test use seed 0 only.

## Rules followed

No pretrained weights, at most 1M parameters, Inception module included, test data never used for tuning,
on-the-fly augmentation only, at most 60 epochs, 3 seeds.
