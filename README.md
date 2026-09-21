# 🧠 Brain Tumor Segmentation & Classification

A PyTorch deep learning project that jointly performs **tumor segmentation** and **multi-class classification** on MRI brain scans, using a ResNet50-based U-Net architecture.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Grad-CAM Visualisation](#grad-cam-visualisation)
- [File Reference](#file-reference)
- [Outputs](#outputs)

---

## Overview

This model simultaneously:

1. **Classifies** each MRI scan into one of four categories: `glioma`, `meningioma`, `pituitary`, or `notumor`
2. **Segments** the tumor region by producing a pixel-level binary mask

The dual-output design allows the shared encoder to learn features useful to both tasks at once. The entire pipeline — training, evaluation, inference, and Grad-CAM — runs on **PyTorch only**.

---

## Architecture

```
Input (256×256×3)
        │
   ResNet50 Encoder (pretrained on ImageNet)
        ├── s1: 128×128×64
        ├── s2:  64×64×256
        ├── s3:  32×32×512
        ├── s4:  16×16×1024
        └── bn:   8×8×2048
        │               │
  Classification     U-Net Decoder (skip connections)
  Head (GAP →           └── 256×256×1
  256 → 4 logits)
        │               │
  cls_output        seg_output (sigmoid)
```

- **Encoder**: ResNet50 with ImageNet weights
- **Decoder**: U-Net style with bilinear upsampling and skip connections
- **Loss**: Combined BCE + Dice loss (segmentation) + CrossEntropy (classification)
- **Schedule**: Warmup + Cosine Decay learning rate
- **Regularisation**: Dropout (0.4), optional encoder freezing

---

## Dataset Structure

```
Dataset/
├── Segmentation/
│   ├── Glioma/
│   │   ├── enh_001.jpg
│   │   ├── enh_001_mask.jpg
│   │   └── ...
│   ├── Meningioma/
│   │   ├── enh_001.jpg
│   │   ├── enh_001_mask.jpg
│   │   └── ...
│   └── PituitaryTumor/
│       ├── enh_001.jpg
│       ├── enh_001_mask.jpg
│       └── ...
└── classification/
    └── Training/
        └── notumor/
            ├── enh_Tr-no_0001.jpg
            └── ...
```

> **Note:** For tumor classes, each image must have a corresponding `_mask` file in the same folder. The `notumor` class uses all-black masks generated automatically in memory — no mask files needed on disk.

### Generating notumor mask files (optional)

If you want mask files for `notumor` images written to disk:

```bash
python generate_notumor_masks.py \
    --notumor_dir Dataset/classification/Training/notumor \
    --output_dir  Dataset/classification/Training/notumor
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch ≥ 2.0 (with CUDA recommended)

### Setup

```bash
# 1. Clone or copy this repository
git clone <your-repo-url>
cd brain-tumor-project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install PyTorch (visit https://pytorch.org for the right CUDA version)
pip install torch torchvision

# 4. Install remaining dependencies
pip install scikit-learn Pillow matplotlib seaborn tqdm
```

---

## Quick Start

### 1 — Prepare your dataset

Place your data following the [Dataset Structure](#dataset-structure) above.

### 2 — Train the model

```bash
python train.py \
    --dataset_root path/to/Dataset \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4
```

### 3 — Run inference on an image

```bash
python predict.py \
    --model outputs/best_model_YYYYMMDD_HHMMSS.pt \
    --image path/to/mri.jpg \
    --save_dir results/
```

---

## Training

```bash
python train.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--dataset_root` | `Dataset` | Path to root dataset folder |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--seg_weight` | `1.0` | Loss weight for segmentation branch |
| `--cls_weight` | `0.5` | Loss weight for classification branch |
| `--freeze_encoder` | `False` | Freeze ResNet50 encoder for initial epochs |
| `--unfreeze_epoch` | `10` | Epoch at which to unfreeze the encoder |
| `--output_dir` | `outputs` | Directory to save checkpoints and logs |
| `--num_workers` | `4` | DataLoader worker processes |

### Training Features

- **Warmup + Cosine Decay** learning rate schedule
- **ReduceLROnPlateau** — halves LR after 5 epochs without `val_loss` improvement
- **Early Stopping** — halts after 12 epochs without `val_loss` improvement
- **Model Checkpoint** — saves best model by `val_dice`
- **TensorBoard** logging (if `tensorboard` is installed)
- **History CSV** saved to `outputs/history_<timestamp>.csv`

### Example with encoder freezing

```bash
python train.py \
    --dataset_root Dataset \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --freeze_encoder \
    --unfreeze_epoch 10
```

The encoder is frozen for the first 10 epochs (only the decoder and heads train), then unfrozen at epoch 10 with the learning rate reduced to `lr × 0.1`.

---

## Evaluation

### Quick test evaluation

```bash
python test_eval.py
```

> Edit `test_eval.py` to point `--model` and `--dataset_root` at your paths before running.

### Full evaluation with confusion matrix and mask visualisations

```bash
python evaluate.py \
    --model_path   outputs/best_model_XXXX.pt \
    --dataset_root Dataset \
    --output_dir   outputs/eval \
    --batch_size   8 \
    --num_vis      8 \
    --threshold    0.5
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Path to `.pt` checkpoint |
| `--dataset_root` | `Dataset` | Path to root dataset folder |
| `--output_dir` | `outputs/eval` | Where to save outputs |
| `--batch_size` | `8` | Batch size for the test loader |
| `--num_vis` | `8` | Number of sample rows in the visualisation figure |
| `--threshold` | `0.5` | Binarisation threshold for predicted masks |

This saves three files to `--output_dir`:

- `classification_report.txt` — per-class precision, recall, F1, and support
- `confusion_matrix.png` — heatmap of predictions vs ground truth
- `segmentation_samples.png` — grid of original image / GT mask / predicted mask

Segmentation metrics (Dice and IoU) are also printed to the terminal.

---

## Prediction

Run inference on a **single image** or an entire **folder**:

```bash
# Single image — print results to terminal
python predict.py \
    --model   outputs/best_model_XXXX.pt \
    --image   path/to/mri.jpg

# Whole folder — save overlay images
python predict.py \
    --model    outputs/best_model_XXXX.pt \
    --folder   path/to/mri_folder/ \
    --save_dir results/

# Adjust mask binarisation threshold (default 0.5)
python predict.py \
    --model     outputs/best_model_XXXX.pt \
    --image     mri.jpg \
    --threshold 0.4
```

Each saved result is a three-panel image:

```
[ Original ] [ Binary Mask ] [ Colour Overlay + Class Label ]
```

### Terminal output example

```
=======================================================
  Image : Te-gl_0010.jpg
  Class : GLIOMA  (94.3% confidence)
  ─────────────────────────────────────────────────────
  Class probabilities:
    glioma          94.3%  ██████████████████████████████
    meningioma       4.1%  █
    pituitary        1.2%
    notumor          0.4%
  Tumor region detected: YES
=======================================================
```

---

## Grad-CAM Visualisation

Produce Grad-CAM saliency maps to understand which spatial regions drove each classification decision.

```bash
python gradcam.py \
    --model_path   outputs/best_model_XXXX.pt \
    --dataset_root Dataset \
    --output_dir   outputs/gradcam \
    --num_vis      8 \
    --batch_size   8
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Path to `.pt` checkpoint |
| `--dataset_root` | `Dataset` | Path to root dataset folder |
| `--output_dir` | `outputs/gradcam` | Where to save the figure |
| `--num_vis` | `8` | Number of sample rows to visualise |
| `--batch_size` | `8` | Batch size for the test loader |

Output: `outputs/gradcam/gradcam_results.png`

Each row shows:

```
[ Original Image ] [ GT Mask ] [ Predicted Mask ] [ Grad-CAM Overlay ]
```

### How it works

The implementation uses **PyTorch forward and backward hooks** registered on `model.enc4[-1]` (the final block of ResNet50's `layer4`, equivalent to `conv5_block3` in Keras). During the forward pass the hook captures the activation maps; a backward pass on the predicted class score captures the corresponding gradients. The gradients are globally average-pooled into per-channel importance weights, multiplied with the activations, summed, and passed through ReLU to produce the raw heatmap. It is then bilinearly upsampled to 256×256 and blended over the original image. Hooks are removed after use via `gradcam.remove()` to avoid memory leaks.

---

## File Reference

| File | Purpose |
|---|---|
| `model.py` | ResNet50 U-Net dual-output model definition |
| `dataset.py` | Dataset loader, augmentation, train/val/test split |
| `losses.py` | BCE+Dice loss, Dice coefficient metric, IoU metric |
| `train.py` | Full training loop with scheduling and early stopping |
| `evaluate.py` | Confusion matrix, classification report, mask visualisation |
| `predict.py` | Single-image or batch inference with overlay export |
| `test_eval.py` | Minimal script to re-run test metrics on a saved checkpoint |
| `gradcam.py` | Grad-CAM via PyTorch hooks on `enc4[-1]` |
| `generate_notumor_masks.py` | Utility to write black mask files for notumor images |
| `requirements.txt` | Dependency list |

---

## Outputs

After training and evaluation, the `outputs/` directory contains:

```
outputs/
├── best_model_<timestamp>.pt       ← Best checkpoint (by val_dice)
├── final_model_<timestamp>.pt      ← Model state at end of training
├── history_<timestamp>.csv         ← Per-epoch metrics table
├── logs/
│   └── <timestamp>/                ← TensorBoard event files
├── eval/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── segmentation_samples.png
└── gradcam/
    └── gradcam_results.png
```

---

## Notes

- All images are resized to **256×256** during preprocessing.
- ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) is applied consistently across all scripts.
- Data augmentation (horizontal flip, vertical flip, brightness/contrast jitter) is applied **only during training**.
- The dataset is split **70% train / 15% val / 15% test** using stratified sampling.
- Classification uses **raw logits + CrossEntropyLoss**; softmax is only applied at inference time in `predict.py`.
- Segmentation uses **sigmoid output + BCE-Dice loss**.
