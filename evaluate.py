"""
Usage:
    python evaluate.py --model_path outputs/best_model_XXX.pt
                       --dataset_root D:/ComputerVision/FinalProject/Dataset
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model   import build_resnet50_segmentation_model
from dataset import build_dataset, CLASS_NAMES
from losses  import bce_dice_loss, DiceCoefficient, IoU


# ── Helpers ───────────────────────────────────────────────────────────────────

def denormalise(img: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation for display. img shape: (C, H, W)"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img  = img * std + mean
    img  = np.transpose(img, (1, 2, 0))   # (C,H,W) → (H,W,C)
    return np.clip(img, 0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   type=str, required=True)
    p.add_argument("--dataset_root", type=str, default="Dataset")
    p.add_argument("--output_dir",   type=str, default="outputs/eval")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_vis",      type=int, default=8)
    p.add_argument("--threshold",    type=float, default=0.5)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_resnet50_segmentation_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"[INFO] Model loaded from {args.model_path}")

    # ── Load test set ─────────────────────────────────────────────────────────
    _, _, test_loader = build_dataset(
        args.dataset_root, batch_size=args.batch_size
    )

    # ── Collect predictions ───────────────────────────────────────────────────
    all_true, all_pred = [], []
    vis_imgs       = []
    vis_true_masks = []
    vis_pred_masks = []
    vis_true_lbl   = []
    vis_pred_lbl   = []

    dice_metric = DiceCoefficient()
    iou_metric  = IoU()

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs     = imgs.to(device)
            seg_true = targets["seg_output"].to(device)   # (B, 1, H, W)
            cls_true = targets["cls_output"]              # (B, 4) one-hot  — keep on CPU

            seg_pred, cls_pred = model(imgs)

            # Classification
            t_idx = cls_true.argmax(dim=1).numpy()
            p_idx = cls_pred.argmax(dim=1).cpu().numpy()
            all_true.extend(t_idx)
            all_pred.extend(p_idx)

            # Segmentation metrics
            dice_metric.update(seg_true, seg_pred)
            iou_metric.update(seg_true,  seg_pred)

            # Collect visualisation samples
            if len(vis_imgs) < args.num_vis:
                n = min(args.num_vis - len(vis_imgs), imgs.size(0))
                vis_imgs.extend(imgs[:n].cpu().numpy())
                vis_true_masks.extend(seg_true[:n].cpu().numpy())
                vis_pred_masks.extend(seg_pred[:n].cpu().numpy())
                vis_true_lbl.extend(t_idx[:n])
                vis_pred_lbl.extend(p_idx[:n])

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(all_true, all_pred, target_names=CLASS_NAMES, digits=4)
    print("\n── Classification Report ──")
    print(report)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] Report saved → {report_path}")

    # ── Segmentation metrics ──────────────────────────────────────────────────
    print(f"── Segmentation Metrics ──")
    print(f"  Dice : {dice_metric.result():.4f}")
    print(f"  IoU  : {iou_metric.result():.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    # ── Segmentation visualisation ────────────────────────────────────────────
    n   = len(vis_imgs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        # Original image
        axes[i][0].imshow(denormalise(vis_imgs[i]))
        axes[i][0].set_title(f"Image\nTrue: {CLASS_NAMES[vis_true_lbl[i]]}")
        axes[i][0].axis("off")

        # Ground truth mask — shape (1, H, W) → (H, W)
        axes[i][1].imshow(vis_true_masks[i][0], cmap="gray")
        axes[i][1].set_title("GT Mask")
        axes[i][1].axis("off")

        # Predicted mask (binarised)
        pred_bin = (vis_pred_masks[i][0] > args.threshold).astype(float)
        axes[i][2].imshow(pred_bin, cmap="gray")
        axes[i][2].set_title(f"Pred Mask\nPred: {CLASS_NAMES[vis_pred_lbl[i]]}")
        axes[i][2].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(args.output_dir, "segmentation_samples.png")
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"[INFO] Segmentation samples saved → {vis_path}")
    print(f"\n[INFO] All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()