"""
Grad-CAM for ResNet50_SegCls (dual-output: segmentation + classification).

Target layer: enc4 (the last ResNet conv block, 8x8 x 2048 bottleneck)
  - Gradients of the predicted class score w.r.t. this layer's activations
    tell us which spatial regions drove the classification decision.

Usage:
    python gradcam.py --model_path outputs/best_model_XXX.pt
                      --dataset_root D:/ComputerVision/FinalProject/Dataset
                      --num_vis 8
                      --output_dir outputs/gradcam
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model   import build_resnet50_segmentation_model
from dataset import build_dataset, CLASS_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hooks into the target layer to capture activations and gradients,
    then computes the Grad-CAM heatmap.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        # Forward hook — saves the layer output
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        # Backward hook — saves gradients flowing back into the layer
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()          # (1, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()    # (1, C, H, W)

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(self, img_tensor: torch.Tensor, class_index: int = None):
        """
        Args:
            img_tensor:  (1, 3, H, W) normalised, on same device as model.
            class_index: Which class to explain. None → use predicted class.

        Returns:
            heatmap:        float32 np array (h, w) in [0, 1]  — feature-map resolution
            pred_class_idx: int
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass — hooks capture activations automatically
        seg_pred, cls_pred = self.model(img_tensor)

        if class_index is None:
            class_index = int(cls_pred.argmax(dim=1).item())

        # Scalar score for the target class
        score = cls_pred[0, class_index]

        # Backward pass — hooks capture gradients automatically
        score.backward()

        # Global-average-pool gradients → importance weight per channel
        # gradients: (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()                             # (h, w)

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_index


def overlay_gradcam(img_array: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, colormap=plt.cm.jet) -> np.ndarray:
    """
    Resize heatmap to image size and blend as a colour overlay.

    Args:
        img_array: Denormalised float32 (H, W, 3) in [0, 1].
        heatmap:   Float32 (h, w) in [0, 1].

    Returns:
        overlay: uint8 (H, W, 3)
    """
    H, W = img_array.shape[:2]

    # Resize heatmap to image resolution using torch interpolation
    heatmap_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)
    heatmap_t = F.interpolate(heatmap_t, size=(H, W), mode="bilinear", align_corners=False)
    heatmap_resized = heatmap_t.squeeze().numpy()                  # (H, W)

    coloured = colormap(heatmap_resized)[:, :, :3]                 # (H, W, 3) — drop alpha
    overlay  = np.clip(img_array * (1 - alpha) + coloured * alpha, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def denormalise(img: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation. img shape: (C, H, W) → returns (H, W, C)"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img  = img * std + mean
    img  = np.transpose(img, (1, 2, 0))    # (C,H,W) → (H,W,C)
    return np.clip(img, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_gradcam(model, test_loader, device, num_vis=8,
                      output_dir="outputs/gradcam"):
    """
    Pull samples from test_loader, run Grad-CAM, save a figure.

    Layout per row:
      [Original image] [GT mask] [Pred mask] [Grad-CAM overlay]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Hook into enc4 — the last ResNet block (8×8 × 2048), same as conv5_block3_out in TF
    gradcam = GradCAM(model, target_layer=model.enc4[-1])

    imgs_list, true_masks, pred_masks_list = [], [], []
    true_lbls, pred_lbls, heatmaps        = [], [], []

    for imgs, targets in test_loader:
        imgs     = imgs.to(device)
        seg_true = targets["seg_output"]   # (B, 1, H, W) — keep on CPU for display
        cls_true = targets["cls_output"]   # (B, 4)

        for i in range(imgs.size(0)):
            if len(imgs_list) >= num_vis:
                break

            img_single = imgs[i].unsqueeze(0)          # (1, 3, H, W)
            true_idx   = int(cls_true[i].argmax().item())

            # Run Grad-CAM (uses predicted class automatically)
            heatmap, pred_idx = gradcam.compute(img_single)

            # Collect for plotting
            imgs_list.append(imgs[i].cpu().numpy())            # (3, H, W)
            true_masks.append(seg_true[i].numpy())             # (1, H, W)

            with torch.no_grad():
                seg_pred, _ = model(img_single)
            pred_masks_list.append(seg_pred[0].cpu().numpy())  # (1, H, W)

            true_lbls.append(true_idx)
            pred_lbls.append(pred_idx)
            heatmaps.append(heatmap)

        if len(imgs_list) >= num_vis:
            break

    gradcam.remove()   # clean up hooks

    # ── Plot ──────────────────────────────────────────────────────────────────
    n    = len(imgs_list)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Original Image", "GT Mask", "Pred Mask", "Grad-CAM"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        img_denorm  = denormalise(imgs_list[i])          # (H, W, 3)
        true_name   = CLASS_NAMES[true_lbls[i]]
        pred_name   = CLASS_NAMES[pred_lbls[i]]
        correct     = "✓" if true_lbls[i] == pred_lbls[i] else "✗"
        cam_overlay = overlay_gradcam(img_denorm, heatmaps[i])

        # Col 0 – original image
        axes[i][0].imshow(img_denorm)
        axes[i][0].set_ylabel(f"True: {true_name}", fontsize=10)
        axes[i][0].set_xticks([]); axes[i][0].set_yticks([])

        # Col 1 – ground-truth mask  (1, H, W) → (H, W)
        axes[i][1].imshow(true_masks[i][0], cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_xticks([]); axes[i][1].set_yticks([])

        # Col 2 – predicted mask (threshold 0.5)
        pred_bin = (pred_masks_list[i][0] > 0.5).astype(float)
        axes[i][2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_xlabel(f"Pred: {pred_name} {correct}", fontsize=10)
        axes[i][2].set_xticks([]); axes[i][2].set_yticks([])

        # Col 3 – Grad-CAM overlay
        axes[i][3].imshow(cam_overlay)
        axes[i][3].set_xlabel(f"Explains: {pred_name}", fontsize=10)
        axes[i][3].set_xticks([]); axes[i][3].set_yticks([])

    plt.suptitle("Grad-CAM Explanations — Brain Tumour Model", fontsize=15, y=1.002)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "gradcam_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Grad-CAM figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   type=str, required=True)
    p.add_argument("--dataset_root", type=str, default="Dataset")
    p.add_argument("--output_dir",   type=str, default="outputs/gradcam")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_vis",      type=int, default=8)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model = build_resnet50_segmentation_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"[INFO] Model loaded from {args.model_path}")

    # Load test set
    _, _, test_loader = build_dataset(args.dataset_root, batch_size=args.batch_size)

    visualise_gradcam(
        model, test_loader, device,
        num_vis=args.num_vis,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()