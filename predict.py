"""
Predict tumor class + segmentation mask for one or more images.

Usage:
    # Single image
    python predict.py --model outputs/best_model_XXXX.pt --image path/to/img.jpg

    # Whole folder
    python predict.py --model outputs/best_model_XXXX.pt --folder path/to/folder

    # Save output images (original + mask overlay)
    python predict.py --model outputs/best_model_XXXX.pt --image img.jpg --save_dir results/
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from model  import build_resnet50_segmentation_model
from dataset import CLASS_NAMES, IMG_SIZE

# ImageNet normalisation (same as training)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Color for each class (RGB) used in overlay
CLASS_COLORS = {
    "glioma":      (255,  60,  60),   # red
    "meningioma":  (255, 165,   0),   # orange
    "pituitary":   ( 60, 180,  60),   # green
    "notumor":     ( 80, 140, 255),   # blue
}


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    model = build_resnet50_segmentation_model(
        input_shape=(256, 256, 3),
        num_classes=len(CLASS_NAMES),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    print(f"[INFO] Model loaded from {ckpt_path}")
    return model


# ── Preprocess one image ───────────────────────────────────────────────────────

def preprocess(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    t   = TF.to_tensor(img)          # (3, H, W)
    t   = (t - MEAN) / STD
    return t.unsqueeze(0)            # (1, 3, H, W)


# ── Run prediction ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_path: str, device: torch.device, mask_threshold: float = 0.5):
    tensor = preprocess(image_path).to(device)

    seg_pred, cls_pred = model(tensor)

    # Classification
    probs      = F.softmax(cls_pred, dim=1).squeeze(0).cpu().numpy()  # (4,)
    class_idx  = int(probs.argmax())
    class_name = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])

    # Segmentation mask
    mask = seg_pred.squeeze().cpu().numpy()          # (H, W) float in [0, 1]
    mask_bin = (mask > mask_threshold).astype(np.uint8) * 255

    return {
        "class":      class_name,
        "class_idx":  class_idx,
        "confidence": confidence,
        "probs":      {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "mask":       mask,
        "mask_bin":   mask_bin,
    }


# ── Visualise: side-by-side original | mask | overlay ─────────────────────────

def make_result_image(image_path: str, result: dict) -> Image.Image:
    orig = Image.open(image_path).convert("RGB").resize(
        (IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR
    )

    # Mask as grayscale
    mask_img = Image.fromarray(result["mask_bin"], mode="L").convert("RGB")

    # Coloured overlay
    color    = CLASS_COLORS[result["class"]]
    overlay  = orig.copy().convert("RGBA")
    mask_rgba = Image.fromarray(result["mask_bin"]).convert("L")

    color_layer = Image.new("RGBA", orig.size, color + (140,))
    overlay.paste(color_layer, mask=mask_rgba)
    overlay = overlay.convert("RGB")

    # Combine side by side
    W, H  = orig.size
    panel = Image.new("RGB", (W * 3, H + 40), (30, 30, 30))
    panel.paste(orig,     (0,       20))
    panel.paste(mask_img, (W,       20))
    panel.paste(overlay,  (W * 2,   20))

    # Labels
    draw = ImageDraw.Draw(panel)
    cls  = result["class"].upper()
    conf = result["confidence"] * 100
    draw.text((5,   2), "Original",              fill=(200, 200, 200))
    draw.text((W+5, 2), "Mask",                  fill=(200, 200, 200))
    draw.text((W*2+5, 2), f"{cls}  ({conf:.1f}%)", fill=CLASS_COLORS[result["class"]])

    return panel


# ── Print result to terminal ───────────────────────────────────────────────────

def print_result(image_path: str, result: dict):
    print(f"\n{'='*55}")
    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Class : {result['class'].upper()}  ({result['confidence']*100:.1f}% confidence)")
    print(f"  {'─'*45}")
    print(f"  Class probabilities:")
    for cls, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
        bar   = "█" * int(prob * 30)
        print(f"    {cls:<15} {prob*100:5.1f}%  {bar}")
    has_tumor = result["mask_bin"].any()
    print(f"  Tumor region detected: {'YES' if has_tumor else 'NO'}")
    print(f"{'='*55}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str, required=True,  help="Path to .pt checkpoint")
    p.add_argument("--image",      type=str, default=None,   help="Single image path")
    p.add_argument("--folder",     type=str, default=None,   help="Folder of images")
    p.add_argument("--save_dir",   type=str, default=None,   help="Where to save result images")
    p.add_argument("--threshold",  type=float, default=0.5,  help="Mask binarisation threshold")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, device)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.folder:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for f in sorted(os.listdir(args.folder)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(args.folder, f))

    if not image_paths:
        print("[ERROR] Provide --image or --folder")
        return

    for img_path in image_paths:
        result = predict(model, img_path, device, mask_threshold=args.threshold)
        print_result(img_path, result)

        if args.save_dir:
            panel     = make_result_image(img_path, result)
            out_name  = os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg"
            out_path  = os.path.join(args.save_dir, out_name)
            panel.save(out_path)
            print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()