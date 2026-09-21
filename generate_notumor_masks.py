"""
Creates all-black mask files for notumor images (optional – only if you want them on disk).
Usage:
    python generate_notumor_masks.py --notumor_dir D:/ComputerVision/FinalProject/Dataset/classification/Training/notumor
"""

import os, argparse
from PIL import Image


def generate_black_masks(notumor_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(notumor_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    created = 0
    for fname in files:
        base, ext = os.path.splitext(fname)
        mask_path = os.path.join(output_dir, base + "_mask" + ext)
        if os.path.exists(mask_path):
            continue
        with Image.open(os.path.join(notumor_dir, fname)) as img:
            w, h = img.size
        Image.new("L", (w, h), 0).save(mask_path)
        created += 1
    print(f"[INFO] Generated {created} black masks in {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--notumor_dir", type=str,
                   default=os.path.join("Dataset","classification","Training","notumor"))
    p.add_argument("--output_dir",  type=str, default=None)
    args = p.parse_args()
    generate_black_masks(args.notumor_dir, args.output_dir or args.notumor_dir)
