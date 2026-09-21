"""
Dataset layout (unchanged):
  Dataset/
    Segmentation/
      Glioma/         enh_XXX.jpg + enh_XXX_mask.jpg
      Meningioma/     enh_XXX.jpg + enh_XXX_mask.jpg
      PituitaryTumor/ enh_XXX.jpg + enh_XXX_mask.jpg
    classification/
      Training/
        notumor/      enh_Tr-no_XXXX.jpg
"""

import os
import random
from typing import List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


CLASS_NAMES  = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_SIZE     = (256, 256)

# ImageNet normalisation (same values as TF version)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Sample collection (identical logic to TF version) ─────────────────────────
def collect_samples(dataset_root: str):
    image_paths, mask_paths, labels = [], [], []

    seg_root = os.path.join(dataset_root, "Segmentation")
    folder_to_class = {
        "Glioma":         "glioma",
        "Meningioma":     "meningioma",
        "PituitaryTumor": "pituitary",
    }

    for folder, cls_name in folder_to_class.items():
        folder_path = os.path.join(seg_root, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Segmentation folder not found: {folder_path}")
            continue
        all_files = set(os.listdir(folder_path))
        for fname in sorted(all_files):
            if "_mask" in fname:
                continue
            base, ext = os.path.splitext(fname)
            mask_fname = base + "_mask" + ext
            if mask_fname not in all_files:
                print(f"[WARNING] Mask not found for {fname}, skipping.")
                continue
            image_paths.append(os.path.join(folder_path, fname))
            mask_paths.append(os.path.join(folder_path, mask_fname))
            labels.append(CLASS_TO_IDX[cls_name])

    notumor_root = os.path.join(dataset_root, "classification", "Training", "notumor")
    if os.path.isdir(notumor_root):
        for fname in sorted(os.listdir(notumor_root)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_paths.append(os.path.join(notumor_root, fname))
            mask_paths.append("BLACK")       # sentinel → zero mask at load time
            labels.append(CLASS_TO_IDX["notumor"])
    else:
        print(f"[WARNING] notumor folder not found: {notumor_root}")

    print(f"[INFO] Total samples: {len(image_paths)}")
    for c, i in CLASS_TO_IDX.items():
        print(f"       {c}: {labels.count(i)}")

    return image_paths, mask_paths, labels


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths:  List[str],
        labels:      List[int],
        training:    bool = False,
        img_size:    Tuple[int, int] = IMG_SIZE,
    ):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.labels      = labels
        self.training    = training
        self.img_size    = img_size

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img = TF.to_tensor(img)          # (3, H, W) float32 in [0, 1]
        img = (img - MEAN) / STD
        return img

    def _load_mask(self, path: str) -> torch.Tensor:
        if path == "BLACK":
            return torch.zeros(1, *self.img_size, dtype=torch.float32)
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        mask = TF.to_tensor(mask)        # (1, H, W) float32 in [0, 1]
        return torch.clamp(mask, 0.0, 1.0)

    def _augment(self, img: torch.Tensor, mask: torch.Tensor):
        # Horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # Vertical flip
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        # Brightness / contrast on un-normalised image, then re-normalise
        img_raw = img * STD + MEAN
        img_raw = TF.adjust_brightness(img_raw, 1.0 + random.uniform(-0.1, 0.1))
        img_raw = TF.adjust_contrast(img_raw,   random.uniform(0.9, 1.1))
        img     = (img_raw - MEAN) / STD
        return img, mask

    def __getitem__(self, idx):
        img  = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])

        if self.training:
            img, mask = self._augment(img, mask)

        label_oh = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
        label_oh[self.labels[idx]] = 1.0

        return img, {"seg_output": mask, "cls_output": label_oh}


# ── build_dataset (same signature as TF version) ──────────────────────────────

def build_dataset(
    dataset_root: str,
    batch_size:   int   = 16,
    val_split:    float = 0.15,
    test_split:   float = 0.15,
    seed:         int   = 42,
    num_workers:  int   = 4,
):
    image_paths, mask_paths, labels = collect_samples(dataset_root)

    idx = list(range(len(image_paths)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_split, stratify=labels, random_state=seed
    )
    adjusted_val = val_split / (1.0 - test_split)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=adjusted_val,
        stratify=[labels[i] for i in idx_trainval],
        random_state=seed,
    )

    def _subset(indices):
        return (
            [image_paths[i] for i in indices],
            [mask_paths[i]  for i in indices],
            [labels[i]      for i in indices],
        )

    def _make_loader(ips, mps, lbs, training=False):
        ds = BrainTumorDataset(ips, mps, lbs, training=training)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    train_ips, train_mps, train_lbs = _subset(idx_train)
    val_ips,   val_mps,   val_lbs   = _subset(idx_val)
    test_ips,  test_mps,  test_lbs  = _subset(idx_test)

    train_loader = _make_loader(train_ips, train_mps, train_lbs, training=True)
    val_loader   = _make_loader(val_ips,   val_mps,   val_lbs)
    test_loader  = _make_loader(test_ips,  test_mps,  test_lbs)

    print(f"[INFO] train: {len(train_ips)}, val: {len(val_ips)}, test: {len(test_ips)}")
    return train_loader, val_loader, test_loader