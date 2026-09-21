"""
Usage:
    python train.py --dataset_root D:/ComputerVision/FinalProject/Dataset
                    --epochs 50 --batch_size 16 --lr 1e-4
"""

import os
import math
import argparse
import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

from model   import build_resnet50_segmentation_model
from dataset import build_dataset, CLASS_NAMES
from losses  import bce_dice_loss, DiceCoefficient, IoU


# ── Device setup (mirrors setup_gpu) ──────────────────────────────────────────

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] No GPU found – running on CPU.")
    return device


# ── Warmup + Cosine Decay LR schedule ─────────────────────────────────────────

class WarmupCosineDecay(torch.optim.lr_scheduler.LambdaLR):
    """Step-level schedule matching the TF WarmupCosineDecay exactly."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            cosine_step  = step - warmup_steps
            cosine_total = max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_total))

        super().__init__(optimizer, lr_lambda=lr_lambda)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",   type=str,   default="Dataset")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--seg_weight",     type=float, default=1.0)
    p.add_argument("--cls_weight",     type=float, default=0.5)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--unfreeze_epoch", type=int,   default=10)
    p.add_argument("--output_dir",     type=str,   default="outputs")
    p.add_argument("--num_workers",    type=int,   default=4)
    return p.parse_args()


# ── Freeze / unfreeze encoder ──────────────────────────────────────────────────

ENCODER_KEYS = ("enc0", "pool", "enc1", "enc2", "enc3", "enc4")

def set_encoder_trainable(model: nn.Module, trainable: bool):
    for name, param in model.named_parameters():
        if any(k in name for k in ENCODER_KEYS):
            param.requires_grad = trainable


# ── One epoch: train or evaluate ──────────────────────────────────────────────

def run_epoch(model, loader, device, optimizer=None, scheduler=None,
              seg_weight=1.0, cls_weight=0.5, epoch=0, total_epochs=0, phase="train"):
    training = optimizer is not None
    model.train() if training else model.eval()

    cls_loss_fn  = nn.CrossEntropyLoss()
    total_loss   = 0.0
    dice_metric  = DiceCoefficient()
    iou_metric   = IoU()
    correct_cls  = 0
    total_cls    = 0

    # Progress bar label: "Epoch 3/50 | train" or "val"
    desc = f"Epoch {epoch:03d}/{total_epochs} [{phase:>5}]"
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False,
                bar_format="{l_bar}{bar:25}{r_bar}")

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs     = imgs.to(device)
            seg_true = targets["seg_output"].to(device)
            cls_true = targets["cls_output"].to(device)

            seg_pred, cls_pred = model(imgs)

            cls_idx  = cls_true.argmax(dim=1)
            loss_seg = bce_dice_loss(seg_true, seg_pred)
            loss_cls = cls_loss_fn(cls_pred, cls_idx)
            loss     = seg_weight * loss_seg + cls_weight * loss_cls

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss  += loss.item()
            dice_metric.update(seg_true.detach(), seg_pred.detach())
            iou_metric.update(seg_true.detach(),  seg_pred.detach())
            correct_cls += (cls_pred.argmax(dim=1) == cls_idx).sum().item()
            total_cls   += cls_idx.size(0)

            # Live metrics shown inside the progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(
                loss  = f"{avg_loss:.4f}",
                dice  = f"{dice_metric.result():.4f}",
                iou   = f"{iou_metric.result():.4f}",
                acc   = f"{correct_cls / max(total_cls, 1):.4f}",
            )

    pbar.close()
    n = len(loader)
    return {
        "loss": total_loss / n,
        "dice": dice_metric.result(),
        "iou":  iou_metric.result(),
        "acc":  correct_cls / max(total_cls, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    train_loader, val_loader, test_loader = build_dataset(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet50_segmentation_model(
        input_shape=(256, 256, 3),
        num_classes=len(CLASS_NAMES),
    ).to(device)

    if args.freeze_encoder:
        set_encoder_trainable(model, trainable=False)
        print("[INFO] Encoder frozen.")

    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = min(5 * steps_per_epoch, total_steps // 10)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = WarmupCosineDecay(optimizer, total_steps, warmup_steps)

    ckpt_path  = os.path.join(args.output_dir, f"best_model_{timestamp}.pt")
    final_path = os.path.join(args.output_dir, f"final_model_{timestamp}.pt")

    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", timestamp))
    except ImportError:
        tb = None
        print("[INFO] TensorBoard not available; skipping TB logging.")

    best_dice        = -1.0
    best_val_loss    = float("inf")
    patience_counter = 0
    patience_limit   = 12        # EarlyStopping (val_loss)
    lr_patience      = 5         # ReduceLROnPlateau
    lr_counter       = 0
    lr_factor        = 0.5
    min_lr           = 1e-7

    # ── History tracking ───────────────────────────────────────────────────────
    history = {
        "epoch":     [],
        "loss":      [], "dice":     [], "iou":     [], "acc":     [],
        "val_loss":  [], "val_dice": [], "val_iou": [], "val_acc": [],
        "lr":        [],
    }

    for epoch in range(args.epochs):

        # ── Unfreeze encoder at the configured epoch ───────────────────────
        if args.freeze_encoder and epoch == args.unfreeze_epoch:
            print(f"\n[INFO] Epoch {epoch}: unfreezing all layers.")
            set_encoder_trainable(model, trainable=True)
            new_lr = args.lr * 0.1
            remaining_steps = (args.epochs - epoch) * steps_per_epoch
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
            scheduler = WarmupCosineDecay(optimizer, remaining_steps, warmup_steps=0)
            print(f"[INFO] lr reset to {new_lr:.2e}")

        train_m = run_epoch(model, train_loader, device, optimizer, scheduler,
                            args.seg_weight, args.cls_weight,
                            epoch=epoch+1, total_epochs=args.epochs, phase="train")
        val_m   = run_epoch(model, val_loader, device,
                            seg_weight=args.seg_weight, cls_weight=args.cls_weight,
                            epoch=epoch+1, total_epochs=args.epochs, phase="val")

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Clean per-epoch summary line (printed after both bars close) ───
        print(
            f"  Epoch {epoch+1:03d}/{args.epochs}"
            f"  loss {train_m['loss']:.4f}  dice {train_m['dice']:.4f}"
            f"  │  val_loss {val_m['loss']:.4f}  val_dice {val_m['dice']:.4f}"
            f"  val_iou {val_m['iou']:.4f}  val_acc {val_m['acc']:.4f}"
            f"  lr {current_lr:.2e}"
        )

        # Record history
        history["epoch"].append(epoch + 1)
        history["loss"].append(train_m["loss"])
        history["dice"].append(train_m["dice"])
        history["iou"].append(train_m["iou"])
        history["acc"].append(train_m["acc"])
        history["val_loss"].append(val_m["loss"])
        history["val_dice"].append(val_m["dice"])
        history["val_iou"].append(val_m["iou"])
        history["val_acc"].append(val_m["acc"])
        history["lr"].append(current_lr)

        if tb:
            for k, v in train_m.items():
                tb.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                tb.add_scalar(f"val/{k}", v, epoch)

        # ModelCheckpoint: monitor val_dice
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Best model saved (val_dice={best_dice:.4f})")

        # ReduceLROnPlateau: monitor val_loss
        if val_m["loss"] < best_val_loss:
            best_val_loss    = val_m["loss"]
            lr_counter       = 0
            patience_counter = 0
        else:
            lr_counter       += 1
            patience_counter += 1
            if lr_counter >= lr_patience:
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] * lr_factor, min_lr)
                print(f"  ReduceLROnPlateau → lr={optimizer.param_groups[0]['lr']:.2e}")
                lr_counter = 0

        # EarlyStopping: monitor val_loss
        if patience_counter >= patience_limit:
            print(f"[INFO] Early stopping at epoch {epoch+1}.")
            break

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\n[INFO] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m = run_epoch(model, test_loader, device,
                       seg_weight=args.seg_weight, cls_weight=args.cls_weight,
                       epoch=1, total_epochs=1, phase="test")
    print("[INFO] Test results:")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")

    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Best  model → {ckpt_path}")
    print(f"[INFO] Final model → {final_path}")

    if tb:
        tb.close()

    # ── Training history summary ───────────────────────────────────────────────
    print_history(history)

    # Save history to CSV
    history_path = os.path.join(args.output_dir, f"history_{timestamp}.csv")
    save_history_csv(history, history_path)
    print(f"[INFO] History saved → {history_path}")


# ── History helpers ────────────────────────────────────────────────────────────

def print_history(history: dict):
    epochs = history["epoch"]
    n      = len(epochs)

    col_w  = 10   # column width
    fields = ["loss", "dice", "iou", "acc", "val_loss", "val_dice", "val_iou", "val_acc", "lr"]
    header = f"{'Epoch':>6}  " + "  ".join(f"{f:>{col_w}}" for f in fields)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  TRAINING HISTORY")
    print("=" * len(header))
    print(header)
    print(sep)

    for i in range(n):
        row = f"{epochs[i]:>6}  "
        for f in fields:
            v = history[f][i]
            row += f"{v:>{col_w}.4f}  " if f != "lr" else f"{v:>{col_w}.2e}  "
        print(row)

    print(sep)

    # Best epoch summary
    best_dice_epoch = history["epoch"][history["val_dice"].index(max(history["val_dice"]))]
    best_loss_epoch = history["epoch"][history["val_loss"].index(min(history["val_loss"]))]
    print(f"\n  Best val_dice : {max(history['val_dice']):.4f}  @ epoch {best_dice_epoch}")
    print(f"  Best val_loss : {min(history['val_loss']):.4f}  @ epoch {best_loss_epoch}")
    print(f"  Best val_iou  : {max(history['val_iou']):.4f}")
    print(f"  Best val_acc  : {max(history['val_acc']):.4f}")
    print("=" * len(header) + "\n")


def save_history_csv(history: dict, path: str):
    import csv
    keys = list(history.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(history["epoch"])):
            writer.writerow([history[k][i] for k in keys])


if __name__ == "__main__":
    main()