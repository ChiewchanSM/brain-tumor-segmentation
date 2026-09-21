import torch
import torch.nn.functional as F


# ── Losses ────────────────────────────────────────────────────────────────────

def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        y_true_f.sum() + y_pred_f.sum() + smooth
    )


def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
    return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)


# ── Metrics ───────────────────────────────────────────────────────────────────

class DiceCoefficient:
    """Stateful Dice metric — mirrors the TF Metric class interface."""
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth    = smooth
        self.reset()

    def reset(self):
        self._sum   = 0.0
        self._count = 0

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_pred_bin   = (y_pred > self.threshold).float()
        y_true_f     = y_true.reshape(-1).float()
        y_pred_f     = y_pred_bin.reshape(-1)
        intersection = (y_true_f * y_pred_f).sum().item()
        dice = (2.0 * intersection + self.smooth) / (
            y_true_f.sum().item() + y_pred_f.sum().item() + self.smooth
        )
        self._sum   += dice
        self._count += 1

    def result(self) -> float:
        return self._sum / (self._count + 1e-8)


class IoU:
    """Stateful IoU metric."""
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth    = smooth
        self.reset()

    def reset(self):
        self._sum   = 0.0
        self._count = 0

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_pred_bin   = (y_pred > self.threshold).float()
        y_true_f     = y_true.reshape(-1).float()
        y_pred_f     = y_pred_bin.reshape(-1)
        intersection = (y_true_f * y_pred_f).sum().item()
        union        = y_true_f.sum().item() + y_pred_f.sum().item() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        self._sum   += iou
        self._count += 1

    def result(self) -> float:
        return self._sum / (self._count + 1e-8)