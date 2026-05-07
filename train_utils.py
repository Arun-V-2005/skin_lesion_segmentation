# train_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import CONFIG


# ══════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss.  Uses sigmoid so gradients flow through the
    prediction without a hard threshold.
    """
    pred   = torch.sigmoid(pred)
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice  = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss(pred: torch.Tensor,
               target: torch.Tensor,
               gamma: float | None = None,
               alpha: float | None = None) -> torch.Tensor:
    """
    Focal loss (Lin et al. 2017) – down-weights easy negatives to
    focus learning on hard, boundary-adjacent pixels.
    """
    cfg   = CONFIG["loss"]
    gamma = gamma if gamma is not None else cfg["focal_gamma"]
    alpha = alpha if alpha is not None else cfg["focal_alpha"]

    bce_raw = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t     = torch.exp(-bce_raw)                          # probability of correct class
    focal   = alpha * (1.0 - p_t) ** gamma * bce_raw
    return focal.mean()


class SegmentationLoss(nn.Module):
    """
    Combined loss = w_bce·BCE + w_dice·Dice + w_focal·Focal

    Weights come from CONFIG["loss"].  A single forward call avoids
    recomputing sigmoid / BCE multiple times.
    """

    def __init__(self):
        super().__init__()
        cfg = CONFIG["loss"]
        self.w_bce   = cfg.get("bce_weight",   1.0)
        self.w_dice  = cfg.get("dice_weight",  1.0)
        self.w_focal = cfg.get("focal_weight", 0.5)
        self.bce     = nn.BCEWithLogitsLoss()

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = (self.w_bce   * self.bce(pred, target)
              + self.w_dice  * dice_loss(pred, target)
              + self.w_focal * focal_loss(pred, target))
        return loss


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

def iou_score(pred:   torch.Tensor,
              target: torch.Tensor,
              thresh: float = 0.5,
              eps:    float = 1e-6) -> torch.Tensor:
    """
    Intersection-over-Union.  Returns a scalar tensor to avoid
    GPU→CPU synchronisation during the training loop.
    """
    pred   = (torch.sigmoid(pred) > thresh).to(target.dtype)
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


def dice_score(pred:   torch.Tensor,
               target: torch.Tensor,
               thresh: float = 0.5,
               eps:    float = 1e-6) -> torch.Tensor:
    """Hard Dice coefficient (a.k.a. F1 for binary segmentation)."""
    pred   = (torch.sigmoid(pred) > thresh).to(target.dtype)
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2.0 * inter + eps) / (union + eps)).mean()


def pixel_accuracy(pred:   torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """Fraction of correctly classified pixels."""
    pred = (torch.sigmoid(pred) > 0.5).to(target.dtype)
    return (pred == target).float().mean()