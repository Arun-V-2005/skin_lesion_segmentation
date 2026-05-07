# trainer.py

import torch
from tqdm import tqdm

from train_utils import SegmentationLoss, iou_score, dice_score, pixel_accuracy


_criterion = SegmentationLoss()


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _unpack_batch(batch: tuple) -> tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor | None]:
    """
    Handles both 2-item and 3-item batches.

    DataLoader may return (imgs, masks) or (imgs, masks, cached_attn).
    cached_attn is None when it's an empty tensor (cache miss).
    """
    imgs, masks = batch[0], batch[1]
    cached_attn = None
    if len(batch) == 3:
        ca = batch[2]
        # empty tensor → cache miss → let model run DINOv2 live
        if ca.ndim == 4:
            cached_attn = ca
    return imgs, masks, cached_attn


# ══════════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, scaler,
                    grad_clip: float = 1.0) -> dict[str, float]:
    model.train()

    total = dict(loss=0.0, iou=0.0, dice=0.0, acc=0.0)
    n = 0

    loop = tqdm(loader, desc="Train", leave=False)

    for batch in loop:
        imgs, masks, cached_attn = _unpack_batch(batch)
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if cached_attn is not None:
            cached_attn = cached_attn.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        # ── forward (AMP) ──
        with torch.cuda.amp.autocast():
            preds, _ = model(imgs, cached_attn=cached_attn)
            loss = _criterion(preds, masks)

        # ── backward (scaled) ──
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── metrics (no grad, stays on GPU) ──
        with torch.no_grad():
            iou  = iou_score(preds, masks)
            dice = dice_score(preds, masks)
            acc  = pixel_accuracy(preds, masks)

        bs = imgs.size(0)
        total["loss"] += loss.item() * bs
        total["iou"]  += iou.item()  * bs
        total["dice"] += dice.item() * bs
        total["acc"]  += acc.item()  * bs
        n += bs

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{iou.item():.4f}",
        )

    return {k: v / n for k, v in total.items()}


# ══════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()

    total = dict(loss=0.0, iou=0.0, dice=0.0, acc=0.0)
    n = 0

    loop = tqdm(loader, desc="Val", leave=False)

    for batch in loop:
        imgs, masks, cached_attn = _unpack_batch(batch)
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if cached_attn is not None:
            cached_attn = cached_attn.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            preds, _ = model(imgs, cached_attn=cached_attn)
            loss = _criterion(preds, masks)
            iou  = iou_score(preds, masks)
            dice = dice_score(preds, masks)
            acc  = pixel_accuracy(preds, masks)

        bs = imgs.size(0)
        total["loss"] += loss.item() * bs
        total["iou"]  += iou.item()  * bs
        total["dice"] += dice.item() * bs
        total["acc"]  += acc.item()  * bs
        n += bs

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{iou.item():.4f}",
        )

    return {k: v / n for k, v in total.items()}