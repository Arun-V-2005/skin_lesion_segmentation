# test.py

"""
Evaluation script for the DINO-guided skin-lesion segmentation model.

Outputs
-------
- Console : per-sample and aggregate metrics (Dice, IoU, Precision, Recall, F1, Accuracy)
- Saved   : metrics.json, metrics_summary.txt
- Figures : research_viz_<N>.png  (un-normalised image | GT mask | pred mask | overlay)

Leak-free design
----------------
get_splits() performs a two-stage GroupShuffleSplit identical to the one
used in get_loaders(). The test_df returned here is guaranteed to have
ZERO overlap with the train/val splits used during training.

The test HAMDataset is constructed with cache_dir=None so it never reads
from the DINO cache (which was built on train images only). The model runs
live DINOv2 inference on test images — clean, no shortcuts.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from configs  import CONFIG
from model    import SegmentationModel
from dataset  import HAMDataset, get_splits


# ══════════════════════════════════════════════════════════════════
#  IMAGENET UN-NORMALISATION  (applied only for visualisation)
# ══════════════════════════════════════════════════════════════════

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalise(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and return an HWC uint8 numpy array.

    Note: the dataset performs /255 rescaling but NOT mean/std
    normalisation, so this function simply clamps and converts.
    If you later add torchvision Normalize() to the pipeline,
    un-comment the two lines below.
    """
    img = img_tensor.cpu().float().clone()
    # img = img * _STD + _MEAN          # ← un-comment if Normalize() added
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
#  PER-BATCH METRICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(pred_logits: torch.Tensor,
                    target:       torch.Tensor,
                    thresh:       float = 0.5,
                    eps:          float = 1e-6) -> dict[str, float]:
    """
    Compute Dice, IoU, Precision, Recall, F1, and Pixel Accuracy.
    All computations stay on the GPU; only `.item()` syncs.
    """
    pred   = (torch.sigmoid(pred_logits) > thresh).to(target.dtype)
    pred_f = pred.view(pred.size(0), -1)
    tgt_f  = target.view(target.size(0), -1)

    tp = (pred_f * tgt_f).sum(dim=1)
    fp = (pred_f * (1 - tgt_f)).sum(dim=1)
    fn = ((1 - pred_f) * tgt_f).sum(dim=1)
    tn = ((1 - pred_f) * (1 - tgt_f)).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    acc       = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "dice":      dice.mean().item(),
        "iou":       iou.mean().item(),
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
        "f1":        f1.mean().item(),
        "accuracy":  acc.mean().item(),
    }


# ══════════════════════════════════════════════════════════════════
#  RESEARCH-STYLE VISUALISATION
# ══════════════════════════════════════════════════════════════════

def save_research_viz(img:      torch.Tensor,    # (3, H, W)  float32
                      gt_mask:  torch.Tensor,    # (1, H, W)  float32
                      pred_logit: torch.Tensor,  # (1, H, W)  float32
                      sample_id: int,
                      save_dir: str,
                      thresh: float = 0.5) -> None:
    """
    Four-panel research figure:
      [0] Un-normalised image
      [1] Ground-truth mask
      [2] Predicted mask
      [3] Overlay  (image + GT contour in green + pred fill in red)
          annotated with per-image Dice and IoU
    """
    # ── tensors → numpy ──────────────────────────────────────────
    img_np   = unnormalise(img)                        # (H, W, 3) uint8
    gt_np    = gt_mask.squeeze().cpu().numpy()         # (H, W)    float32
    pred_np  = (torch.sigmoid(pred_logit).squeeze().cpu().numpy() > thresh).astype(np.float32)

    # ── per-image metrics ────────────────────────────────────────
    eps = 1e-6
    tp  = (pred_np * gt_np).sum()
    fp  = (pred_np * (1 - gt_np)).sum()
    fn  = ((1 - pred_np) * gt_np).sum()
    dice_val = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou_val  = (tp + eps) / (tp + fp + fn + eps)

    # ── overlay ──────────────────────────────────────────────────
    overlay = img_np.copy().astype(np.float32) / 255.0

    # red fill for predicted region
    pred_bool = pred_np.astype(bool)
    overlay[pred_bool, 0] = np.clip(overlay[pred_bool, 0] + 0.45, 0, 1)
    overlay[pred_bool, 1] = np.clip(overlay[pred_bool, 1] - 0.15, 0, 1)
    overlay[pred_bool, 2] = np.clip(overlay[pred_bool, 2] - 0.15, 0, 1)

    # green fill for GT region (semi-transparent)
    gt_bool = gt_np.astype(bool)
    overlay[gt_bool, 0] = np.clip(overlay[gt_bool, 0] - 0.15, 0, 1)
    overlay[gt_bool, 1] = np.clip(overlay[gt_bool, 1] + 0.35, 0, 1)
    overlay[gt_bool, 2] = np.clip(overlay[gt_bool, 2] - 0.15, 0, 1)

    # ── figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Sample #{sample_id}   |   Dice: {dice_val:.4f}   IoU: {iou_val:.4f}",
                 fontsize=13, fontweight="bold")

    titles = ["Input Image (un-normalised)", "Ground Truth Mask",
              "Predicted Mask",              "Overlay (GT=green | Pred=red)"]
    images = [img_np, gt_np, pred_np, overlay]
    cmaps  = [None, "gray", "gray", None]

    for ax, title, im, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1 if im.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # legend for overlay
    green_patch = mpatches.Patch(color=(0, 0.7, 0), alpha=0.6, label="Ground Truth")
    red_patch   = mpatches.Patch(color=(0.9, 0, 0), alpha=0.6, label="Prediction")
    axes[3].legend(handles=[green_patch, red_patch],
                   loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"research_viz_{sample_id:04d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  TEST LOADER  (held-out split, no augmentation, no cache)
# ══════════════════════════════════════════════════════════════════

def get_test_loader(img_dir:      str | Path,
                    mask_dir:     str | Path,
                    metadata_csv: str | Path,
                    batch_size:   int = 8) -> DataLoader:
    """
    Returns a DataLoader for the held-out test split.

    Key design decisions
    --------------------
    1. Uses get_splits() — the same two-stage GroupShuffleSplit used in
       get_loaders() — so train/val/test boundaries are always identical.

    2. cache_dir=None — test images are deliberately NOT loaded from the
       DINO cache. The cache was built on train images only. Passing
       cache_dir=None forces the model to run live DINOv2 inference on
       test images, giving a fully clean evaluation.

    3. augment=False — no flips or colour jitter at test time.
    """
    cfg = CONFIG

    _, _, test_df = get_splits(
        metadata_csv = metadata_csv,
        mask_dir     = mask_dir,
        val_split    = cfg["data"]["val_split"],
        test_split   = cfg["data"].get("test_split", 0.2),
        seed         = cfg["data"]["seed"],
    )

    # cache_dir=None → no cache lookup → live DINOv2 inference on test images
    test_ds = HAMDataset(
        test_df,
        img_dir,
        mask_dir,
        cache_dir = None,   # ← intentional: keeps test set leak-free
        augment   = False,
    )

    loader = DataLoader(
        test_ds,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = cfg["data"]["num_workers"],
        pin_memory        = cfg["data"]["pin_memory"],
        persistent_workers= cfg["data"]["num_workers"] > 0,
    )
    print(f"🧪  Test set : {len(test_ds):,} samples  (cache disabled — live DINO inference)")
    return loader


# ══════════════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════

def main(
    checkpoint: str | None = None,
    num_viz:    int         = 8,
    thresh:     float       = 0.5,
) -> None:
    """
    Parameters
    ----------
    checkpoint : path to .pth checkpoint (defaults to save_dir/best_model.pth)
    num_viz    : number of research-style visualisations to save
    thresh     : binarisation threshold for predicted masks
    """
    cfg      = CONFIG
    device   = cfg["device"]
    save_dir = cfg["paths"]["save_dir"]
    viz_dir  = os.path.join(save_dir, "test_viz")
    os.makedirs(viz_dir, exist_ok=True)

    if checkpoint is None:
        checkpoint = os.path.join(save_dir, "best_model.pth")

    # ── model ──
    model = SegmentationModel(config=cfg, device=device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"✅  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val dice = {ckpt.get('val_dice', float('nan')):.4f})")

    # ── data — cache disabled for clean test evaluation ──
    test_loader = get_test_loader(
        img_dir      = cfg["paths"]["img_dir"],
        mask_dir     = cfg["paths"]["mask_dir"],
        metadata_csv = cfg["paths"]["metadata"],
        batch_size   = cfg["data"]["batch_size"],
    )

    # ── evaluation ──
    aggregated: dict[str, float] = {k: 0.0 for k in
                                    ["dice", "iou", "precision", "recall", "f1", "accuracy"]}
    n_samples  = 0
    viz_count  = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            imgs  = batch[0].to(device, non_blocking=True)
            masks = batch[1].to(device, non_blocking=True)
            # batch[2] will be empty tensors (cache_dir=None) — model runs live DINO
            cached_attn = batch[2].to(device) if batch[2].ndim == 4 else None

            with torch.cuda.amp.autocast():
                preds, _ = model(imgs, cached_attn=cached_attn)

            metrics = compute_metrics(preds, masks, thresh=thresh)
            bs = imgs.size(0)
            for k, v in metrics.items():
                aggregated[k] += v * bs
            n_samples += bs

            # ── research visualisations ──
            if viz_count < num_viz:
                for i in range(min(bs, num_viz - viz_count)):
                    save_research_viz(
                        img         = imgs[i].cpu(),
                        gt_mask     = masks[i].cpu(),
                        pred_logit  = preds[i].cpu(),
                        sample_id   = viz_count,
                        save_dir    = viz_dir,
                        thresh      = thresh,
                    )
                    viz_count += 1

    # ── aggregate ──
    final: dict[str, float] = {k: v / n_samples for k, v in aggregated.items()}

    # ── print ──
    print(f"\n{'='*55}")
    print(f"  TEST RESULTS  ({n_samples:,} samples)")
    print(f"{'='*55}")
    print(f"  {'Metric':<15} {'Score':>8}")
    print(f"  {'-'*25}")
    for metric, value in final.items():
        print(f"  {metric.capitalize():<15} {value:>8.4f}")
    print(f"{'='*55}\n")

    # ── save ──
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final, f, indent=2)

    summary_path = os.path.join(save_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Test Results — {n_samples} samples\n")
        f.write("=" * 40 + "\n")
        for metric, value in final.items():
            f.write(f"{metric.capitalize():<15} {value:.4f}\n")

    print(f"📄  metrics.json      → {metrics_path}")
    print(f"📄  metrics_summary   → {summary_path}")
    print(f"🖼️   Research figures  → {viz_dir}/  ({viz_count} saved)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: save_dir/best_model.pth)")
    parser.add_argument("--num_viz",    type=int, default=8,
                        help="Number of research visualisations to generate")
    parser.add_argument("--thresh",     type=float, default=0.5,
                        help="Binarisation threshold for predicted masks")
    args = parser.parse_args()

    main(checkpoint=args.checkpoint, num_viz=args.num_viz, thresh=args.thresh)