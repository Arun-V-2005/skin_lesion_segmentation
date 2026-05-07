# train.py

import os
import json
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from configs   import CONFIG
from model     import SegmentationModel
from dataset   import get_loaders
from trainer   import train_one_epoch, validate


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════

def _plot_curves(history: dict, save_dir: str) -> None:
    """Save two side-by-side plots: train curves and val curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── train ──
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="tab:blue", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color="tab:red")
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax.set_title("Training: Loss vs Accuracy")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # ── val ──
    ax = axes[1]
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
    ax3 = ax.twinx()
    ax3.plot(epochs, history["val_acc"], label="Val Acc", color="tab:green", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color="tab:orange")
    ax3.set_ylabel("Accuracy", color="tab:green")
    ax.set_title("Validation: Loss vs Accuracy")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📈  Curves saved → {out_path}")


# ══════════════════════════════════════════════════════════════════
#  OPTIMISER + SCHEDULER
# ══════════════════════════════════════════════════════════════════

def _build_optimizer(model: torch.nn.Module) -> AdamW:
    """
    Build AdamW with per-component learning rates.

    DINO is excluded when frozen because:
      1. All its params have requires_grad=False → AdamW skips them anyway.
      2. When the cache is used, self.dino(x) is never called during
         training → no gradient can flow to DINO regardless.
    Including frozen params wastes optimizer state memory for no benefit.

    If you set dino.freeze=False and unfreeze_last_n > 0, DINO params
    are added back so they receive gradient updates.
    """
    cfg = CONFIG["train"]

    param_groups = [
        {"params": model.encoder.parameters(), "lr": cfg["encoder_lr"]},
        {"params": model.gate.parameters(),    "lr": cfg["lr"]},
        {"params": model.decoder.parameters(), "lr": cfg["lr"]},
    ]

    # Only optimise DINO if it's actually trainable
    if not CONFIG["dino"]["freeze"]:
        trainable_dino = [p for p in model.dino.parameters() if p.requires_grad]
        if trainable_dino:
            param_groups.insert(0, {"params": trainable_dino, "lr": cfg["dino_lr"]})
            print(f"🔓  DINO unfrozen — {len(trainable_dino)} param tensors added to optimizer")
        else:
            print("ℹ️   DINO freeze=False but no trainable params found — skipping")
    else:
        print("🔒  DINO frozen — excluded from optimizer (cache bypasses forward pass)")

    return AdamW(param_groups, weight_decay=cfg["weight_decay"])


def _build_scheduler(optimizer: AdamW, total_epochs: int) -> SequentialLR:
    cfg = CONFIG["train"]
    warmup_epochs = cfg["warmup_epochs"]

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-7,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ══════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    cfg       = CONFIG
    device    = cfg["device"]
    epochs    = cfg["train"]["epochs"]
    grad_clip = cfg["train"]["grad_clip"]
    save_dir  = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ── data ──
    train_loader, val_loader = get_loaders(
        img_dir      = cfg["paths"]["img_dir"],
        mask_dir     = cfg["paths"]["mask_dir"],
        metadata_csv = cfg["paths"]["metadata"],
    )

    # ── model ──
    model = SegmentationModel(config=cfg, device=device)

    # ── optimiser / scheduler / scaler ──
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer, epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"])

    # ── sanity check: confirm DINO is not running during training ──
    cache_dir = cfg["paths"].get("cache_dir")
    if cache_dir and len(list(__import__("pathlib").Path(cache_dir).glob("*.pt"))) > 0:
        print("✅  DINO cache detected — ViT forward pass will be SKIPPED every epoch")
    else:
        print("⚠️   No DINO cache found — ViT will run live inference every batch (slow)")

    # ── history ──
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":  [],
    }

    best_val_dice = 0.0
    best_ckpt     = os.path.join(save_dir, "best_model.pth")

    print(f"\n{'='*60}")
    print(f"  Training for {epochs} epochs on {device.upper()}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # ── train ──
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler, grad_clip
        )

        # ── validate ──
        val_metrics = validate(model, val_loader, device)

        # ── scheduler step ──
        scheduler.step()
        current_lr = optimizer.param_groups[-1]["lr"]

        # ── log ──
        print(
            f"Epoch [{epoch:>3}/{epochs}] "
            f"| Train  loss: {train_metrics['loss']:.4f}  "
            f"dice: {train_metrics['dice']:.4f}  "
            f"acc: {train_metrics['acc']:.4f} "
            f"| Val  loss: {val_metrics['loss']:.4f}  "
            f"dice: {val_metrics['dice']:.4f}  "
            f"acc: {val_metrics['acc']:.4f} "
            f"| LR: {current_lr:.2e}"
        )

        # ── record ──
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # ── checkpoint (best val dice) ──
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_dice":   best_val_dice,
                },
                best_ckpt,
            )
            print(f"  ✅  Best model saved (val dice = {best_val_dice:.4f})")

    # ── save history + curves ──
    history_path = os.path.join(save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    _plot_curves(history, save_dir)
    print(f"\n✅  Training complete. Best val dice: {best_val_dice:.4f}")
    print(f"   Checkpoint → {best_ckpt}")


if __name__ == "__main__":
    main()