# configs.py

CONFIG = {
    # ─────────────────────────── DEVICE ────────────────────────────
    "device":  "cuda",
    "use_amp": True,          # Automatic Mixed Precision (~2× speedup)

    # ─────────────────────────── PATHS ─────────────────────────────
    "paths": {
        "img_dir":  "/root/.cache/kagglehub/datasets/nightfury007/"
                    "ham10000-isic2018-raw/versions/1/dataverse_files/"
                    "HAM10000_images_combined_600x450",
        "mask_dir": "/root/.cache/kagglehub/datasets/nightfury007/"
                    "ham10000-isic2018-raw/versions/1/dataverse_files/"
                    "HAM10000_segmentations_lesion_tschandl",
        "metadata": "/root/.cache/kagglehub/datasets/nightfury007/"
                    "ham10000-isic2018-raw/versions/1/dataverse_files/"
                    "HAM10000_metadata",
        "save_dir":  "/content/skin_lesion_new/outputs",
        "cache_dir": "/content/skin_lesion_new/dino_cache",  # Pre-computed DINO features
    },

    # ─────────────────────────── DATA ──────────────────────────────
    "data": {
        "image_size":  (448, 448),
        "batch_size":  16,
        "num_workers": 4,           # bumped: cache-hits are CPU-light
        "pin_memory":  True,
        "val_split":   0.2,
        "test_split":  0.2,   # Held-out test split — carved BEFORE caching
        "seed":        42,
        # Online augmentation (no geometry – keeps mask alignment trivial)
        "augment": True,
    },

    # ─────────────────────────── DINO ──────────────────────────────
    "dino": {
        "model_name":     "dinov2_vits14_reg",
        "patch_size":     14,
        "freeze":         True,       # Frozen; run once → cache to disk
        "unfreeze_last_n": 0,
        "smooth_kernel":  7,          # Must be odd; Gaussian blur on GPU
    },

    # ─────────────────────────── CNN ───────────────────────────────
    "cnn": {
        "encoder":    "convnext_tiny",
        "pretrained": True,
        "freeze":     False,          # Trainable – learns skin textures
        "channels":   [96, 192, 384, 768],
    },

    # ──────────────────────── ATTENTION ────────────────────────────
    "attention": {
        "use_attention": True,
        "gating_alpha":  0.5,
        "floor":         0.1,
        "clamp_min":     0.0,
        "clamp_max":     1.0,
    },

    # ─────────────────────────── DECODER ───────────────────────────
    "decoder": {
        "type":                "prior_guided",
        "use_skip_residual_add": True,
    },

    # ─────────────────────────── TRAINING ──────────────────────────
    "train": {
        "epochs":       20,
        "lr":           1e-4,
        "weight_decay": 1e-5,
        "dino_lr":      1e-5,
        "encoder_lr":   1e-4,
        # Gradient clipping prevents AMP instability with large batches
        "grad_clip":    1.0,
        # Warm-up epochs before cosine annealing kicks in
        "warmup_epochs": 5,
    },

    # ─────────────────────────── LOSS ──────────────────────────────
    "loss": {
        "bce_weight":   1.0,
        "dice_weight":  1.0,
        "focal_weight": 0.5,   # Extra supervision on hard negatives
        "focal_gamma":  2.0,
        "focal_alpha":  0.25,
        "kl_weight":    0.05,
    },
}