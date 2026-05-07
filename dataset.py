# dataset.py

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import torchvision.transforms.functional as TF
import random

from configs import CONFIG


# ══════════════════════════════════════════════════════════════════
#  SPLIT HELPER  (single source of truth for all splits)
# ══════════════════════════════════════════════════════════════════

def _load_df(metadata_csv: str | Path, mask_dir: str | Path) -> pd.DataFrame:
    """Load metadata and drop rows whose mask file is missing."""
    mask_dir = Path(mask_dir)
    df = pd.read_csv(metadata_csv)
    return df[df["image_id"].apply(
        lambda x: (mask_dir / f"{x}_segmentation.png").exists()
    )].copy()


def get_splits(metadata_csv: str | Path,
               mask_dir:     str | Path,
               val_split:    float | None = None,
               test_split:   float | None = None,
               seed:         int   | None = None,
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a TWO-STAGE group-aware split:

        full dataset  ──(test_split)──►  train+val  |  test
        train+val     ──(val_split)───►  train      |  val

    All three returned DataFrames share NO lesion_id across boundaries.

    Returns
    -------
    train_df, val_df, test_df
    """
    val_split  = val_split  or CONFIG["data"]["val_split"]   # default 0.2
    test_split = test_split or CONFIG["data"].get("test_split", 0.2)
    seed       = seed       or CONFIG["data"]["seed"]

    df = _load_df(metadata_csv, mask_dir)
    groups = df["lesion_id"]

    # ── stage 1: carve out test ──
    spl1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_idx, test_idx = next(spl1.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx].copy()
    test_df     = df.iloc[test_idx].copy()

    # ── stage 2: split train+val ──
    spl2 = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(spl2.split(trainval_df,
                                          groups=trainval_df["lesion_id"]))

    train_df = trainval_df.iloc[train_idx].copy()
    val_df   = trainval_df.iloc[val_idx].copy()

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def resize_and_pad(image: np.ndarray, is_mask: bool = False) -> torch.Tensor:
    """Resize with aspect-ratio preservation, then zero/reflect pad."""
    target_h, target_w = CONFIG["data"]["image_size"]

    h, w = image.shape[:2] if not is_mask else image.shape

    scale = min(target_h / h, target_w / w)
    nh, nw = int(h * scale), int(w * scale)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    image = cv2.resize(image, (nw, nh), interpolation=interp)

    pad_h, pad_w = target_h - nh, target_w - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    if is_mask:
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=0)
        image = (image > 127).astype(np.float32)
        return torch.tensor(image).unsqueeze(0)          # (1, H, W)
    else:
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_REFLECT)
        image = image.astype(np.float32) / 255.0
        return torch.tensor(image).permute(2, 0, 1)     # (3, H, W)


# ══════════════════════════════════════════════════════════════════
#  AUGMENTATION  (colour-only – no spatial transforms that break masks)
# ══════════════════════════════════════════════════════════════════

def color_augment(img: torch.Tensor) -> torch.Tensor:
    """Random colour jitter applied consistently to the image only."""
    if random.random() > 0.5:
        img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        img = TF.adjust_saturation(img, saturation_factor=random.uniform(0.8, 1.2))
    if random.random() > 0.3:
        img = TF.adjust_hue(img, hue_factor=random.uniform(-0.05, 0.05))
    return img.clamp(0.0, 1.0)


def joint_flip(img: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal + vertical flip applied identically to image and mask."""
    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask)
    return img, mask


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class HAMDataset(Dataset):
    """
    HAM10000 dataset.

    Returns (img, mask, dino_prior) where dino_prior is:
      - loaded from cache_dir/<image_id>.pt  when available
      - an empty tensor otherwise (cache_dir=None or file missing)

    For the test set, pass cache_dir=None so no cached prior is
    loaded — the model will run live DINOv2 inference instead.
    This guarantees the test set was never touched during caching.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 img_dir: str | Path,
                 mask_dir: str | Path,
                 cache_dir: str | Path | None = None,
                 augment: bool = False):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_id = self.df.iloc[idx]["image_id"]

        img_path  = self.img_dir  / f"{image_id}.jpg"
        mask_path = self.mask_dir / f"{image_id}_segmentation.png"

        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing files for: {image_id}")

        # ── load & preprocess ──
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img_t  = resize_and_pad(img,  is_mask=False)   # (3, H, W) float32
        mask_t = resize_and_pad(mask, is_mask=True)    # (1, H, W) float32

        # ── augmentation (train only) ──
        if self.augment:
            img_t, mask_t = joint_flip(img_t, mask_t)
            img_t = color_augment(img_t)

        # ── DINO prior from cache ──
        attn_t = self._load_cache(image_id)

        return img_t, mask_t, attn_t

    # ── helpers ──────────────────────────────────────────────────

    def _load_cache(self, image_id: str) -> torch.Tensor:
        if self.cache_dir is None:
            return torch.tensor([])
        cache_file = self.cache_dir / f"{image_id}.pt"
        if cache_file.exists():
            return torch.load(cache_file, map_location="cpu", weights_only=True)
        return torch.tensor([])


# ══════════════════════════════════════════════════════════════════
#  DINO CACHE BUILDER  (train-only — called AFTER splitting)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_dino_cache(img_dir:      str | Path,
                     mask_dir:     str | Path,
                     metadata_csv: str | Path,
                     cache_dir:    str | Path,
                     dino_model,                 # DinoPrior (already on device)
                     device:       str   = "cuda",
                     batch_size:   int   = 32,
                     train_df:     "pd.DataFrame | None" = None,
                     ) -> None:
    """
    Pre-compute DINO attention maps and save them to disk.

    ⚠️  IMPORTANT — always pass `train_df` (the train split only).
    Caching only train images ensures test images are never processed
    by the pipeline before evaluation, eliminating any risk of
    inadvertent data leakage through the caching step.

    If `train_df` is None (legacy behaviour), all images with masks
    are cached — only do this if you are certain DINO is fully frozen
    and you accept the theoretical leakage risk.

    Usage
    -----
    train_df, val_df, test_df = get_splits(metadata_csv, mask_dir)

    dino = DinoPrior(...).to("cuda")
    build_dino_cache(img_dir, mask_dir, metadata_csv,
                     cache_dir   = "/content/dino_cache",
                     dino_model  = dino,
                     train_df    = train_df)      # ← train split only
    """
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if train_df is not None:
        df = train_df.reset_index(drop=True)
        print(f"🔒  Cache restricted to {len(df):,} TRAIN images only "
              f"(test set is leak-free).")
    else:
        # Legacy: cache everything — only safe for frozen DINO
        df = _load_df(metadata_csv, mask_dir).reset_index(drop=True)
        print(f"⚠️   Caching ALL {len(df):,} images (legacy mode — "
              f"only safe with fully frozen DINO).")

    tmp_ds = HAMDataset(df, img_dir, mask_dir, cache_dir=None, augment=False)
    loader = DataLoader(tmp_ds, batch_size=batch_size, shuffle=False,
                        num_workers=CONFIG["data"]["num_workers"],
                        pin_memory=True)

    dino_model.eval()
    written = 0

    for batch_imgs, _, _ in tqdm(loader, desc="Building DINO cache"):
        batch_imgs = batch_imgs.to(device)
        attn_maps  = dino_model(batch_imgs)           # (B, 1, H, W)

        start = written
        for i, attn in enumerate(attn_maps):
            image_id = df.iloc[start + i]["image_id"]
            out_path = cache_dir / f"{image_id}.pt"
            if not out_path.exists():
                torch.save(attn.cpu(), out_path)
        written += len(attn_maps)

    print(f"✅  Cached {written} DINO maps → {cache_dir}")


# ══════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════

def get_loaders(img_dir:      str | Path,
                mask_dir:     str | Path,
                metadata_csv: str | Path,
                batch_size:   int | None = None,
                val_split:    float | None = None,
                ) -> tuple[DataLoader, DataLoader]:
    """
    Returns train and val DataLoaders.

    Splits are derived via get_splits() so they are consistent with
    the test split used in test.py — the same seed and same two-stage
    GroupShuffleSplit logic guarantees zero patient leakage.
    """
    batch_size = batch_size or CONFIG["data"]["batch_size"]
    cache_dir  = CONFIG["paths"].get("cache_dir")
    do_augment = CONFIG["data"].get("augment", False)

    train_df, val_df, _ = get_splits(metadata_csv, mask_dir,
                                     val_split=val_split)

    train_ds = HAMDataset(train_df, img_dir, mask_dir,
                          cache_dir=cache_dir, augment=do_augment)
    val_ds   = HAMDataset(val_df,   img_dir, mask_dir,
                          cache_dir=cache_dir, augment=False)

    common_kw = dict(
        num_workers=CONFIG["data"]["num_workers"],
        pin_memory=CONFIG["data"]["pin_memory"],
        persistent_workers=CONFIG["data"]["num_workers"] > 0,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  **common_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **common_kw)

    cache_hits = (sum(1 for _ in Path(cache_dir).glob("*.pt"))
                  if cache_dir and Path(cache_dir).exists() else 0)
    print(f"📦  Train: {len(train_ds):,} | Val: {len(val_ds):,} samples")
    print(f"🗂️   DINO cache: {cache_hits:,} files in {cache_dir}")

    return train_loader, val_loader