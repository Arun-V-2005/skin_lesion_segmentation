# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_encoder  import ConvNeXtEncoder
from dino_prior   import DinoPrior
from attention_gate import AttentionGate
from decoder      import PriorGuidedDecoder
from configs      import CONFIG


class SegmentationModel(nn.Module):
    """
    DINO-guided skin-lesion segmentation model.

    Forward contract
    ----------------
    forward(x, cached_attn=None)
        x            : (B, 3, H, W)  – input images
        cached_attn  : (B, 1, H, W)  – pre-computed DINO priors (optional)
                       Pass this when the dataset returns tensors loaded from
                       the DINO cache directory to skip the DINOv2 forward
                       pass entirely.

    Returns
    -------
    logits : (B, 1, H, W)  – raw (un-sigmoided) segmentation logits
    attn   : (B, 1, H, W)  – spatial prior used (from cache or live)
    """

    def __init__(self, config: dict | None = None, device: str = "cuda"):
        super().__init__()

        cfg = config or CONFIG
        self.device = device

        self.encoder = ConvNeXtEncoder(pretrained=cfg["cnn"]["pretrained"]).to(device)
        self.dino    = DinoPrior(
            model_name=cfg["dino"]["model_name"],
            patch_size=cfg["dino"]["patch_size"],
            device=device,
        )
        self.gate    = AttentionGate(cfg).to(device)
        self.decoder = PriorGuidedDecoder().to(device)

    # ──────────────────────────────────────────────────────────────

    def forward(self,
                x:            torch.Tensor,
                cached_attn:  torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── DINO prior (use cache when available) ──
        if cached_attn is not None and cached_attn.ndim == 4:
            # Dataset returned a valid cached map
            attn = cached_attn.to(x.device, dtype=x.dtype)
        else:
            attn = self.dino(x)            # live inference (first epoch / no cache)

        # ── CNN encoder ──
        f1, f2, f3, f4 = self.encoder(x)

        # ── attention gate on deepest features ──
        f4 = self.gate(f4, attn)

        # ── decode ──
        out = self.decoder(f4, f3, f2, f1, attn)

        # ── upsample to input resolution ──
        out = F.interpolate(out, size=x.shape[-2:],
                            mode="bilinear", align_corners=False)

        return out, attn