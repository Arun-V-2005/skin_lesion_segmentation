# decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineBlock(nn.Module):
    """
    Two-layer conv residual refinement block.

    Uses depthwise-separable convolutions for ~3× fewer parameters
    and faster throughput vs plain Conv2d, while retaining full
    receptive field.  A residual shortcut stabilises training.
    """

    def __init__(self, channels: int):
        super().__init__()

        # depthwise → pointwise → BN → ReLU  ×2
        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),   # depthwise
            nn.Conv2d(channels, channels, 1, bias=False),  # pointwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # block 2
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)   # residual connection


class PriorGuidedDecoder(nn.Module):
    """
    Three-level decoder that fuses CNN skip features with the DINO
    spatial prior at each resolution.

    Architecture (high → low resolution)
    -------------------------------------
    f4 (14²) → up1 → prior-gate → + f3 (28²)  → ref1
             → up2 → prior-gate → + f2 (56²)  → ref2
             → up3 → prior-gate → + f1 (112²) → ref3
             → final 1×1 → logits (1, H, W)
    """

    def __init__(self):
        super().__init__()

        # ── upsamplers ──────────────────────────────────────────
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(192,  96, kernel_size=2, stride=2)

        # ── refinement blocks ────────────────────────────────────
        self.ref1 = RefineBlock(384)
        self.ref2 = RefineBlock(192)
        self.ref3 = RefineBlock(96)

        # ── head ─────────────────────────────────────────────────
        self.final = nn.Conv2d(96, 1, kernel_size=1)

    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _gate(x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Resize-and-apply prior gate: x * (1 + attn)."""
        if attn.shape[-2:] != x.shape[-2:]:
            attn = F.interpolate(attn, size=x.shape[-2:],
                                 mode="bilinear", align_corners=False)
        return x * (1.0 + attn)

    def forward(self,
                f4:   torch.Tensor,   # (B, 768, 14, 14)
                f3:   torch.Tensor,   # (B, 384, 28, 28)
                f2:   torch.Tensor,   # (B, 192, 56, 56)
                f1:   torch.Tensor,   # (B,  96, 112,112)
                attn: torch.Tensor,   # (B,   1, H,  W )
                ) -> torch.Tensor:

        # cast prior to match feature dtype (AMP safety)
        attn = attn.to(dtype=f4.dtype)

        # ── level 1: 14² → 28² ──
        x = self.up1(f4)
        x = self._gate(x, attn)
        x = x + f3
        x = self.ref1(x)

        # ── level 2: 28² → 56² ──
        x = self.up2(x)
        x = self._gate(x, attn)
        x = x + f2
        x = self.ref2(x)

        # ── level 3: 56² → 112² ──
        x = self.up3(x)
        x = self._gate(x, attn)
        x = x + f1
        x = self.ref3(x)

        return self.final(x)   # (B, 1, 112, 112) – upsampled in model.py