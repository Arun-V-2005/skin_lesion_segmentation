# attention_gate.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import CONFIG


class AttentionGate(nn.Module):
    """
    Applies a DINO-derived spatial prior onto CNN feature maps.

    Gating formula
    --------------
        F_out = F * (1 + α * A_stabilised)

    where
        A_stabilised = (1 - floor) * A + floor

    so the effective multiplier is always ≥ (1 + α·floor) > 1,
    meaning no spatial region is ever fully suppressed.

    Notes
    -----
    *  Resize is only performed when the attention and feature spatial
       dimensions actually differ – avoiding a wasted interpolation call
       at the deepest (14×14) level where they already match.
    *  dtype cast ensures AMP (float16) compatibility.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg        = config or CONFIG
        self.alpha = cfg["attention"]["gating_alpha"]
        self.floor = cfg["attention"]["floor"]

    # ──────────────────────────────────────────────────────────────

    def forward(self,
                features:  torch.Tensor,   # (B, C, Hf, Wf)
                attention: torch.Tensor,   # (B, 1, H,  W )
                ) -> torch.Tensor:

        # ── 1. dtype alignment ──
        attention = attention.to(dtype=features.dtype)

        # ── 2. spatial alignment (lazy – only when needed) ──
        if attention.shape[-2:] != features.shape[-2:]:
            attention = F.interpolate(attention,
                                      size=features.shape[-2:],
                                      mode="bilinear",
                                      align_corners=False)

        # ── 3. stabilise + gate (single fused expression) ──
        stabilised = (1.0 - self.floor) * attention + self.floor
        return features * (1.0 + self.alpha * stabilised)