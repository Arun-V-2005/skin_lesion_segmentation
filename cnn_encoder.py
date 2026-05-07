# cnn_encoder.py

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from configs import CONFIG


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-Tiny multi-scale feature extractor.

    Returns four feature maps at strides 4, 8, 16, 32:
        f1: (B,  96, H/4,  W/4 )
        f2: (B, 192, H/8,  W/8 )
        f3: (B, 384, H/16, W/16)
        f4: (B, 768, H/32, W/32)

    For a 448×448 input that gives spatial sizes:
        f1: 112×112 | f2: 56×56 | f3: 28×28 | f4: 14×14
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights  = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        features = convnext_tiny(weights=weights).features

        # ── explicit stage split ──────────────────────────────────
        self.stem   = features[0]   # stride-4 stem
        self.stage1 = features[1]

        self.down1  = features[2]   # stride-8 downsample
        self.stage2 = features[3]

        self.down2  = features[4]   # stride-16 downsample
        self.stage3 = features[5]

        self.down3  = features[6]   # stride-32 downsample
        self.stage4 = features[7]

        if CONFIG["cnn"].get("freeze", False):
            for p in self.parameters():
                p.requires_grad = False

    # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        # Cast to match weight dtype (critical for AMP correctness)
        x = x.to(dtype=self.stem[0].weight.dtype)

        x  = self.stem(x)            # prepare
        f1 = self.stage1(x)          # (B,  96, H/4,  W/4 )

        x  = self.down1(f1)
        f2 = self.stage2(x)          # (B, 192, H/8,  W/8 )

        x  = self.down2(f2)
        f3 = self.stage3(x)          # (B, 384, H/16, W/16)

        x  = self.down3(f3)
        f4 = self.stage4(x)          # (B, 768, H/32, W/32)

        return f1, f2, f3, f4