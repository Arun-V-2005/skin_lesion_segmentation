# dino_prior.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from configs import CONFIG


class DinoPrior(nn.Module):
    """
    Wraps a frozen DINOv2 backbone and extracts a spatial attention prior.

    Pipeline
    --------
    image → DINOv2 patch tokens → mean-pool over D → reshape → normalise
          → bilinear upsample → GPU Gaussian smooth → floor clamp → (B,1,H,W)

    Notes
    -----
    *  The forward pass is decorated with @torch.no_grad() when freeze=True,
       eliminating the autograd graph for the entire DINO branch and saving
       both memory and compute.
    *  All operations (including blur) stay on-device – no CPU round-trips.
    """

    def __init__(self, model_name: str, patch_size: int, device: str):
        super().__init__()

        self.device       = device
        self.patch_size   = patch_size
        self.smooth_k     = CONFIG["dino"]["smooth_kernel"]
        self.floor        = CONFIG["attention"]["floor"]
        self._frozen      = CONFIG["dino"]["freeze"]
        unfreeze_last_n   = CONFIG["dino"]["unfreeze_last_n"]

        # ── load backbone ──
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(device)

        # ── freeze / unfreeze ──
        for p in self.model.parameters():
            p.requires_grad = False

        if not self._frozen and unfreeze_last_n > 0:
            for block in self.model.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True

    # ──────────────────────────────────────────────────────────────

    def _compute_prior(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # 1. patch tokens  →  (B, N, D)
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]

        # 2. scalar score per patch via mean-pooling over D
        attn = tokens.mean(dim=-1)                              # (B, N)

        # 3. reshape to spatial grid
        Gh, Gw = H // self.patch_size, W // self.patch_size
        attn = attn.reshape(B, 1, Gh, Gw)                      # (B, 1, Gh, Gw)

        # 4. per-sample min-max normalise
        flat   = attn.view(B, -1)
        a_min  = flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
        a_max  = flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
        attn   = (attn - a_min) / (a_max - a_min + 1e-6)

        # 5. upsample to full resolution
        attn = F.interpolate(attn, size=(H, W),
                             mode="bilinear", align_corners=False)

        # 6. GPU-native Gaussian smoothing (no cv2, no .cpu())
        if self.smooth_k > 0:
            k = self.smooth_k if self.smooth_k % 2 != 0 else self.smooth_k + 1
            attn = TF.gaussian_blur(attn, [k, k])

        # 7. floor so no region is completely suppressed
        attn = (1.0 - self.floor) * attn + self.floor

        return attn                                             # (B, 1, H, W)

    # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  – input images, already on self.device

        Returns
        -------
        attn : (B, 1, H, W)  – spatial prior in [floor, 1]
        """
        if self._frozen:
            with torch.no_grad():
                return self._compute_prior(x)
        return self._compute_prior(x)