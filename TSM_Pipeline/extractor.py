# tsm_pipeline/extractor.py
# ---------------------------------------------------------------------------
# TSM feature extractor built on ImageNet-pretrained ResNet-50.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .constants import VISUAL_DIM, NUM_FRAMES


class TSMExtractor(nn.Module):
    """
    Temporal Shift Module feature extractor.

    Architecture:
        ResNet-50 (ImageNet pretrained)
        → final FC layer removed
        → each frame processed independently → (NUM_FRAMES, 2048)
        → temporal average pooling → (1, 2048)
    """

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Flatten()
        )

    def forward(self, x):
        """
        Args:
            x: (1, C, T, H, W)
        Returns:
            (1, 2048)
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)       # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)       # (B*T, C, H, W)
        x = self.feature_extractor(x)        # (B*T, 2048)
        x = x.reshape(B, T, VISUAL_DIM)     # (B, T, 2048)
        x = x.mean(dim=1)                    # (B, 2048)
        return x
