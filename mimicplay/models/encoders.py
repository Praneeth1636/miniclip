"""Vision encoder backbones."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18_encoder(in_channels: int) -> nn.Module:
    """Create a ResNet-18 encoder that outputs 512-dim features."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    resnet.fc = nn.Identity()
    return resnet


def infer_encoder_out_dim(encoder: nn.Module, obs_shape: Tuple[int, int, int]) -> int:
    """Infer output feature dimension by a dummy forward pass."""
    c, h, w = _to_chw(obs_shape)
    x = torch.zeros(1, c, h, w)
    with torch.no_grad():
        y = encoder(x)
    return int(y.shape[-1])


def _to_chw(obs_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    h, w, c = obs_shape
    return c, h, w

