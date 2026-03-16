"""Behavioral cloning policy network."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mimicplay.models.encoders import build_resnet18_encoder, infer_encoder_out_dim


class BCPolicy(nn.Module):
    """Frame → encoder → MLP head → action logits."""

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        n_actions: int,
        encoder_name: str = "resnet18",
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        h, w, c = obs_shape
        if in_channels is None:
            in_channels = c

        if encoder_name == "resnet18":
            self.encoder = build_resnet18_encoder(in_channels=in_channels)
        else:
            raise ValueError(f"Unsupported encoder '{encoder_name}'")

        feat_dim = infer_encoder_out_dim(self.encoder, (h, w, in_channels))
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Tensor of shape (B, C, H, W).
        """
        features = self.encoder(obs)
        logits = self.head(features)
        return logits

