"""Vision-Language-Action policy with FiLM conditioning."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from mimicplay.models.encoders import build_resnet18_encoder, infer_encoder_out_dim


class VLAPolicy(nn.Module):
    """Miniature VLA policy.

    (Frame, Language) → fused features → action logits.
    """

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
            self.vision_encoder = build_resnet18_encoder(in_channels=in_channels)
        else:
            raise ValueError(f"Unsupported encoder '{encoder_name}'")

        feat_dim = infer_encoder_out_dim(self.vision_encoder, (h, w, in_channels))
        self.language_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.language_encoder.requires_grad_(False)
        self.language_proj = nn.Linear(384, feat_dim)

        self.film_gamma = nn.Linear(feat_dim, feat_dim)
        self.film_beta = nn.Linear(feat_dim, feat_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def encode_language(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of language instructions."""
        emb = self.language_encoder.encode(texts, convert_to_tensor=True)
        return emb  # (B, 384)

    def forward(self, obs: torch.Tensor, language_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Tensor of shape (B, C, H, W).
            language_emb: Tensor of shape (B, 384).
        """
        vis_features = self.vision_encoder(obs)
        lang_features = self.language_proj(language_emb)
        gamma = self.film_gamma(lang_features)
        beta = self.film_beta(lang_features)
        fused = gamma * vis_features + beta
        logits = self.head(fused)
        return logits

