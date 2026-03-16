"""Learning rate scheduling helpers."""

from __future__ import annotations

from typing import Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(optimizer: Optimizer, max_epochs: int) -> CosineAnnealingLR:
    """Create a cosine annealing scheduler."""
    return CosineAnnealingLR(optimizer, T_max=max_epochs)

