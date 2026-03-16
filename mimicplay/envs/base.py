"""Base environment interface for MimicPlay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class BaseEnv(ABC):
    """Abstract base class for all MimicPlay environments."""

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Returns:
            Tuple of observation array and info dictionary.
        """

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment by one action.

        Args:
            action: Integer index of the chosen action.

        Returns:
            Observation, reward, terminated flag, truncated flag, and info dict.
        """

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render the current state as an RGB array (H, W, 3)."""

    @abstractmethod
    def get_action_space(self) -> List[str]:
        """Return the list of action names in fixed order."""

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, int, int]:
        """Return the shape of observations as (H, W, C)."""

