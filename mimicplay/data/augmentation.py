"""Visual data augmentation utilities."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def apply_augmentations(obs: np.ndarray) -> np.ndarray:
    """Apply light augmentations to an observation stack.

    Args:
        obs: Observation array of shape (H, W, C).
    """
    aug = obs.astype(np.float32)
    aug = _random_crop(aug, padding=8)
    aug = _color_jitter(aug, brightness=0.1, contrast=0.1)
    aug = np.clip(aug, 0.0, 255.0).astype(np.uint8)
    return aug


def _random_crop(img: np.ndarray, padding: int) -> np.ndarray:
    h, w, c = img.shape
    padded = cv2.copyMakeBorder(
        img,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_REFLECT_101,
    )
    y = np.random.randint(0, 2 * padding)
    x = np.random.randint(0, 2 * padding)
    return padded[y : y + h, x : x + w, :c]


def _color_jitter(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Apply simple brightness and contrast jitter."""
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    img = img * c_factor + (b_factor - 1.0) * 127.5
    return img

