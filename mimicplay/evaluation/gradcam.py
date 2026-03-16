"""Grad-CAM style visualizations for policies."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: int,
) -> np.ndarray:
    """Compute a simple Grad-CAM heatmap for a given input and class."""
    activations: torch.Tensor
    gradients: torch.Tensor

    def forward_hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal activations
        activations = out.detach()

    def backward_hook(_module: nn.Module, grad_in: tuple[torch.Tensor, ...], grad_out: tuple[torch.Tensor, ...]) -> None:
        nonlocal gradients
        gradients = grad_out[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(input_tensor)
    score = logits[:, target_class].sum()
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay heatmap on top of an RGB frame."""
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_color = cv2.applyColorMap((255 * heatmap_resized).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color[..., ::-1], alpha, 0)
    return overlay

