"""Perceptual loss based on LPIPS."""

import torch
import torch.nn as nn
import lpips


class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss for MRI image translation.

    Uses AlexNet features by default. Handles grayscale images by
    repeating to 3 channels.

    Args:
        net: Backbone network ('alex', 'vgg', 'squeeze').
        weight: Scaling factor for the loss.
    """

    def __init__(self, net: str = "alex", weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.lpips_fn = lpips.LPIPS(net=net)
        # Freeze LPIPS parameters
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def _to_3ch(self, x: torch.Tensor) -> torch.Tensor:
        """Convert single-channel to 3-channel by repeating."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image (B, 1, H, W).
            target: Target image (B, 1, H, W).

        Returns:
            Scalar perceptual loss.
        """
        pred_3ch = self._to_3ch(pred)
        target_3ch = self._to_3ch(target)
        return self.weight * self.lpips_fn(pred_3ch, target_3ch).mean()
