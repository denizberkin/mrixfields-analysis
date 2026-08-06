"""Adversarial losses for GAN training."""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """GAN loss supporting vanilla and least-squares variants.

    Args:
        mode: Loss type - 'vanilla' (BCE) or 'lsgan' (MSE).
        real_label: Target value for real samples.
        fake_label: Target value for fake samples.
    """

    def __init__(
        self,
        mode: str = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        if mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")

    def _get_target(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target_val = self.real_label if is_real else self.fake_label
        return target_val.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Args:
            prediction: Discriminator output.
            is_real: True for real, False for fake.

        Returns:
            Scalar loss.
        """
        target = self._get_target(prediction, is_real)
        return self.loss(prediction, target)
