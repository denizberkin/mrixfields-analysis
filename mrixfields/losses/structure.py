"""Structure preservation loss for extreme field-gap translation (Task 2).

Combines SSIM-based structural similarity with edge consistency
to prevent structure loss when translating from very low (0.1T) to high fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM).

    Args:
        window_size: Size of the Gaussian window.
        sigma: Gaussian standard deviation.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)  # 2D Gaussian
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2

        mu_x = F.conv2d(x, self.kernel, padding=pad)
        mu_y = F.conv2d(y, self.kernel, padding=pad)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, self.kernel, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, self.kernel, padding=pad) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.kernel, padding=pad) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


class StructureLoss(nn.Module):
    """Combined structure preservation loss: SSIM + edge consistency.

    Args:
        ssim_weight: Weight for SSIM loss.
        edge_weight: Weight for edge consistency loss.
    """

    def __init__(self, ssim_weight: float = 1.0, edge_weight: float = 0.5):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        # Pre-compute Sobel kernels as buffers (moved to device with module)
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).reshape(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).reshape(1, 1, 3, 3),
        )

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge map."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ssim_weight * self.ssim_loss(pred, target)
        if self.edge_weight > 0:
            edge_pred = self._edge_map(pred)
            edge_target = self._edge_map(target)
            loss += self.edge_weight * F.l1_loss(edge_pred, edge_target)
        return loss
