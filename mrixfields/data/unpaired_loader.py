"""Unpaired data loader for CUT-style training.

Provides synchronized iteration over two domains (source and target)
without requiring paired samples.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class UnpairedDataLoader:
    """Wraps two DataLoaders for unpaired domain A and domain B.

    Returns batches of (source, target) where source and target are
    independently sampled from their respective domains.

    Args:
        dataset_a: Source domain dataset.
        dataset_b: Target domain dataset.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        shuffle: Whether to shuffle datasets.
    """

    def __init__(
        self,
        dataset_a: Dataset,
        dataset_b: Dataset,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        self.loader_a = DataLoader(
            dataset_a, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )
        self.loader_b = DataLoader(
            dataset_b, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)

        # Iterate for the length of the shorter domain
        for _ in range(min(len(self.loader_a), len(self.loader_b))):
            try:
                batch_a = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                batch_a = next(iter_a)
            try:
                batch_b = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                batch_b = next(iter_b)
            yield batch_a, batch_b

    def __len__(self):
        return min(len(self.loader_a), len(self.loader_b))


class ImagePool:
    """Buffer of previously generated images for discriminator training.

    Stores up to pool_size images. When full, randomly replaces existing
    images with 50% probability (Shrivastava et al., 2017).

    Args:
        pool_size: Maximum number of stored images (0 = disabled).
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        """Return images from pool (with replacement) or pass through."""
        if self.pool_size == 0:
            return images

        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img.clone())
                result.append(img)
            elif np.random.random() > 0.5:
                idx = np.random.randint(0, self.pool_size)
                old = self.images[idx].clone()
                self.images[idx] = img.clone()
                result.append(old)
            else:
                result.append(img)
        return torch.cat(result, dim=0)
