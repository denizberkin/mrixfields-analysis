"""Preprocessing transforms for MRI volumes and 2D slices."""

from typing import Optional, Tuple

import numpy as np
import torch


class CenterCropOrPad:
    """Center-crop or zero-pad a 3D volume or 2D slice to a target shape."""

    def __init__(self, target_shape: Tuple[int, ...]):
        self.target_shape = target_shape

    def __call__(self, data: np.ndarray) -> np.ndarray:
        result = np.zeros(self.target_shape, dtype=data.dtype)
        slices_src = []
        slices_dst = []
        for i, (s, t) in enumerate(zip(data.shape, self.target_shape)):
            if s > t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(None))
            else:
                start = (t - s) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + s))
        result[tuple(slices_dst)] = data[tuple(slices_src)]
        return result
