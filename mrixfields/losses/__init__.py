from .perceptual import PerceptualLoss
from .adversarial import GANLoss
from .patchnce import PatchNCELoss
from .structure import SSIMLoss, StructureLoss

__all__ = [
    "PerceptualLoss",
    "GANLoss",
    "PatchNCELoss",
    "SSIMLoss",
    "StructureLoss",
]
