"""NIfTI I/O utilities for MRIxFields2026.

Data directory structure:
    {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

    Splits: Training_retrospective, Training_prospective,
            Validating_prospective, Testing_prospective
    Modalities: T1W, T2W, T2FLAIR
    Field strengths: 0.1T, 1.5T, 3T, 5T, 7T
"""

from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np


# 5 field strengths
FIELD_STRENGTHS = ["0.1T", "1.5T", "3T", "5T", "7T"]

# 3 modalities
MODALITIES = ["T1W", "T2W", "T2FLAIR"]

# Data splits — directory name → short abbreviation used in CLI args, yaml configs, and slice filenames.
# Two orthogonal dimensions: collection period (retro/pro) × dataset partition (train/val/test).
# Independent from --mode (retro_scratch / pro_scratch / pro_pretrained), which controls pipeline stage.
SPLIT_ABBR = {
    "Training_retrospective": "retro_train",
    "Training_prospective":   "pro_train",
    "Validating_prospective": "pro_val",
    "Testing_prospective":    "pro_test",
}
ABBR_TO_SPLIT = {v: k for k, v in SPLIT_ABBR.items()}
SPLITS = list(SPLIT_ABBR.keys())  # for backward-compat / iteration

# Domain index mapping for StarGAN v2 (single-modality: only 5 field strengths).
FIELD_TO_DOMAIN = {fs: i for i, fs in enumerate(FIELD_STRENGTHS)}

# Joint (modality, field) → domain mapping for multi-modality StarGAN v2 (Task 3).
# Flat index: modality_idx * len(FIELD_STRENGTHS) + field_idx
#   0-4:   T1W × {0.1T, 1.5T, 3T, 5T, 7T}
#   5-9:   T2W × {0.1T, 1.5T, 3T, 5T, 7T}
#   10-14: T2FLAIR × {0.1T, 1.5T, 3T, 5T, 7T}
NUM_JOINT_DOMAINS = len(MODALITIES) * len(FIELD_STRENGTHS)


def get_joint_domain(modality: str, field_strength: str) -> int:
    """Map (modality, field_strength) to a flat domain index in [0, NUM_JOINT_DOMAINS)."""
    return MODALITIES.index(modality) * len(FIELD_STRENGTHS) + FIELD_STRENGTHS.index(field_strength)


def joint_domain_to_pair(domain: int) -> Tuple[str, str]:
    """Inverse of get_joint_domain: flat index → (modality, field_strength)."""
    m_idx, f_idx = divmod(domain, len(FIELD_STRENGTHS))
    return MODALITIES[m_idx], FIELD_STRENGTHS[f_idx]

# Task definitions
TASK1_PAIRS = [(fs, "7T") for fs in FIELD_STRENGTHS if fs != "7T"]  # 4 pairs
TASK2_PAIRS = [("0.1T", fs) for fs in FIELD_STRENGTHS if fs != "0.1T"]  # 4 pairs


def load_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data, affine).

    The image is reoriented to RAS+ canonical form for consistency
    across different scanner conventions.
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    """Save numpy array as a NIfTI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    nib.save(img, str(path))


def get_voxel_size(path: str | Path) -> np.ndarray:
    """Get voxel dimensions (mm) from a NIfTI file without loading data."""
    img = nib.load(str(path))
    return np.array(img.header.get_zooms()[:3], dtype=np.float32)


def list_nifti_files(
    data_root: str | Path,
    split: str,
    modality: str,
    field_strength: str,
) -> List[Path]:
    """List all NIfTI files for a given split/modality/field_strength.

    Path pattern: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

    Args:
        data_root: Root data directory (e.g., Data_to_be_released/).
        split: e.g., "Training_retrospective".
        modality: e.g., "T1W".
        field_strength: e.g., "0.1T", "3T".

    Returns:
        Sorted list of .nii.gz file paths.
    """
    data_root = Path(data_root)
    split_root = data_root / split
    if not split_root.exists() and data_root.is_dir():
        split_root = next(
            (path for path in data_root.iterdir() if path.is_dir() and path.name.casefold() == split.casefold()),
            split_root,
        )
    root = split_root / modality / field_strength
    if not root.exists():
        return []
    return sorted(root.glob("*.nii.gz"))


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from standardized filename for cross-field pairing.

    Filename format: {R,P}_{modality}_{field}_{ID}.nii.gz
    Examples:
        P_T1W_0.1T_0001.nii.gz -> P_0001
        R_T1W_3T_0042.nii.gz   -> R_0042
    """
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return f"{parts[0]}_{parts[-1]}"
    return base


def field_strength_to_float(field_str: str) -> float:
    """Convert field strength string like '0.1T' to float 0.1.

    For names like '3T', returns the numeric part (3.0).
    """
    # Extract numeric part before 'T'
    numeric = field_str.split("T")[0]
    return float(numeric)


def float_to_field_strength(value: float) -> str:
    """Convert float field strength to string, e.g. 0.1 -> '0.1T'."""
    if value == int(value):
        return f"{int(value)}T"
    return f"{value}T"
