"""Cached PyTorch Dataset classes for preprocessed npz slices.

These load pre-extracted 2D slices from npz files (created by preprocess.py
extract-slices), avoiding the overhead of re-reading 3D NIfTI volumes.

Data structure:
    {preprocessed_dir}/{split}/{modality}/{field_strength}/*.npz
    Each npz contains: image (H, W) float32 in [0,1], slice_idx int32

Output tensors are scaled to [-1, 1] (standard for GAN training).

Three dataset types mirror the on-the-fly versions in dataset.py:
    - CachedUnpairedDataset: Single-domain slices (CUT/CycleGAN)
    - CachedPairedDataset: Matched source-target slices (Hybrid fine-tuning)
    - CachedMultiDomainDataset: All domains (StarGAN v2)
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import CenterCropOrPad, ToTensor, ScaleToMinusOneOne, Compose
from .utils import FIELD_STRENGTHS, FIELD_TO_DOMAIN, MODALITIES, get_joint_domain


def _cached_transform(crop_size: Optional[Tuple[int, int]] = None) -> Compose:
    """Transform for cached data (npz stored as [0,1], scaled to [-1,1]).

    Args:
        crop_size: Target (H, W) for CenterCropOrPad. None to skip cropping.
    """
    steps = []
    if crop_size is not None:
        steps.append(CenterCropOrPad(crop_size))
    steps += [ToTensor(), ScaleToMinusOneOne()]
    return Compose(steps)


def _list_npz_files(base_dir: Path, split: str, modality: str, field: str) -> List[Path]:
    """List npz files for a given split/modality/field."""
    d = base_dir / split / modality / field
    if not d.exists():
        return []
    return sorted(d.glob("*.npz"))


class CachedUnpairedDataset(Dataset):
    """Single-domain 2D slice dataset from preprocessed npz files.

    Used by CUT and CycleGAN. Two instances (source + target) are
    paired via UnpairedDataLoader.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: str,
        field_strength: str,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.field_strength = field_strength
        self.transform = transform or _cached_transform(crop_size)

        self.files = _list_npz_files(self.preprocessed_dir, split, modality, field_strength)
        if not self.files:
            raise FileNotFoundError(
                f"No npz files found in {self.preprocessed_dir / split / modality / field_strength}. "
                f"Run: python scripts/preprocess.py extract-slices --input_dir <data_root> --output_dir {preprocessed_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        npz = np.load(self.files[index])
        image = npz["image"]  # (H, W), float32, stored as [0, 1], scaled to [-1, 1]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "field_strength": self.field_strength,
        }


class CachedPairedDataset(Dataset):
    """Paired source-target 2D slice dataset from preprocessed npz files.

    Used by Hybrid fine-tuning. Matches slices by filename — both source
    and target directories must have been extracted from the same subjects
    (Prospective data).

    Pairing: slices are matched by volume name prefix (e.g., "001_GX").
    For each source slice, the corresponding target slice at the same
    slice index is used.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: str,
        source_field: str,
        target_field: str,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.transform = transform or _cached_transform(crop_size)

        source_files = _list_npz_files(self.preprocessed_dir, split, modality, source_field)
        target_files = _list_npz_files(self.preprocessed_dir, split, modality, target_field)

        if not source_files or not target_files:
            raise FileNotFoundError(
                f"No npz files found for {source_field} or {target_field} in "
                f"{self.preprocessed_dir / split / modality}"
            )

        # Build lookup by subject_id + slice_id (last two parts before .npz)
        # e.g., "pro_T1W_0.1T_P001_s053.npz" -> key = "P001_s053"
        def _pair_key(path: Path) -> str:
            parts = path.stem.split("_")
            # Find subject ID (P### or R_*) and slice (s###)
            return "_".join(parts[-2:])  # e.g., "P001_s053"

        target_lookup = {_pair_key(f): f for f in target_files}

        # Match pairs by subject + slice index
        self.pairs: List[Tuple[Path, Path]] = []
        for src_path in source_files:
            key = _pair_key(src_path)
            if key in target_lookup:
                self.pairs.append((src_path, target_lookup[key]))

        if not self.pairs:
            raise ValueError(
                f"No matching pairs found between {source_field} and {target_field}. "
                f"Ensure both were extracted from the same subjects (e.g., Prospective)."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src_path, tgt_path = self.pairs[index]
        src_img = np.load(src_path)["image"]
        tgt_img = np.load(tgt_path)["image"]

        if self.transform:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)

        return {"source": src_img, "target": tgt_img}


class CachedMultiDomainDataset(Dataset):
    """Multi-domain 2D slice dataset from preprocessed npz for StarGAN v2.

    Returns: image + domain label + reference image from a different domain.

    Supports multi-modality training. When multiple modalities are provided,
    domain labels use ``get_joint_domain`` (0..14 for 3 modalities × 5 fields).
    For a single modality, labels fall back to ``FIELD_TO_DOMAIN`` (0..4).
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        field_strengths: List[str] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        if modalities is not None:
            self.modalities = list(modalities)
        elif modality is not None:
            self.modalities = [modality]
        else:
            self.modalities = list(MODALITIES)
        self.field_strengths = field_strengths or FIELD_STRENGTHS
        self.transform = transform or _cached_transform(crop_size)
        self._use_joint = len(self.modalities) > 1

        # Index files per domain
        self.domain_files: Dict[int, List[Path]] = {}
        self.samples: List[Tuple[Path, int, str, str]] = []  # (npz_path, domain_idx, modality, field_strength)
        self._domain_to_pair: Dict[int, Tuple[str, str]] = {}

        for modality_name in self.modalities:
            for fs in self.field_strengths:
                domain_idx = (
                    get_joint_domain(modality_name, fs) if self._use_joint
                    else FIELD_TO_DOMAIN[fs]
                )
                files = _list_npz_files(self.preprocessed_dir, split, modality_name, fs)
                if files:
                    self.domain_files.setdefault(domain_idx, []).extend(files)
                    self._domain_to_pair[domain_idx] = (modality_name, fs)
                    for f in files:
                        self.samples.append((f, domain_idx, modality_name, fs))

        if not self.samples:
            where = self.preprocessed_dir / split
            raise FileNotFoundError(
                f"No npz files found under {where} for modalities={self.modalities}, "
                f"fields={self.field_strengths}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        npz_path, domain_idx, modality_name, field_strength = self.samples[index]
        image = np.load(npz_path)["image"]

        # Sample reference from a different domain
        other_domains = [d for d in self.domain_files if d != domain_idx]
        if other_domains:
            ref_domain = random.choice(other_domains)
        else:
            ref_domain = domain_idx
        ref_path = random.choice(self.domain_files[ref_domain])
        ref_image = np.load(ref_path)["image"]

        if self.transform:
            image = self.transform(image)
            ref_image = self.transform(ref_image)

        return {
            "image": image,
            "domain": domain_idx,
            "field_strength": field_strength,
            "modality": modality_name,
            "ref_image": ref_image,
            "ref_domain": ref_domain,
        }
