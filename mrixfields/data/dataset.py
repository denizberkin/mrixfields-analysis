"""PyTorch Dataset classes for MRIxFields2026 challenge.

Data structure:
    {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

Output tensors are scaled to [-1, 1] (standard for GAN training).

Three dataset types:
    - UnpairedMRIDataset: Single-domain slices (for CUT/CycleGAN unpaired training)
    - PairedMRIDataset: Matched source-target slices (for validation/finetune)
    - MultiDomainMRIDataset: All domains for StarGAN v2 (Task 3)
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    FIELD_STRENGTHS,
    FIELD_TO_DOMAIN,
    MODALITIES,
    extract_subject_id,
    get_joint_domain,
    list_nifti_files,
    load_nifti,
)
from .transforms import CenterCropOrPad, NormalizeMinMax, ToTensor, ScaleToMinusOneOne, Compose


def _default_transform(crop_size: Optional[Tuple[int, int]] = None) -> Compose:
    """Default preprocessing: optional center crop + normalize to [0,1] + to tensor + scale to [-1,1].

    Args:
        crop_size: Target (H, W) for CenterCropOrPad. None to skip cropping
                   (CycleGAN/CUT use native slice size; StarGAN v2 pads to 512x512).
    """
    steps = []
    if crop_size is not None:
        steps.append(CenterCropOrPad(crop_size))
    steps += [NormalizeMinMax(), ToTensor(), ScaleToMinusOneOne()]
    return Compose(steps)


class UnpairedMRIDataset(Dataset):
    """Single-domain 2D slice dataset for unpaired training.

    Used by CUT and CycleGAN to load slices from one field strength.
    Two instances (source + target) are paired via UnpairedDataLoader.

    Data path: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: str,
        field_strength: str,
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.modality = modality
        self.field_strength = field_strength
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform(crop_size)

        # Index: list of (nifti_path, slice_index)
        self.samples: List[Tuple[Path, int]] = []
        self._index_data()

    def _index_data(self):
        nifti_files = list_nifti_files(
            self.data_root, self.split, self.modality, self.field_strength
        )
        for nifti_path in nifti_files:
            data, _ = load_nifti(nifti_path)
            n_slices = data.shape[self.slice_axis]
            for i in range(n_slices):
                slc = self._get_slice(data, i)
                if slc.std() > self.min_slice_std:
                    self.samples.append((nifti_path, i))

    def _get_slice(self, volume: np.ndarray, idx: int) -> np.ndarray:
        slicing = [slice(None)] * volume.ndim
        slicing[self.slice_axis] = idx
        return volume[tuple(slicing)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        nifti_path, slice_idx = self.samples[index]
        data, _ = load_nifti(nifti_path)
        slc = self._get_slice(data, slice_idx)
        slc = self.transform(slc)

        return {
            "image": slc,
            "field_strength": self.field_strength,
            "modality": self.modality,
            "filename": nifti_path.name,
            "slice_idx": slice_idx,
        }


class PairedMRIDataset(Dataset):
    """Paired source-target 2D slice dataset.

    Used for:
    - Validation/testing (Validating_prospective, Testing_prospective)
    - Paired fine-tuning phase (Training_prospective)

    Pairs slices from source and target field strengths of the SAME subject.
    Subjects are identified by matching filenames across field strength dirs.

    Data path: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: str,
        source_field: str,
        target_field: str,
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.modality = modality
        self.source_field = source_field
        self.target_field = target_field
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform(crop_size)

        # Index: list of (src_path, tgt_path, slice_index)
        self.samples: List[Tuple[Path, Path, int]] = []
        self._index_data()

    def _index_data(self):
        src_files = list_nifti_files(
            self.data_root, self.split, self.modality, self.source_field
        )
        tgt_files = list_nifti_files(
            self.data_root, self.split, self.modality, self.target_field
        )

        # Match by subject ID (filenames differ across field strengths)
        tgt_by_id = {extract_subject_id(f.name): f for f in tgt_files}
        for src_path in src_files:
            src_id = extract_subject_id(src_path.name)
            tgt_path = tgt_by_id.get(src_id)
            if tgt_path is None:
                continue

            src_data, _ = load_nifti(src_path)
            tgt_data, _ = load_nifti(tgt_path)

            n_slices = min(
                src_data.shape[self.slice_axis],
                tgt_data.shape[self.slice_axis],
            )
            for i in range(n_slices):
                src_slc = self._get_slice(src_data, i)
                tgt_slc = self._get_slice(tgt_data, i)
                if (src_slc.std() > self.min_slice_std and
                        tgt_slc.std() > self.min_slice_std):
                    self.samples.append((src_path, tgt_path, i))

    def _get_slice(self, volume: np.ndarray, idx: int) -> np.ndarray:
        slicing = [slice(None)] * volume.ndim
        slicing[self.slice_axis] = idx
        return volume[tuple(slicing)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        src_path, tgt_path, slice_idx = self.samples[index]

        src_data, _ = load_nifti(src_path)
        tgt_data, _ = load_nifti(tgt_path)

        src_slc = self.transform(self._get_slice(src_data, slice_idx))
        tgt_slc = self.transform(self._get_slice(tgt_data, slice_idx))

        return {
            "source": src_slc,
            "target": tgt_slc,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "modality": self.modality,
            "filename": src_path.name,
            "slice_idx": slice_idx,
        }


class MultiDomainMRIDataset(Dataset):
    """Multi-domain dataset for StarGAN v2 (Task 3: Any -> Any).

    Combines slices from all (modality, field_strength) combinations with a
    flat domain label. When multiple modalities are provided, domain labels
    use ``get_joint_domain`` (0..NUM_JOINT_DOMAINS-1). For a single modality,
    labels fall back to ``FIELD_TO_DOMAIN`` (0..4) for backward compatibility.

    Data path: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        field_strengths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = list(modalities)
        elif modality is not None:
            self.modalities = [modality]
        else:
            self.modalities = list(MODALITIES)
        self.field_strengths = field_strengths or FIELD_STRENGTHS
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform(crop_size)
        self._use_joint = len(self.modalities) > 1

        # Index: list of (nifti_path, slice_index, domain_label, modality, field_strength)
        self.samples: List[Tuple[Path, int, int, str, str]] = []
        self.domain_samples: Dict[int, List[int]] = {}
        self._domain_to_pair: Dict[int, Tuple[str, str]] = {}
        self._index_data()

    def _domain_label(self, modality: str, field_strength: str) -> int:
        if self._use_joint:
            return get_joint_domain(modality, field_strength)
        return FIELD_TO_DOMAIN[field_strength]

    def _index_data(self):
        for modality in self.modalities:
            for fs in self.field_strengths:
                domain_idx = self._domain_label(modality, fs)
                self.domain_samples.setdefault(domain_idx, [])
                self._domain_to_pair[domain_idx] = (modality, fs)
                nifti_files = list_nifti_files(
                    self.data_root, self.split, modality, fs
                )
                for nifti_path in nifti_files:
                    data, _ = load_nifti(nifti_path)
                    n_slices = data.shape[self.slice_axis]
                    for i in range(n_slices):
                        slicing = [slice(None)] * data.ndim
                        slicing[self.slice_axis] = i
                        slc = data[tuple(slicing)]
                        if slc.std() > self.min_slice_std:
                            sample_idx = len(self.samples)
                            self.samples.append((nifti_path, i, domain_idx, modality, fs))
                            self.domain_samples[domain_idx].append(sample_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        nifti_path, slice_idx, domain_idx, modality, field_strength = self.samples[index]
        data, _ = load_nifti(nifti_path)
        slicing = [slice(None)] * data.ndim
        slicing[self.slice_axis] = slice_idx
        slc = data[tuple(slicing)]
        slc = self.transform(slc)

        # Sample a random reference image from a different domain
        other_domains = [d for d in self.domain_samples if d != domain_idx
                         and len(self.domain_samples[d]) > 0]
        if other_domains:
            ref_domain = int(np.random.choice(other_domains))
            ref_idx = np.random.choice(self.domain_samples[ref_domain])
            ref_path, ref_slice, _, _, _ = self.samples[ref_idx]
            ref_data, _ = load_nifti(ref_path)
            ref_slicing = [slice(None)] * ref_data.ndim
            ref_slicing[self.slice_axis] = ref_slice
            ref_slc = self.transform(ref_data[tuple(ref_slicing)])
        else:
            ref_slc = slc
            ref_domain = domain_idx

        return {
            "image": slc,
            "domain": domain_idx,
            "ref_image": ref_slc,
            "ref_domain": ref_domain,
            "field_strength": field_strength,
            "modality": modality,
            "filename": nifti_path.name,
            "slice_idx": slice_idx,
        }

    def get_random_from_domain(self, domain_idx: int) -> Dict:
        """Get a random sample from a specific domain."""
        indices = self.domain_samples[domain_idx]
        idx = np.random.choice(indices)
        return self[idx]
