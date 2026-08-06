#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.data.transforms import CenterCropOrPad
from mrixfields.data.utils import get_joint_domain, save_nifti
from mrixfields.models.stargan_v2 import MappingNetwork, StarGANv2Generator


FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")
CASES = (("0.1T", "0001"), ("7T", "0016"))
Z_INDEX = 165
SLAB_START = 150


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("epoch") != 50:
        raise ValueError(f"Expected epoch 50 checkpoint, got {checkpoint.get('epoch')}")
    generator = StarGANv2Generator(img_size=512, style_dim=64, max_conv_dim=512, input_nc=1)
    mapping = MappingNetwork(latent_dim=16, style_dim=64, num_domains=15)
    generator.load_state_dict(checkpoint["nets_ema"]["generator"])
    mapping.load_state_dict(checkpoint["nets_ema"]["mapping_network"])
    del checkpoint
    generator.to(device).eval()
    mapping.to(device).eval()

    random = torch.Generator(device=device).manual_seed(args.seed)
    domains = torch.arange(15, device=device)
    with torch.inference_mode():
        styles = mapping(torch.randn(15, 16, generator=random, device=device), domains)

    crop = CenterCropOrPad((512, 512))
    uncrop = CenterCropOrPad((364, 436))
    for modality in MODALITIES:
        for source, subject_id in CASES:
            source_path = (
                args.data_root / modality / source / f"P_{modality}_{source}_{subject_id}.nii.gz"
            )
            original = nib.load(str(source_path))
            canonical = nib.as_closest_canonical(original)
            source_slice = np.asarray(canonical.dataobj[:, :, Z_INDEX], dtype=np.float32)
            image = torch.from_numpy(crop(source_slice)).unsqueeze(0).unsqueeze(0).to(device)
            targets = [field for field in FIELDS if field != source]
            target_domains = [get_joint_domain(modality, target) for target in targets]
            images = image.repeat(len(targets), 1, 1, 1).mul(2).sub(1)
            with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
                predictions = generator(images, styles[target_domains])
            predictions = predictions.float().cpu().numpy()[:, 0]

            canonical_orientation = nib.io_orientation(canonical.affine)
            original_orientation = nib.io_orientation(original.affine)
            orientation = nib.orientations.ornt_transform(canonical_orientation, original_orientation)
            original_mask = np.asarray(original.dataobj[:, :, Z_INDEX]) > 1e-6
            for target, prediction in zip(targets, predictions, strict=True):
                slab = np.zeros((364, 436, 30), dtype=np.float32)
                slab[:, :, Z_INDEX - SLAB_START] = uncrop(np.clip(prediction, -1, 1) * 0.5 + 0.5)
                slab = nib.orientations.apply_orientation(slab, orientation)
                slab[:, :, Z_INDEX - SLAB_START] *= original_mask
                output = (
                    args.output_root
                    / modality
                    / f"{source}_to_{target}"
                    / "pred"
                    / f"P_{modality}_{target}_{subject_id}.nii.gz"
                )
                save_nifti(slab, original.affine, output, header=original.header)
                print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
