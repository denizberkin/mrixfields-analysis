#!/usr/bin/env python3
"""
Build a Task 3 submission ZIP from an experiment-pipeline U-Net checkpoint.

Mirrors scripts/make_source_submission.py exactly in layout, slab clipping and
shape validation -- that script is the reference for the Task 3 archive format.
The only difference is that predictions come from a model instead of being
copies of the source volume.

Written directly rather than through scripts/generate_challenge_submission.py,
because that script's validate_inputs() hard-requires official/MRIxFields2026/
before it ever reaches the inference_command path, and that checkout is not
present here.

Task 3 accepts no segmentations.

Usage:
    python scripts/make_task3_submission.py \
        --checkpoint experiment-pipeline/runs/task3_unet_unconditional/artifacts/task3_unet_finetune_10.pt \
        --architecture unconditional \
        --name task3_unet_unconditional_epoch10

    python scripts/make_task3_submission.py --checkpoint ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from mrixfields.data.transforms import CenterCropOrPad  # noqa: E402
from mrixfields.data.utils import get_joint_domain, load_nifti  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")

#: Validation subject ids per source field. They are disjoint across fields by design, so the ids in
#: an output filename always come from the source folder even though the name carries the target.
SOURCE_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}

CROP_SIZE = (368, 448)
EXPECTED_SHAPE = (364, 436, Z_CLIP_RANGE[1] - Z_CLIP_RANGE[0])


def find_split(data_dir: Path) -> Path:
    """Validation split directory, tolerating the dataset's inconsistent casing."""
    for name in ("Validating_prospective", "validating_prospective"):
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No validation split under {data_dir}")


def derive_shape(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Read (base_channels, max_channels, levels) back out of a state dict.

    Hardcoding these invites a silent mismatch whenever a config is edited
    between training and packaging, so infer them instead:
      base_channels  = width of the first encoder conv
      levels         = number of encoder blocks
      max_channels   = bottleneck width, which reproduces the channel schedule
                       whether or not the cap actually binds
    """
    base_channels = state["encoders.0.0.weight"].shape[0]
    levels = len({key.split(".")[1] for key in state if key.startswith("encoders.")})
    max_channels = state["bottleneck.0.weight"].shape[0]
    return int(base_channels), int(max_channels), int(levels)


def derive_decoder_channels(state: dict[str, torch.Tensor]) -> int:
    """Decoder width of a ConditionalViT, read back from its first reassemble conv.

    The backbone identity is not recoverable from the tensors alone, so it stays
    a flag; a wrong one fails loudly in load_state_dict rather than silently.
    """
    return int(state["reassemble.0.weight"].shape[0])


def build_model(
    architecture: str, checkpoint: Path, device: torch.device, backbone: str
) -> torch.nn.Module:
    """Construct the architecture and load weights strictly.

    All three forwards accept (image, target_domain, source_domain); the two
    unconditional variants ignore the domain tensors, so predict_slab can call
    them identically.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    weights = state.get("model", state)
    if architecture == "vit":
        decoder_channels = derive_decoder_channels(weights)
        print(f"derived from checkpoint: decoder_channels={decoder_channels} "
              f"backbone={backbone}", flush=True)
    elif architecture == "swin":
        # Read the shape off the tensors rather than trusting a flag, so a
        # checkpoint can never be loaded into a differently-sized model.
        feature_size = int(weights["net.swinViT.patch_embed.proj.weight"].shape[0])
        num_domains = int(weights["source_embedding.weight"].shape[0])
        print(f"derived from checkpoint: feature_size={feature_size} "
              f"num_domains={num_domains}", flush=True)
    else:
        base_channels, max_channels, levels = derive_shape(weights)
        print(
            f"derived from checkpoint: base_channels={base_channels} "
            f"max_channels={max_channels} levels={levels}",
            flush=True,
        )

    if architecture == "conditional":
        from components.models.conditional_unet import ConditionalUNet

        model: torch.nn.Module = ConditionalUNet(
            base_channels=base_channels, max_channels=max_channels, levels=levels
        )
    elif architecture == "unconditional":
        from components.models.unconditional_unet import UnconditionalUNet

        model = UnconditionalUNet(
            base_channels=base_channels, max_channels=max_channels, levels=levels
        )
    elif architecture == "vanilla":
        from components.models.vanilla_unet import VanillaUNet

        model = VanillaUNet(base_channels=base_channels, levels=levels)
    elif architecture == "vit":
        from components.models.conditional_vit import ConditionalViT

        # pretrained=False: every weight comes from the checkpoint, so fetching
        # the backbone from the hub first would only be overwritten.
        model = ConditionalViT(
            backbone=backbone, pretrained=False, decoder_channels=decoder_channels
        )
    elif architecture == "swin":
        from components.models.conditional_swin_unetr import ConditionalSwinUNETR

        # pretrained_path=None: the BraTS weights this was fine-tuned from are
        # already inside the checkpoint, and loading them first would only be
        # overwritten.
        model = ConditionalSwinUNETR(
            feature_size=feature_size, num_domains=num_domains, pretrained_path=None
        )
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(f"Unknown architecture {architecture!r}")

    model.load_state_dict(weights)
    return model.to(device).eval()


def predict_slab(
    model: torch.nn.Module,
    volume: np.ndarray,
    source_domain: int,
    target_domain: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Predict the submission slab only. Copied from
    experiment-pipeline/scripts/inference_task3_conditional_unet.py so the two
    paths cannot drift in intensity handling or cropping."""
    start, stop = Z_CLIP_RANGE
    crop = CenterCropOrPad(CROP_SIZE)
    uncrop = CenterCropOrPad(volume.shape[:2])
    output = np.zeros_like(volume, dtype=np.float32)

    for batch_start in range(start, stop, batch_size):
        indices = range(batch_start, min(batch_start + batch_size, stop))
        slices = np.stack([crop(volume[:, :, index]) for index in indices])
        images = torch.from_numpy(slices).unsqueeze(1).to(device=device, dtype=torch.float32)
        images = images.mul(2).sub(1)
        source = torch.full((len(slices),), source_domain, device=device, dtype=torch.long)
        target = torch.full((len(slices),), target_domain, device=device, dtype=torch.long)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            predictions = model(images, target, source)
        predictions = predictions.float().cpu().numpy()[:, 0]
        for index, prediction in zip(indices, predictions, strict=True):
            output[:, :, index] = uncrop(np.clip(prediction, -1, 1) * 0.5 + 0.5)
    return output


def predict_slab_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    source_domain: int,
    target_domain: int,
    device: torch.device,
    roi: int,
    sw_batch_size: int,
    overlap: float,
) -> np.ndarray:
    """Sliding-window 3D inference, for the volumetric models.

    The 2D path predicts a whole slice at once; a 3D model is trained on
    cubic crops instead, so the volume is covered by overlapping windows and
    blended. Everything downstream of this - the [-1, 1] to [0, 1] map, the
    background mask, the reorientation, the z clip - is shared with the 2D
    path, so the two cannot drift in how a prediction becomes a file.

    The accumulator lives on the CPU: the windows are what need the GPU, and a
    full 364x436x364 float32 output plus its blending weights would otherwise
    sit in VRAM for the whole volume.
    """
    from monai.inferers import sliding_window_inference

    # ascontiguousarray: reorienting to canonical can flip an axis, and a
    # negative stride is not something torch.from_numpy accepts. The 2D path
    # never hit this because stacking slices copies anyway.
    image = torch.from_numpy(np.ascontiguousarray(volume)).to(device=device, dtype=torch.float32)
    image = image[None, None].mul(2).sub(1)
    source = torch.tensor([source_domain], device=device, dtype=torch.long)
    target = torch.tensor([target_domain], device=device, dtype=torch.long)

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        count = patch.shape[0]
        return model(patch, target.expand(count), source.expand(count))

    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
        prediction = sliding_window_inference(
            image,
            (roi, roi, roi),
            sw_batch_size,
            predictor,
            overlap=overlap,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
        )
    return np.clip(prediction.float().numpy()[0, 0], -1, 1) * 0.5 + 0.5


def predict_to_slab_image(
    model: torch.nn.Module,
    source_path: Path,
    modality: str,
    source: str,
    target: str,
    device: torch.device,
    batch_size: int,
    volumetric: bool = False,
    roi: int = 96,
    overlap: float = 0.25,
) -> nib.Nifti1Image:
    """Run the model on one volume and return the clipped submission slab.

    The volume is predicted in canonical orientation, then mapped back to the
    file's own orientation before clipping, exactly as the inference script
    does. The z axis is unaffected by that transform for this dataset, so the
    Z_CLIP_RANGE indices stay aligned with the ground-truth pack.
    """
    original = nib.load(str(source_path))
    original_orientation = nib.io_orientation(original.affine)
    volume, canonical_affine = load_nifti(source_path)

    source_domain = get_joint_domain(modality, source)
    target_domain = get_joint_domain(modality, target)
    if volumetric:
        prediction = predict_slab_volume(
            model, volume, source_domain, target_domain, device, roi, batch_size, overlap
        )
    else:
        prediction = predict_slab(
            model, volume, source_domain, target_domain, device, batch_size
        )

    # Zero the background using the source volume, matching the inference script.
    start, stop = Z_CLIP_RANGE
    prediction[:, :, start:stop] *= volume[:, :, start:stop] > 1e-6

    canonical_orientation = nib.io_orientation(canonical_affine)
    transform = nib.orientations.ornt_transform(canonical_orientation, original_orientation)
    prediction = nib.orientations.apply_orientation(prediction, transform)

    image = nib.Nifti1Image(prediction.astype(np.float32), original.affine, original.header)
    image.set_data_dtype(np.float32)
    return image.slicer[:, :, start:stop]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("unconditional", "conditional", "vanilla", "vit", "swin"),
        default="unconditional",
    )
    parser.add_argument(
        "--backbone",
        default="vit_large_patch16_dinov3.lvd1689m",
        help="timm backbone name, only used by --architecture vit",
    )
    parser.add_argument("--roi", type=int, default=96,
                        help="sliding-window size, only used by --architecture swin; "
                             "match the crop_size the checkpoint was trained on")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="sliding-window overlap fraction, only used by --architecture swin")
    parser.add_argument("--name", default=None, help="subdirectory under SUBMISSION_DIR")
    parser.add_argument("--data-dir", type=Path, default=None, help="defaults to DATA_DIR from .env")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="defaults to SUBMISSION_DIR/<name> from .env")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="list the work without writing")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    from mrixfields.env import get_data_dir, get_submission_dir

    name = args.name or f"task3_{args.architecture}_unet"
    data_dir = args.data_dir or Path(get_data_dir())
    out_dir = args.out_dir or (Path(get_submission_dir()) / name)
    split = find_split(data_dir)

    pairs = [(source, target) for source in FIELDS for target in FIELDS if source != target]
    total = len(MODALITIES) * len(pairs) * 3
    print(f"checkpoint  : {args.checkpoint}", flush=True)
    print(f"architecture: {args.architecture}", flush=True)
    print(f"source      : {split}", flush=True)
    print(f"output      : {out_dir}", flush=True)
    print(
        f"{len(pairs)} field pairs x {len(MODALITIES)} modalities x 3 subjects = {total} files",
        flush=True,
    )

    if args.dry_run:
        for modality in MODALITIES[:1]:
            for source, target in pairs[:2]:
                for subject in SOURCE_IDS[source]:
                    print(
                        f"  {split.name}/{modality}/{source}/P_{modality}_{source}_{subject}.nii.gz"
                        f"  ->  task3/{modality}/{source}_to_{target}/pred/"
                        f"P_{modality}_{target}_{subject}.nii.gz",
                        flush=True,
                    )
        print("... (nothing written)", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")
    model = build_model(args.architecture, args.checkpoint, device, args.backbone)

    task_dir = out_dir / "task3"
    if task_dir.exists():
        shutil.rmtree(task_dir)

    written: list[Path] = []
    for modality in MODALITIES:
        for source, target in pairs:
            for subject in SOURCE_IDS[source]:
                origin = split / modality / source / f"P_{modality}_{source}_{subject}.nii.gz"
                if not origin.is_file():
                    raise FileNotFoundError(origin)
                destination = (
                    task_dir / modality / f"{source}_to_{target}" / "pred"
                    / f"P_{modality}_{target}_{subject}.nii.gz"
                )
                clipped = predict_to_slab_image(
                    model, origin, modality, source, target, device, args.batch_size,
                    volumetric=args.architecture == "swin",
                    roi=args.roi,
                    overlap=args.overlap,
                )
                if clipped.shape != EXPECTED_SHAPE:
                    raise RuntimeError(
                        f"{destination} has shape {clipped.shape}, expected {EXPECTED_SHAPE}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                nib.save(clipped, str(destination))
                written.append(destination)
        print(f"{modality}: {len(written)} files so far", flush=True)

    if len(written) != total:
        raise RuntimeError(f"wrote {len(written)} files, expected {total}")

    zip_path = out_dir / "task3.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(written):
            archive.write(path, path.relative_to(task_dir.parent))
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"corrupt archive member: {bad}")
    if len(members) != total:
        raise RuntimeError(f"archive has {len(members)} members, expected {total}")

    print(f"wrote {len(written)} files", flush=True)
    print(f"archive: {zip_path} ({zip_path.stat().st_size / 2**20:.0f} MiB, {len(members)} members)",
          flush=True)


if __name__ == "__main__":
    main()
