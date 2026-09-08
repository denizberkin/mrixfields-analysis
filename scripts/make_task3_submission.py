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
BACKGROUND_THRESHOLD = 1e-3
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


def derive_unet_flags(state: dict[str, torch.Tensor]) -> tuple[bool, bool]:
    """Read ConditionalUNet's two architecture flags back out of a state dict.

    Both are visible in the tensors, so neither is taken on trust: a residual
    checkpoint carries residual_head.weight where a Tanh one carries
    output.0.weight, and per-scale FiLM adds film_projections.*. A wrong guess
    would fail in load_state_dict rather than pass silently -- both flags move
    key names -- but there is no reason to guess in the first place.
    """
    residual_output = "residual_head.weight" in state
    film_conditioning = any(key.startswith("film_projections.") for key in state)
    return residual_output, film_conditioning


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
    elif architecture == "tubelet":
        print("tubelet: LeJEPA ViT-B encoder + conditioned UNETR decoder", flush=True)
    else:
        base_channels, max_channels, levels = derive_shape(weights)
        print(
            f"derived from checkpoint: base_channels={base_channels} "
            f"max_channels={max_channels} levels={levels}",
            flush=True,
        )

    if architecture == "conditional":
        from components.models.conditional_unet import ConditionalUNet

        residual_output, film_conditioning = derive_unet_flags(weights)
        # Input width is visible in the first encoder conv, so a 4-channel
        # multi-contrast checkpoint builds the model it was trained as instead of
        # failing in load_state_dict against the 1-channel default.
        input_channels = int(weights["encoders.0.0.weight"].shape[1])
        print(
            f"derived from checkpoint: residual_output={residual_output} "
            f"film_conditioning={film_conditioning} input_channels={input_channels}",
            flush=True,
        )
        model: torch.nn.Module = ConditionalUNet(
            input_channels=input_channels,
            base_channels=base_channels,
            max_channels=max_channels,
            levels=levels,
            residual_output=residual_output,
            film_conditioning=film_conditioning,
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
    elif architecture == "tubelet":
        # The pipeline component owns the (image, target, source) adapter and the
        # axial-slab permute, so import it rather than re-deriving either here.
        # Its factory would also fetch the pretrained encoder off disk, which the
        # checkpoint is about to overwrite, so build the parts directly.
        from components.models.conditional_tubelet import ConditionalTubeletAdapter
        from mrixfields_downstream.tubelet_model import (
            build_conditional_tubelet_translator,
        )

        model = ConditionalTubeletAdapter(build_conditional_tubelet_translator())
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
    tta: bool = False,
    auxiliary: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Predict the submission slab only. Copied from
    experiment-pipeline/scripts/inference_task3_conditional_unet.py so the two
    paths cannot drift in intensity handling or cropping.

    ``tta`` averages the four in-plane flips, matching scripts/eval_holdout.py so a
    gain measured locally is the same computation that ships. Flips only, not the
    eight-way dihedral: the data modules augment with flips, never rotations, so a
    rotated brain is out of distribution.

    ``auxiliary`` carries the other contrasts of the same subject at the same field, in
    the fixed (T1W, T2W, T2FLAIR) order of CachedMultiContrastDataset. They become
    channels 1..3 behind ``volume`` itself on channel 0, which is the layout the
    multi-contrast model was trained on -- channel 0 duplicates the contrast being
    predicted, and the duplication is what let the single-channel weights transfer."""
    start, stop = Z_CLIP_RANGE
    crop = CenterCropOrPad(CROP_SIZE)
    uncrop = CenterCropOrPad(volume.shape[:2])
    output = np.zeros_like(volume, dtype=np.float32)

    for batch_start in range(start, stop, batch_size):
        indices = range(batch_start, min(batch_start + batch_size, stop))
        channels = [volume, *auxiliary] if auxiliary else [volume]
        slices = np.stack([[crop(channel[:, :, index]) for channel in channels]
                           for index in indices])
        images = torch.from_numpy(slices).to(device=device, dtype=torch.float32)
        images = images.mul(2).sub(1)
        source = torch.full((len(slices),), source_domain, device=device, dtype=torch.long)
        target = torch.full((len(slices),), target_domain, device=device, dtype=torch.long)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            if tta:
                accumulated = None
                for dims in ((), (-1,), (-2,), (-2, -1)):
                    view = torch.flip(images, dims) if dims else images
                    result = model(view, target, source)
                    result = torch.flip(result, dims) if dims else result
                    accumulated = result if accumulated is None else accumulated + result
                predictions = accumulated / 4
            else:
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


def predict_slab_tubelet(
    model: torch.nn.Module,
    volume: np.ndarray,
    source_domain: int,
    target_domain: int,
    device: torch.device,
    batch_size: int,
    depth: int = 16,
) -> np.ndarray:
    """Predict in 16-slice axial slabs, the shape the tubelet encoder was pretrained on.

    Unlike the Swin path there is no sliding window: the tubelet stem is (2, 16, 16) and
    the model is fixed at 16 slices, so the volume is tiled along z instead. The tail is
    zero-padded to a full slab and cut back afterwards, and the in-plane axes are padded
    up to a multiple of 16 rather than centre-cropped, so no anatomy is discarded.
    """
    height, width, slices = volume.shape
    pad_h = (-height) % depth
    pad_w = (-width) % depth
    output = np.zeros_like(volume, dtype=np.float32)
    source = torch.tensor([source_domain], device=device, dtype=torch.long)
    target = torch.tensor([target_domain], device=device, dtype=torch.long)

    for start in range(0, slices, depth * batch_size):
        stop = min(start + depth * batch_size, slices)
        block = np.ascontiguousarray(volume[:, :, start:stop], dtype=np.float32)
        pad_d = (-block.shape[2]) % depth
        if pad_d:
            block = np.pad(block, ((0, 0), (0, 0), (0, pad_d)))
        tensor = torch.from_numpy(block).to(device)
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_w, 0, pad_h))
        count = tensor.shape[2] // depth
        # [H, W, count*depth] -> [count, 1, H, W, depth], the adapter's axial layout
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], count, depth)
        tensor = tensor.permute(2, 0, 1, 3).unsqueeze(1).mul(2).sub(1)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            prediction = model(tensor, target.expand(count), source.expand(count))
        prediction = prediction.float().squeeze(1).permute(1, 2, 0, 3)
        prediction = prediction.reshape(tensor.shape[2], tensor.shape[3], count * depth)
        prediction = prediction[:height, :width, : stop - start].cpu().numpy()
        output[:, :, start:stop] = np.clip(prediction, -1, 1) * 0.5 + 0.5
    return output


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
    architecture: str = "",
    tta: bool = False,
    auxiliary_paths: list[Path] | None = None,
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
    if architecture == "tubelet":
        prediction = predict_slab_tubelet(
            model, volume, source_domain, target_domain, device, batch_size
        )
    elif volumetric:
        prediction = predict_slab_volume(
            model, volume, source_domain, target_domain, device, roi, batch_size, overlap
        )
    else:
        # Loaded through the same load_nifti as the primary, so every channel is in the
        # same canonical orientation before any of them reaches the model.
        auxiliary = ([load_nifti(path)[0] for path in auxiliary_paths]
                     if auxiliary_paths else None)
        prediction = predict_slab(
            model, volume, source_domain, target_domain, device, batch_size, tta=tta,
            auxiliary=auxiliary,
        )

    # Zero the background using the source volume, matching the inference script.
    # The threshold is 1e-3, not 1e-6: P_T2W_1.5T_0006 and _0009 carry a flat
    # non-zero pedestal in air (1.2e-4 and 1.1e-5, uniform), so at 1e-6 their
    # foreground mask covers the whole volume and nothing is zeroed -- those two
    # sources would ship predicted texture where the target is flat air. At 1e-3
    # every volume masks consistently (a healthy one moves 0.228 -> 0.226, i.e.
    # 0.2% of voxels at the brain edge, where both sides are already near zero).
    # Rejected an adaptive per-volume pedestal estimate: more precise, but it
    # makes the mask depend on statistics we cannot check against the evaluator.
    start, stop = Z_CLIP_RANGE
    prediction[:, :, start:stop] *= volume[:, :, start:stop] > BACKGROUND_THRESHOLD

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
        choices=("unconditional", "conditional", "vanilla", "vit", "swin", "tubelet"),
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
    parser.add_argument("--tta", action="store_true",
                        help="average the four in-plane flips; 2D architectures only")
    parser.add_argument("--dry-run", action="store_true", help="list the work without writing")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.tta and args.architecture in {"swin", "tubelet"}:
        # Better to stop than to accept a flag that silently does nothing: only the
        # 2D slice path implements it, and a submission built with --tta that ignored
        # it would be indistinguishable from one that applied it.
        raise SystemExit(f"--tta is not implemented for --architecture {args.architecture}")

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
    # The model itself decides whether the other contrasts are needed; nothing here is
    # a flag the caller could get wrong.
    first_conv = next((p for n, p in model.named_parameters()
                       if n == "encoders.0.0.weight"), None)
    multicontrast = first_conv is not None and first_conv.shape[1] > 1
    if multicontrast:
        print(f"multi-contrast: channel 0 duplicates the predicted contrast, then "
              f"{', '.join(MODALITIES)} at the source field", flush=True)

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
                auxiliary_paths = None
                if multicontrast:
                    auxiliary_paths = [
                        split / other / source / f"P_{other}_{source}_{subject}.nii.gz"
                        for other in MODALITIES
                    ]
                    for path in auxiliary_paths:
                        # Every validation subject has all three contrasts at its one
                        # source field; stop rather than zero-fill if that ever changes,
                        # because a zero channel is a legitimate intensity here.
                        if not path.is_file():
                            raise FileNotFoundError(path)
                clipped = predict_to_slab_image(
                    model, origin, modality, source, target, device, args.batch_size,
                    volumetric=args.architecture == "swin",
                    roi=args.roi,
                    overlap=args.overlap,
                    architecture=args.architecture,
                    tta=args.tta,
                    auxiliary_paths=auxiliary_paths,
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
