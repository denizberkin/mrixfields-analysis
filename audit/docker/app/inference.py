#!/usr/bin/env python3
"""MRIxFields2026 Task 3 test-phase inference. Container entrypoint.

Reads /input/manifest.json, writes one full-volume prediction per sample to the
sample's own ``output`` path under /output.

Two things differ from scripts/make_task3_submission.py, which builds the
validation-phase ZIP:

  * No z clip. The validation pack is the 30-slice slab Z_CLIP_RANGE = (150, 180);
    the test phase requires the full (364, 436, 364) volume and explicitly rejects
    "cropped slabs or validation-style z-clipped volumes". Training only covered
    slices 72..291 (scripts/preprocess.py), so slices 0..71 and 292..363 are out of
    distribution -- but measured against ground truth on four held cases the model
    still beats the identity baseline there (+0.002 to +0.039 SSIM below z=72, a
    +-0.0002 wash above z=291, where the volume is essentially air), so every slice
    goes through the model rather than being copied from the source.

  * The file list comes from the manifest, not from a hardcoded subject table. The
    test case ids are hidden, and the guide is explicit that the manifest is the
    only authority: "Do not derive additional cases or mappings outside the
    manifest.json file."

Everything else -- the [0,1] to [-1,1] map, the 368x448 centre crop, the four-channel
multi-contrast layout, the 1e-3 background mask, the canonical-orientation round trip --
is the same computation as the scored validation path, so a local score carries over.

Usage (the container's ENTRYPOINT supplies the defaults):
    python inference.py --input /input --output /output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELD_STRENGTHS = ("0.1T", "1.5T", "3T", "5T", "7T")
CROP_SIZE = (368, 448)
#: scripts/preprocess.py keeps slices 72..291, and slice_position() normalises on that
#: span. Hardcoded here rather than imported: the container vendors its dependencies.
SLICE_INDEX_RANGE = (72, 291)
EXPECTED_SHAPE = (364, 436, 364)

#: 1e-3, not 1e-6: two source volumes in the training set carry a flat non-zero
#: pedestal in air (1.2e-4 and 1.1e-5, uniform), and at 1e-6 their foreground mask
#: covers the whole volume, so predicted texture would ship where the target is flat
#: air. At 1e-3 a healthy volume moves 0.228 -> 0.226 foreground, i.e. 0.2% of voxels
#: at the brain edge where both sides are already near zero.
BACKGROUND_THRESHOLD = 1e-3

WEIGHTS = Path(__file__).resolve().parent / "weights" / "task3.pt"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrx.model import ConditionalUNet  # noqa: E402
from mrx.transforms import CenterCropOrPad  # noqa: E402


def get_joint_domain(modality: str, field_strength: str) -> int:
    """Flat index into the 15 joint (modality, field) domains the model conditions on."""
    return MODALITIES.index(modality) * len(FIELD_STRENGTHS) + FIELD_STRENGTHS.index(field_strength)


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load reoriented to RAS+ canonical, matching mrixfields.data.utils.load_nifti."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    return image.get_fdata(dtype=np.float32), image.affine


def normalise(sample: dict) -> dict:
    """Fill in any manifest field that is missing, from the sample's own paths.

    The published schema carries modality, source_field, target_field and case_id
    outright, and the guide is explicit that the manifest is the only authority on what
    to produce. But every one of them is also recoverable from
    ``{MOD}/{SRC}_to_{TGT}/src/P_{MOD}_{SRC}_{ID}.nii.gz``, and a KeyError at sample 1
    would cost the whole run for a schema detail that does not change what to predict.
    Present keys always win; nothing here overrides the manifest.
    """
    parts = Path(sample["input"]).parts
    modality, pair, name = parts[0], parts[1], parts[-1]
    source_field, target_field = pair.split("_to_")
    sample.setdefault("modality", modality)
    sample.setdefault("source_field", source_field)
    sample.setdefault("target_field", target_field)
    sample.setdefault("case_id", name.removesuffix(".nii.gz").split("_")[-1])
    sample.setdefault("output", str(Path(modality) / pair / "pred"
                                    / f"P_{modality}_{target_field}_{sample['case_id']}.nii.gz"))
    return sample


def build_model(device: torch.device) -> torch.nn.Module:
    """Construct the network from the checkpoint's own tensor shapes.

    Nothing about the architecture is hardcoded here: width, depth, input channels
    and both architecture flags are read back off the state dict, so swapping in a
    different checkpoint cannot silently build the wrong model -- a mismatch fails
    in load_state_dict instead.
    """
    payload = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    input_channels = int(state["encoders.0.0.weight"].shape[1])
    base_channels = int(state["encoders.0.0.weight"].shape[0])
    max_channels = int(state["bottleneck.0.weight"].shape[0])
    levels = len({key.split(".")[1] for key in state if key.startswith("encoders.")})
    residual_output = "residual_head.weight" in state
    film_conditioning = any(key.startswith("film_projections.") for key in state)
    # Read, never assumed: a slice-conditioned checkpoint carries slice_embedding.*, and
    # building without it would either raise on those keys or -- if they were dropped --
    # silently ignore the axial-position input the weights were trained to use.
    slice_conditioning = any(key.startswith("slice_embedding.") for key in state)
    if input_channels != 4:
        raise SystemExit(
            f"{WEIGHTS} expects input_channels={input_channels}; this entrypoint builds the "
            "4-channel multi-contrast input (primary + T1W + T2W + T2FLAIR). A 2.5D "
            "checkpoint (12 channels) needs the neighbour-slice block as well."
        )
    print(f"model: base={base_channels} max={max_channels} levels={levels} "
          f"in={input_channels} residual={residual_output} film={film_conditioning} "
          f"slice={slice_conditioning}", flush=True)
    model = ConditionalUNet(
        input_channels=input_channels,
        base_channels=base_channels,
        max_channels=max_channels,
        levels=levels,
        residual_output=residual_output,
        film_conditioning=film_conditioning,
        slice_conditioning=slice_conditioning,
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_volume(model, volume, auxiliary, source_domain, target_domain, device, batch_size,
                   tta=True):
    """Run every axial slice through the model and return the full-volume prediction.

    ``auxiliary`` is the other contrasts of the same case at the same source field,
    in the fixed (T1W, T2W, T2FLAIR) order the dataset uses. They occupy channels
    1..3 behind the primary on channel 0 -- channel 0 duplicates the contrast being
    predicted, which is what let the single-channel weights transfer into this model.

    ``tta`` averages the four in-plane flips, which is worth +0.0042 challenge SSIM on
    the shipped weights and is therefore part of what the checkpoint's score means -- not
    an optional extra. Flips only, never rotations: the data module augments with flips,
    so a flipped brain is in distribution and a rotated one never is. It costs little
    here because the job is dominated by reading and writing volumes, not by the forward.

    The axial position of each slice is passed alongside, normalised on the same
    SLICE_INDEX_RANGE the training filenames encode in their _s<idx> suffix. ``z`` is the
    volume's own z index, which is exactly the quantity that suffix records, so the two
    agree by construction. None for a checkpoint without slice conditioning.
    """
    crop = CenterCropOrPad(CROP_SIZE)
    uncrop = CenterCropOrPad(volume.shape[:2])
    channels = [volume, *auxiliary]
    output = np.zeros_like(volume, dtype=np.float32)

    for start in range(0, volume.shape[2], batch_size):
        indices = range(start, min(start + batch_size, volume.shape[2]))
        slices = np.stack([[crop(channel[:, :, z]) for channel in channels] for z in indices])
        images = torch.from_numpy(slices).to(device=device, dtype=torch.float32).mul(2).sub(1)
        source = torch.full((len(slices),), source_domain, device=device, dtype=torch.long)
        target = torch.full((len(slices),), target_domain, device=device, dtype=torch.long)
        positions = None
        if getattr(model, "slice_embedding", None) is not None:
            low, high = SLICE_INDEX_RANGE
            positions = torch.tensor(
                [min(max((z - low) / (high - low), 0.0), 1.0) for z in indices],
                device=device, dtype=torch.float32,
            )
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            views = ((), (-1,), (-2,), (-2, -1)) if tta else ((),)
            accumulated = None
            for dims in views:
                view = torch.flip(images, dims) if dims else images
                result = model(view, target, source, positions)
                result = torch.flip(result, dims) if dims else result
                accumulated = result if accumulated is None else accumulated + result
            predictions = accumulated / len(views)
        predictions = predictions.float().cpu().numpy()[:, 0]
        for z, prediction in zip(indices, predictions):
            output[:, :, z] = uncrop(np.clip(prediction, -1, 1) * 0.5 + 0.5)
    return output


def find_auxiliary(input_root: Path, modality: str, source_field: str, case_id: str) -> Path | None:
    """Locate one contrast of a case at its source field, anywhere in the input tree.

    The same volume appears under every mapping that uses it, so which mapping
    directory it is found in does not matter -- the filename pins the case, the
    modality and the field. Searching rather than constructing a path keeps this
    working if the organizers' directory layout differs from the published example.
    """
    name = f"P_{modality}_{source_field}_{case_id}.nii.gz"
    matches = sorted((input_root / modality).rglob(name)) or sorted(input_root.rglob(name))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, nargs="?", default="/input", help="input directory")
    parser.add_argument("--output", type=str, nargs="?", default="/output", help="output directory")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    input_root, output_root = Path(args.input), Path(args.output)
    print(f"input : {input_root}", flush=True)
    print(f"output: {output_root}", flush=True)

    manifest = json.loads((input_root / "manifest.json").read_text())
    samples = [normalise(sample) for sample in manifest["samples"]]
    print(f"manifest: task={manifest.get('task')} phase={manifest.get('phase')} "
          f"samples={len(samples)}", flush=True)

    device = torch.device(args.device)
    model = build_model(device)

    # Group by (case, source field): every sample in a group shares the same three
    # source volumes and differs only in which contrast is predicted and at which
    # target field, so each volume is read from disk once instead of once per mapping.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        groups[(sample["case_id"], sample["source_field"])].append(sample)

    started = time.time()
    done = failed = 0
    for (case_id, source_field), group in sorted(groups.items()):
        volumes: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            path = find_auxiliary(input_root, modality, source_field, case_id)
            if path is not None:
                volumes[modality] = load_nifti(path)[0]
            else:
                print(f"  WARNING: no {modality} for case {case_id} at {source_field}", flush=True)

        for sample in group:
            destination = output_root / sample["output"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_path = input_root / sample["input"]
            modality, target_field = sample["modality"], sample["target_field"]
            try:
                original = nib.load(str(source_path))
                original_orientation = nib.io_orientation(original.affine)
                volume, canonical_affine = load_nifti(source_path)

                # A contrast the input tree does not carry falls back to the primary,
                # which is what channel 0 already holds. That is a degraded prediction,
                # not a wrong one -- unlike zeros, which are a real intensity the model
                # would read as air.
                auxiliary = [volumes.get(m, volume) for m in MODALITIES]

                # The model was trained on volumes already scaled to [0, 1] and maps
                # them to [-1, 1] itself. An unscaled input (raw scanner units, say)
                # would not fail -- it would quietly predict nonsense for all 120
                # volumes. Warn rather than rescale: min-max normalising here would
                # also move a legitimate volume whose brightest voxel is below 1, and
                # that would silently break the calibration on the expected case.
                low, high = float(volume.min()), float(volume.max())
                if low < 0.0 or high > 1.0 or high < 0.2:
                    print(f"  WARNING: {sample['input']} spans "
                          f"[{low:.4g}, {high:.4g}], expected [0, 1]", flush=True)

                prediction = predict_volume(
                    model, volume, auxiliary,
                    get_joint_domain(modality, source_field),
                    get_joint_domain(modality, target_field),
                    device, args.batch_size,
                )
                prediction *= volume > BACKGROUND_THRESHOLD

                transform = nib.orientations.ornt_transform(
                    nib.io_orientation(canonical_affine), original_orientation)
                prediction = nib.orientations.apply_orientation(prediction, transform)
            except Exception:
                # A crash here would otherwise cost the whole run: the platform rejects
                # a submission for *missing* files, so a copy of the source (which the
                # challenge already scores at SSIM 0.836) is strictly better than a gap.
                traceback.print_exc()
                print(f"  FALLBACK to source copy: {sample['output']}", flush=True)
                prediction = np.asarray(nib.load(str(source_path)).dataobj, dtype=np.float32)
                original = nib.load(str(source_path))
                failed += 1

            prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)
            prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32)
            if prediction.shape != EXPECTED_SHAPE:
                print(f"  WARNING: {sample['output']} has shape {prediction.shape}, "
                      f"expected {EXPECTED_SHAPE}", flush=True)

            image = nib.Nifti1Image(prediction, original.affine, original.header)
            image.set_data_dtype(np.float32)
            nib.save(image, str(destination))
            done += 1
            elapsed = time.time() - started
            print(f"[{done}/{len(samples)}] {sample['output']}  "
                  f"{elapsed:.0f}s elapsed, {elapsed / done:.1f}s/volume", flush=True)

    print(f"wrote {done} predictions to {output_root} in {time.time() - started:.0f}s "
          f"({failed} fell back to the source copy)", flush=True)


if __name__ == "__main__":
    main()
