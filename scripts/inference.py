"""Inference script: generate predictions from trained models.

Supports CUT, CycleGAN (ResnetGenerator), and StarGAN v2.

Usage:
    # CUT / CycleGAN
    python scripts/inference.py --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
        --checkpoint $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cut/retro_scratch/weights/checkpoint_epoch100.pth \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/

    # StarGAN v2
    python scripts/inference.py --config configs/task3/stargan/any_to_any_all_modalities.yaml \
        --checkpoint $OUTPUT_DIR/task3_any_to_any_multimodal/stargan_v2/retro_scratch/weights/checkpoint_500000.pth \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/ \
        --target_field 7T

After inference, run SynthSeg segmentation:
    python Evaluation/segment.py \
        --input_dir $INFERENCE_DIR/ \
        --output_dir ${INFERENCE_DIR}_seg/
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.models.networks import ResnetGenerator
from mrixfields.models.stargan_v2 import build_stargan_v2
from mrixfields.data.utils import load_nifti, save_nifti, FIELD_TO_DOMAIN
from mrixfields.data.transforms import CenterCropOrPad, NormalizeMinMax
from mrixfields.env import load_env, get_inference_dir, get_device


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_generator_state(state_dict: dict) -> dict:
    """Extract generator weights from various checkpoint formats.

    Supports:
        - Full CUT/CycleGAN model checkpoint (key "model" with "netG." prefix)
        - Hybrid fine-tune checkpoint (key "generator")
        - Raw generator state_dict (no wrapper keys)

    Raises:
        ValueError: If no generator weights can be found.
    """
    if "model" in state_dict:
        for prefix in ("netG.", "netG_AB."):
            gen_state = {k[len(prefix):]: v for k, v in state_dict["model"].items()
                         if k.startswith(prefix)}
            if gen_state:
                return gen_state
        raise ValueError(
            f"Checkpoint has 'model' key but no generator prefix found. "
            f"Available prefixes: {set(k.split('.')[0] for k in state_dict['model'])}"
        )
    if "generator" in state_dict:
        return state_dict["generator"]
    if all(not k.startswith("net") for k in state_dict):
        return state_dict
    raise ValueError(
        f"Unrecognized checkpoint format. Top-level keys: {list(state_dict.keys())}"
    )


def load_generator(cfg: dict, checkpoint_path: str, device: torch.device):
    """Load a generator from a training checkpoint.

    Dispatches on ``cfg['method']``:
    - ``cut`` / ``cyclegan`` / ``hybrid``: build a ``ResnetGenerator`` and
      load weights via ``_extract_generator_state`` (handles the ``netG_A``,
      ``netG``, ``generator``, and plain state-dict layouts).
    - ``stargan_v2``: build the full StarGAN v2 net set and load the EMA
      generator and EMA mapping network.

    Returns ``(model, model_type)`` where ``model_type`` is ``'resnet'`` or
    ``'stargan_v2'`` — used by ``predict_volume`` to dispatch the forward call.
    """
    method = cfg["method"]
    model_cfg = cfg["model"]

    if method in ("cut", "cyclegan", "hybrid"):
        model = ResnetGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
        )
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        gen_state = _extract_generator_state(state_dict)
        model.load_state_dict(gen_state)
        model.to(device).eval()
        return model, "resnet"

    elif method == "stargan_v2":
        nets, nets_ema = build_stargan_v2(
            img_size=model_cfg.get("img_size", 128),
            style_dim=model_cfg.get("style_dim", 64),
            latent_dim=model_cfg.get("latent_dim", 16),
            num_domains=model_cfg.get("num_domains", 5),
            max_conv_dim=model_cfg.get("max_conv_dim", 512),
            input_nc=model_cfg.get("input_nc", 1),
        )
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "nets_ema" in state_dict:
            for k in nets_ema:
                nets_ema[k].load_state_dict(state_dict["nets_ema"][k])
        for k in nets_ema:
            nets_ema[k].to(device).eval()
        return nets_ema, "stargan_v2"

    else:
        raise ValueError(f"Unknown method: {method}")


def predict_volume(
    model,
    model_type: str,
    volume: np.ndarray,
    crop_size: tuple = None,
    slice_axis: int = 2,
    device: torch.device = torch.device("cpu"),
    target_domain: int = None,
    latent_dim: int = 16,
    style_dim: int = 64,
) -> np.ndarray:
    """Run inference on a full 3D volume, slice by slice.

    Input volume is expected in [0,1]; scaled to [-1,1] for the model.
    Output is mapped back to [0,1] for saving.

    Args:
        crop_size: If None, slices are fed at native size (CycleGAN/CUT).
                   If set (e.g. (512,512)), pads/crops before model, then restores.
    """
    use_crop = crop_size is not None
    if use_crop:
        crop = CenterCropOrPad(crop_size)

    original_shape = volume.shape
    n_slices = volume.shape[slice_axis]

    vol_norm = volume.astype(np.float32)

    output = np.zeros_like(volume)

    with torch.no_grad():
        # For StarGAN v2: compute style code ONCE for the entire volume
        # to ensure consistent style across all slices
        s_trg_vol = None
        if model_type == "stargan_v2":
            if target_domain is None:
                raise ValueError(
                    "StarGAN v2 inference requires --target_domain or --target_field"
                )
            z = torch.randn(1, latent_dim, device=device)
            y = torch.tensor([target_domain], device=device).long()
            s_trg_vol = model["mapping_network"](z, y)

        for i in range(n_slices):
            slicing = [slice(None)] * volume.ndim
            slicing[slice_axis] = i
            s = tuple(slicing)
            slc = vol_norm[s]

            slc_input = crop(slc) if use_crop else slc
            # Scale [0,1] → [-1,1] to match training data range
            slc_scaled = slc_input * 2.0 - 1.0
            tensor = torch.from_numpy(slc_scaled).float().unsqueeze(0).unsqueeze(0).to(device)

            if model_type == "resnet":
                pred = model(tensor)
            elif model_type == "stargan_v2":
                pred = model["generator"](tensor, s_trg_vol)

            pred_np = pred.squeeze().cpu().numpy()

            # Un-crop: restore to original slice spatial dimensions
            if use_crop:
                if slice_axis == 2:
                    orig_slice_shape = (original_shape[0], original_shape[1])
                elif slice_axis == 1:
                    orig_slice_shape = (original_shape[0], original_shape[2])
                else:
                    orig_slice_shape = (original_shape[1], original_shape[2])
                uncrop = CenterCropOrPad(orig_slice_shape)
                pred_np = uncrop(pred_np)

            # Model outputs [-1, 1] (Tanh); map back to [0, 1] for saving
            pred_np = np.clip(pred_np, -1, 1) * 0.5 + 0.5
            output[s] = pred_np

    return output


def main():
    """Run inference on every NIfTI in ``--input_dir``.

    For each input volume: load it in canonical (RAS+) orientation, run the
    generator slice-by-slice along ``slice_axis``, map outputs back to the
    input's original orientation, mask the (skull-stripped) background using
    the input as a brain mask, and save under ``--output_dir`` preserving the
    relative path. Defaults for ``--output_dir`` / ``--device`` come from
    ``$INFERENCE_DIR`` / ``$DEVICE`` in ``.env``.
    """
    parser = argparse.ArgumentParser(description="MRIxFields2026 Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    load_env()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: INFERENCE_DIR from .env)")
    parser.add_argument("--target_domain", type=int, default=None,
                        help="Target domain index (StarGAN v2 only)")
    parser.add_argument("--target_field", type=str, default=None,
                        help="Target field strength name, e.g. '7T' (auto-converts to domain idx)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. 'cuda', 'cuda:0' (default: DEVICE from .env; falls back to cpu when CUDA is unavailable)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = get_inference_dir()

    cfg = load_config(args.config)
    device_str = args.device or get_device()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model, model_type = load_generator(cfg, args.checkpoint, device)
    print(f"Loaded model ({model_type}) from: {args.checkpoint}")

    # Resolve target domain for StarGAN v2
    target_domain = args.target_domain
    if target_domain is None and args.target_field:
        target_domain = FIELD_TO_DOMAIN.get(args.target_field)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    crop_size_raw = cfg["data"].get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    slice_axis = cfg["data"].get("slice_axis", 2)
    model_cfg = cfg["model"]

    for nifti_path in tqdm(nifti_files, desc="Inference"):
        # Load original image to preserve affine/orientation
        original_img = nib.load(str(nifti_path))
        original_affine = original_img.affine
        original_ornt = nib.io_orientation(original_affine)

        # Load canonical (RAS+) for inference
        data, canonical_affine = load_nifti(nifti_path)
        canonical_ornt = nib.io_orientation(canonical_affine)

        pred = predict_volume(
            model, model_type, data,
            crop_size=crop_size,
            slice_axis=slice_axis,
            device=device,
            target_domain=target_domain,
            latent_dim=model_cfg.get("latent_dim", 16),
            style_dim=model_cfg.get("style_dim", 64),
        )

        # Convert pred back from canonical to original orientation
        transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
        pred = nib.orientations.apply_orientation(pred, transform)

        # Restore brain mask from input. pack1_mask inputs are skull-stripped,
        # but ResnetGenerator/StarGAN-v2 tanh+bias leaks small nonzero values
        # outside the brain — see analysis/README.md.
        src_arr = original_img.get_fdata(dtype=np.float32)
        pred = pred * (src_arr > 1e-6).astype(pred.dtype)

        rel_path = nifti_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        save_nifti(pred, original_affine, out_path, header=original_img.header)

    print(f"Inference complete. Results saved to: {output_dir}")
    print(f"\nNext step: run SynthSeg segmentation")
    print(f"  python Evaluation/segment.py --input_dir {output_dir} --output_dir {output_dir}_seg/")


if __name__ == "__main__":
    main()
