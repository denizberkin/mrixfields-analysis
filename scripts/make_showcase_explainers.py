#!/usr/bin/env python3
"""Explainer figures for the showcase deck (reports/showcase_presentation_v0.tex).

1. `metric_ssim_<k>.pdf`: one target slice and three distortions scaled to the same nRMSE
   (brighter, blurred, noisy), with the SSIM of each. Step k shows panels 1..k.
2. `lpips_*.png`: the pieces of the LPIPS diagram, for our prediction of one 0.1 T -> 7 T
   T1W slice against its target: both images, three channels of AlexNet layer 1's
   unit-normalised activations for each, and the LPIPS distance map summed over the five
   layers. `lpips_values.txt` holds the scalar; the deck assembles the diagram in TikZ.
3. `kspace_<k>.pdf`: a 3 T T1W slice, its k-space, fastMRI's 4x equispaced Cartesian mask
   (centre 8% kept, 25% of lines in all) and the zero-filled image. Simulated from the magnitude image;
   fastMRI itself ships the measured multi-coil k-space.

The prediction comes from the shipped weights through make_showcase_qualitative.py (TTA on).
Subject 0006 is a training subject.

    ~/anaconda3/envs/mri/bin/python scripts/make_showcase_explainers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_showcase_qualitative import compute_rows  # noqa: E402
from make_slides_qualitative import (  # noqa: E402
    Z_CLIP_RANGE, brain_bbox, build_model, get_data_dir, load_nifti, volume_path)

NRMSE_LEVEL = 0.15
ACCELERATION, CENTRE_FRACTION = 4, 0.08     # fastMRI's 4x setting
INK, ACCENT = "#102A43", "#B03030"


def nrmse(image: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((image - target)[mask]) / np.linalg.norm(target[mask]))


def ssim_panels(target: np.ndarray, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Three distortions of the target, each scaled to nRMSE = NRMSE_LEVEL on the brain."""
    mask = target > 1e-6
    scale = NRMSE_LEVEL * np.linalg.norm(target[mask]) / np.sqrt(mask.sum())
    brighter = target + scale * mask
    noise = np.random.default_rng(seed).standard_normal(target.shape) * mask
    noisy = target + noise * NRMSE_LEVEL * np.linalg.norm(target[mask]) / np.linalg.norm(noise[mask])
    low, high = 0.05, 20.0                    # bisection on the blur width
    for _ in range(60):
        sigma = 0.5 * (low + high)
        if nrmse(gaussian_filter(target, sigma) * mask, target, mask) < NRMSE_LEVEL:
            low = sigma
        else:
            high = sigma
    blurred = gaussian_filter(target, sigma) * mask
    return [("target", target), ("brighter", brighter), ("blurred", blurred),
            ("noisy", noisy)]


def draw_ssim(target: np.ndarray, out_dir: Path) -> list[str]:
    panels = ssim_panels(target)
    mask = target > 1e-6
    lines = []
    for step in range(1, len(panels) + 1):
        figure, axes = plt.subplots(1, len(panels), figsize=(6.6, 2.35))
        for index, (axis, (name, image)) in enumerate(zip(axes, panels)):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if index >= step:
                continue                      # laid out, left blank
            axis.imshow(np.rot90(image), cmap="gray", vmin=0.0, vmax=1.0,
                        interpolation="nearest")
            axis.set_title(name, fontsize=13, color=INK, pad=4)
            if index == 0:
                continue
            ssim = structural_similarity(target, image, data_range=1.0)
            axis.set_xlabel(f"nRMSE {nrmse(image, target, mask):.2f}\nSSIM {ssim:.3f}",
                            fontsize=12.5, color=INK, labelpad=3, linespacing=1.35)
            if step == len(panels):
                lines.append(f"{name}: nRMSE {nrmse(image, target, mask):.4f} SSIM {ssim:.4f}")
        figure.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.90, bottom=0.2)
        figure.savefig(out_dir / f"metric_ssim_{step}.pdf", bbox_inches="tight")
        plt.close(figure)
    return lines


def save_image(image: np.ndarray, path: Path, cmap: str = "gray", vmin=0.0, vmax=1.0) -> None:
    plt.imsave(path, np.rot90(image), cmap=cmap, vmin=vmin, vmax=vmax)


def draw_lpips(target: np.ndarray, prediction: np.ndarray, out_dir: Path,
               device: torch.device) -> list[str]:
    import lpips

    model = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()

    def tensor(image):
        return torch.from_numpy(image.copy()).float()[None, None].repeat(1, 3, 1, 1).mul(2).sub(1).to(device)

    y, p = tensor(target), tensor(prediction)
    with torch.no_grad():
        total, layers = model(p, y, retPerLayer=True)
        scalar = lpips.LPIPS(net="alex", verbose=False).to(device).eval()(p, y)
        features = [model.net.forward(model.scaling_layer(t)) for t in (y, p)]
    layer = 0                                 # AlexNet relu1: 64 channels, 1/4 resolution
    fy, fp = (lpips.normalize_tensor(f[layer])[0, :, 2:-2, 2:-2].cpu().numpy()
              for f in features)              # 2-pixel padding border dropped
    channels = np.argsort(fy.reshape(fy.shape[0], -1).var(axis=1))[::-1][:3]
    for rank, channel in enumerate(channels):
        both = np.concatenate([fy[channel].ravel(), fp[channel].ravel()])
        low, high = np.percentile(both, [1, 99])
        for name, feature in (("y", fy), ("p", fp)):
            save_image(feature[channel], out_dir / f"lpips_feat_{name}_{rank + 1}.png",
                       cmap="magma", vmin=low, vmax=high)
    save_image(target, out_dir / "lpips_target.png")
    save_image(prediction, out_dir / "lpips_pred.png")
    distance = total[0, 0].cpu().numpy()
    save_image(distance, out_dir / "lpips_dist.png", cmap="Reds", vmin=0.0,
               vmax=float(np.percentile(distance, 99.5)))
    lines = [f"LPIPS (spatial mean) {distance.mean():.4f}",
             f"LPIPS (lpips package, non-spatial) {float(scalar):.4f}",
             "per layer: " + ", ".join(f"{float(r.mean()):.4f}" for r in layers),
             f"layer {layer + 1} channels shown: {list(map(int, channels))}"]
    (out_dir / "lpips_values.txt").write_text("\n".join(lines) + "\n")
    return lines


def draw_kspace(image: np.ndarray, out_dir: Path) -> None:
    kspace = np.fft.fftshift(np.fft.fft2(image))
    width = image.shape[1]                    # phase encoding along the second axis
    # fastMRI's equispaced rule: the centre lines, plus lines at a spacing chosen so that
    # the total is 1 / ACCELERATION of all lines
    mask = np.zeros(width, dtype=bool)
    centre = int(round(CENTRE_FRACTION * width))
    mask[width // 2 - centre // 2: width // 2 - centre // 2 + centre] = True
    spacing = (width - centre) / (width / ACCELERATION - centre)
    mask[np.round(np.arange(0, width - 1, spacing)).astype(int)] = True
    sampled = kspace * mask[None, :]
    zero_filled = np.abs(np.fft.ifft2(np.fft.ifftshift(sampled)))
    log_full = np.log1p(np.abs(kspace))
    log_sampled = np.log1p(np.abs(sampled))
    top = float(np.percentile(log_full, 99.9))
    panels = [("image", image, "gray", 0.0, 1.0),
              ("k-space", log_full, "gray", 0.0, top),
              (f"{ACCELERATION}\u00d7 undersampled", log_sampled, "gray", 0.0, top),
              ("zero-filled image", zero_filled, "gray", 0.0, 1.0)]
    for step in range(1, len(panels) + 1):
        figure, axes = plt.subplots(1, len(panels), figsize=(7.4, 2.3))
        for index, (axis, (title, data, cmap, vmin, vmax)) in enumerate(zip(axes, panels)):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if index >= step:
                continue
            axis.imshow(np.rot90(data), cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="nearest")
            axis.set_title(title, fontsize=14, color=INK, pad=4)
        figure.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.88, bottom=0.02)
        figure.savefig(out_dir / f"kspace_{step}.pdf", bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "docker/task3/weights/task3.pt")
    parser.add_argument("--subject", default="0006")
    parser.add_argument("--z-index", type=int, default=sum(Z_CLIP_RANGE) // 2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/figures/showcase")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model("conditional", args.checkpoint, device, "")
    (row,) = compute_rows(model, device, args.subject, args.z_index, args.batch_size,
                          [("T1W", "0.1T", "7T")])
    for line in draw_ssim(row["target"], args.out_dir):
        print(line)
    for line in draw_lpips(row["target"], row["prediction"], args.out_dir, device):
        print(line)

    split = Path(get_data_dir()) / "training_prospective"
    slice_3t = load_nifti(volume_path(split, "T1W", "3T", args.subject))[0][:, :, args.z_index]
    rows, columns = brain_bbox([slice_3t], margin=12)
    draw_kspace(slice_3t[rows, columns], args.out_dir)
    print(f"wrote metric_ssim_1..4.pdf, lpips_*.png and kspace_1..4.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
