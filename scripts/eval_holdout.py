#!/usr/bin/env python3
"""Score a Task 3 checkpoint on held-out subjects, per transition and per modality.

The challenge reports one modality-averaged number per submission, which cannot separate
a model that works at 0.1 T from one that copies at 5 T, and it only arrives after an
upload. This scores the same three metrics locally on a subject the model never saw.

These are LOCAL metrics. The platform's normalisation is not published, so absolute values
are not leaderboard-comparable -- use them to rank checkpoints against each other, which is
what checkpoint selection and the per-transition breakdown actually need.

Usage:
    python scripts/eval_holdout.py \
        --config experiment-pipeline/configs/task3_tubelet_pro.toml \
        --checkpoint experiment-pipeline/runs/task3_tubelet_pro/artifacts/<ckpt>.pt

    # sweep every checkpoint of a run (A0.5)
    python scripts/eval_holdout.py --config <cfg> --sweep experiment-pipeline/runs/<run>/artifacts
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "experiment-pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
SPLIT_DIRS = {"pro_train": "training_prospective", "pro_val": "Validating_prospective"}


def joint_domain(modality: str, field: str) -> int:
    return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)


def load_config(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def build_model(config: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Import the config's registry paths, build the model, restore weights."""
    from eval_pipeline.core.importing import import_module_from_path
    from eval_pipeline.core.registry import get_registered_component

    config_dir = PIPELINE_ROOT / "configs"
    for entry in config.get("registry", {}).get("paths", []):
        import_module_from_path(str((config_dir / entry).resolve()))

    spec = config["model"]
    factory = get_registered_component("model", spec["name"])(**spec.get("params", {}))
    model = factory.build()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # The trainer writes {"model", "optimizer", "phase", "progress"}; accept the other
    # common spellings too. Loading strictly matters here: a silently empty load still
    # produces plausible-looking metrics, which is worse than a crash.
    state = None
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(candidate, dict) and candidate:
            state = candidate
            break
    if state is None:
        raise ValueError(
            f"{checkpoint}: no state dict under 'model'/'model_state_dict'/'state_dict' "
            f"(top-level keys: {sorted(payload)[:8] if isinstance(payload, dict) else type(payload)})"
        )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def resolve_split_dir(cache_dir: Path, split: str) -> Path:
    """The cache spells the split 'Training_prospective'; configs say 'pro_train'.

    Match case-insensitively rather than hardcoding, and fall back to the only
    subdirectory when there is exactly one, so a rename upstream does not silently
    produce an empty evaluation.
    """
    wanted = SPLIT_DIRS.get(split, split).lower()
    candidates = [p for p in cache_dir.iterdir() if p.is_dir()] if cache_dir.is_dir() else []
    for candidate in candidates:
        if candidate.name.lower() == wanted:
            return candidate
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"no split directory for {split!r} under {cache_dir} (found: "
        f"{[c.name for c in candidates]})"
    )


def cached_volume(split_dir: Path, modality: str, field: str, subject: str):
    directory = split_dir / modality / field
    if not directory.is_dir():
        return None
    matches = sorted(p for p in directory.glob("*.npy") if subject in p.name)
    return np.load(matches[0], mmap_mode="r") if matches else None


def to_multiple(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


@torch.no_grad()
def predict(model, source: np.ndarray, src_domain: int, tgt_domain: int,
            device: torch.device, depth: int, batch: int,
            tta: bool = False) -> np.ndarray:
    """Run the model over a whole volume. depth=0 means a 2D slice-wise model.

    ``source`` is either one volume (D, H, W) or a channel stack (C, D, H, W) for the
    multi-contrast input; the single-volume case is the C=1 stack, so both paths run the
    same code and a 4-channel model cannot silently be fed one channel.
    """
    stack = source if source.ndim == 4 else source[None]
    C, D, H, W = stack.shape
    pad_h, pad_w = to_multiple(H, 16) - H, to_multiple(W, 16) - W
    out = np.empty((D, H, W), dtype=np.float32)
    sd = torch.tensor([src_domain], device=device)
    td = torch.tensor([tgt_domain], device=device)
    chunk = depth if depth else 1

    for start in range(0, D, chunk * batch):
        stop = min(start + chunk * batch, D)
        block = np.asarray(stack[:, start:stop], dtype=np.float32)
        pad_d = (-block.shape[1]) % chunk if depth else 0
        if pad_d:
            block = np.concatenate([block, np.zeros((C, pad_d, H, W), np.float32)], axis=1)
        tensor = torch.from_numpy(block).to(device)
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
        n = tensor.shape[1] // chunk
        # (C, T, H, W) -> (n, C, chunk, H, W) volumetric, or (T, C, H, W) slice-wise.
        tensor = (tensor.view(C, n, chunk, *tensor.shape[-2:]).permute(1, 0, 2, 3, 4)
                  if depth else tensor.permute(1, 0, 2, 3))
        tensor = tensor.mul(2).sub(1)
        with torch.amp.autocast(device.type):
            if tta:
                # Average over the in-plane flips only -- NOT the eight-way dihedral the
                # plan named. Both data modules augment with flips (horizontal_flip for
                # the 2D path, per-axis flip for the volumes), so a flipped brain is in
                # distribution; a 90-degree rotation never is, and on non-square axial
                # slices it would also swap H and W. Averaging over transforms the model
                # was never trained on measures the wrong thing.
                accumulated = None
                for dims in ((), (-1,), (-2,), (-2, -1)):
                    view = torch.flip(tensor, dims) if dims else tensor
                    result = model(view, td.expand(n), sd.expand(n))
                    result = torch.flip(result, dims) if dims else result
                    accumulated = result if accumulated is None else accumulated + result
                pred = accumulated / 4
            else:
                pred = model(tensor, td.expand(n), sd.expand(n))
        pred = pred.float().add(1).div(2).clamp(0, 1)
        pred = pred.reshape(-1, tensor.shape[-2], tensor.shape[-1])[: stop - start]
        out[start:stop] = pred[:, :H, :W].cpu().numpy()
    return out


def score(pred: np.ndarray, target: np.ndarray, lpips_fn, device) -> dict[str, float]:
    from skimage.metrics import structural_similarity

    mask = target > 1e-6
    if not mask.any():
        return {}
    diff = pred[mask] - target[mask]
    nrmse = float(np.linalg.norm(diff) / (np.linalg.norm(target[mask]) + 1e-12))
    ssim = float(np.mean([
        structural_similarity(target[z], pred[z], data_range=1.0)
        for z in range(target.shape[0]) if mask[z].any()
    ]))
    values = []
    with torch.no_grad():
        for start in range(0, pred.shape[0], 16):
            p = torch.from_numpy(pred[start:start + 16]).unsqueeze(1).to(device)
            t = torch.from_numpy(target[start:start + 16]).unsqueeze(1).to(device)
            values.append(float(lpips_fn(p.mul(2).sub(1), t.mul(2).sub(1))))
    return {"nRMSE": nrmse, "SSIM": ssim, "LPIPS": float(np.mean(values))}


def evaluate(model, split_dir: Path, subjects: list[str], device: torch.device,
             depth: int, batch: int, axial_first: bool = False,
             tta: bool = False, multicontrast: bool = False) -> list[dict]:
    """Score one model over every field transition of every modality.

    With ``multicontrast`` the input is the 4-channel (primary, T1W, T2W, T2FLAIR) stack
    of CachedMultiContrastDataset: channel 0 duplicates the contrast being predicted and
    the auxiliaries keep the fixed MODALITIES order, all read at the *source* field.
    """
    from mrixfields.losses.perceptual import PerceptualLoss

    lpips_fn = PerceptualLoss(net="alex").to(device).eval()
    rows = []
    for subject in subjects:
        cache = {m: {f: cached_volume(split_dir, m, f, subject) for f in FIELDS}
                 for m in MODALITIES}
        if axial_first:
            # Match the training layout: (x, y, z) -> (z, x, y), so slabs are cut
            # along the axial axis and SSIM is averaged over axial slices.
            cache = {m: {f: (None if v is None else v.transpose(2, 0, 1))
                         for f, v in per_field.items()}
                     for m, per_field in cache.items()}
        for modality in MODALITIES:
            volumes = cache[modality]
            for src in FIELDS:
                for tgt in FIELDS:
                    if src == tgt or volumes[src] is None or volumes[tgt] is None:
                        continue
                    if model is None:
                        # Identity control: predict the source unchanged. The challenge
                        # scores this at SSIM 0.836497 (submission 9778330), so it anchors
                        # the local scale against a number we can see on the leaderboard.
                        pred = np.asarray(volumes[src], np.float32)
                    else:
                        source = volumes[src]
                        if multicontrast:
                            auxiliary = [cache[m][src] for m in MODALITIES]
                            if any(v is None for v in auxiliary):
                                # A missing contrast is skipped rather than zero-filled:
                                # zero is a legitimate intensity here, so an imputed
                                # channel would be indistinguishable from data.
                                continue
                            source = np.stack([np.asarray(v, np.float32)
                                               for v in [source, *auxiliary]])
                        pred = predict(model, source, joint_domain(modality, src),
                                       joint_domain(modality, tgt), device, depth, batch,
                                       tta=tta)
                    metrics = score(pred, np.asarray(volumes[tgt], np.float32), lpips_fn, device)
                    if metrics:
                        rows.append({"subject": subject, "modality": modality,
                                     "source": src, "target": tgt, **metrics})
                        print(f"  {subject} {modality:8s} {src:5s}->{tgt:5s}  "
                              f"SSIM {metrics['SSIM']:.4f}  nRMSE {metrics['nRMSE']:.4f}  "
                              f"LPIPS {metrics['LPIPS']:.4f}", flush=True)
    return rows


def summarise(rows: list[dict], label: str) -> dict[str, float]:
    means = {k: float(np.mean([r[k] for r in rows])) for k in ("SSIM", "nRMSE", "LPIPS")}
    print(f"\n{label}: SSIM {means['SSIM']:.4f} | nRMSE {means['nRMSE']:.4f} | "
          f"LPIPS {means['LPIPS']:.4f}   ({len(rows)} transitions)", flush=True)
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sweep", type=Path, help="score every *.pt in this directory")
    parser.add_argument("--tta", action="store_true",
                        help="average predictions over the four in-plane flips")
    parser.add_argument("--identity", action="store_true",
                        help="score the source copied unchanged, no model (calibration control)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="defaults to the config's holdout_subjects")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "holdout")
    args = parser.parse_args()
    if not args.checkpoint and not args.sweep and not args.identity:
        parser.error("pass --checkpoint, --sweep or --identity")

    config = load_config(args.config)
    data = config["data"]["params"]
    subjects = args.subjects or [str(s) for s in data.get("holdout_subjects", [])]
    if not subjects:
        parser.error("no holdout_subjects in the config; pass --subjects")

    crop = list(data.get("crop_size", []))
    depth = int(crop[0]) if len(crop) == 3 else 0
    cache_dir = Path(data.get("cache_dir") or "")
    split_dir = resolve_split_dir(cache_dir, str(data.get("prospective_split", "pro_train")))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.identity:
        checkpoints = [None]
    else:
        checkpoints = sorted(args.sweep.glob("*.pt")) if args.sweep else [args.checkpoint]
    # The number of input channels is a property of the model, so read it there rather
    # than from the data section: a config whose model takes 4 channels must be fed 4.
    multicontrast = int(config["model"]["params"].get("input_channels", 1)) > 1
    print(f"held-out subjects: {subjects} | {'volumetric depth %d' % depth if depth else '2D'} "
          f"| {len(checkpoints)} checkpoint(s)"
          f"{' | 4-channel multi-contrast input' if multicontrast else ''}\n", flush=True)

    summary = []
    for path in checkpoints:
        name = "identity" if path is None else path.name
        print(f"=== {name} ===", flush=True)
        model = None if path is None else build_model(config, path, device)
        rows = evaluate(model, split_dir, subjects, device, depth, args.batch,
                        bool(data.get("axial_first", False)), tta=args.tta,
                        multicontrast=multicontrast)
        if not rows:
            print("  no transitions scored -- check cache_dir and subject ids", flush=True)
            continue
        means = summarise(rows, name)
        summary.append({"checkpoint": name, **means})
        stem = "identity" if path is None else path.stem
        with open(args.out_dir / f"{stem}_per_transition.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        del model
        torch.cuda.empty_cache()

    if summary:
        out = args.out_dir / "sweep_summary.csv"
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        best = max(summary, key=lambda r: r["SSIM"])
        print(f"\nbest SSIM: {best['checkpoint']} at {best['SSIM']:.4f}")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
