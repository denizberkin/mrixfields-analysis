#!/usr/bin/env python3
"""Score an ensemble of Task 3 2D checkpoints, and each member, on the same transitions.

Every single-model lever measured so far (TTA +0.0008, checkpoint averaging +0.0009,
retrospective pretraining +0.0021, FiLM+residual -0.0043 at convergence) fell short of
`task3_unet_pro`. Prediction averaging is the one combination not yet tried, and it is the
only one whose gain grows with how *differently* the members fail rather than how good the
best member is: section 17 found FiLM+residual beats plain on a different 17/60 transitions
than plain wins, which is exactly the decorrelation an ensemble converts into a gain.

Members may have different architectures. Each is rebuilt from its own state dict via
`unet_from_state_dict`, so a plain Tanh checkpoint and a FiLM+residual one ensemble together
without a config per member -- the widths and both flags are read from the tensors.

Averaging happens in the [0, 1] output space, after `predict` has undone the tanh range and
clamped, so a residual-head member and a Tanh member contribute on the same scale.

Usage:
    python scripts/ensemble_eval.py --config <eval.toml> --subjects 0009 \
        --checkpoint runs/task3_unet_pro/artifacts/task3_unet_finetune_60.pt \
        --checkpoint runs/task3_film_long/artifacts/task3_unet_finetune_20.pt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "experiment-pipeline"
for p in (str(PIPELINE_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("eval_holdout", REPO_ROOT / "scripts" / "eval_holdout.py")
EH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EH)


def load_member(path: Path, device: torch.device):
    """Rebuild one member from its own checkpoint, architecture read off the tensors."""
    from components.models.conditional_unet import unet_from_state_dict

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = None
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(candidate, dict) and candidate:
            state = candidate
            break
    if state is None:
        raise ValueError(f"{path}: no state dict under model/model_state_dict/state_dict")
    return unet_from_state_dict(state).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="repeat once per ensemble member")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--members", action="store_true",
                        help="also score each member alone, on the same transitions")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "ensemble")
    args = parser.parse_args()

    from mrixfields.losses.perceptual import PerceptualLoss

    config = EH.load_config(args.config)
    data = config["data"]["params"]
    subjects = args.subjects or [str(s) for s in data.get("holdout_subjects", [])]
    if not subjects:
        parser.error("no holdout_subjects in the config; pass --subjects")
    split_dir = EH.resolve_split_dir(Path(data.get("cache_dir") or ""),
                                     str(data.get("prospective_split", "pro_train")))
    axial_first = bool(data.get("axial_first", False))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_member(p, device) for p in args.checkpoint]
    names = [p.parent.parent.name + "/" + p.stem for p in args.checkpoint]
    print(f"ensemble of {len(models)}: " + ", ".join(names), flush=True)
    print(f"subjects: {subjects}\n", flush=True)

    lpips_fn = PerceptualLoss(net="alex").to(device).eval()
    # column order: the ensemble first, then members, so the CSV reads best-first
    columns = ["ensemble"] + (names if args.members else [])
    rows = []

    for subject in subjects:
        for modality in EH.MODALITIES:
            volumes = {f: EH.cached_volume(split_dir, modality, f, subject) for f in EH.FIELDS}
            if axial_first:
                # Match eval_holdout: (x, y, z) -> (z, x, y) so SSIM is averaged over axial
                # slices, which is the plane the training layout and the challenge use.
                # Omitting this scores a different anatomical plane and silently produces
                # numbers that cannot be compared with any other report in this repo.
                volumes = {f: (None if v is None else v.transpose(2, 0, 1))
                           for f, v in volumes.items()}
            for src in EH.FIELDS:
                for tgt in EH.FIELDS:
                    if src == tgt or volumes[src] is None or volumes[tgt] is None:
                        continue
                    source = np.asarray(volumes[src], np.float32)
                    target = np.asarray(volumes[tgt], np.float32)
                    sd, td = EH.joint_domain(modality, src), EH.joint_domain(modality, tgt)

                    preds = [EH.predict(m, source, sd, td, device, 0, args.batch)
                             for m in models]
                    row = {"subject": subject, "modality": modality,
                           "source": src, "target": tgt}
                    mean_pred = np.mean(preds, axis=0)
                    ens = EH.score(mean_pred, target, lpips_fn, device)
                    if not ens:
                        continue
                    row["ensemble_SSIM"] = ens["SSIM"]
                    row["ensemble_nRMSE"] = ens["nRMSE"]
                    row["ensemble_LPIPS"] = ens["LPIPS"]
                    line = f"  {subject} {modality:8s} {src:5s}->{tgt:5s}  ens {ens['SSIM']:.4f}"
                    if args.members:
                        for name, pred in zip(names, preds):
                            one = EH.score(pred, target, lpips_fn, device)
                            row[f"{name}_SSIM"] = one["SSIM"]
                            line += f" | {name.split('/')[0]} {one['SSIM']:.4f}"
                    rows.append(row)
                    print(line, flush=True)

    if not rows:
        print("no transitions scored -- check cache_dir and subject ids")
        return

    with open(args.out_dir / "per_transition.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'model':45s} {'SSIM':>8s}  {'vs ensemble':>12s}")
    ens_mean = float(np.mean([r["ensemble_SSIM"] for r in rows]))
    print(f"{'ENSEMBLE':45s} {ens_mean:8.5f}  {'--':>12s}")
    for name in (names if args.members else []):
        key = f"{name}_SSIM"
        if key not in rows[0]:
            continue
        m = float(np.mean([r[key] for r in rows]))
        print(f"{name:45s} {m:8.5f}  {m - ens_mean:+12.5f}")
    print(f"\n({len(rows)} transitions)  wrote {args.out_dir / 'per_transition.csv'}")


if __name__ == "__main__":
    main()
