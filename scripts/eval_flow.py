#!/usr/bin/env python3
"""Score a conditional flow matching checkpoint (section 38).

Separate from ``scripts/eval_holdout.py`` because the calling convention differs in two
ways that would have meant a second code path inside it either way: the network takes
``(x, t, target, source)`` rather than ``(image, target, source)``, and a prediction is an
ODE integration rather than a forward pass.

What is *not* re-implemented is the measurement. ``cached_volume``, ``score`` and
``summarise`` are imported from eval_holdout, so SSIM here is the same per-slice,
data_range=1.0 average that produced every other number in this project. Section 20.1 cost
0.006 SSIM to a scoring path that had quietly drifted from the training one; re-deriving
the metric would be the same mistake with extra steps.

Step count is not a detail: section 38.6 measured the preference inverting with adversarial
refinement. Score a **pre-GAN** checkpoint at ``--steps 1`` and a **refined** one at
``--steps 5``; using 5 on a pre-GAN model understates it (their 0.928 vs 0.914), and 1 on a
refined model collapses it (0.909 -> 0.817).

    ~/anaconda3/envs/mri/bin/python scripts/eval_flow.py \
        --checkpoint experiment-pipeline/runs/task3_cfm/artifacts/task3_flow_20.pt \
        --subjects 0006 0007 0009 --steps 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "experiment-pipeline"
# This file's own directory is included deliberately: `eval_holdout` is a sibling script,
# not a package module, so it is only importable because sys.path[0] happens to be
# scripts/ when this is run directly. Naming it makes the import work however it is loaded.
for entry in (str(REPO_ROOT / "scripts"), str(PIPELINE_ROOT), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from eval_holdout import cached_volume, resolve_split_dir, score, summarise  # noqa: E402
from mrixfields.data.utils import FIELD_STRENGTHS, MODALITIES, get_joint_domain  # noqa: E402

CROP_MULTIPLE = 16


def build(checkpoint: Path, device: torch.device, use_ema: bool = True):
    """Construct the velocity net from the checkpoint's own tensors and load it.

    Width, depth, channel count and the domain-table size are all read back off the state
    dict rather than taken from a config: the config that trained a checkpoint is not
    always the config at hand, and a mismatch should fail in load_state_dict rather than
    quietly build a differently-shaped model.
    """
    from components.models.conditional_flow_unet import ConditionalFlowUNet

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    key = "model_ema" if use_ema and "model_ema" in payload else "model"
    state = payload[key] if key in payload else payload
    channels = int(state["encoders.0.0.weight"].shape[1])
    base_channels = int(state["encoders.0.0.weight"].shape[0])
    max_channels = int(state["bottleneck.0.weight"].shape[0])
    levels = len({k.split(".")[1] for k in state if k.startswith("encoders.")})
    num_domains = int(state["source_embedding.weight"].shape[0])
    film = any(k.startswith("film_projections.") for k in state)
    print(f"{checkpoint.name} [{key}]: channels={channels} base={base_channels} "
          f"max={max_channels} levels={levels} num_domains={num_domains} film={film}",
          flush=True)
    model = ConditionalFlowUNet(channels=channels, num_domains=num_domains,
                                base_channels=base_channels, max_channels=max_channels,
                                levels=levels, film_conditioning=film)
    model.load_state_dict(state)
    return model.to(device).eval(), num_domains


def domains(modalities, field: str, num_domains: int, device):
    """Conditioning indices for a whole batch: joint (modality, field), or plain field."""
    if num_domains == len(MODALITIES) * len(FIELD_STRENGTHS):
        values = [get_joint_domain(m, field) for m in modalities]
        return torch.tensor(values, device=device).unsqueeze(0)
    return torch.tensor([FIELD_STRENGTHS.index(field)], device=device)


@torch.no_grad()
def integrate(model, stack: np.ndarray, source_domain, target_domain,
              device: torch.device, batch: int, steps: int) -> np.ndarray:
    """Heun-integrate every axial slice of a [C, D, H, W] volume. Returns [C, D, H, W]."""
    from components.models.conditional_flow_unet import heun_sample

    C, D, H, W = stack.shape
    pad_h = (-H) % CROP_MULTIPLE
    pad_w = (-W) % CROP_MULTIPLE
    out = np.empty((C, D, H, W), dtype=np.float32)
    for start in range(0, D, batch):
        stop = min(start + batch, D)
        block = np.asarray(stack[:, start:stop], dtype=np.float32)
        tensor = torch.from_numpy(block).to(device).permute(1, 0, 2, 3)   # [n, C, H, W]
        # Centred, not bottom-right. The data modules crop every training slice with
        # CenterCropOrPad((368, 448)) and make_task3_submission.py does the same, so an
        # asymmetric pad puts the head at an offset the model has never seen -- and
        # InstanceNorm normalises over the whole plane, so the shift is not confined to
        # the border. Worth ~2e-3 mean intensity against the submission path.
        top, left = pad_h // 2, pad_w // 2
        tensor = torch.nn.functional.pad(tensor, (left, pad_w - left, top, pad_h - top))
        tensor = tensor.mul(2).sub(1)
        n = tensor.shape[0]
        with torch.amp.autocast(device.type):
            predicted = heun_sample(model, tensor, target_domain.expand(n, -1)
                                    if target_domain.dim() == 2 else target_domain.expand(n),
                                    source_domain.expand(n, -1)
                                    if source_domain.dim() == 2 else source_domain.expand(n),
                                    steps=steps)
        predicted = predicted.float().add(1).div(2).clamp(0, 1)
        out[:, start:stop] = predicted[:, :, top:top + H, left:left + W].permute(1, 0, 2, 3).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["0006", "0007", "0009"])
    parser.add_argument("--steps", type=int, default=1,
                        help="Heun steps. 1 for a pre-GAN model, 5 after refinement (38.6)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw", action="store_true", help="score the raw weights, not the EMA")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/home/ruru/Documents/deniz/mri_fields/data_preprocessed")
                        / "volumes_3d")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "flow")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, num_domains = build(args.checkpoint, device, use_ema=not args.raw)
    split_dir = resolve_split_dir(args.cache_dir, "pro_train")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from mrixfields.losses.perceptual import PerceptualLoss

    lpips_fn = PerceptualLoss(net="alex").to(device).eval()
    print(f"subjects {args.subjects} | {args.steps} Heun step(s) | "
          f"{'joint (modality, field)' if num_domains == 15 else 'field'} conditioning\n",
          flush=True)

    rows = []
    for subject in args.subjects:
        volumes = {(m, f): cached_volume(split_dir, m, f, subject)
                   for m in MODALITIES for f in FIELD_STRENGTHS}
        for src in FIELD_STRENGTHS:
            for tgt in FIELD_STRENGTHS:
                if src == tgt:
                    continue
                stack = [volumes[(m, src)] for m in MODALITIES]
                targets = [volumes[(m, tgt)] for m in MODALITIES]
                if any(v is None for v in stack + targets):
                    continue
                source = np.stack([np.asarray(v, np.float32) for v in stack])
                predicted = integrate(
                    model, source,
                    domains(MODALITIES, src, num_domains, device),
                    domains(MODALITIES, tgt, num_domains, device),
                    device, args.batch, args.steps)
                # Scored per contrast, so the numbers sit on the same scale as every other
                # table in this project rather than being averaged over channels first.
                for index, modality in enumerate(MODALITIES):
                    metrics = score(predicted[index],
                                    np.asarray(targets[index], np.float32), lpips_fn, device)
                    if metrics:
                        rows.append({"subject": subject, "modality": modality,
                                     "source": src, "target": tgt, **metrics})
                        print(f"  {subject} {modality:8s} {src:5s}->{tgt:5s}  "
                              f"SSIM {metrics['SSIM']:.4f}  nRMSE {metrics['nRMSE']:.4f}  "
                              f"LPIPS {metrics['LPIPS']:.4f}", flush=True)

    if not rows:
        raise SystemExit("no transitions scored -- check --cache-dir and the subject ids")
    means = summarise(rows, f"{args.checkpoint.name} @ {args.steps} Heun step(s)")
    destination = args.out_dir / f"{args.checkpoint.stem}_steps{args.steps}.csv"
    with open(destination, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {destination}", flush=True)
    return means


if __name__ == "__main__":
    main()
