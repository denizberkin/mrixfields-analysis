#!/usr/bin/env python3
"""Assert the container's vendored model is the pipeline's model, bit for bit.

docker/task3/mrx/model.py is a copy of experiment-pipeline/components/models/
conditional_unet.py with the registry imports dropped. A copy can drift, and a drift
here would be invisible -- both files define a ConditionalUNet that loads the same
state dict without complaint, and the container would just quietly predict something
else. So compare the two directly: same weights, same input, same domains, and require
max |difference| == 0 on the CPU in float64-free float32.

Also checks the shipped weights are the same tensors as the run's own checkpoint, so
docker/task3/weights/task3.pt cannot silently be an older epoch.

    ~/anaconda3/envs/mri/bin/python scripts/verify_docker_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline"), str(ROOT / "docker" / "task3")]

SHIPPED = ROOT / "docker" / "task3" / "weights" / "task3.pt"
# The mean of epochs 6-8 of task3_mc_ssim_slice -- the weights behind challenge SSIM
# 0.913652, which is what the container has to reproduce.
ORIGINAL = ROOT / "experiment-pipeline/runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt"


def weights_of(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("model", payload)


def build(cls, state):
    model = cls(
        input_channels=int(state["encoders.0.0.weight"].shape[1]),
        base_channels=int(state["encoders.0.0.weight"].shape[0]),
        max_channels=int(state["bottleneck.0.weight"].shape[0]),
        levels=len({key.split(".")[1] for key in state if key.startswith("encoders.")}),
        residual_output="residual_head.weight" in state,
        film_conditioning=any(key.startswith("film_projections.") for key in state),
        slice_conditioning=any(key.startswith("slice_embedding.") for key in state),
    )
    model.load_state_dict(state)
    return model.eval()


def main() -> None:
    shipped = weights_of(SHIPPED)
    original = weights_of(ORIGINAL)
    assert set(shipped) == set(original), "shipped weights have different tensor names"
    drift = [k for k in original if not torch.equal(shipped[k], original[k])]
    assert not drift, f"shipped weights differ from {ORIGINAL.name}: {drift[:5]}"
    print(f"weights: {len(shipped)} tensors identical to {ORIGINAL.name}")

    from components.models.conditional_unet import ConditionalUNet as Pipeline
    from mrx.model import ConditionalUNet as Vendored

    torch.manual_seed(0)
    images = torch.randn(2, int(shipped["encoders.0.0.weight"].shape[1]), 368, 448)
    target = torch.tensor([7, 3])
    source = torch.tensor([2, 11])
    # Non-None, and not 0.5: the slice pathway only runs when a position is passed, so a
    # None here would compare the two models with the new branch switched off in both.
    positions = torch.tensor([0.356, 0.493])

    with torch.inference_mode():
        a = build(Pipeline, shipped)(images, target, source, positions)
        b = build(Vendored, shipped)(images, target, source, positions)

    difference = (a - b).abs().max().item()
    print(f"output: shape {tuple(a.shape)}, max |pipeline - vendored| = {difference:.3e}")
    assert difference == 0.0, "vendored model has drifted from the pipeline model"
    print("OK: the container ships the same function that was scored.")


if __name__ == "__main__":
    main()
