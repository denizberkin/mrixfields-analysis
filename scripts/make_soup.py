#!/usr/bin/env python3
"""Weight-average two fine-tunes that started from the same seed (a "model soup").

`task3_mc_ssim_fine` and `task3_mc_ssim_slice` differ by exactly one config line
(`slice_conditioning`). Both start from `retro_pretrain_big/widened4_from_e25.pt` and run
8 epochs at lr 1e-4 on the same data, so they are short fine-tunes from one converged point
and should still lie in a single basin -- which is what makes averaging their *weights*
legitimate. Averaging their *predictions* is a different operation and one we have already
measured as harmful (-0.00239): two models place an edge a pixel apart and the mean of two
edges is a blur. Averaged weights produce one model that draws one edge.

The slice branch carries four tensors the control has no counterpart for. They are not
simply copied over. The control is exactly a slice-conditioned network whose embedding
output projection is zero -- that is what zero-init means and why the two were comparable
in the first place -- so the well-defined average is against that zero: the final Linear of
`slice_embedding.projection` is scaled by the slice branch's weight, and the layers feeding
it ride along unchanged (they only matter through a projection that has been scaled).

    ~/anaconda3/envs/mri/bin/python scripts/make_soup.py \
        --member runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt:0.5 \
        --member runs/task3_mc_ssim_fine/artifacts/task3_unet_finetune_8.pt:0.5 \
        --out runs/task3_mc_ssim_slice/artifacts/soup_50_50.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Only the tensors that are a function of the conditioning path, not of the image path:
# these are the ones the control implicitly holds at zero.
ZEROED_IN_CONTROL = ("slice_embedding.projection.2.weight", "slice_embedding.projection.2.bias")


def weights_of(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        raise SystemExit(f"{path} is a bare tensor, not a checkpoint")
    state = payload.get("model_ema") or payload.get("model") or payload
    return {k: v for k, v in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", action="append", required=True,
                        help="path:weight, repeatable. Weights are normalised to sum to 1.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    members = []
    for entry in args.member:
        path, _, weight = entry.rpartition(":")
        members.append((Path(path), float(weight)))
    total = sum(w for _, w in members)
    members = [(p, w / total) for p, w in members]

    states = [(weights_of(p), w, p) for p, w in members]
    shared = set.intersection(*(set(s) for s, _, _ in states))
    union = set().union(*(set(s) for s, _, _ in states))
    extra = union - shared
    unexpected = extra - set().union(*(
        {k for k in s if k.startswith("slice_embedding.")} for s, _, _ in states))
    if unexpected:
        raise SystemExit(f"members disagree on keys beyond slice conditioning: {sorted(unexpected)}")

    for key in sorted(shared):
        shapes = {tuple(s[key].shape) for s, _, _ in states}
        if len(shapes) > 1:
            raise SystemExit(f"{key}: incompatible shapes {shapes}")

    souped: dict[str, torch.Tensor] = {}
    for key in sorted(shared):
        reference = states[0][0][key]
        if not reference.is_floating_point():
            # Integer buffers (num_batches_tracked and friends) are counters, not
            # parameters; a fractional mean of two counters is meaningless.
            values = {int(s[key].reshape(-1)[0]) for s, _, _ in states} if reference.numel() else {0}
            if len(values) > 1:
                raise SystemExit(f"{key}: integer buffers differ {values}, refusing to average")
            souped[key] = reference.clone()
            continue
        acc = torch.zeros_like(reference, dtype=torch.float64)
        for state, weight, _ in states:
            acc += state[key].to(torch.float64) * weight
        souped[key] = acc.to(reference.dtype)

    for key in sorted(extra):
        owners = [(s, w, p) for s, w, p in states if key in s]
        if len(owners) != 1:
            raise SystemExit(f"{key}: expected exactly one owner, got {len(owners)}")
        state, weight, path = owners[0]
        scale = weight if key in ZEROED_IN_CONTROL else 1.0
        souped[key] = (state[key].to(torch.float64) * scale).to(state[key].dtype)
        print(f"  {key}: from {path.name} x{scale:g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": souped}, args.out)
    print(f"{len(souped)} tensors ({len(shared)} averaged, {len(extra)} carried) -> {args.out}")
    for path, weight in members:
        print(f"  {weight:.3f}  {path}")


if __name__ == "__main__":
    main()
