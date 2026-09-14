#!/usr/bin/env python3
"""Copy a run checkpoint into the container as weights only.

docker/task3/weights/task3.pt is covered by .gitignore's ``*.pt``, so a fresh clone has
to regenerate it. The optimizer state in a run artifact is ~70 MB of moment estimates the
container has no use for, so only the model tensors are carried across -- and they are
compared afterwards, because a silently truncated copy would still load.

    ~/anaconda3/envs/mri/bin/python scripts/export_docker_weights.py
    ~/anaconda3/envs/mri/bin/python scripts/export_docker_weights.py --checkpoint <other.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "experiment-pipeline/runs/task3_mc_ssim_fine/artifacts/task3_unet_finetune_8.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=ROOT / "docker/task3/weights/task3.pt")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, args.output)

    back = torch.load(args.output, map_location="cpu", weights_only=True)["model"]
    assert set(back) == set(state), "tensor names changed in the copy"
    drift = [k for k in state if not torch.equal(back[k], state[k])]
    assert not drift, f"tensors changed in the copy: {drift[:5]}"

    print(f"{args.checkpoint.stat().st_size / 1e6:.1f} MB -> "
          f"{args.output.stat().st_size / 1e6:.1f} MB")
    print(f"{len(state)} tensors, all bit-identical -> {args.output}")


if __name__ == "__main__":
    main()
