#!/usr/bin/env python3
"""Widen a single-channel ConditionalUNet checkpoint to multi-contrast input.

The multi-contrast data module feeds (primary, T1W, T2W, T2FLAIR), where channel 0
duplicates the contrast being predicted -- exactly what the single-channel model was fed.
So the pretrained first convolution moves onto channel 0 unchanged and the three auxiliary
channels are zero-initialised: the widened model computes a bit-identical function to the
checkpoint it came from, and the extra contrasts start contributing nothing and can only
add. Every other tensor is copied untouched.

This is the same zero-init discipline as the residual head and the FiLM projections: begin
at a model whose score is already known, so any change in the score is attributable.

Usage:
    python scripts/widen_input_channels.py --checkpoint <in.pt> --output <out.pt> --channels 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

FIRST_CONV = "encoders.0.0.weight"


def widen(state: dict[str, torch.Tensor], channels: int) -> dict[str, torch.Tensor]:
    weight = state[FIRST_CONV]
    if weight.shape[1] == channels:
        return state
    if weight.shape[1] != 1:
        raise ValueError(f"{FIRST_CONV} has {weight.shape[1]} input channels, expected 1")
    widened = torch.zeros(weight.shape[0], channels, *weight.shape[2:], dtype=weight.dtype)
    widened[:, :1] = weight
    out = dict(state)
    out[FIRST_CONV] = widened
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=4)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    key = next((k for k in ("model", "model_state_dict", "state_dict")
                if isinstance(payload, dict) and isinstance(payload.get(k), dict)), None)
    if key is None:
        raise SystemExit(f"{args.checkpoint}: no state dict found")
    before = payload[key][FIRST_CONV].shape
    # Only the weights are carried over. The optimizer state is deliberately dropped: its
    # moment estimates are shaped for the old first convolution and are meaningless for the
    # widened one.
    widened = widen(payload[key], args.channels)
    torch.save({"model": widened}, args.output)
    print(f"{FIRST_CONV}: {tuple(before)} -> {tuple(widened[FIRST_CONV].shape)}")
    print(f"  channel 0 carries the pretrained weight, channels 1..{args.channels - 1} are zero")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
