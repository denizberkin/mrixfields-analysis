"""Distributed training utilities for MRIxFields2026.

Supports single-GPU (default) and multi-GPU via PyTorch DDP.
Multi-GPU is activated by launching with torchrun and passing --dist.

Usage:
    # Single GPU (unchanged)
    python scripts/train.py --config ... --mode retro_scratch

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=2 scripts/train.py --config ... --mode retro_scratch --dist
"""

import os

import torch
import torch.distributed as dist


def init_dist(backend: str = "nccl"):
    """Initialize distributed process group.

    Reads RANK, WORLD_SIZE, LOCAL_RANK from environment (set by torchrun).
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return local_rank


def get_dist_info():
    """Return (rank, world_size). Returns (0, 1) if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_main_process() -> bool:
    """True if this is rank 0 (or non-distributed)."""
    rank, _ = get_dist_info()
    return rank == 0


def cleanup_dist():
    """Clean up distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
