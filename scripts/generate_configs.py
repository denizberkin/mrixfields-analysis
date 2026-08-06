"""Generate all training configs in unified format.

Each yaml contains both pretrain and finetune sections.
Use --mode retro_scratch/pro_scratch/pro_pretrained at training time to select which steps to run.

Paths (data_dir, preprocessed_dir, device, etc.) are NOT stored in configs.
They are loaded from .env at runtime. See .env.example in the repo root.

Directory structure:
    configs/task{N}/{method}/{src}_to_{tgt}_{mod}.yaml
    configs/task3/stargan/any_to_any_{mod}.yaml

Config counts:
    CUT:      4 pairs x 3 mods x 2 tasks = 24
    CycleGAN: 4 pairs x 3 mods x 2 tasks = 24
    StarGAN:  3 mods                      =  3
    Total: 51 configs

Usage:
    python scripts/generate_configs.py
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.data.utils import MODALITIES

# 5 field strengths
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FIELD_SHORT = {"0.1T": "0.1T", "1.5T": "1.5T", "3T": "3T", "5T": "5T", "7T": "7T"}

TASK1_PAIRS = [(fs, "7T") for fs in FIELDS if fs != "7T"]
TASK2_PAIRS = [("0.1T", fs) for fs in FIELDS if fs != "0.1T"]


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _fs(field):
    return FIELD_SHORT[field]


def make_cut_config(source, target, task, modality):
    name = f"task{task}_{_fs(source)}_to_{_fs(target)}_{modality}"
    return {
        "task": task,
        "task_name": name,
        "method": "cut",
        "model": {
            "input_nc": 1, "output_nc": 1,
            "ngf": 64, "ndf": 64, "n_blocks": 9,
            "nce_layers": [0, 4, 8, 12, 16], "nce_T": 0.07, "num_patches": 256,
            "lambda_GAN": 1.0, "lambda_NCE": 1.0, "nce_idt": True,
            "gan_mode": "lsgan", "netF_nc": 256,
        },
        "data": {
            "source_field": source, "target_field": target,
            "modalities": [modality],
            "crop_size": None, "slice_axis": 2,
        },
        "pretrain": {
            "split": "retro_train",
            "batch_size": 8, "num_workers": 4,
            "lr": 0.0002, "beta1": 0.5, "beta2": 0.999,
            "lr_policy": "linear", "n_epochs": 100, "n_epochs_decay": 100,
            "save_every": 10, "print_every": 1000,
        },
        "finetune": {
            "split": "pro_train",
            "batch_size": 8, "num_workers": 4,
            "lr": 5e-5, "beta1": 0.5, "beta2": 0.999,
            "epochs": 50, "save_every": 10,
            "loss_l1": 1.0, "loss_lpips": 0.1, "loss_ssim": 0.0, "loss_edge": 0.0,
        },
        "evaluation": {
            "metrics": ["nrmse", "ssim", "lpips"],
        },
    }


def make_cyclegan_config(source, target, task, modality):
    name = f"task{task}_{_fs(source)}_to_{_fs(target)}_{modality}"
    return {
        "task": task,
        "task_name": name,
        "method": "cyclegan",
        "model": {
            "input_nc": 1, "output_nc": 1,
            "ngf": 64, "ndf": 64, "n_blocks": 9,
            "lambda_cycle": 10.0, "lambda_idt": 0.5,
            "pool_size": 50, "gan_mode": "lsgan",
        },
        "data": {
            "source_field": source, "target_field": target,
            "modalities": [modality],
            "crop_size": None, "slice_axis": 2,
        },
        "pretrain": {
            "split": "retro_train",
            "batch_size": 8, "num_workers": 4,
            "lr": 0.0002, "beta1": 0.5,
            "lr_policy": "linear", "n_epochs": 100, "n_epochs_decay": 100,
            "save_every": 10, "print_every": 1000,
        },
        "finetune": {
            "split": "pro_train",
            "batch_size": 8, "num_workers": 4,
            "lr": 5e-5, "beta1": 0.5, "beta2": 0.999,
            "epochs": 50, "save_every": 10,
            "loss_l1": 1.0, "loss_lpips": 0.1, "loss_ssim": 0.0, "loss_edge": 0.0,
        },
        "evaluation": {
            "metrics": ["nrmse", "ssim", "lpips"],
        },
    }


def make_stargan_config(modality):
    name = f"task3_any_to_any_{modality}"
    return {
        "task": 3,
        "task_name": name,
        "method": "stargan_v2",
        "model": {
            "img_size": 512, "style_dim": 64, "latent_dim": 16,
            "num_domains": len(FIELDS), "max_conv_dim": 512, "input_nc": 1,
        },
        "domains": list(FIELDS),
        "data": {
            "modalities": [modality],
            "crop_size": [512, 512], "slice_axis": 2,
        },
        "pretrain": {
            "split": "retro_train",
            "batch_size": 8, "num_workers": 4,
            "total_iters": 200000, "lr": 0.0001, "f_lr": 1e-6,
            "beta1": 0.0, "beta2": 0.99, "weight_decay": 0.0001,
            "lambda_reg": 1.0, "lambda_sty": 1.0, "lambda_ds": 1.0, "lambda_cyc": 1.0,
            "ds_iter": 100000, "ema_beta": 0.999,
            "save_every": 10000, "print_every": 1000,
        },
        "finetune": {
            "split": "pro_train",
            "batch_size": 8, "num_workers": 4,
            "lr": 5e-5, "beta1": 0.5, "beta2": 0.999,
            "epochs": 50, "save_every": 10,
            "loss_l1": 1.0, "loss_lpips": 0.1, "loss_ssim": 0.0, "loss_edge": 0.0,
        },
        "evaluation": {
            "metrics": ["nrmse", "ssim", "lpips"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MRIxFields2026 training configs")
    parser.add_argument("--output_dir", type=str, default="configs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    count = 0

    print(f"Fields:     {[_fs(f) for f in FIELDS]}")
    print(f"Modalities: {MODALITIES}")
    print()

    all_pairs = {"task1": TASK1_PAIRS, "task2": TASK2_PAIRS}

    for method, make_fn in [("cut", make_cut_config), ("cyclegan", make_cyclegan_config)]:
        print(f"{method.upper()}:")
        for task_key, pairs in all_pairs.items():
            task_num = int(task_key[-1])
            for mod in MODALITIES:
                for src, tgt in pairs:
                    cfg = make_fn(src, tgt, task_num, mod)
                    p = out / task_key / method / f"{_fs(src)}_to_{_fs(tgt)}_{mod}.yaml"
                    write_config(p, cfg)
                    print(f"  {p}")
                    count += 1

    print("STARGAN:")
    for mod in MODALITIES:
        cfg = make_stargan_config(mod)
        p = out / "task3" / "stargan" / f"any_to_any_{mod}.yaml"
        write_config(p, cfg)
        print(f"  {p}")
        count += 1

    print(f"\nGenerated {count} configs in {out}/")


if __name__ == "__main__":
    main()
