# Training Configurations

51 YAML configs for all baseline models across three tasks and three modalities.

Configs contain **only model and training hyperparameters**. All paths (data, output, device) are configured in `.env` at the repo root. See `.env.example`.

## Overview

| Task | Method | Pairs | Configs |
|------|--------|-------|---------|
| Task 1 (Any -> 7T) | CUT | 4 source fields x 3 modalities | 12 |
| Task 1 (Any -> 7T) | CycleGAN | 4 source fields x 3 modalities | 12 |
| Task 2 (0.1T -> Higher) | CUT | 4 target fields x 3 modalities | 12 |
| Task 2 (0.1T -> Higher) | CycleGAN | 4 target fields x 3 modalities | 12 |
| Task 3 (Any -> Any) | StarGAN v2 | 1 (all fields) x 3 modalities | 3 |
| | | **Total** | **51** |

## Directory Layout

```text
configs/
├── task1/                              # Task 1: Any → 7T
│   ├── cut/                            #   CUT (12 configs: 4 pairs × 3 modalities)
│   │   ├── 0.1T_to_7T_T1W.yaml
│   │   ├── 0.1T_to_7T_T2W.yaml
│   │   ├── 0.1T_to_7T_T2FLAIR.yaml
│   │   ├── 1.5T_to_7T_{T1W,T2W,T2FLAIR}.yaml
│   │   ├── 3T_to_7T_{T1W,T2W,T2FLAIR}.yaml
│   │   └── 5T_to_7T_{T1W,T2W,T2FLAIR}.yaml
│   └── cyclegan/                       #   CycleGAN (12 configs, same pairs)
│       └── ...
├── task2/                              # Task 2: 0.1T → Higher
│   ├── cut/                            #   CUT (12 configs: 4 targets × 3 modalities)
│   │   ├── 0.1T_to_1.5T_{T1W,T2W,T2FLAIR}.yaml
│   │   ├── 0.1T_to_3T_{T1W,T2W,T2FLAIR}.yaml
│   │   ├── 0.1T_to_5T_{T1W,T2W,T2FLAIR}.yaml
│   │   └── 0.1T_to_7T_{T1W,T2W,T2FLAIR}.yaml
│   └── cyclegan/                       #   CycleGAN (12 configs)
│       └── ...
└── task3/                              # Task 3: Any → Any
    └── stargan/                        #   StarGAN v2 (3 configs)
        ├── any_to_any_T1W.yaml
        ├── any_to_any_T2W.yaml
        └── any_to_any_T2FLAIR.yaml
```

## Config Structure

Each config contains both `pretrain` and `finetune` sections. Use `--mode retro_scratch/pro_scratch/pro_pretrained` at training time to select which steps to run.

The mode name encodes data split × init source:

| Mode | Data | Init | What it does |
|------|------|------|--------------|
| `retro_scratch` | **Retro**spective (unpaired) | **Scratch** | Pretrain only — unpaired training on `Training_retrospective/`. Output is reusable as the starting checkpoint for multiple `pro_pretrained` runs. |
| `pro_scratch` | **Pro**spective (paired) | **Scratch** | Supervised paired training on `Training_prospective/` from random init. Ablation-only. |
| `pro_pretrained` | **Pro**spective (paired) | **Pretrained** | Supervised paired fine-tune on `Training_prospective/`, starting from a `retro_scratch` checkpoint. Full pipeline; **recommended**. |

Top-level keys:

| Key | Description |
|-----|-------------|
| `task` | Task number (1, 2, or 3) |
| `task_name` | Unique name used for output directory (e.g. `task1_0.1T_to_7T_T1W`) |
| `method` | Model type: `cut`, `cyclegan`, or `stargan_v2` |
| `model` | Architecture hyperparameters (method-specific) |
| `domains` | (StarGAN v2 only) List of field strengths `[0.1T, 1.5T, 3T, 5T, 7T]` |
| `data` | Source/target fields, modalities, crop_size, slice_axis |
| `pretrain` | Unpaired pretraining settings (split, lr, epochs, batch_size, etc.) |
| `finetune` | Paired fine-tuning settings (split, lr, epochs, loss weights) |
| `evaluation` | Metrics list for auto-evaluation |

## Naming Convention

Format: `task{N}/{method}/{source}_to_{target}_{modality}.yaml`

- Field strengths: `0.1T`, `1.5T`, `3T`, `5T`, `7T`
- Modalities: `T1W`, `T2W`, `T2FLAIR`

## Regenerating Configs

To modify default hyperparameters for all configs at once:

```bash
python scripts/generate_configs.py
```
