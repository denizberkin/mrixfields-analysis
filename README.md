# MRIxFields2026 Baseline

Baseline implementations for cross-field brain MRI translation.

## Models

| Method | Paper | Tasks | Key Idea |
|--------|-------|-------|----------|
| **CUT** | Park et al., ECCV 2020 | Task 1, 2 | Single G+D, PatchNCE contrastive loss |
| **CycleGAN** | Zhu et al., ICCV 2017 | Task 1, 2 | Dual G+D, cycle consistency loss |
| **StarGAN v2** | Choi et al., CVPR 2020 | Task 3 | Multi-domain with AdaIN style injection |

Pre-trained checkpoints (all 147 task × method × mode combinations) are available on Hugging Face: [`mrixfields/MRIxFields2026-Baseline`](https://huggingface.co/mrixfields/MRIxFields2026-Baseline).

## Prerequisites

### Install Dependencies

```bash
# From repo root
conda env create -f environment.yml
conda activate mf
```

This installs PyTorch (CUDA 12.1), TensorFlow 2.15, and the `mrixfields` package.

For CUDA 11.8: edit `environment.yml` and replace `cu121` with `cu118` before running.

### Configure Paths

```bash
cp .env.example .env
vim .env  # Edit paths below
```

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Dataset root (contains Training_retrospective, etc.) |
| `PREPROCESSED_DIR` | Extracted 2D slices for training |
| `OUTPUT_DIR` | Training checkpoints and logs |
| `INFERENCE_DIR` | Inference outputs |
| `SYNTHSEG_DIR` | SynthSeg installation (for evaluation) |
| `DEVICE` | GPU device (e.g. `cuda:0`) |

Python scripts auto-load `.env` via `mrixfields.env.load_env()`. For shell commands using `$DATA_DIR` etc., run `source .env` first.

## Quick Start

> The Step 1–5 below are the **local Baseline pipeline** (preprocess → train → inference → segmentation → local evaluation). They are distinct from the top-level [README § 6](../README.md#6-submit) submission sub-steps (segment → pack → zip → upload), which describe what to do *after* this pipeline to assemble the Synapse upload.

### Step 1: Preprocess

Extract 2D axial slices. Input volumes are in [0, 1]; preprocessing casts to float32 and slices. The training transform scales slices to [-1, 1] for tanh-output GANs.

```bash
cd Baseline
python scripts/preprocess.py extract-slices --splits retro_train pro_train
```

### Step 2: Train

```bash
# Task 1 (Any -> 7T): CUT, 0.1T -> 7T
python scripts/train.py \
    --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
    --mode retro_scratch

# Task 2 (0.1T -> Higher): CycleGAN, 0.1T -> 3T
python scripts/train.py \
    --config configs/task2/cyclegan/0.1T_to_3T_T1W.yaml \
    --mode retro_scratch

# Task 3 (Any -> Any): StarGAN v2
python scripts/train.py \
    --config configs/task3/stargan/any_to_any_T1W.yaml \
    --mode retro_scratch
```

### Step 3: Inference

```bash
# Task 1 (CUT, 0.1T -> 7T)
python scripts/inference.py \
    --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
    --checkpoint $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cut/retro_scratch/weights/checkpoint_epoch100.pth \
    --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
    --output_dir $INFERENCE_DIR/

# Task 2 (CycleGAN, 0.1T -> 3T)
python scripts/inference.py \
    --config configs/task2/cyclegan/0.1T_to_3T_T1W.yaml \
    --checkpoint $OUTPUT_DIR/task2_0.1T_to_3T_T1W/cyclegan/retro_scratch/weights/checkpoint_epoch100.pth \
    --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
    --output_dir $INFERENCE_DIR/

# Task 3 (StarGAN v2, 0.1T -> 7T — specify --target_field)
python scripts/inference.py \
    --config configs/task3/stargan/any_to_any_all_modalities.yaml \
    --checkpoint $OUTPUT_DIR/task3_any_to_any_multimodal/stargan_v2/retro_scratch/weights/checkpoint_500000.pth \
    --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
    --output_dir $INFERENCE_DIR/ \
    --target_field 7T
```

> Inference output goes into `$INFERENCE_DIR/` with **source** field tags (e.g. `P_T1W_0.1T_0001.nii.gz`). Synapse expects target field tags in a per-task tree — repack with [Submission/build_submission/](../Submission/build_submission/) before zipping. Do not `zip -r` `$INFERENCE_DIR/` directly.

### Step 4: Segmentation (SynthSeg) — Task 1 / Task 2 only

Run SynthSeg on your predictions to produce segmentation maps for the
Dice and Volume metrics. **Skip this step entirely for Task 3** — Task 3 is
voxel-level only and does not accept segmentation submissions.

`scripts/segment_predictions.py` mirrors `$INFERENCE_DIR/` into
`$PREDICTIONS_SEG_DIR/` (sibling tree of seg NIfTI files), using the same
sweep coordinates as [`Submission/build_submission/`](../Submission/build_submission/):

```bash
# Default: task1 + task2 with build_submission's defaults
python scripts/segment_predictions.py

# Single task / dry-run / force re-segment — see --help for full options
python scripts/segment_predictions.py --tasks task1
python scripts/segment_predictions.py --dry-run
python scripts/segment_predictions.py --overwrite
```

SynthSeg loads once (~30 s), then ~30–60 s per file. 72 files total ≈
30–60 min on GPU. Existing seg outputs are skipped; pass `--overwrite` to
force re-segmentation.

For SynthSeg installation, see [Evaluation/README.md](../Evaluation/README.md). For a generic single-directory wrapper, use [Evaluation/segment.py](../Evaluation/segment.py).

### Step 5: Evaluation

Evaluate your predictions against ground truth:

- **Task 1 / Task 2**: 5 metrics — nRMSE, SSIM, LPIPS (voxel-level) plus Dice, Volume (require Step 4 segmentations).
- **Task 3**: 3 metrics — nRMSE, SSIM, LPIPS only (no segmentations).

See [Evaluation/README.md](../Evaluation/README.md) for evaluation commands and metric details.

Local evaluation is an optional pre-submission sanity check — it's not required by the Synapse evaluator, which scores submissions independently after upload.

## Training Modes

Each config contains both `pretrain` and `finetune` sections. Use `--mode` to select:

| Mode | Step 1 | Step 2 | Use Case |
|------|--------|--------|----------|
| `retro_scratch` | Unpaired on retrospective | - | Pretrain only |
| `pro_scratch` | - | Paired on prospective (from scratch) | Ablation |
| `pro_pretrained` | Unpaired pretrain | Then paired fine-tune | Full pipeline (recommended) |

**Recommended workflow**: use `pro_pretrained` for the strongest baseline
(unpaired pretrain on retrospective + paired finetune on prospective).
`pro_scratch` is for ablation only. `retro_scratch` produces a pretrain-only
checkpoint that can be reused as the starting point for multiple
`pro_pretrained` finetune runs.

Output: `$OUTPUT_DIR/{task_name}/{method}/{mode}/`

## Scripts

| Script | Description |
|--------|-------------|
| `preprocess.py` | Extract 2D slices from NIfTI to npz |
| `train.py` | Unified training (CUT, CycleGAN, StarGAN v2) |
| `inference.py` | Generate predictions from trained models |
| `visualize.py` | Visualize input/prediction/target comparisons |
| `generate_configs.py` | Regenerate all 51 task configs |
| `generate_metadata.py` | Generate dataset metadata tables |

## Directory Structure

```
Baseline/
├── mrixfields/              # Python package
│   ├── data/                #   Dataset classes, transforms
│   ├── models/              #   CUT, CycleGAN, StarGAN v2
│   └── losses/              #   GAN, PatchNCE, LPIPS, SSIM
├── configs/                 # 51 task configs
│   ├── task1/{cut,cyclegan}/
│   ├── task2/{cut,cyclegan}/
│   └── task3/stargan/
├── scripts/                 # Entry-point scripts
└── setup.py                 # Editable install (used by environment.yml)
```

## Config Naming

Format: `{source}_to_{target}_{modality}.yaml`

- Fields: `0.1T`, `1.5T`, `3T`, `5T`, `7T`
- Modalities: `T1W`, `T2W`, `T2FLAIR`

See [configs/README.md](configs/README.md) for details.

## Citation

If you use these baselines, please cite the original method papers:

```bibtex
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}

@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
