# mrixfields-analysis

Analysis and experiment workspace for the MRIxFields 2026 challenge (cross-field
brain MRI translation: given a brain MRI at one field strength/modality, synthesise
it at another). Team: Berkin Deniz Kahya, Furkan Yüceyalçın, Racha Badreddine,
Ahmet Zelka.

Field strengths: `0.1T`, `1.5T`, `3T`, `5T`, `7T`. Modalities: `T1W`, `T2W`,
`T2FLAIR`. 15 domains = 5 fields × 3 modalities.

This repo hosts the shared `mrixfields` package, generated baseline configs
(CUT/CycleGAN/StarGAN v2), analysis/reporting scripts, and two git submodules:
`experiment-pipeline` (where current training actually happens) and `SynthSeg`
(segmentation metrics). For the current research plan and status, see
[`TODO.md`](TODO.md).

## Status

Task 3, modality-averaged, best-per-model (see `baseline_experiment_logs.csv`
and `reports/` for the full breakdown). Lower is better for nRMSE/LPIPS,
higher is better for SSIM.

| Model | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| INPUT (identity control) | 0.5216 | 0.8365 | 0.1573 |
| StarGAN v2 (challenge baseline) | 0.3436 | 0.7397 | 0.1545 |
| FPS-Former (203 ep) | 0.2471 | 0.8795 | 0.1492 |
| Conditional U-Net (50 ep) | 0.2412 | 0.8976 | 0.0895 |
| U-Net 3D + tubelet LeJEPA encoder | 0.2383 | 0.8893 | 0.0889 |
| **U-Net 3D (10 ep)** | **0.2329** | **0.9026** | **0.0816** |

Current leader is the plain 3D U-Net at only 10 epochs, ahead of both the
conditional variant and the LeJEPA-pretrained encoder. See [`TODO.md`](TODO.md)
for why, and what's being tried next.

## Setup

**Python environment** already exists — do not `conda env create`. There is no
`environment.yml` in this repo.

```bash
~/anaconda3/envs/mri/bin/python --version   # 3.13, torch 2.13.0+cu126 (CUDA), mlflow 3.14.0
```

**LaTeX**: `~/anaconda3/envs/tex/bin/tectonic` (no system `pdflatex`/`latexmk`; see
Reports below).

**Environment variables**: a real `.env` already sits at the repo root (there is
no `.env.example` to copy from). It is loaded automatically by
`mrixfields.env.load_env()` in Python; for shell commands using `$DATA_DIR`
etc., run `source .env` first.

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Dataset root |
| `PREPROCESSED_DIR` | Extracted 2D slices for training |
| `OUTPUT_DIR` | Training checkpoints and logs |
| `INFERENCE_DIR` | Inference outputs |
| `SUBMISSION_DIR` | Assembled submission archives |

**Submodules**: `experiment-pipeline` and `SynthSeg` are git submodules — after
cloning, run `git submodule update --init --recursive`.

## Repo layout

```
mrixfields/                 # installed Python package (editable, setup.py)
├── data/                   #   dataset classes, transforms, metadata
├── models/                 #   CUT, CycleGAN, StarGAN v2, networks
├── losses/                 #   adversarial, patchnce, perceptual, structure
├── env.py                  #   .env loader (load_env())
├── utils_dist.py
└── zclip_constants.py

experiment-pipeline/        # git submodule: denizberkin/experiment-pipeline
├── components/             #   Task 3 models/data/losses/training
│   ├── data/                #   task3.py, task3_volume.py
│   ├── models/              #   conditional_unet.py, conditional_swin_unetr.py,
│   │                        #   conditional_vit.py, unconditional_unet.py, vanilla_unet.py
│   ├── losses/               #   reconstruction.py
│   └── training/             #   task3.py
├── configs/task3_*.toml    #   TOML experiment configs
└── runs/<name>/artifacts/  #   checkpoints (*.pt) land here

SynthSeg/                   # git submodule: denizberkin/SynthSeg (Dice/Volume metrics, Tasks 1-2)

configs/                    # 51 generated baseline task configs
├── task1/{cut,cyclegan}/
├── task2/{cut,cyclegan}/
├── task3/stargan/
└── submission_manifests/

scripts/                    # 19 entry-point scripts, notably:
├── preprocess.py             # extract 2D slices
├── inference.py
├── segment_predictions.py    # SynthSeg over predictions (Tasks 1/2)
├── spectral_analysis.py      # RAPSD / power-law alpha fits
├── make_task3_submission.py  # build + validate the Task 3 ZIP
├── make_source_submission.py # identity-control archive
├── make_report_tables.py / make_slide_figures.py / spectral_figures.py
├── clean_latex.py
├── generate_configs.py
└── visualize*.py

reports/                    # LaTeX sources + tracked figures/tables/spectral outputs
docs/metric_formulas.md
tests/test_custom_submission_inference.py
baseline_experiment_logs.csv   # gitignored local mirror of the shared results sheet
TODO.md                        # research plan / status tracker
```

`lejepa_pretraining/` is gitignored: a sparse checkout of `furkanycy/MRIxFields`
(the LeJEPA tubelet encoder work) under `repo/`, its deploy key under
`.ssh_keys/`, and the pretrained encoder artifact under `artifacts/`.

## Common workflows

### Train (Task 3, experiment-pipeline)

All current training runs through the `eval_pipeline` TOML runner. **Run these
commands from inside `experiment-pipeline/`**, not the repo root (see Gotchas).

```bash
cd experiment-pipeline

# 1. Validate the config and imports first
~/anaconda3/envs/mri/bin/python -m eval_pipeline validate configs/task3_unet_pro.toml --check-imports

# 2. Run training + validation (the default --stages is "test" only, so specify explicitly)
~/anaconda3/envs/mri/bin/python -m eval_pipeline run configs/task3_unet_pro.toml --stages training validation
```

Checkpoints land in `experiment-pipeline/runs/<name>/artifacts/*.pt`. Other
Task 3 configs live alongside `task3_unet_pro.toml` in `experiment-pipeline/configs/`
(unconditional, vanilla, ViT, Swin UNETR, retro/pro finetune variants).

### Build a Task 3 submission

```bash
~/anaconda3/envs/mri/bin/python scripts/make_task3_submission.py \
    --checkpoint experiment-pipeline/runs/<run>/artifacts/<ckpt>.pt \
    --architecture {unconditional,conditional,vanilla,vit,swin} \
    --name <name> \
    --dry-run   # run once without --dry-run to actually write the archive
```

A complete Task 3 archive is 20 field pairs × 3 modalities × 3 subjects = 180
files; the script asserts this count. Task 3 accepts **no segmentations** —
only voxel metrics (nRMSE, SSIM, LPIPS). Tasks 1 and 2 additionally need Dice
and Volume, which require `scripts/segment_predictions.py` (SynthSeg) over the
predictions first.

For an identity-control baseline archive, use `scripts/make_source_submission.py`.

### Spectral analysis

```bash
~/anaconda3/envs/mri/bin/python scripts/spectral_analysis.py --splits <...> --out-dir reports/spectral
```

Produces the RAPSD / power-law α fits consumed by the report tables below.

### Build the reports / slides

Use the `report` skill, or manually:

```bash
~/anaconda3/envs/mri/bin/python scripts/make_report_tables.py --in-dir reports/spectral --out-dir reports/tables
~/anaconda3/envs/mri/bin/python scripts/make_slide_figures.py --out-dir reports/figures
cd reports && ~/anaconda3/envs/tex/bin/tectonic <target>.tex
~/anaconda3/envs/mri/bin/python scripts/clean_latex.py
```

`make_report_tables.py` reads `reports/spectral/alpha_summary.csv` — rerun
`spectral_analysis.py` first if those numbers are stale. `make_slide_figures.py`
hardcodes numbers transcribed from the experiment log and report text; edit its
constants rather than the generated PDF. Compiled PDFs are gitignored — only
`.tex`, `reports/figures/`, `reports/spectral/`, `reports/tables/` and
`reports/visuals/` are tracked.

## Gotchas

- **Wrong `eval_pipeline`**: run `eval_pipeline` commands from inside
  `experiment-pipeline/`. From the repo root, `import eval_pipeline` resolves
  to an unrelated editable checkout at
  `/home/ruru/Documents/deniz/cmrx/projects/experiment-pipeline`.
- **Default stages**: `eval_pipeline run` defaults to `--stages test`. Training
  needs `--stages training validation` explicitly, as shown above.
- **Task 3 has no segmentations**: don't run `segment_predictions.py` against
  Task 3 predictions — it's voxel-metrics only (nRMSE, SSIM, LPIPS).
- **`make_task3_submission.py`** asserts the full 180-file archive shape; use
  `--dry-run` first to catch mismatches before writing the ZIP.

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
