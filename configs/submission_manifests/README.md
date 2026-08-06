# Manifest-driven submission generation

Use [`scripts/generate_challenge_submission.py`](../../scripts/generate_challenge_submission.py)
to map checkpoints to validation inference pairs and generate an isolated
submission archive.

## Command

```bash
/home/denizberkin/miniconda3/envs/mri/bin/python \
  scripts/generate_challenge_submission.py \
  --manifest configs/submission_manifests/my_run.yaml
```

Validate a manifest and print the inference plan without writing:

```bash
/home/denizberkin/miniconda3/envs/mri/bin/python \
  scripts/generate_challenge_submission.py \
  --manifest configs/submission_manifests/my_run.yaml \
  --dry-run
```

Run one stage only with `--stage inference`, `--stage segment`, or
`--stage package`. Existing complete inference and segmentation outputs are
reused unless `--overwrite-inference` or `--overwrite-segmentations` is set.

## Required manifest parameters

| Parameter | Description |
|---|---|
| `name` | Unique run name. Keeps methods/runs in isolated output directories. |
| `task` | `task1`, `task2`, or `task3`. |
| `method` | Output label. Without `inference_command`: Task 1/2 `cut` or `cyclegan`; Task 3 `stargan_v2`. |
| `models` | List of checkpoint-to-mapping assignments. |
| `models[].checkpoint` | Checkpoint file or directory path. |
| `models[].mappings` | One or more mappings handled by that checkpoint. |
| `modality` | `T1W`, `T2W`, or `T2FLAIR`. |
| `source` | Source field: `0.1T`, `1.5T`, `3T`, `5T`, or `7T`. |
| `target` | Target field. Must form a valid pair for the selected task. |

## Optional parameters

| Parameter | Default | Description |
|---|---|---|
| `mode` | `pro_pretrained` | Sweep coordinate used for output and packaging. |
| `epoch_tag` | `epoch100` (`epoch50` Task 3) | Sweep coordinate used for output and packaging. |
| `device` | `.env DEVICE` | PyTorch inference device. |
| `seed` | `0` | NumPy/PyTorch seed, particularly relevant to StarGAN v2 style sampling. |
| `inference_command` | Built-in baseline inference | Command template for any custom model. |
| `models[].config` | Derived for baselines | Config shared by all mappings for a checkpoint. |
| `mappings[].config` | Model config | Per-mapping config override. |
| `paths.predictions` | `.env INFERENCE_DIR` | Base prediction directory. |
| `paths.segmentations` | `.env PREDICTIONS_SEG_DIR` | Base segmentation directory. |
| `paths.submissions` | `.env SUBMISSION_DIR` | Base submission directory. |
| `paths.synthseg` | `.env SYNTHSEG_DIR` | SynthSeg installation. |

Outputs are isolated below `<configured-root>/<name>/`.

## Custom models

Set any filesystem-safe `method` name and provide `inference_command` as a list
of command arguments. The command runs once per mapping and may use these
placeholders: `{checkpoint}`, `{config}`, `{input_dir}`, `{output_dir}`,
`{device}`, `{modality}`, `{source}`, `{target}`, `{task}`, `{method}`, and
`{seed}`.

```yaml
name: task2_my_model
task: task2
method: my_model
device: cuda:0
inference_command:
  - python
  - /models/my_model/predict.py
  - --weights
  - "{checkpoint}"
  - --input
  - "{input_dir}"
  - --output
  - "{output_dir}"
  - --target
  - "{target}"
  - --device
  - "{device}"
models:
  - checkpoint: /models/my_model/weights.pt
    mappings:
      - {modality: T1W, source: 0.1T, target: 1.5T}
```

The adapter may use any model architecture, framework, or Python environment.
It must write exactly three NIfTI files to `{output_dir}`, retaining the input
names (`P_{modality}_{source}_{ID}.nii.gz`). The pipeline then performs the
same segmentation, clipping, renaming, validation, and ZIP packaging used for
the baselines. Relative checkpoint and config paths are resolved from the
manifest; a checkpoint may be a file or directory.

## Model mapping patterns

Task 1/2 baseline runs usually assign one checkpoint to one mapping:

```yaml
models:
  - checkpoint: /models/task1_3T_to_7T_T1W/cut/pro_pretrained/weights/checkpoint_epoch100.pth
    mappings:
      - {modality: T1W, source: 3T, target: 7T}
```

Task 3 must use one unique checkpoint, which can fan out to many mappings:

```yaml
task: task3
method: stargan_v2
models:
  - checkpoint: /models/task3/stargan_v2/pro_pretrained/weights/checkpoint_epoch50.pth
    mappings:
      - {modality: T1W, source: 0.1T, target: 7T}
      - {modality: T1W, source: 7T, target: 0.1T}
      - {modality: T2W, source: 3T, target: 5T}
```

The script rejects multiple unique Task 3 checkpoints.
