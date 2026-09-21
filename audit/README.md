# MRIxFields2026 — Task 3 audit materials, team `inzva_mri`

Reproducibility package and training logs for our Task 3 test-phase submission, per the
[Data Integrity Policy](https://www.synapse.org/Synapse:syn72060672/wiki/642237).

| | |
|---|---|
| Team | `inzva mri` (Synapse team id 3599505) — Berkin Deniz Kahya, Furkan Yüceyalçın, Racha Badreddine, Ahmet Zelka |
| Test-phase Docker image | `docker.synapse.org/syn76236366/task3:v2` — digest `sha256:7165fe78625729b778f37e3752221891a62d61014cb6876e5e60f4a22cf04c5a` |
| Shipped weights | `/app/weights/task3.pt`, sha256 `be132195d8afc96494aa9c3a252065cadb97084547d421111b79dd392567063b` |
| Corresponding validation-leaderboard submission | **ID 9780368**, name **`mc_ssim_slice_avg_tta`**, 2026-09-11 05:51 UTC, SSIM 0.913652 / nRMSE 0.237970 / LPIPS 0.090899 |
| Model | 2D conditional U-Net (32 base ch., 4 levels, InstanceNorm, FiLM + zero-init residual head), 4-channel multi-contrast input, axial-position token; MIM-pretrained on the unpaired retrospective split; final weights = mean of fine-tune epochs 6–8; 4-flip test-time augmentation at inference |

`submission_record.md` lists every leaderboard submission in the lineage with its ID.

## 1. What is in this package

```
task3/
├── code/                       this directory
│   ├── README.md               this file
│   ├── requirements.txt        pip environment (torch 2.x + CUDA; exact versions used are in logs/environment_*.txt)
│   ├── mrixfields/             shared package: dataset classes, transforms, LPIPS loss, audit bridge
│   ├── experiment-pipeline/    the training pipeline (config-driven runner + Task 3 components)
│   │   ├── train_task3_audit.py            ONE-COMMAND TRAINING entry point (monitored)
│   │   ├── audit_monitor.py, audit_utils.py official audit tools, unmodified (v1.0.0)
│   │   ├── audit_dataset.py                official dataset wrapper, adapted: every file read goes through it
│   │   ├── audit_mirror.py                 copies the logs to persistent storage during the run
│   │   ├── configs/task3_audit_stage1_pretrain_finetune.toml
│   │   ├── configs/task3_audit_stage2_multicontrast_ssim_slice.toml
│   │   └── components/{data,models,losses,training}/   Task 3 data modules, ConditionalUNet, losses, trainer
│   ├── scripts/                preprocess.py, widen_input_channels.py, make_task3_submission.py (inference), ...
│   ├── docker/                 the test-phase container: app/{inference.py, mrx/, requirements.txt, weights/task3.pt}, Dockerfile.reconstructed, PULL_INFO.md
│   └── task3_audit_colab.ipynb the notebook the audited run was launched from
├── data-split.md               training / validation subject lists (generated from the files actually read)
└── logs/
    ├── audit-logs/             official logs: monitor_*, file-loading_*, training_* (hash-chained, one file per process)
    ├── console-logs/           full stdout/stderr of every launch, environment_*.txt, segments.jsonl
    ├── runs/<experiment>/      config.source.toml, config.resolved.json, logs/losses.jsonl, artifacts/ (checkpoints, see §6)
    └── export/                 task3_final_weights.pt, checksums.sha256, data-split.md, preprocess_manifest.json
```

## 2. Environment

* Linux, Python ≥ 3.12, an NVIDIA GPU. The audited run used Google Colab; the exact
  `pip freeze`, torch/CUDA/cuDNN versions, GPU model and driver are recorded in
  `logs/console-logs/environment_<timestamp>.txt` at the start of every launch.
* `pip install -r requirements.txt` (torch is installed from the CUDA index of your choice;
  the run used the torch that ships with Colab).
* The official audit tools need Linux (`os.getuid`, `/proc` I/O counters).

## 3. Data

The challenge release as distributed, unmodified, laid out as

```
<raw-dir>/training_retrospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/R_<mod>_<field>_<id>.nii.gz   1,939 volumes
<raw-dir>/training_prospective/  {T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/P_<mod>_<field>_<id>.nii.gz   3 subjects x 15
<raw-dir>/validating_prospective/...                                                                  leaderboard inference only
```

(directory names are matched case-insensitively). Preprocessing extracts axial slices
72..291 of every volume as `float32` npz in `[0, 1]` (`scripts/preprocess.py`); the training
stages read only those files. The withdrawn `Training_prospective.zip` was never
downloaded or used.

## 4. One-command training

From inside `experiment-pipeline/`, with an unbuffered interpreter (the runtime monitor
checks that `import audit_monitor` is the first statement of the entry program):

```bash
cd experiment-pipeline
python3 -u train_task3_audit.py --stages all \
    --raw-dir          /path/to/challenge/data \
    --preprocessed-dir /local/disk/preprocessed \
    --output-dir       /persistent/runs \
    --mirror-dir       /persistent/logs
python3 audit_mirror.py --src audit-logs console-logs --dst /persistent/logs --delete   # after exit: picks up the .log -> .log.csv rename
```

This runs, in order, skipping anything already done and resuming any interrupted stage
from its latest checkpoint:

| stage | what | config / script | output |
|---|---|---|---|
| `preprocess` | NIfTI → npz slices, `retro_train` + `pro_train` | `scripts/preprocess.py` | `<preprocessed-dir>/` |
| `stage1` | 30,000 steps masked-image pretraining (batch 64, 50 % of 16 px blocks zeroed, L1 on the holes) on the 1,939 retrospective volumes, then 25 supervised epochs (L1 + 0.05 LPIPS, lr 1e-4, Adam 0.9/0.999, AMP, horizontal flips) on the 3 paired subjects | `configs/task3_audit_stage1_pretrain_finetune.toml` | `runs/task3_audit_stage1_pretrain_finetune/artifacts/task3_unet_{pretrain_*,finetune_1..25}.pt` |
| `widen` | epoch-25 weights, first convolution 1 → 4 input channels, new channels zero | `scripts/widen_input_channels.py` | `.../widened4_from_e25.pt` |
| `stage2` | 8 epochs, 4-channel input (predicted contrast + T1W/T2W/T2FLAIR at the source field), L1 + 0.5·(1−SSIM) + 0.05 LPIPS, axial-position conditioning, same seed/lr/optimizer | `configs/task3_audit_stage2_multicontrast_ssim_slice.toml` | `runs/task3_audit_stage2_multicontrast_ssim_slice/artifacts/task3_unet_finetune_1..8.pt` |
| `average` | mean of epochs 6, 7, 8 | (in the entry script) | `.../avg_e6_e8.pt` — **the submitted weights** |
| `export` | weights-only copy, sha256 of every checkpoint, `data-split.md`, config copies | | `runs/export/` |
| `predict` | validation-phase ZIP with 4-flip TTA, i.e. the leaderboard submission | `scripts/make_task3_submission.py --tta` | `runs/submission_reproduced/task3.zip` |

`--smoke` runs the same chain on two subjects per domain, two field strengths, 20
pretraining steps and 2 epochs, to check the plumbing in minutes.

### Hardware independence: micro-batching

Every optimizer step uses 64 slices, as in the original runs. On a GPU that cannot hold
64 slices of 368×448 the script lowers the loader batch (`--micro-batch`, auto-chosen from
GPU memory: 64 ≥ 38 GB, 32 ≥ 22 GB, 16 ≥ 14 GB) and accumulates gradients so the product
is 64. The network is InstanceNorm throughout and every loss is a per-sample mean over
equal-sized micro-batches, so the accumulated gradient equals the batch-64 gradient up to
floating-point rounding. The training log records `BatchSize` (micro), `EffectiveBatchSize`
(64), and `MicroStep:i/k`.

### Resuming

Checkpoints carry model, optimizer and AMP-scaler state. Re-running the same command
after an interruption continues from the newest loadable checkpoint of the interrupted
stage (a checkpoint cut off mid-write is renamed `*.pt.corrupt` and the previous one is
used). Each launch is a *segment*; `logs/console-logs/segments.jsonl` lists them with
their pid, and the audit tools name every log file `<log>_<pid>_<time>`, so the segment a
file belongs to is read off its name.

## 5. One-command inference

**Test phase (what was submitted).** The container's entrypoint, included here under
`docker/app/`:

```bash
docker run --gpus all -v /path/to/input:/input -v /path/to/output:/output \
    docker.synapse.org/syn76236366/task3:v2          # = python3 /app/inference.py --input /input --output /output
```

It reads `/input/manifest.json`, builds the network from the checkpoint's own tensor
shapes, predicts every axial slice of the full 364-slice volume with the four in-plane
flips averaged, masks air (source < 1e-3), and writes one NIfTI per sample.
`docker/Dockerfile.reconstructed` is the build recipe read back from the image's layer
history. The reproduced weights (`logs/export/task3_final_weights.pt`) are saved in the
same `{"model": state_dict}` form the entrypoint loads, so rebuilding the container from
them is `cp task3_final_weights.pt docker/app/weights/task3.pt` and `docker build`.

**Validation phase (the leaderboard twin).** From the repository root:

```bash
python3 scripts/make_task3_submission.py --architecture conditional --tta \
    --checkpoint experiment-pipeline/runs/task3_audit_stage2_multicontrast_ssim_slice/artifacts/avg_e6_e8.pt \
    --data-dir /path/to/challenge/data --out-dir /path/to/out
```

Same computation on the 30-slice validation slab (`Z_CLIP_RANGE`); the two paths were
verified to agree to 1.8e-4 (max) on the shipped weights. Both accept any checkpoint
produced by the pipeline.

## 6. The logs (Rule I) and how to read them

All under `logs/audit-logs/`, written by the unmodified official tools:

| log | file(s) | produced by |
|---|---|---|
| 1 Runtime monitoring | `monitor_<pid>_<time>.log.csv` | `import audit_monitor` as the first statement of `train_task3_audit.py`: samples memory and I/O every 2 s for the whole process tree |
| 2 Dataset access | `file-loading_<pid>_<time>.log[.csv]` | `audit_dataset.py::load_image / load_volume` (`audit_file_loading` before every read). One file per process: the main process (`.csv`, archived at exit) and every DataLoader / preprocessing worker (`.log`; workers are terminated by the parent, so the tools' exit-time rename does not apply to them). The `Official dataset, <path>, <digest>` line in each records the wrapper's fingerprint. |
| 3 Training | `training_<pid>_<time>.log.csv` | one line per micro-batch: `Epoch, Iteration (optimizer step), MicroStep, LR, Scheduler, BatchSize, EffectiveBatchSize, Loss, Losses{...}, LossWeight{...}, MemUsage, Stage`. Pretraining lines carry `Epoch:0` and `Losses:{'masked_l1'}`. |
| 4 Validation | same file, `[Validation] Epoch:…, ValTotalLoss, ValLosses{…}, ValMetrics{ssim,…}` | every fine-tune epoch; every 2,500 pretraining steps (`Iteration` given) |
| 5 Checkpoint | same file, `[Checkpoint] Saved to: … \| Epoch, IsBest, ValLoss, ValMetrics{…}` | every saved checkpoint: every 5,000 pretraining steps and every epoch of both fine-tunes |

Every line ends with `| <hash>`, the tools' chain hash; nothing was deleted, merged,
filtered or rewritten. The full stdout/stderr of each launch (which also echoes every
training-log line, the tqdm progress and the tracker's per-epoch means) is in
`logs/console-logs/`.

**About the validation records.** Only three paired subjects exist and the submitted
model was trained on all three. No subject was held out — doing so would have changed the
model — so the per-epoch `[Validation]` records are computed on a fixed, evenly strided,
unaugmented subset of the *training* slices (2,400 slices; during pretraining 1,024
retrospective slices under a fixed mask). They document a continuous loss/SSIM curve and
are labelled as training-subset values; the challenge validation split has no targets and
was used only to build leaderboard submissions. Checkpoints were chosen by leaderboard
score (see `submission_record.md`), never by these values.

**Checkpoint history.** Every checkpoint of both stages is included, unaltered, under
`logs/runs/<experiment>/artifacts/` (model + optimizer + AMP-scaler state, ~105-112 MB
each): 6 pretraining checkpoints (every 5,000 steps), fine-tune epochs 1-25, the widened
seed, stage-2 epochs 1-8 and `avg_e6_e8.pt`. `logs/export/checksums.sha256` lists the sha256
of each, and `logs/export/task3_final_weights.pt` is the weights-only copy of `avg_e6_e8.pt`
in the form the container loads.

**Pretrained model.** The only pretraining is stage 1's masked-image pretraining on the
challenge's own retrospective split, fully logged here (its `[Validation]` lines give the
masked-L1 and inpainting SSIM through the 30,000 steps). No external pretrained weights
are used; the LPIPS loss uses the standard `lpips` package's AlexNet weights, which are
frozen and part of the loss, not of the model.

**Launch history.** `logs/console-logs/segments.jsonl` records the one launch that produced
the run (pid 3120, 2026-09-20 18:48:00 UTC). Three launches in the preceding two minutes
(pids 2825, 2888, 2985; 18:46:56, 18:47:09, 18:47:30 UTC) exited within a second at
start-up -- mistyped command-line flags (`--mini_batch`, `--micro_batch`) and a re-launch --
before any data was accessed; each left the tools' three 1 KB start-up stubs
(`monitor_/file-loading_/training_<pid>_*.log.csv`), which are kept as they are.

## 7. The audited run

| | |
|---|---|
| Launched | 2026-09-20 18:48:00 UTC, Google Colab, one process (pid 3120), all stages in sequence |
| Hardware / software | NVIDIA A100-SXM4-40GB; Python 3.13.15; torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900 (`logs/console-logs/environment_20260920_184806.txt`) |
| Batching | 64 slices per optimizer step, no accumulation (`--micro-batch 64`), 8 DataLoader workers -- the original setting |
| Wall-clock | 5.12 h: preprocessing 29.5 min (1,939 + 45 volumes -> 436,480 slices), pretraining 19:18-21:36, stage-1 fine-tune 21:36-23:05, stage-2 23:05-23:40, average + export + validation ZIP to 23:55 |
| Pretraining | validation masked-L1 (1,024 fixed retrospective slices, fixed mask) 0.0771 at step 2,500 -> 0.0655 at 30,000; final training masked-L1 0.0716 |
| Stage 1 | training-subset validation SSIM 0.8520 (e1) -> 0.9437 (e5) -> 0.9482 (e10) -> 0.9536 (e20) -> **0.9550 (e25)**; total loss 0.0637 -> 0.0157 |
| Stage 2 | 0.9580 (e1) -> 0.9624 (e6) -> 0.9625 (e7) -> **0.9632 (e8)**; total loss 0.0382 -> 0.0332 |
| Final weights | `avg_e6_e8.pt` (mean of stage-2 e6/e7/e8) = `logs/export/task3_final_weights.pt`, 78 tensors |
| Leaderboard twin | `runs/submission_reproduced/task3.zip` (180 files, 4-flip TTA) -- submitted to the validation queue as recorded in `submission_record.md` |

## 8. Correspondence with the original runs

The submitted weights were produced on 2026-09-06/07 (stage 1, run `task3_retro_pretrain_big`)
and 2026-09-11 (stage 2, run `task3_mc_ssim_slice`) on the team's workstation, before the
audit tools were available, and the machine's logs are not of the required form. The
audited run in §7 is a re-execution of the same two configs (`configs/task3_retro_pretrain_big.toml`
and `configs/task3_mc_ssim_slice.toml` are kept in `experiment-pipeline/configs/` for
comparison; the `task3_audit_*` copies differ only in tracker, paths, checkpoint cadence,
the validation subset and the micro-batch mechanism described above). Training on
different hardware with cuDNN's non-deterministic kernels does not reproduce the weights
bit for bit; the consistency check is the `predict` stage's validation ZIP scored on the
leaderboard against submission 9780368. The shipped weights are included at
`code/docker/app/weights/task3.pt` next to the reproduced `logs/export/task3_final_weights.pt`,
so a tensor-wise comparison is a `torch.load` of each; the run itself did not write
`reference_comparison.json` because the shipped weights were not on the audit machine.

## 9. Contact

Berkin Deniz Kahya — Synapse `denizberkin` — via the challenge Discussion board or the
team's Synapse project.
