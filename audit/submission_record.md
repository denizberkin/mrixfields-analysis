# Submission record — team `inzva_mri`, Task 3

Source: Synapse submission view `syn74915588` (Task 3 validation queue 9619636), pulled
2026-09-20 with `scripts/leaderboard.py`. Scores are the challenge's modality-averaged
metrics; SSIM is the ranked one. All entries are by the same submitter (team id 3599505).

## The submission these logs correspond to

| | |
|---|---|
| Submission ID | **9780368** |
| Submission name | **`mc_ssim_slice_avg_tta`** |
| Submitted | 2026-09-11 05:51 UTC, by `denizberkin` (Synapse user 3584795) |
| Entity | syn77355152 |
| SSIM / nRMSE / LPIPS | **0.913652** / 0.237970 / 0.090899 |
| Per modality SSIM | T1W 0.913460 · T2W 0.919841 · T2FLAIR 0.907654 |
| Rank (best entry per team, 2026-09-20 snapshot) | 9 of 40 |
| Weights | `runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt` = mean of fine-tune epochs 6, 7, 8 of run `task3_mc_ssim_slice` |
| Inference | `scripts/make_task3_submission.py --architecture conditional --tta` (4-flip TTA) |
| Test-phase container | `docker.synapse.org/syn76236366/task3:v2`, digest `sha256:7165fe78625729b778f37e3752221891a62d61014cb6876e5e60f4a22cf04c5a`, same weights (`/app/weights/task3.pt`, sha256 `be132195d8afc96494aa9c3a252065cadb97084547d421111b79dd392567063b`), same TTA, full 364-slice volumes |

## The audited re-run

The logs in this package come from a re-execution of the chain on 2026-09-20 (README §7).
Its own validation-phase ZIP (`runs/submission_reproduced/task3.zip`, 4-flip TTA) was
submitted to the same queue so the two can be compared directly:

| | |
|---|---|
| Submission ID | _(fill in after submitting)_ |
| Submission name | `audit_rerun_mc_ssim_slice_avg_tta` |
| SSIM / nRMSE / LPIPS | _(fill in)_ |

## Lineage on the leaderboard (the chain the audited run reproduces)

Each row keeps everything above it and changes one thing. Rows in bold are the two
training runs the audit reproduces; the last row is training-free post-processing.

| ID | name | date | change | SSIM | nRMSE | LPIPS |
|---|---|---|---|---|---|---|
| 9778330 | `task3_input_validation` | 08-28 | identity (copy the source) | 0.836497 | 0.521604 | 0.157279 |
| 9778464 | `unet-vanilla` | 08-30 | plain U-Net | 0.869856 | 0.402221 | 0.119849 |
| 9778494 | `unet_conditioned_50ep` | 08-30 | + source/target domain embeddings | 0.897646 | 0.241173 | 0.089527 |
| 9779441 | `unet_film_residual_ep8` | 09-05 | + zero-init FiLM in the decoder + zero-init residual head | 0.892839 | 0.263835 | 0.092091 |
| 9779581 | `retro_pretrain_film` | 09-06 | + MIM pretraining, 12k steps on 883 retrospective volumes | 0.901633 | 0.242206 | 0.085563 |
| **9779618** | **`unet_cond_retro30k_pro20ep`** | 09-07 | **MIM pretraining scaled to 30k steps on all 1,939 volumes, then 25 supervised epochs — stage 1 (`task3_retro_pretrain_big`, its e25)** | 0.903110 | 0.245574 | 0.083882 |
| 9779729 | `unet_multi_contrast_e20` | 09-08 | + 4-channel multi-contrast input, seeded from the row above widened to 4 channels | 0.906773 | 0.243607 | 0.080831 |
| 9779800 | `unet_multicontrast_ssim_ep8` | 09-08 | + 0.5·(1−SSIM) in the loss, 8 epochs from the same widened seed | 0.908873 | 0.241785 | 0.088697 |
| **9780364** | **`mc_ssim_slice_e8`** | 09-11 | **+ axial-position token, same seed and schedule — stage 2 (`task3_mc_ssim_slice`, its e8)** | 0.909433 | 0.240994 | 0.088294 |
| **9780368** | **`mc_ssim_slice_avg_tta`** | 09-11 | **mean of e6–e8 of the row above + 4-flip TTA (no training)** | **0.913652** | **0.237970** | 0.090899 |

Stage 2 is an 8-epoch fine-tune seeded from stage 1's epoch 25 (widened); rows 9779729
and 9779800 were separate fine-tunes from that same widened seed that established the
4-channel input and the SSIM term, and are not on the checkpoint path of the final model.

## Every other accepted Task 3 submission by the team

Branches that were measured and not shipped, and teammates' models. None contributes to
the submitted weights.

| ID | name | date | by | note | SSIM |
|---|---|---|---|---|---|
| 9771786 | `val_task3` | 07-15 | 3599511 | StarGAN v2 challenge baseline | 0.739694 |
| 9772388 | `unet-ft-dandik` | 07-18 | 3584795 | early conditional U-Net, 10 epochs (artifact lost) | 0.901442 |
| 9778241 | `delitubet` | 08-28 | 3592219 | teammate, tubelet encoder | 0.876630 |
| 9778284 | `epoch_30_fpsformer` | 08-28 | 3599511 | teammate, FPS-Former 30 ep | 0.864881 |
| 9778381 | `StS-T-lj-un-v1` | 08-29 | 3592219 | teammate, LeJEPA tubelet | 0.889338 |
| 9778454 | `unet-unconditioned` | 08-30 | 3584795 | conditioning ablated | 0.869784 |
| 9778500 | `vit_100` | 08-30 | 3599511 | teammate, DINOv3 ViT | 0.869782 |
| 9778534 | `b-u` | 08-30 | 3592219 | teammate, 3D U-Net | 0.898601 |
| 9778590 | `swin_cond_10ep` | 08-31 | 3584795 | Swin UNETR 3D | 0.831536 |
| 9778594 | `swin_cond_50ep` | 08-31 | 3584795 | Swin UNETR 3D | 0.858214 |
| 9778734 | `task3-16t16.zip` | 08-31 | 3592219 | teammate, 3D U-Net 2 ep | 0.894233 |
| 9778735 | `vit_256_150` | 08-31 | 3599511 | teammate, DINOv3 ViT | 0.878918 |
| 9779018 | `task_3_overfit_fix_try_8b` | 09-02 | 3599511 | teammate, FPS-Former 203 ep | 0.879531 |
| 9779021 | `ğ` | 09-02 | 3592219 | teammate, 3D U-Net | 0.902596 |
| 9779418 | `tubelet_encoder_unetr_decoder_ft23ep` | 09-05 | 3584795 | LeJEPA tubelet + UNETR decoder | 0.859000 |
| 9780001 | `mc_ssim_long_ep12` | 09-09 | 3584795 | SSIM-loss run, epoch 12 instead of 8 | 0.908645 |
| 9780056 | `task3.zip` | 09-09 | 3592219 | teammate, 3D U-Net | 0.905596 |
| 9780070 | `task3-str16.zip` | 09-09 | 3592219 | teammate, 3D U-Net | 0.904645 |
| 9780113 | `plain_e15` | 09-10 | 3584795 | the chain without FiLM/residual | 0.901476 |
| 9780175 | `mc_25d_e10` | 09-10 | 3584795 | 2.5D input (z±2) | 0.909353 |
| 9780354 | `mc_25d_slice_ep8` | 09-11 | 3584795 | 2.5D + axial token | 0.908354 |
| 9780375 | `soup_5050_tta` | 09-11 | 3584795 | 50/50 soup of avg_e6_e8 with the control run's e8 | 0.913203 |
| 9780386 | `cfm_adv10_tta` | 09-11 | 3584795 | conditional flow matching, 3 stages | 0.886725 |
