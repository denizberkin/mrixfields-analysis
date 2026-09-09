# TODO — MRIxFields 2026

Working tracker. Status marks: `[x]` done · `[~]` in progress · `[ ]` not started · `[-]` descoped.

Two horizons, because the challenge deadline is under two weeks and the work continues afterwards:

- **Horizon A — challenge submission.** Rank-sum on the leaderboard. Cheap, high-certainty moves only.
- **Horizon B — fastMRI + conference.** The parts that generalise and are worth writing up.

Last updated 2026-09-05. Log pulled from the shared sheet at 70 rows.

---

## 0. Where we actually are

**SSIM is the ranked metric we optimise for.** Task 3, modality-averaged, sorted by SSIM:

| SSIM ↑ | nRMSE ↓ | LPIPS ↓ | Model | Experiment | Date | |
|---|---|---|---|---|---|---|
| **0.9026** | 0.2329 | 0.0816 | UNet-3D | 3d unet baseline — 10 epoch | 02/09 | **current best** |
| **0.9014** | 0.2364 | 0.0890 | UNet | "New baseline?" — the original model, 10 ep | **18/07** | **artifact lost** |
| 0.8986 | 0.2336 | 0.0819 | UNet | furkan implementasyon unet conditioned | 30/08 | |
| 0.8976 | 0.2412 | 0.0895 | UNet | unet conditioned 50ep | 30/08 | |
| 0.8942 | 0.2436 | 0.0967 | UNet-3D | 3d unet baseline — 2 epoch | 31/08 | |
| 0.8893 | 0.2383 | 0.0889 | UNet-3D | tubelet LeJEPA encoder, 2 ep | 29/08 | |
| 0.8795 | 0.2471 | 0.1492 | FPS-Former | 203 ep | 02/09 | |
| 0.8699 | 0.4022 | 0.1198 | UNet | unet unconditioned | 30/08 | |
| **0.8590** | 0.2658 | **0.1436** | Tubelet-UNETR | LeJEPA enc + cond UNETR, 10ep-equiv | **05/09** | **submitted; see §7** |
| 0.8365 | 0.5216 | 0.1573 | INPUT | source control | 28/08 | zero point |
| 0.8315 | 0.3002 | 0.1713 | Swin UNetR 3D | swin unet 10ep | 31/08 | |
| 0.7397 | 0.3436 | 0.1545 | StarGANv2 | challenge baseline | 15/07 | |

Four things this table says, and the plan follows from them.

1. **Six weeks of architecture search has bought +0.0012 SSIM.** The original 10-epoch conditional U-Net from 18 July scored 0.9014. FPS-Former, Swin UNETR, the tubelet LeJEPA encoder and the 3D U-Net have since produced 0.9026. The top four entries sit inside 0.005 of each other and **nobody has measured the seed-noise floor**, so it is not established that they differ at all.

2. **The re-implementations regressed.** The same conditional-U-Net idea, rebuilt in August, scores 0.8976 (50 epochs) and 0.8986 — both *below* the July 10-epoch original. More training, worse SSIM.

3. **There is a mechanism for that, and it is unmonitored.** `configs/task3_unet_pro.toml` sets `epochs = 100`, `early_stopping_patience = 10`, `early_stopping_restore_best = true`, and defines **no validation or holdout keys at all**. So `validation_loader is None` and `components/training/task3.py:228` falls back to monitoring **training** loss; "restore best" then restores the best *training* loss, i.e. the most overfit weights. The run reached at least epoch 60 (`runs/task3_unet_pro/artifacts/` holds epochs 10–60).

4. **Conditioning remains the only large measured effect.** Unconditioned 0.8699 against conditioned 0.8976 SSIM at matched settings.

**The July model cannot be recovered.** No artifacts exist before 2026-08-30 in `OUTPUT_DIR` or `experiment-pipeline/runs/`; mlflow's earliest run is 2026-08-30; this repo's first commit is 2026-08-06. Only the submission ID 9772388 survives.

The reading that follows: a substantial part of the last six weeks may have been resolving checkpoint-selection noise rather than architecture. That is cheap to test and the test comes first.

---

## 1. Horizon A — challenge (next 12 days)

### A0 — Instrument, then read the checkpoints we already have · ~2 days · **do this first**

We are chasing SSIM differences of ~0.005 with no local held-out signal, on models whose early stopping watches training loss. Four runs under these conditions produce four numbers that cannot be ordered.

- [x] **A0.1 — Done, then deliberately switched off.** The split works (880/440 exact partition), but with only 3 paired subjects it costs 33% of the training data, so all configs now run `holdout_subjects = []` with a fixed epoch budget. Kept behind the flag for the day there is more data. Subjects `0006`, `0007`, `0009` are the three possible folds.
- [ ] **A0.2 — Report per transition × per modality.** 20 transitions whose Δα differs by an order of magnitude are currently averaged into one number. A model that works at 0.1 T and one that copies at 5 T are indistinguishable today.
- [ ] **A0.3 — Seed-noise floor.** Run the current best config twice with different seeds. If the spread exceeds the gaps we are trying to resolve, stop comparing and say so. Given that the top four models span 0.005 SSIM, this is not a formality.
- [x] **A0.5 — Done on the new arm (§9): SSIM saturates at epoch 8.** `runs/task3_unet_pro/artifacts/` holds epochs 10, 20, 30, 40, 50 and 60. Score all six on the held-out fold, SSIM primary, per transition. This directly tests conclusion 3 in section 0: if SSIM peaks near epoch 10 and decays, then the July result is explained, the August regression is explained, and checkpoint selection is worth more SSIM than any architecture tried since. Hours of compute, and it decides what A1–A5 are even for.
- [ ] **A0.4 — Early stopping on the held-out fold.** `components/training/task3.py:228` falls back to *training* loss when `validation_loader is None`, which is every 2D config. Once A0.1 lands, set `early_stopping_monitor = "validation"`.

### A1 — Free wins on the current best model · ~2 days

None of these need a new architecture and all are near-zero risk.

- [x] **A1.1 — Predict the residual, not the image.** **Worth +0.0126 SSIM with A3.1** (see §9). Implemented, zero-init verified (max|out-in| = 0.000e+00 at step 0); training in `task3_unet_film_residual`. Done in the bounded [-1,1] range, not the tubelet's logit space: ~85% of a slice is air at exactly -1, where `atanh` puts the base near -5 and the tanh derivative is ~1e-4 — the saturation `conditional_vit.py` documents as having frozen a run at a constant.
- [ ] **A1.2 — Eight-way dihedral TTA at inference.** Flips × rot90, average. Not in the log anywhere.
- [ ] **A1.3 — Checkpoint weight averaging** over the artifacts already sitting in `experiment-pipeline/runs/*/artifacts/`. Costs one evaluation pass.
- [ ] **A1.4 — Submit the best combination** of A1.1–A1.3.

### A2 — The calibration control · ~0.5 day · one submission slot

- [ ] **A2.1** Fit the per-(modality, source, target) monotone quantile map from Report finding F4 on all three paired subjects and submit it. Zero parameters, no training.

Why this matters more than it looks: F4 measured that this map removes 46% of the identity's error, and F3 measured that the U-Net's target conditioning is 12–96% explained by a single global affine. If a parameter-free histogram match lands near 0.2329, then every architecture comparison in section 0 has been resolving differences in calibration quality, and we should say so in the report rather than run a fifth architecture.

### A3 — Conditioning, the one lever with headroom · ~3 days

- [x] **A3.1 — FiLM at every decoder scale.** Run jointly with A1.1; see §9. Implemented alongside A1.1, same run. Per-stage scale-and-shift after each `InstanceNorm`, zero-init. Both flags opt-in; the six `task3_unet_pro` checkpoints still strict-load into a default model. The bottleneck broadcast add is kept, not replaced — dropping it would change what a loaded checkpoint computes.
- [ ] **A3.2 — Condition on physics, not fifteen opaque indices.** Feed `(modality one-hot, log B₀_src, log B₀_tgt, Δα(src→tgt))` through a small MLP to produce the FiLM parameters, so the twenty transitions share structure instead of each learning its own embedding from three subjects.
- [ ] **A3.3 — Control: delete the source embedding.** F2 measured that changing the declared source field moves the output by 1.1–2.1% of its range while changing the target moves it by 19.7–53.7%. If removing it costs nothing, the model simplifies; if it costs something, F2 was a one-checkpoint measurement and needs weakening. Either way we learn something.

### A4 — Tasks 1 and 2, where nobody is competing · ~2 days

Tasks 1 and 2 are scored on **five** metrics; Task 3 on three. We have spent almost all recent effort on Task 3.

Finding F1: on Task 1 the identity control has the **highest Dice of any model in the log** (0.8549 against CUT 0.8542 and CycleGAN 0.8498) and its volume consistency is within 0.002 of the best. Two of five ranked metrics are being scored and not competed for.

- [ ] **A4.1** Apply the A2 calibration map to Task 1 / Task 2 inputs and submit. It should keep Dice and volume near identity level while improving nRMSE.
- [ ] **A4.2** Run `scripts/segment_predictions.py` on the result and confirm Dice/Volume locally before submitting. Note this needs the TensorFlow environment, not `mri`.

### A5 — Encoder transfer · **promoted to first experiment** · ~3 days

`ConditionalTubeletTranslator` (`lejepa_pretraining/repo/mrixfields_downstream/tubelet_model.py`) already **is** "LeJEPA encoder + conditioned U-Net", and it already contains the two changes A1.1 and A3.1 propose for the 2D model:

- **Per-stage zero-init FiLM** — `gamma, beta = condition_projections[stage](condition).chunk(2, dim=1)`, applied as `value * (1 + gamma) + beta` at all four decoder stages, with both projection weight and bias zero-initialised (`tubelet_model.py:131-135, 172-173`). Conditioning is a concat of separate 32-d source-field, target-field and modality embeddings, not one 15-way index.
- **Zero-init residual head** in logit space — `sigmoid(logit(x) + residual)` with `output_head` zero-initialised.

So the architecture we were going to build by hand exists, is trained, and was only ever run for two epochs with no control.

- [x] **A5.1 — Pull the artifact.** `lejepa_pretraining/artifacts/encoder_step_00040000.pt`, 86.66 M params, schema v2, step 40 000 of a planned 90 000. Strict load verified; all four provenance hashes match `run_metadata.json`.
- [ ] **A5.2 — The matched control.** Same Tubelet-UNETR, same LOSO fold, same 10-epoch budget as the current leader, two arms: LeJEPA init vs. random init. Settles the encoder question and is the paper's central result; Furkan's design record names it as the one omission. **Caveat on priority:** at matched 2 epochs the encoder wins nRMSE (0.2383 vs 0.2436) and LPIPS (0.0889 vs 0.0967) but *loses SSIM* (0.8893 vs 0.8942). Since SSIM is the ranked metric, this is no longer the first experiment — but 0.0049 is inside the unmeasured seed spread, so it is not settled against either.
- [x] **A5.2/A5.3 — Run and submitted. It lost.** 23 epochs (budget 40), early-stopped on a held-out
  fold, best epoch 15 restored. Challenge: **SSIM 0.8590 / nRMSE 0.2658 / LPIPS 0.1436** against the
  leader's 0.9026 / 0.2329 / 0.0816. nRMSE is competitive; **LPIPS is barely better than identity's
  0.1573**, and §7 says why. Treat the encoder question as answered *for this decoder*: the tubelet
  stem's geometry, not the pretraining, is what caps it.

---

## 2. Horizon B — fastMRI and a paper

Ordered by how much of it is already in hand.

- [ ] **B1 — The dense/CLS gap in tubelet LeJEPA.** Section 3. This is a clean, already-measured negative result and the most publishable thing we have.
- [ ] **B2 — Synthetic pretraining transfer test.** The generator reached 0.86–0.98 diversity ratio at 0.1/3/5 T (1.5 T still short at 0.65) but no model has been pretrained on it, so the premise of the whole exercise is untested. Run the 2×2: {random, pretrained} × {real pairs only, synthetic→real}.
- [ ] **B3 — Rebuild the generator on SynthSeg's own machinery.** `estimate_priors.build_intensity_stats` gives per-label GMM statistics over ~32 disjoint labels instead of three soft tissues, which removes the over-summing term in the current remap outright. `BrainGenerator` adds bias field, slice-thickness and nonlinear deformation, none of which the four-step operator models.
- [ ] **B4 — A contrast-agnostic perceptual loss.** The LPIPS term uses AlexNet features trained on natural photographs. SynthSeg's encoder was trained to be invariant to MRI contrast and resolution — exactly the invariance a field-translation perceptual metric needs. Distil it into a small 3D U-Net (its generator supplies unlimited training data) and use its features.
- [ ] **B5 — Spectral-profile loss.** Penalise `|log P_pred(f) − log P_tgt(f)|` over the 0.02–0.20 fit band instead of hand-set band weights. Gets the sign right by construction at every transition, which fixed band weights cannot (three of nine Δα entries are negative). `scripts/spectral_analysis.py` already has the RAPSD machinery.
- [ ] **B6 — Port to fastMRI** and expand the study beyond the three travelling-volunteer subjects.

---

## 3. New evidence: the tubelet encoder's dense features

Report II said the three-part validation of the tubelet encoder was unrun and made no transfer claim. It has in fact been run — the numbers are in `lejepa_pretraining/artifacts/monitor_step_00040000.json`, pulled from the Drive run folder on 2026-09-05. They split cleanly in two.

**The CLS path — what the loss actually trains — is healthy.**

| | value | Δ vs step 0 (random init) |
|---|---|---|
| CLS effective rank (max 299) | 95.70 | **+66.00** |
| Retrieval top-1, global → local views | 0.883 – 0.983 | **+0.46 to +0.65** |

No collapse, and a large genuine gain in view-invariant slab identity over a random ViT.

**The dense path — what the translator actually consumes — is worse than random initialisation.**

| | value | Δ vs step 0 |
|---|---|---|
| Dense effective rank, mean (max 768) | 115.06 | **−69.98** |
| Dense effective rank, min | 96.07 | −41.37 |
| Dense scale, mean | 0.214 | −0.078 |

| correspondence (cosine) | value |
|---|---|
| matched location, matched subject | 0.9734 |
| wrong location, same subject | 0.9556 |
| wrong subject | 0.9415 |
| **matched − wrong-location gap** | **0.0177** (+0.0103 vs random init) |

Every dense token sits at cosine ≈ 0.94–0.97 with every other one regardless of anatomy. The features that are supposed to tell the decoder *where it is* barely distinguish the right location from the wrong one, and they have lost 70 points of effective rank relative to an untrained network.

Report II named this exact risk — *"the gap between those two paths is the design's main open risk"*. The monitor says the risk materialised. It also explains the leaderboard row that otherwise looks like noise: the LeJEPA-initialised 3D U-Net scores 0.2383 against 0.2329 for the same architecture without it. The encoder is not neutral, it is a small tax.

**Consequences.** For Horizon A, do not spend the remaining days on encoder transfer (A5.2 descoped). For Horizon B this becomes B1 and is the most interesting result in the project: *a CLS-only LeJEPA objective can improve pooled representation quality substantially while degrading dense token quality below random initialisation.* The fix to test is a coordinate-level token term alongside the CLS objective. The negative result is already measured; the positive control is one training run.

---

## 4. Suggested order

| When | What | Why |
|---|---|---|
| Days 1–2 | **A0.1, A0.5** | The held-out fold, then the six-checkpoint sweep. No training. Tests whether six weeks of architecture search has been reading checkpoint noise. |
| Days 2–3 | A0.3, A1.2, A1.3 | Seed floor, dihedral TTA and checkpoint averaging — all three help SSIM specifically, none needs a new architecture. |
| Days 3–4 | A2 | Zero-parameter calibration control. Its SSIM effect is unmeasured, unlike its nRMSE effect, so treat it as one cheap submission rather than a threat to the leader. |
| Days 4–7 | A3 | Conditioning is still the only large measured effect on SSIM (0.8699 → 0.8976). |
| Days 5–9 | A5.2 | The matched encoder control. Demoted from first: on SSIM the encoder is 0.0049 *behind* scratch at matched 2 epochs. Still the paper's core result, so run it, but after the free SSIM has been taken. |
| Days 6–9 | A4 | Different task, different person, in parallel. |
| Days 6–9 | A4 | Run in parallel with A3 — different task, different person. Clearest unclaimed rank-sum win. |
| Days 9–12 | Submissions, freeze, write up | |
| After | B1 → B2 → B4 | B1 is already measured and only needs the positive control. |

---

## 5. Open items

- [ ] Drive folder holds `encoder_step_00040000.pt` as the newest export, but the config targets 90 000 steps and `run_metadata.json` records `stop_after_steps: 60000`. Confirm with Furkan whether the run is finished, stalled, or still going.
- [ ] `scripts/generate_challenge_submission.py` cannot run — its `validate_inputs()` hard-requires an `official/MRIxFields2026/` checkout that is not in this repo. Either vendor the checkout or delete the script in favour of `make_task3_submission.py`.
- [ ] `mlflow.db` and `mlruns/` sit at the repo root, but `experiment-pipeline/mlruns/` also exists — tracking URIs are cwd-relative, so runs have been launched from both places. Pick one.

---

## 6. Feasibility checks (run 2026-09-05, before committing to the plan)

Everything Horizon A depends on is present on this machine.

| Item | Requirement | Status |
|---|---|---|
| A0.1, A2, A4 | `DATA_DIR` with paired prospective subjects | 81 G, `training_prospective` / `training_retrospective` / `Validating_prospective` |
| A0.1, A1, A3 | `PREPROCESSED_DIR` slices | 14 G |
| A1.3 | ≥2 checkpoints to average | `task3_unet_pro` 6, `task3_swin_unetr` 6, `task3_vit_pro` 3 |
| A3.2 | Measured Δα for all 20 transitions × 3 modalities | all 15 retro cells present in `reports/spectral/alpha_summary.csv`, 30 volumes each |
| A5.1 | Tubelet encoder artifact | `lejepa_pretraining/artifacts/encoder_step_00040000.pt`, strict load verified |

### Δα table for A3.2

Derived from `reports/spectral/alpha_summary.csv`, `retro` split, as
`Δα = α_target − α_source`. All 20 transitions resolve for all three modalities,
so the physics conditioning vector is constructible today with no new measurement.

| transition | T1W | T2W | T2FLAIR | | transition | T1W | T2W | T2FLAIR |
|---|---|---|---|---|---|---|---|---|
| 0.1T→1.5T | −0.851 | −1.402 | −0.944 | | 3T→5T | −0.334 | −0.154 | −0.162 |
| 0.1T→3T | −1.180 | −1.562 | −0.950 | | 3T→7T | +0.171 | +0.060 | −0.344 |
| 0.1T→5T | −1.514 | −1.716 | −1.112 | | 5T→0.1T | +1.514 | +1.716 | +1.112 |
| 0.1T→7T | −1.009 | −1.502 | −1.294 | | 5T→1.5T | +0.663 | +0.314 | +0.167 |
| 1.5T→0.1T | +0.851 | +1.402 | +0.944 | | 5T→3T | +0.334 | +0.154 | +0.162 |
| 1.5T→3T | −0.329 | −0.160 | −0.005 | | 5T→7T | +0.505 | +0.214 | −0.182 |
| 1.5T→5T | −0.663 | −0.314 | −0.167 | | 7T→0.1T | +1.009 | +1.502 | +1.294 |
| 1.5T→7T | −0.158 | −0.100 | −0.350 | | 7T→1.5T | +0.158 | +0.100 | +0.350 |
| 3T→0.1T | +1.180 | +1.562 | +0.950 | | 7T→3T | −0.171 | −0.060 | +0.344 |
| 3T→1.5T | +0.329 | +0.160 | +0.005 | | 7T→5T | −0.505 | −0.214 | +0.182 |

**Exactly 30 of the 60 entries are negative.** Report II argued from a nine-entry
table that UniField's fixed band weights "would be harmful above 1.5 T"; on the full
grid the situation is sharper — a transition-independent, fixed-sign band weighting
has the wrong sign for precisely half the transition × modality cells. That is the
argument for B5 (spectral-profile loss, correct by construction) over deriving signed
band weights by hand.


---

## 7. Why the tubelet submission lost (2026-09-05)

Measured on held-out subject 0009 with the epoch-15 weights that were submitted. The failure is
**not** blur — the first hypothesis, and the measurement contradicted it. Predictions carry *more*
high-frequency energy than targets (ratio 1.06, 0.90, 23.0, 2.28 across four transitions; the 23x is
T2W 3T->0.1T, where the target is nearly smooth and the model paints texture into it).

Two periodic artifacts, both structural:

| Artifact | Measurement | Cause |
|---|---|---|
| In-plane patch grid | error power in the 14-18 px band is **709x the median**, peak at 16.15 px | 16x16 tubelet stem upsampled 16x by `ConvTranspose3d` |
| Axial slab seams | z-gradient at every 16th slice **2.17x** the surrounding slices | 16-slice slabs predicted independently, never blended |

LPIPS punishes periodic structure hard, which is why it came back at 0.1436 against identity's 0.1573
while nRMSE stayed a competitive 0.2658. **More epochs cannot fix either.** If the tubelet is revisited,
the two cheap inference-side tests come first, neither needing retraining:

- [ ] **7.1** Overlapping slab inference with blending, as the swin path already does (`--overlap 0.25`).
- [ ] **7.2** Shifted-grid TTA — average predictions over in-plane offsets, which cancels a fixed
  16 px grid by construction.

Only if those close most of the LPIPS gap is a decoder change (resize+conv instead of transposed
convolution, the standard checkerboard fix) worth the retraining cost.

## 8. The local scorer is calibrated (2026-09-05)

`scripts/eval_holdout.py` predicted the tubelet's challenge SSIM at **0.8645**; it scored **0.859**.
Error 0.0055. The offset between local and challenge SSIM is ~+0.060 and is stable across models:

| Model | local SSIM | challenge SSIM | offset |
|---|---|---|---|
| Identity (no training, no memorisation) | 0.8955 | 0.8365 | +0.0590 |
| 2D U-Net e50 (0009 **was** in its training set) | 0.9579 | 0.8976 | +0.0602 |
| Tubelet e15 (0009 held out) | 0.9235 | 0.8590 | +0.0645 |

The first two agree to 0.0012 despite one having memorised the subject and the other having no
parameters at all. That says the gap is dominated by evaluation protocol (different subjects, full
volume vs the z-clipped 30-slice slab, unpublished normalisation) rather than by memorisation —
which is what makes it defensible to score on a training subject now that the holdout is off.

**Use it before spending a submission slot.** Subtract ~0.060 from local SSIM.


## 9. FiLM + zero-init residual on the 2D conditional U-Net (2026-09-05)

`task3_unet_film_residual`, 10 epochs, all 3 subjects, no holdout. Scored on subject 0009 — a
**training** subject, so these rank; they do not estimate generalisation.

### The matched comparison (both at epoch 10, same subject, same script)

| | SSIM | nRMSE | LPIPS |
|---|---|---|---|
| Baseline conditional U-Net | 0.93852 | 0.13473 | **0.03611** |
| **+ FiLM + zero-init residual** | **0.95114** | **0.13050** | 0.07151 |

**SSIM +0.01262, sd 0.00622, better on 57/60 transitions.** Six weeks of architecture search
(FPS-Former, Swin UNETR, tubelet, 3D U-Net) moved the leaderboard **+0.0012** total. This is 10x that
on the ranked metric. LPIPS roughly doubles — the same perceptual weakness the tubelet showed, now
localised to the residual head rather than to any tubelet-specific geometry.

### Checkpoint sweep (A0.5)

| Epoch | SSIM | nRMSE | LPIPS |
|---|---|---|---|
| 2 | 0.93982 | 0.17322 | 0.05594 |
| 4 | 0.94429 | 0.15366 | 0.06136 |
| 6 | 0.94977 | 0.13636 | 0.06513 |
| **8** | 0.95109 | **0.12985** | **0.06790** |
| 10 | **0.95114** | 0.13050 | 0.07151 |

SSIM saturates at epoch 8; e10 beats e8 by 5.3e-05, which is 50x smaller than the per-transition sd
(2.7e-03). **Take epoch 8** — same SSIM in practice, better nRMSE and LPIPS. Note `eval_holdout`'s
"best SSIM" line picks the first maximum in glob order and does not break ties; do not trust it blind.

### Not submitted, and why

Calibrated estimate 0.95114 - 0.060 = **~0.891** against the 0.9026 bar, and the offset grows with
model quality (+0.0590 identity, +0.0602 U-Net e50, +0.0645 tubelet) while a training-subject score
inflates further. Realistic **0.881-0.891**. A slot here buys a number we can already predict.

- [x] ~~**9.1 — Port FiLM + zero-init residual to the 3D U-Net.**~~ **RETRACTED — nothing to port.**
  `lejepa_pretraining/repo/mrixfields_downstream/baseline_unet3d/model.py` already has both: per-stage
  zero-init FiLM (`condition_projections`, `Linear(96, 2*C)`, `chunk(2)`, zeroed in `_initialize`) and a
  zero-init residual in logit space (`sigmoid(logit(padded) + residual)`). The 3D leader was never
  missing these. The 2D +0.0126 was the 2D model *catching up* to what the 0.9026 model already had —
  which also explains why it lands near, not above, the leader.


## 10. The FiLM+residual submission scored, and the calibration held (2026-09-05)

Submitted `task3_unet_film_residual` epoch 8. **Challenge SSIM 0.892839.** Predicted ~0.891 in section
9 before submitting; the error was **+0.0018**. That is the second consecutive correct prediction.

| Model | local SSIM | challenge SSIM | offset |
|---|---|---|---|
| Identity (no training) | 0.8955 | 0.836497 | +0.0590 |
| 2D U-Net e50 | 0.9579 | 0.897646 | +0.0603 |
| Tubelet e15 | 0.9235 | 0.859000 | +0.0645 |
| FiLM+residual e8 | 0.95109 | 0.892839 | +0.0583 |

**Offset mean 0.0605, sd 0.0028.** The local scorer is now a usable predictor: `challenge ~= local -
0.060 +/- 0.006`. Use it to refuse submission slots, not just to explain them afterwards.

**CORRECTED.** An earlier version of this section claimed local ranking does not transfer. It does;
the comparison behind that claim was not like-for-like.

| | local (0009) | challenge | offset |
|---|---|---|---|
| `task3_unet_pro` **e50** | 0.95789 | 0.897646 | 0.06024 |
| FiLM+residual **e8** | 0.95109 | 0.892839 | 0.05825 |

Local says plain-e50 is ahead by **+0.0068**; the challenge says **+0.0048**. Same sign, same order of
magnitude. **Local scores predict leaderboard order correctly** — provided the models compared got the
same training budget.

The real confound is the budget. The +0.0126 FiLM gain in the table above is e10 vs e10. But the two
models actually *submitted* were FiLM at **e8** and plain at **e50**. FiLM+residual has never been
trained past 10 epochs, and its loss was still falling when it stopped: e8 0.0197, e9 0.0193, e10
0.0190, against the plain model's 0.0120 after 63 epochs — **37% lower**. The submitted FiLM model is
under-trained, not out-classed, and no like-for-like leaderboard comparison of the two has ever run.

## 11. Test-time augmentation is dead (2026-09-05)

Four in-plane flips, averaged, on FiLM+residual e8, subject 0009:

| | SSIM | nRMSE | LPIPS |
|---|---|---|---|
| Plain | 0.95109 | 0.12985 | 0.06790 |
| 4-flip TTA | 0.95190 | 0.12590 | 0.06220 |

**+0.0008 SSIM** for 4x inference cost. nRMSE and LPIPS do improve, but neither is ranked
(`primary_metric: SSIM`). Below the noise floor being measured in section 13. Not pursuing;
`--tta` stays in `scripts/eval_holdout.py` and `make_task3_submission.py` as a diagnostic only.

## 12. The actual ceiling is the paired-data count (2026-09-05)

Audit of `training_retrospective`:

- **1939 volumes across 1056 subjects**, all 15 (modality, field) domains populated.
- **Zero subjects have two or more field strengths within one modality** — T1W `{1: 703}`,
  T2W `{1: 587}`, T2FLAIR `{1: 649}`. The split is *entirely unpaired*.
- Volumes per domain range 43 (T2W 5T) to 235 (T1W 7T).

So the supervised objective is trained on **3 paired subjects** while 1056 unpaired ones sit unused.
Everything clustering at 0.89-0.90 is what three brains buys. No architecture change addresses this;
sections 3, 7 and 9 were all re-ranking models fit to the same three subjects.

Cross-subject intensity variation is modest (1.07-1.46x) **except T1W 7T, where 0009 is a 2.34x
outlier** (fg mean 0.0907 vs 0006's 0.2126). That single outlier explains both the exploding
`->7T` nRMSE in every sweep and why the section-8 quantile map landed 3x too bright on 0009 T1W 3T->7T.

## 13. In flight (started 2026-09-05 evening)

### 13.1 The seed was never applied — A0.3 was unmeasurable, not just unmeasured

`[experiment].seed` was parsed, validated, and carried into every context object, and then **never
used**. The only seeding in the pipeline was each `DataLoader`'s own generator, which fixes shuffle
order but not weight initialisation; module constructors draw from the global torch RNG. **Every run
in this project started from different random weights**, and every comparison already contains that
noise — including section 9's +0.0126.

Fixed in `eval_pipeline/cli.py`: `_seed_rngs(config.seed)` seeds `random`, `numpy` and `torch` before
`build_model_factory().build()`. It has to be there, not in a trainer — by the time a training context
exists the weights are already drawn. Verified: seed 1 twice gives an identical
`encoders.0.0.weight` sum (3.680016), seed 2 gives -3.807679.

- [ ] **A0.3 — the noise floor.** `task3_film_seed1` / `task3_film_seed2`: `task3_unet_film_residual`
  with seed 1 and 2, everything else identical, 10 epochs each, scored on 0009 by the same script.
  The spread is the number that decides whether any result in this file means anything. Running.

### 13.2 Self-supervised pretraining on the unpaired 1056

`components/training/task3.py::_pretrain` already implements block-masked L1 inpainting and takes
`{image, domain}` — single images, **no pairs**. `forward(image, target_domain, source_domain=None)`
falls back to `source_domain = target_domain`, so `model(images, domains)` is a valid self-
reconstruction. The loss counts only masked, non-air voxels, so the zero-init residual cannot shortcut
it. `include_retro=true` builds the loader; `pretrain_steps > 0` runs the stage.

None of it has ever run, because **the retrospective slices were never extracted** — the cache held
only `pro_train`, `pro_val`, `volumes_3d`, so `include_retro=true` raised `FileNotFoundError` on the
`require_complete` check. That, not a design decision, is why `include_retro = false` everywhere.

Added `--max_subjects` to `scripts/preprocess.py` (there was no subject cap). Extracting **60 volumes
per domain = 883 volumes, ~15 GB at float16**; the full 1939 would be ~33 GB for diversity the
pretraining stage cannot consume. float16 is free: `ToTensor` ends in `.float()`. Verified the
retrospective normalisation matches prospective (fg mean 0.2057 vs 0.2006 on T1W 7T).

- [ ] **13.2 — pretrain on 883 unpaired volumes, then fine-tune on the 3 paired subjects.** ~294x the
  images the supervised stage sees. The only structural lever found that is not architecture.

## 14. A2 answered: calibration is not what the architectures were resolving (2026-09-05)

`scripts/quantile_calibration.py --score`, leave-one-subject-out, 180 transitions. Parameter-free
monotone histogram match, 512 knots, no network, nothing trained.

**SSIM 0.9083 | nRMSE 0.3345 | LPIPS 0.0856**

Paired against the identity transform on subject 0009 (both 60 transitions, 0009 unused in fitting
for both):

| | SSIM |
|---|---|
| Identity | 0.8955 |
| Quantile calibration (LOSO) | **0.9201** |
| FiLM+residual e8 — *0009 was in training* | 0.95109 |

**+0.0246 over identity, sd 0.0418, better on only 37/60.** It closes roughly 44% of the identity-to-
network gap for free, but loses on 23/60 transitions — 8 of the first 9 losses are T1W, the modality
with the outlier subject.

Calibrated estimate: 0.9083 - 0.059 = **~0.849** on the challenge, against the 0.9026 bar and the
0.897646 best. **So the answer to A2 is no**: a pure calibration lands ~0.05 below the trained models,
so the architecture comparisons were not merely re-ranking calibration quality. The network earns its
keep. Not worth a submission slot.

### 14.1 Subject variance is sd 0.011 — larger than most effects in this file

The same method scored on each held-out subject in turn:

| Held out | SSIM |
|---|---|
| 0006 | 0.9068 |
| 0007 | 0.8981 |
| **0009** | **0.9201** |

Mean 0.9083, **sd 0.0111** (n=3). Subject 0009 is **+0.0118 above the 3-subject mean** — and 0009 is
the subject every local score in this file is computed on. So absolute local numbers carry an
optimistic subject bias of roughly +0.01 on top of the +0.060 calibration offset.

Matched comparisons on the same subject (section 9) are still valid — the bias cancels on both sides.
Absolute local scores are not, and neither is any cross-method comparison run on different subjects.

This is the first measured variance in the project and it lands **before** the seed noise floor
(section 13.1) is even known. It is 14x the TTA effect (+0.0008) and comparable to the FiLM+residual
effect (+0.0126), which is the concrete form of the objection that the +0.02-chasing had to stop.

### 14.2 The ->7T failures are one subject, not a method problem

nRMSE `->7T` **0.5783** vs non-7T **0.2736**. The four worst transitions in all 180 are 0009 T1W ->7T
(nRMSE 2.01, 1.94, 1.48, 1.12): the map is fitted on 0006+0007, whose T1W 7T foreground means are
~0.21, and applied to 0009 at 0.0907 — 2.3x too bright, exactly the outlier from section 12. Any
per-subject intensity normalisation would remove most of this; it is not a flaw in the quantile map.

## 15. A1.3 closed: checkpoint averaging is real and too small (2026-09-05)

Weight averaging over `task3_unet_film_residual` checkpoints, scored on 0009, paired per transition
against the single best checkpoint (e8):

| | SSIM | nRMSE | LPIPS | delta vs e8 | better on |
|---|---|---|---|---|---|
| e8 (reference) | 0.95109 | 0.12985 | 0.06790 | — | — |
| `avg_all5` (e2,4,6,8,10) | 0.95034 | 0.13435 | **0.05903** | -0.00075 | 10/60 |
| `avg_late3` (e6,8,10) | **0.95196** | **0.12738** | 0.06638 | **+0.00087** | **52/60** |

`avg_late3` wins on 52/60 with sd 0.00124 — the effect is consistent, not chance. It is also
**+0.00087**, within rounding of the TTA effect (+0.0008) and about 13x smaller than the subject sd of
0.011 from section 14.1. `avg_all5` is *worse* than the checkpoint it averages over: e2 and e4 are
still 0.0068-0.0113 below e8 and drag the mean down. If averaging is used at all, average only the
converged tail.

**Closes A1.3.** Along with section 11 (TTA, +0.0008) this is the second free win measured this
session and the second one below the noise floor. Both are now known quantities and neither is worth
a submission slot; the remaining gap to the 0.9026 bar is ~0.010 on the leaderboard, an order of
magnitude more than either buys.

## 16. A0.3 answered: the seed noise floor is ~0.001 (2026-09-06)

`task3_film_seed1` / `task3_film_seed2`: `task3_unet_film_residual` re-run with seed 1 and seed 2,
identical in every other respect, scored on 0009 by the same script. With the original run that is
three independent initialisations of one config.

| | e8 | e10 |
|---|---|---|
| original (unseeded) | 0.95109 | 0.95114 |
| seed 1 | 0.94869 | 0.95116 |
| seed 2 | 0.95015 | 0.95098 |
| mean | 0.94998 | 0.95109 |
| **sd (n=3)** | **0.00121** | **0.00010** |

**Seed noise is ~0.001 at worst, and ~0.0001 at the converged epoch.** This was only measurable at all
after the fix in 13.1 — before that the seed was inert and could not be varied deliberately.

What it settles:

- The FiLM+residual gain (**+0.0126**) is **10x the seed floor**. It is a real effect, not an
  initialisation artefact. The worry that motivated this measurement is answered: no.
- TTA (**+0.0008**, section 11) and `avg_late3` (**+0.00087**, section 15) are **at or below** the e8
  floor of 0.00121. Both correctly discarded.
- The dominant noise term is **not** the seed. Subject variance is sd **0.0111** (section 14.1),
  **~11x** the seed floor at e8 and **~100x** at e10. Local scoring on a single subject is the
  limiting source of error in every comparison in this file.

### 16.1 Small correction to section 9

Section 9 concluded "take epoch 8", from an e10-minus-e8 gap of +5.3e-05 in one run. Across all three
runs e10 beats e8 every time (+0.00005, +0.00247, +0.00083). The original run's near-tie was the
outlier. **e10 >= e8 consistently**; the practical difference is still inside the noise floor, so
nothing submitted needs revisiting, but do not repeat the "SSIM saturates at e8" claim.

## 17. The 60-epoch run refutes section 9: FiLM+residual converges faster to a WORSE optimum (2026-09-06)

`task3_film_long` — the same config as `task3_unet_film_residual` at the plain U-Net's budget, seed 1,
constant LR. Scored on 0009, same script.

| Epoch | FiLM+residual | plain U-Net |
|---|---|---|
| 10 | 0.95114 | 0.93852 |
| **20** | **0.95356** (peak) | — |
| 30 | 0.95340 | — |
| 40 | 0.95323 | — |
| 50 | — | **0.95789** |

Paired against plain e50 on the same 60 transitions: **-0.00433, sd 0.00797, FiLM better on only
17/60.**

The run was **stopped at e47**, not carried to 60: SSIM had been flat within 0.0004 for 20 epochs and
the GPU was worth more to section 13.2. Checkpoints e5-e45 are on disk, so the curve is preserved.

**The prediction was wrong.** I expected FiLM+residual to pick up the +0.0194 the plain model gained
from e10->e50 and reach ~0.970 local. It gained **+0.0025** (e10 -> e20) and then plateaued and drifted
*down*. The e20->e40 decline (-0.0004) is inside the seed floor, so the honest reading is flat from
e20, not degrading.

**So section 9's +0.0126 was a convergence-rate artefact, not a quality gain.** The two architectures
were compared at e10, where FiLM+residual is simply further along its curve. At convergence the plain
U-Net is **better by 0.0043**, which is 4x the seed floor and therefore real. The zero-init residual
and FiLM buy early progress and a slightly worse optimum.

Note the objective and the metric diverge after e20: total loss keeps falling (0.0189 at e10, 0.0134
at e40, 0.0130 at e46) while SSIM is flat-to-down. L1 + 0.05*LPIPS is still being minimised on a
subject that is *in the training set*, and SSIM does not follow it. Any future "train it longer"
argument has to contend with this: **loss is not a proxy for the ranked metric here.**

Calibrated: 0.95356 - 0.0605 = **~0.893**, below the 0.897646 already banked. **Not submitted.**

### 17.1 What this costs the plan

Sections 9, 11, 15 and 17 are now four consecutive architecture/inference tweaks that did not beat
`task3_unet_pro` e50. Combined with section 14 (calibration alone reaches ~0.849) and section 12
(3 paired subjects vs 1056 unpaired), the evidence points one way: **the 2D supervised setup is
saturated at ~0.898 and more variations of it will not clear 0.9026.** The retrospective pretraining
run (13.2) is the only remaining untried lever that changes the amount of data rather than the shape
of the model.

## 18. Retrospective pretraining works, and is 5x too small (2026-09-06)

`task3_retro_pretrain`: 12,000 steps of block-masked L1 inpainting over 883 unpaired retrospective
volumes (194,260 slices), then the same 10-epoch supervised fine-tune. Seed 1, so `task3_film_seed1`
is the controlled baseline -- identical weight init, identical budget, pretraining the only difference.
`masked_l1` fell 0.1006 -> 0.0628 over the 12k steps, so the stage did train.

Paired per transition on 0009, matched epoch, same seed:

| | pretrained | control | delta | better on |
|---|---|---|---|---|
| e8 | 0.95103 | 0.94869 | **+0.00233** | 44/60 |
| e10 | **0.95324** | 0.95116 | **+0.00209** | 46/60 |

**The effect is real.** +0.0021 is 2x the seed floor at e8 (0.00121) and 20x the floor at e10
(0.00010), and it reproduces at both epochs on ~45/60 transitions. 294x more images for the encoder
buys a genuine, reproducible improvement.

**It is also nowhere near enough.**

| | local | calibrated |
|---|---|---|
| pretrained e10 | 0.95324 | ~0.8927 |
| long run e20 (no pretraining, 2x budget) | 0.95356 | ~0.8930 |
| **plain U-Net e50 — already banked** | **0.95789** | **0.897646** |

Pretrained-e10 lands within 0.0003 of what the *non*-pretrained run reached at e20. So what the
pretraining actually bought is roughly **2x convergence speed to the same plateau** -- and section 17
already showed that plateau (~0.9535) sits **0.0046 below** the plain U-Net at convergence. Beating
the banked 0.897646 needs +0.0047 local; pretraining delivered +0.0021.

**Not submitted.**

### 18.1 Where that leaves it

Five levers have now been measured against `task3_unet_pro` e50 and all five fall short:

| Lever | delta local | verdict |
|---|---|---|
| TTA (4-flip) | +0.0008 | at the noise floor |
| Checkpoint averaging (`avg_late3`) | +0.0009 | at the noise floor |
| FiLM + zero-init residual | -0.0043 at convergence | faster, worse optimum |
| Quantile calibration alone | ~0.849 calibrated | far below |
| **Retrospective MIM pretraining** | **+0.0021** | real, 2x too small |

The one thing never tried is the combination the evidence actually points at: pretraining is
architecture-agnostic and gave +0.0021 on the *weaker* of the two decoders. Applying it to the plain
conditional U-Net -- the 0.897646 model -- at its own 50-60 epoch budget is the only remaining
configuration where a measured gain sits on top of the best measured baseline rather than a worse one.
Expected ~0.9600 local, ~0.8996 calibrated: still short of the 0.9026 bar, which is the honest reason
to consider whether Task 3 is the right place to keep spending.

## 19. The pretrained model scored 0.901633 and broke the calibration (2026-09-06)

`task3_retro_pretrain` e10 submitted. **Challenge SSIM 0.901633** — a new best, past
`task3_unet_pro` e50's 0.897646, and **0.001 short of the 0.9026 bar.**

Per modality: T1W 0.901179, T2W 0.906179, T2FLAIR 0.897542. nRMSE 0.242206, LPIPS 0.085563.

**I predicted ~0.8927. The error was +0.0089, and it was structural, not luck.**

| Model | local | challenge | offset |
|---|---|---|---|
| Identity | 0.89550 | 0.836497 | 0.05900 |
| 2D U-Net e50 | 0.95790 | 0.897646 | 0.06025 |
| Tubelet e15 | 0.92350 | 0.859000 | 0.06450 |
| FiLM+res e8 | 0.95109 | 0.892839 | 0.05825 |
| **Retro-pretrained e10** | 0.95324 | **0.901633** | **0.05161** |

First four: mean 0.06050, sd 0.00279. The pretrained model sits **3.2 sd below** that. The
constant-offset predictor from section 10 is refuted for this class of model.

### 19.1 Why, and what it invalidates

**The offset is the generalisation gap.** Every local number in this file is measured on subject
0009, which is *in the training set*; the challenge scores three subjects no model has seen.
Retrospective pretraining is an intervention aimed squarely at generalisation, so of course it moves
the offset — and a constant-offset model assumes exactly that cannot happen.

On the metric that is actually ranked:

| | local delta | challenge delta |
|---|---|---|
| pretraining vs none, matched budget and seed | +0.0021 | **+0.0088** |

**Pretraining is worth ~4x more on unseen subjects than local scoring can see.** Sections 17 and 18
both judged levers with a ruler blind to the one property that matters. Specifically:

- Section 18's "real, 5x too small" verdict was **wrong**. It was 4x larger than measured.
- Section 18.1's estimate for pretraining the plain U-Net (~0.8996) used the constant offset and is
  therefore too low; on the corrected offset it is closer to **~0.906**.
- Section 17's "plateaus at ~0.9535, below plain e50" was measured on the *non-pretrained* model and
  locally. It does not license the claim that a *pretrained* model plateaus, and it says nothing about
  where either plateaus on unseen subjects.
- The rule going forward: **local score ranks models trained the same way. It cannot compare models
  that differ in how much data their encoder saw.** For those, only the leaderboard or a genuine
  held-out-subject protocol (LOSO) is admissible.

## 20. Ensembling: real but marginal, and a scoring bug worth remembering (2026-09-07)

`scripts/ensemble_eval.py` (new). Averages several 2D checkpoints' predictions in the [0,1] output
space; each member is rebuilt from its own state dict via `unet_from_state_dict`, so a plain Tanh
checkpoint and a FiLM+residual one ensemble without a config per member.

Three most-decorrelated strong members, subject 0009, 60 transitions:

| | SSIM | vs ensemble |
|---|---|---|
| **ENSEMBLE** | **0.95836** | — |
| `task3_unet_pro` e60 | 0.95728 | -0.00108 |
| `task3_film_long` e20 | 0.95356 | -0.00480 |
| `task3_retro_pretrain` e10 | 0.95324 | -0.00511 |

**+0.00108 over the best member**, and +0.00047 over `task3_unet_pro` e50 (0.95789). Real but marginal
-- the same order as TTA and checkpoint averaging, not the +0.005 hoped for.

**Not submitted.** Section 19.1 showed local scoring understates the *pretrained* model specifically,
because the offset tracks generalisation. Averaging it with two non-pretrained members should pull the
blend's offset back toward theirs, so the ensemble's challenge score is likely at or below
`task3_retro_pretrain` alone at 0.901633.

### 20.1 The bug: `axial_first` was omitted, and it moved every number by ~0.006

The first version of `ensemble_eval.py` did not apply the `axial_first` transpose that
`eval_holdout.evaluate` does -- `(x, y, z) -> (z, x, y)`, so SSIM is averaged over **axial** slices.
Scoring the wrong plane read every model ~0.006 low and, worse, *unevenly*:

| | wrong plane | correct (axial) | error |
|---|---|---|---|
| ENSEMBLE | 0.94895 | 0.95836 | -0.0094 |
| `task3_unet_pro` e60 | 0.94252 | 0.95728 | **-0.0148** |
| `task3_film_long` e20 | 0.94262 | 0.95356 | -0.0109 |
| `task3_retro_pretrain` e10 | 0.94745 | 0.95324 | -0.0058 |

The error is 2.5x larger for one member than another, so it did not cancel: on the wrong plane the
ensemble appeared to beat its best member by **+0.0015** and `unet_pro` looked like the *worst*
member. Correctly scored, `unet_pro` is the *best* member and the ensemble gain is +0.0011.

The check that caught it: members whose standalone scores are already known must reproduce them. After
the fix `film_long` e20 gives 0.95356 and `retro_pretrain` e10 gives 0.95324, matching
`reports/long_run/` and `reports/retro_pretrain/` to five decimals. **Any new scoring path must be
validated against an existing number before its results are believed.**

## 21. Scaling the pretraining is exhausted; the fine-tune is not (2026-09-07)

`task3_retro_pretrain_big`: 30,000 MIM steps over the **full** retrospective split (1,939 volumes,
426,580 slices) vs the first run's 12,000 over 883 volumes, then a 25-epoch fine-tune. Seed 1
throughout, so every comparison below is matched on initialisation.

| checkpoint | SSIM | nRMSE | LPIPS |
|---|---|---|---|
| e5 | 0.94715 | 0.13914 | 0.04747 |
| e10 | 0.95392 | 0.12248 | 0.04313 |
| e15 | 0.95139 | 0.14290 | 0.04617 |
| e20 | 0.95367 | 0.13127 | 0.04832 |
| **e25** | **0.95535** | 0.13176 | **0.04178** |

### 21.1 More pretraining bought almost nothing

| comparison (paired, 60 transitions) | delta | better on |
|---|---|---|
| big e10 (30k steps / 1939 vol) vs prev e10 (12k / 883 vol) | +0.00067 | **36/60** |
| big e10 vs no pretraining (seed1 e10) | +0.00276 | 48/60 |
| prev e10 vs no pretraining | +0.00209 | 46/60 |

Pretraining itself reproduces cleanly: **+0.002 to +0.003 on ~47/60**, twice measured. But **2.5x the
steps and 2.2x the data added +0.0007 on 36/60** -- 36/60 is barely off a coin flip. The lever
saturates almost immediately. **Do not run 60k steps.**

This also retro-justifies not trusting `masked_l1`: its power-law fit predicted a further -0.0032 from
16k to 30k and delivered it, and that loss reduction bought +0.0007 downstream. The reconstruction
loss and the ranked metric are effectively unrelated.

### 21.2 But a pretrained model does not plateau where a non-pretrained one does

| comparison | delta | better on |
|---|---|---|
| big e25 vs big e10 | +0.00143 | **55/60** |
| big e25 vs big e20 | +0.00168 | 48/60 |

Section 17's *non*-pretrained run peaked at e20 and declined (0.95356, 0.95340, 0.95323 at e20/30/40).
This one is at its maximum at e25 and still rising. The per-transition means are noisy (sd 0.020 --
a few transitions swing hard between epochs) but **55/60** makes the direction credible where the mean
alone would not.

So the two levers behave oppositely: pretraining *scale* is done, fine-tuning *length* is not.

### 21.3 Submitted

big e25, local **0.95535**. On the pretrained-model offset (0.05161, the only relevant anchor, n=1)
that predicts **~0.9037** -- the first candidate in this file to project above the 0.9026 bar.

- [ ] **21.4 — `task3_retro_long_ft`.** 50 supervised epochs resuming from
  `task3_unet_pretrain_30000.pt` via `[model.params].checkpoint`, `pretrain_steps = 0`. Reuses the
  finished 5h of MIM instead of repeating it, and locates where a pretrained model actually turns
  over. Queued behind the submission build.

## 22. 0.903110 — past the bar (2026-09-07)

`task3_retro_pretrain_big` e25 submitted. **Challenge SSIM 0.903110**, against Furkan's 3D U-Net at
**0.9026**. Ahead by **+0.00051**. Per modality: T1W 0.902832, T2W 0.908522, T2FLAIR 0.897977.
nRMSE 0.245574, LPIPS 0.083882.

Predicted ~0.9037 from local 0.95535 on the pretrained offset; **error -0.0006**.

### 22.1 The calibration has two regimes, and both are now pinned

| regime | offset | n | sd |
|---|---|---|---|
| not pretrained | 0.06050 | 4 | 0.00279 |
| **pretrained** | **0.05192** | 2 | **0.00045** |

They differ by **0.00858 — 3.1 sd** of the non-pretrained spread, and the pretrained offset is 6x
*tighter*. Section 19.1's claim that the offset tracks generalisation is now measured twice rather
than inferred once.

Practical rule, refined from 19.1: **local score ranks models within a regime, never across.** Within
the pretrained regime a local delta transfers at roughly 0.7x (e10 -> e25: local +0.00211, challenge
+0.00148). Across regimes it is meaningless -- section 18 called pretraining "5x too small" from a
local +0.0021 that was worth +0.0088 on the leaderboard.

### 22.2 Progression

| submission | challenge |
|---|---|
| identity | 0.836497 |
| tubelet e15 | 0.859000 |
| FiLM+residual e8 | 0.892839 |
| plain U-Net e50 | 0.897646 |
| retro-pretrain e10 | 0.901633 |
| **retro-pretrain-big e25** | **0.903110** |

Everything from 0.892839 up came from the same architecture; the only things that moved it were
**pretraining on unpaired data** (+0.0088) and **fine-tuning longer** (+0.0015). No architecture
change contributed after the plain conditional U-Net.

### 22.3 Projection for the run in flight

`task3_retro_long_ft` (50 epochs from `task3_unet_pretrain_30000.pt`), on offset 0.05192:

| local | predicted challenge |
|---|---|
| 0.9560 | 0.9041 |
| 0.9570 | 0.9051 |
| 0.9578 | 0.9059 |
| 0.9590 | 0.9071 |

e25 reached 0.95535 and was still climbing (better than e10 on 55/60), so 0.9560-0.9580 by e50 is the
reasonable band -- i.e. **0.904-0.906**.

## 23. Multi-contrast input — implemented, queued (2026-09-07)

**Why.** Task 3 is pure contrast mapping: source and target are the same subject, already
registered, so anatomy is free (identity scores 0.836497) and only the intensity relationship
has to be learned. From **one** contrast that relationship is under-determined — a single
intensity per voxel cannot separate white matter from grey from a partial-volume edge, so how
that voxel behaves at another field strength is guesswork. **Three** contrasts at the same
field largely pin down the tissue parameters, and from those the target field is close to
deterministic. We had been feeding the network one channel and making a well-posed problem
ill-posed.

It costs nothing: the training split has all 3 modalities x 5 fields per subject, and the
validation split gives each subject all three modalities at its one source field (0001-0003
appear in T1W, T2W and T2FLAIR, all at 0.1T). Verified directly from `Validating_prospective`.

A linear probe (fit 0006+0007, test 0009) is too crude to size the gain — the T1W ->7T rows
are wrecked by the 2.34x intensity outlier — but where it is well behaved the signal is
clear: T2FLAIR 0.1T->3T goes **R^2 0.001 -> 0.399**, 0.1T->7T **0.156 -> 0.504**, purely from
adding the other two contrasts.

### 23.1 Design: 4 channels, exact transfer

Input is **(primary, T1W, T2W, T2FLAIR)**, where channel 0 duplicates the contrast being
predicted. The duplication is what makes the seeding exact. A fixed (T1W, T2W, T2FLAIR)
ordering cannot transfer a single-channel first convolution, because the "right" channel
moves per sample; with a primary channel the pretrained weight moves onto channel 0 and the
three auxiliaries are zero-initialised.

**Verified bit-identical**: the widened model reproduces `task3_retro_pretrain_big` e25 at
max|diff| **0.000e+00** on real data, and perturbing the auxiliary weights moves the output
by 1.19, so they are wired in rather than inert. First-conv gradients reach them immediately
(aux 1.08e-02 vs channel-0 7.83e-03). **The run starts at exactly 0.903110 and any change is
attributable** — the same zero-init discipline as the residual head and FiLM.

New code, all additive:

- `mrixfields/data/cached_dataset.py::CachedMultiContrastDataset`
- `components/data/task3.py::Task3MultiContrastDataModule` (`task3_multicontrast`) — kept
  separate from `Task3DataModule`, which every existing checkpoint was trained through
- `conditional_unet.py` — residual base is channel 0 when input and output channel counts
  differ; `unet_from_state_dict` now reads `input_channels` off the first conv
- `scripts/widen_input_channels.py`
- `configs/task3_multicontrast.toml` — 25 epochs, seed 1, 39,600 samples / 619 batches
  (identical to previous runs, so epochs stay comparable)

Existing single-channel checkpoints still load strictly — confirmed.

- [ ] **23.2** Queued behind `task3_retro_long_ft`. Compare against that run's e25 at matched
  epoch: same seed, same data order, same budget, only the input channels differ.


## 24. A0.5 on `task3_retro_long_ft`: the pretrained fine-tune turns over at e25 too, and past it the model is unstable rather than saturated (2026-09-08)

**First: epochs 5-25 of this run are the same computation as `task3_retro_pretrain_big`'s
fine-tune, bit for bit.** Same 30k-step pretrain checkpoint via `[model.params].checkpoint`,
same seed, same data order, and the trainer re-seeds at the start of the supervised stage, so
resuming reproduced rather than continued. Train L1 matches step for step (0.05838, 0.01841,
0.01606, 0.01497, 0.01420 at e1/6/11/16/21) and e5 scores 0.94715 against 0.94715 on **0/60**
transitions. Half the 5h45m run re-measured numbers we already had; only e30-e50 are new.

That is worth keeping as a fact about the pipeline: **a resumed run is reproducible to the
last decimal**, which is what makes the paired comparisons in this file admissible at all.

| epoch | SSIM | nRMSE | LPIPS | vs e25 | beats e25 on | worse by >0.005 |
|---|---|---|---|---|---|---|
| 25 | 0.95535 | 0.13176 | 0.04178 | — | — | — |
| 30 | 0.95404 | 0.13313 | 0.04408 | -0.00131 | 23/60 | 6 |
| 35 | 0.95416 | 0.15217 | 0.04907 | -0.00119 | 31/60 | 6 |
| **40** | **0.95693** | **0.13049** | **0.04264** | **+0.00158** | **49/60** | **0** |
| 45 | 0.95383 | 0.14590 | 0.04736 | -0.00152 | 48/60 | 6 |
| 50 | 0.95515 | 0.13457 | 0.04452 | -0.00020 | 42/60 | 2 |

### 24.1 The mean and the win count disagree, and the tail is why

e45 beats e25 on **48/60** transitions and still has a **worse mean**. Six transitions crater,
led by T2W 3T->1.5T at **-0.0976**; e50 breaks the same transition at -0.0563. Past e25 the
median transition keeps improving (+0.0010 at e45, +0.0009 at e50) while individual transitions
blow up. **The late fine-tune is unstable, not saturated** -- which is a different diagnosis
from section 17's non-pretrained run, where the mean declined because everything declined.

Since the challenge scores the mean, an unstable checkpoint is worth nothing however good its
median is. e40 is the one late checkpoint with **no transition worse by more than 0.0022**, and
its gain is spread evenly across targets (+0.0006 to +0.0025) and modalities (+0.0011 to
+0.0024) rather than carried by a few. That makes its +0.00158 a broad improvement rather than
the maximum of a noisy curve -- but it is measured on one subject, so e40 being clean on 0009
is not evidence that it is clean on 0001-0003.

**Closes 21.4.** Both regimes turn over at the same place; the pretrained model's advantage is
in where it starts, not in how long it keeps climbing. Do not run 50-epoch fine-tunes again.

## 25. 23.2 answered: multi-contrast input is the largest local gain since pretraining (2026-09-08)

`task3_multicontrast`, 25 epochs from `widened4_from_e25.pt`, seed 1, same 619 batches/epoch.
Scored through a new 4-channel path in `scripts/eval_holdout.py`.

**The scoring path was validated before any of its numbers were read** (section 20.1's rule).
`widened4_from_e25` is bit-identical to `big_e25`, so through the 4-channel path it must return
big_e25's score: it returns **0.9553502 vs 0.9553502, delta +0.00e+00, 0/60 transitions differ**.
The patched 1-channel path likewise re-scored big_e25 at 0.95535 on 0/60.

| checkpoint | SSIM | nRMSE | LPIPS | vs big_e25 |
|---|---|---|---|---|
| e5 | 0.95623 | 0.13139 | 0.04130 | +0.00088 |
| e10 | 0.95706 | 0.14210 | 0.03940 | +0.00171 |
| e15 | 0.95756 | 0.12995 | 0.04271 | +0.00221 |
| **e20** | **0.95836** | **0.12888** | 0.04329 | **+0.00301** |
| e25 | 0.95742 | 0.14754 | 0.04398 | +0.00207 |

**e20 is +0.00301 over the shipped 0.903110 model on 52/60 transitions**, and unlike section
24's checkpoints the curve is smooth: e20 beats e15 on 49/60 and e25 on 42/60, so the peak is
real rather than the argmax of noise. Against big_e25 the median transition moves +0.00253,
**13 transitions improve by more than 0.005 and one degrades by more than 0.005**.

The gains sit where the argument predicted they would. T2W, the modality whose intensity
relationship is least determined by its own contrast alone, moves most: 1.5T->3T **+0.0182**,
1.5T->5T +0.0134, 0.1T->3T +0.0098. The one consistent loser is T2W 0.1T->1.5T (-0.0151), which
is the transition every model scores worst in absolute terms (0.77-0.80 against ~0.95 elsewhere).

For reference against section 24: mc e20 beats `lft_e40` by +0.00142 on 40/60, using 20 epochs
rather than 40.

### 25.1 The extra channels survive the move to the validation split

Local scoring reads training subjects; a submission reads validation subjects, and the extra
channels are only worth anything if they are registered to the primary there too. Foreground
Dice between contrasts at each subject's own source field:

| split | subject | T1W/T2W | T1W/T2FLAIR | T2W/T2FLAIR |
|---|---|---|---|---|
| train | 0006 @ 0.1T | 0.9704 | 0.9696 | 0.9564 |
| train | 0009 @ 0.1T | 0.9788 | 0.9663 | 0.9639 |
| **validation** | 0001 @ 0.1T | 0.9713 | 0.9592 | 0.9508 |
| **validation** | 0004 @ 1.5T | 0.9863 | 0.9828 | 0.9840 |
| **validation** | 0016 @ 7T | 0.9683 | 0.9637 | 0.9815 |

Validation is at least as well registered as training, so the input the model gets at
submission time is the input it was trained on.

### 25.2 Submission

`scripts/make_task3_submission.py` now reads `input_channels` off the first encoder conv and
loads the sibling contrasts from the validation split itself; nothing is a flag the caller can
get wrong. The path was validated the same way as the scorer: with `widened4_from_e25` the
4-channel path reproduces the 1-channel path at **max|diff| 0.000e+00** on T2W 1.5T->3T 0004
and T1W 0.1T->7T 0001.

Archive built at `submissions/task3_multicontrast_e20/task3.zip` (1310 MiB, 180 members,
60 per modality, 20 field pairs, no degenerate or wrong-shape volume). Replacing the real
auxiliaries with copies of the primary moves the prediction by 0.0454 mean absolute in brain
against a prediction mean of 0.3088, so the shipped volumes genuinely use the extra contrasts.

Uploaded as `unet_multi_contrast_e20`; scored **0.906773** (section 26). Local **0.95836**. On the pretrained offset (0.05192) that projects to
**~0.9064**; at the 0.7x within-regime transfer measured in 22.1, **~0.9052**. Either way ahead
of 0.903110. The caveat is that the offset regime was pinned on models that see one channel, and
this one sees four -- the projection is an extrapolation, and the submission is what tests it.

- [ ] **25.3** If the offset holds, the next question is whether MC and the 40-epoch schedule
  compose: `widened4_from_e25` fine-tuned to e40 rather than e25. Section 24 says epochs past 25
  are unstable, so this is not free.


## 26. 0.906773 — the multi-contrast submission scored, and the calibration survived four channels (2026-09-08)

`unet_multi_contrast_e20` (submission 9779729): **challenge SSIM 0.906773**, up **+0.003663**
from `unet_cond_retro30k_pro20ep`'s 0.903110. nRMSE 0.243607, LPIPS 0.080831.

**Predicted ~0.9064 from local 0.95836 on the pretrained offset; error +0.00037.** The implied
offset is **0.051587** against the regime's 0.05192 (n=2, sd 0.00045) -- 0.7 sd out. Section
22.1's offset was pinned on models that see one channel, and 25.2 flagged the projection as an
extrapolation; it was not. The offset tracks *how the encoder was trained*, not what the input
looks like.

| regime | offset | n | sd |
|---|---|---|---|
| not pretrained | 0.06050 | 4 | 0.00279 |
| **pretrained** | **0.05181** | **3** | **0.00040** |

The within-regime transfer rate needs revising upward: local +0.00301 became challenge
+0.00366, a ratio of **1.22x**, where e10 -> e25 gave 0.7x. Two points, opposite sides of 1.0 --
treat local deltas as transferring roughly one-for-one and stop claiming a discount.

### 26.1 The local per-modality signal predicted the leaderboard's, in order

| modality | local delta | challenge delta |
|---|---|---|
| T2W | +0.00353 | **+0.005451** |
| T1W | +0.00293 | +0.004461 |
| T2FLAIR | +0.00256 | +0.001076 |

Same ordering, T2W first and T2FLAIR last, which is what section 23's argument predicted: T2W is
the contrast whose intensity relationship is least determined by its own signal alone, so it has
the most to gain from the other two. The local eval on one seen subject is a weaker instrument
than the leaderboard, but on this comparison it got the direction, the magnitude and the ranking.

**This is also the first submission to improve all three metrics at once.** The previous step
(0.901633 -> 0.903110) bought SSIM at the cost of nRMSE (0.242206 -> 0.245574); this one moves
SSIM +0.0037, nRMSE -0.0020 and LPIPS -0.0031 together.

### 26.2 Standing

11th of 38 teams with an accepted 180/180 submission -- 10th in practice, since the leader at
exactly **1.000000** is `task3_broken.zip`, which is the ground truth by accident. The real
leader is 0.931402, and 4th-8th sit in 0.9137-0.9173, so **the gap to a top-five place is now
about 0.007**: roughly two more multi-contrast-sized steps.

Progression:

| submission | challenge |
|---|---|
| identity | 0.836497 |
| FiLM+residual e8 | 0.892839 |
| plain U-Net e50 | 0.897646 |
| retro-pretrain e10 | 0.901633 |
| retro-pretrain-big e25 | 0.903110 |
| **multi-contrast e20** | **0.906773** |

Multi-contrast input is the second-largest single lever found so far, after pretraining on the
unpaired retrospective data (+0.0088).

## 27. The leaderboard is queryable, and reading it changes the target (2026-09-08)

`scripts/leaderboard.py` pulls `syn74915588` live (token from `.env`, `synapseclient` 4.13.0 is
already in the `mri` env), keeps ACCEPTED 180/180 rows, drops the per-modality columns and ranks
teams by best SSIM. `--mine`, `--submissions`, `--csv <snapshot>`, `--out`.

### 27.1 The all-time top three are not a target

Its top three -- 0.9314, 0.9281, 0.9263, all with nRMSE near 0.11 against the field's ~0.23 --
come from **two teams inside a five-week window that closed 2026-07-07**. Every submission in the
whole challenge with nRMSE below 0.15 falls in that window; since then the best anyone has posted,
those two teams included, is 0.2151. One of the submissions that jumps is named `new_no_leak`.

**The evaluator did not change underneath them.** Four teams have submitted an unmodified copy of
the source and all four scored **0.836497 / 0.521604** to six decimals -- 2026-06-16, 08-19, 08-20
and our own `task3_input_validation` on 08-28. That fingerprint pins the metric as stable across
the entire period, so the discontinuity is in those submissions, not in the scoring.

Restricted to 2026-08-01 onward (336 submissions, 23 teams) we are **10th of 23**, the frontier is
**0.916747**, and the gap to close is **~0.010**, not 0.025.

| | SSIM | nRMSE | LPIPS |
|---|---|---|---|
| ours (`multi_contrast_e20`) | 0.906773 (10/23) | 0.243607 (15/23) | 0.080831 (**3/23**) |
| top-5 median | 0.914467 | 0.220767 | 0.098120 |
| best in period | 0.916747 | 0.216049 | 0.069441 |

### 27.2 We optimise the metric we already lead

3rd of 23 on LPIPS, 15th on nRMSE. **The five teams above us beat us on both SSIM and nRMSE while
being worse than us on LPIPS.** Our loss is `L1 + 0.05*LPIPS` with no structural term -- and SSIM
is what is ranked. Mining the other teams' submission names (they are free text) for recurring
words turns up `ssim` in 10 submissions across **6 teams**, alongside `synth` (2 teams),
`multimodal`, `rician`, `fieldmorph`, `seedavg`. The single best `ssim`-named submission is
`lesssteps_moressim?` -- but it is inside the pre-07-07 window, so it is evidence about what
people try, not about what it is worth.

- [ ] **27.3** Add a differentiable SSIM (or MS-SSIM) term to the training loss and sweep its
  weight. Smallest change on the list, aimed straight at the ranked metric, and the one lever
  where the field's naming says six teams got there first.

## 28. 0.908873 — SSIM in the loss is worth +0.0021, and the sampling grid nearly hid it (2026-09-08)

`components/losses/reconstruction.py::SSIMLoss` (`task3_ssim`), weight 0.5, on top of the
multi-contrast model. Submitted as `unet_multicontrast_ssim_ep8`: **challenge SSIM 0.908873**,
up **+0.002100** from 0.906773. nRMSE 0.241785 (-0.0018), LPIPS 0.088697 (**+0.0079**).

The loss reproduces the *scorer's* SSIM rather than the Gaussian-windowed version most libraries
ship: 7x7 uniform window, unbiased `NP/(NP-1)` covariance, border dropped -- skimage reflects and
then crops `(win-1)//2`, which is exactly a valid-mode filter, so `avg_pool2d` with no padding is
identical and cheaper. **Validated against skimage before use at max |diff| 3.2e-07.**

### 28.1 The trade is real and it is the one we wanted

LPIPS got materially worse and that is the point. SSIM with a uniform window rewards matching
local means and variances and tolerates losing high-frequency texture; LPIPS is precisely
sensitive to that texture. We were **3rd of 23** on LPIPS and **15th** on nRMSE with SSIM ranked,
so spending the metric we led to buy the two that are ranked is the correct currency. It showed
up in training at e5 (LPIPS 0.04748 against the plain run's 0.03923) and carried to the
leaderboard at +9.7%. nRMSE improved at the same time, so this is not simply blur.

### 28.2 A coarse grid nearly killed a working lever

The first run used `save_every = 5`, copied from `task3_multicontrast`, whose peak was at e20.
The SSIM term does not behave like that. Scored at e10 first, it read **0.9527** against the
matched mc e10's 0.95706 -- and was written off as a failure. e5 then scored **0.9598**, above
the best checkpoint in the project at the time.

Replaying the run at `save_every = 1` (bit-identical: e5 returned 0.9598 again, and every epoch
total matched to six decimals) gives the real curve:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | (10) |
|---|---|---|---|---|---|---|---|---|---|
| SSIM | 0.9478 | 0.9605 | 0.9585 | 0.9506 | 0.9598 | 0.9516 | 0.9522 | **0.9606** | 0.9527 |

**Adjacent epochs differ by up to 0.010 -- ten times the seed noise floor.** At weight 0.5 the
term does not just move the optimum, it destabilises the trajectory. The rule this earns:
**a checkpoint grid inherited from a different loss is an assumption, not a default.** One
number from one checkpoint of an unswept schedule is not evidence about a lever.

### 28.3 Checkpoint averaging does not fix an oscillation

The obvious hypothesis was that averaging cancels the swing. It does not -- it interpolates:

| | SSIM |
|---|---|
| `avg_late4` (e5-e8) | 0.9556 |
| arithmetic mean of those four | 0.9560 |
| `avg_good3` (e2, e5, e8) | 0.9603 |
| best member (e8) | 0.9606 |
| `avg_all8` | 0.9557 |

So the peaks and troughs are genuinely different-quality points in weight space, not a wobble
around a better midpoint. Section 15's +0.0009 does not improve on an unstable run.

### 28.4 Cross-subject scoring picked a different checkpoint than the argmax would have

On 0009 alone e2 (0.9605) and e8 (0.9606) are indistinguishable. Over all three paired subjects:

| | 0009 | pooled 180 transitions | vs mc e20 | better on |
|---|---|---|---|---|
| mc e20 | 0.95836 | 0.95478 | — | — |
| **mc+ssim e8** | 0.9606 | **0.95708** | **+0.00230** | **155/180** |
| mc+ssim e2 | 0.9605 | 0.95656 | +0.00177 | 103/180 |

e2 and e8 have nearly the same mean, and e8 wins on 155/180 where e2 wins on 103/180. **Picking
by single-subject argmax would have shipped the weaker model.** Given section 14.1's subject
spread of 0.011, scoring the final candidate on all three subjects should now be standard before
any submission -- it costs 8 minutes.

### 28.5 The calibration held for a model trained on the ranked metric

Predicted ~0.909 from local 0.9606; actual 0.908873, error 0.0001. Implied offset **0.051727**
against the pretrained regime's 0.051587 and 0.05181. Section 25.2's worry -- that a model
trained partly on SSIM would have a different generalisation gap -- did not materialise.

| regime | offset | n | sd |
|---|---|---|---|
| not pretrained | 0.06050 | 4 | 0.00279 |
| **pretrained** | **0.05171** | **4** | **0.00030** |

Local delta +0.00230 became challenge +0.00210, a transfer of **0.91x**.

- [x] **28.6 Swept, and negative: 0.5 stays.** Three weights, 41 checkpoints scored on 0009.

| weight | 0.10 (25 ep) | 0.25 (8 ep) | **0.50 (8 ep)** |
|---|---|---|---|
| best local SSIM | 0.9585 (e16) | 0.9548 (e6) | **0.9606 (e8)** |

`task3_mc_ssim_w010_long`, the full 25-epoch curve at weight 0.10:

| e2 | e4 | e6 | e8 | e10 | e12 | e14 | e16 | e18 | e20 | e22 | e24 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9542 | 0.9554 | 0.9574 | 0.9580 | 0.9526 | 0.9573 | 0.9551 | **0.9585** | 0.9560 | 0.9577 | 0.9560 | 0.9516 |

Three things this kills at once. **The premise was wrong**: a lower weight does not buy a stable
curve, it oscillates by the same +/-0.003 between adjacent epochs while the training loss falls
monotonically. **The response is not monotone in the weight**: 0.25 scored below both of its
neighbours, so there is no gradient to follow and no interpolation to trust. **Length does not
rescue it**: 25 epochs at 0.10 peak at 0.9585, below the 8-epoch run at 0.50, and e24 falls to
0.9516.

Epochs 2-8 of the 25-epoch run reproduce the 8-epoch run to the last digit (0.9542 / 0.9554 /
0.9574 / 0.9580), so the trajectory is deterministic and the swings are a property of the SSIM
term, not of the run.

The lever's gains have to be *harvested by dense sampling*, not engineered away: `save_every = 1`
and score every epoch. That is also the standing lesson from 28.2, arrived at twice now.

## 29. 25.3 answered: multi-contrast plateaus at e20, it does not compose with a longer schedule (2026-09-08)

`task3_mc_long`: resumed from `task3_multicontrast` e25 for 15 more epochs, same loss, same data.

| schedule epoch | 5 | 10 | 15 | **20** | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|
| SSIM | 0.95623 | 0.95706 | 0.95756 | **0.95836** | 0.95742 | 0.95741 | 0.95681 | 0.95848 |

e40 vs e20: **+0.00012 on 32/60** -- 32/60 is a coin flip and the delta is an eighth of the seed
noise floor. Fifteen epochs and ~2 h of GPU bought nothing.

One difference from section 24 worth keeping: the single-channel model became *unstable* past
e25, with individual transitions collapsing (T2W 3T->1.5T at -0.0976). The 4-channel model does
not do that -- it simply flattens, and nRMSE and LPIPS stay sane throughout (0.1417 / 0.0412 at
e40). More input information appears to buy stability as well as accuracy.

**Both fine-tuning schedules are now closed.** Neither the 1-channel nor the 4-channel line has
anything past its peak, and those peaks are at e25 and e20. Long fine-tunes are done as a lever.

## 31. The 0.1T deficit is information-limited, not under-fitted: weighting its loss makes it worse (2026-09-09)

`task3_mc_low01` = `task3_mc_ssim_long` plus two keys, `low_field_weight = 0.5` and
`low_field_ends = "either"`. The weight is not a guess: 40% of samples touch 0.1T, so
0.4(1+w)/(0.4(1+w)+0.6) = 0.5 solves at w = 0.5, the value that makes 0.1T's share of the loss
equal its share of the error. 20 epochs, `save_every = 1`, everything else identical.

**Best-of-20 0.9603 (e3) against the baseline's 0.9620 (e12); mean delta over all twenty matched
epochs -0.0017.** Per-field, comparing each run's best checkpoint on 0009:

| cell | baseline | weighted | delta |
|---|---|---|---|
| 0.1T as source *(weighted)* | 0.9514 | 0.9471 | **-0.0043** |
| 0.1T as target *(weighted)* | 0.9604 | 0.9614 | **+0.0010** |
| 0.1T-touching (24) | 0.9559 | 0.9543 | -0.0016 |
| everything else (36) | 0.9661 | 0.9643 | -0.0018 |

**Mapping from 0.1T got worse despite being the thing upweighted**, and the two groups degraded by
the same amount, so there was no trade-off to speak of -- the extra gradient bought nothing
anywhere. The single cell that improved is 0.1T *as target*, the information-*destruction*
direction. That is the asymmetry section 30's error decomposition hinted at, now measured:
low-field to high-field cannot be fixed by allocating more loss to it, because the information is
not in the input. LPIPS also degrades monotonically through the run (0.045 -> 0.115) while the
baseline stays near 0.05.

With the noise analysis (section 27) reaching the same conclusion from the marginal-distribution
side, the 0.1T line is closed on evidence: **synthetic data and noise reparameterisation would
have been attacking a ceiling, not a deficit.**

Implementation note: `low_field_weight` defaults to 0.0 and every new path is behind
`if low_field_weight > 0.0`, so all configs predating this keep their trajectories bit-identical.
