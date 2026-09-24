# Routes After Report II

Proposed study routes for MRIxFields 2026, with the measurements that motivate them.
Companion to `progress_report_2.tex` / `progress_slides_2.tex`.

Written 2026-08-31. Evidence: `runs/task3_unet_pro`, `data/training_prospective`,
`Baseline Experiment Logs - Baselines.csv` (through submission 9778735).

---

## 1. Findings

Five measurements taken against the current checkpoints and paired subjects. They are the
reason the routes are ordered the way they are.

### F1 — On Task 1, doing nothing wins the segmentation metrics

The `INPUT` control has the highest Dice of any model in the log, and its volume consistency is
within 0.002 of the best.

| Task 1, modality-averaged | nRMSE ↓ | SSIM ↑ | LPIPS ↓ | Dice ↑ | Volume ↑ |
|---|---|---|---|---|---|
| **INPUT (identity control)** | 0.4085 | 0.8935 | 0.0897 | **0.8549** | 0.8586 |
| CUT | 0.3041 | **0.9053** | **0.0750** | 0.8542 | **0.8601** |
| CycleGAN | **0.3002** | 0.9052 | 0.0764 | 0.8498 | 0.8485 |
| DPT-LeJEPA | 0.3300 | 0.8970 | 0.0817 | 0.8510 | 0.8551 |
| DPT-DINO | 0.3248 | 0.8978 | 0.0812 | 0.8468 | 0.8513 |
| DPT-Random | 0.3235 | 0.8974 | 0.0835 | 0.8474 | 0.8508 |

Two of the five ranked Task 1 metrics are being scored but not competed for — and unlike nRMSE,
they are metrics a translator ought to be able to win outright.

### F2 — The conditional U-Net ignores the source-field label

Holding the input slice fixed and changing only the declared source domain moves the output by
**1.1–2.1%** of its dynamic range. Changing the target domain moves it by **19.7–53.7%**. The
trained model is effectively `f(image, target_field)`.

Δα ranges from +1.50 at 0.1 T to −0.51 at 5 T, so the twenty transitions demand qualitatively
different behaviour — and a model blind to its source cannot supply it. Plausibly there is no
gradient pressure: with three subjects the source field is trivially readable off the image, so
the label is redundant during training.

> Measured on `task3_unet_finetune_60.pt`, subject 0006, axial slice 180, all three modalities,
> target fixed at 7 T. Mean absolute output change relative to the 1–99% output range.

### F3 — Target conditioning mostly sets brightness and contrast

Fit one global affine `a·out(→0.1T) + b` to `out(→t)` and the residual mostly disappears:

| Modality | Fraction of conditioning effect explained by one global affine |
|---|---|
| T1W | 74–96% |
| T2W | 36–79% |
| T2FLAIR | 12–92% (highest at the 7 T target) |

There is real spatially-varying behaviour left over, but the dominant term is a global intensity
control. That is not a defect on its own — it becomes a problem next to F4.

### F4 — A zero-parameter intensity map removes 46% of the identity's error

Fit a per-(modality, source, target) monotone quantile map on two of the three paired subjects,
apply it to the third. No network anywhere.

| Transition (LOSO) | Identity | Calibrated | Δ |
|---|---|---|---|
| 0.1T → 7T | 1.6233 | 0.5816 | −1.0418 |
| 3T → 7T | 1.5243 | 0.6626 | −0.8617 |
| 1.5T → 7T | 1.3465 | 0.6094 | −0.7371 |
| 0.1T → 5T | 0.9309 | 0.3214 | −0.6095 |
| 5T → 7T | 0.7898 | 0.5222 | −0.2676 |
| … | | | |
| 3T → 1.5T | 0.2262 | 0.2296 | +0.0034 |
| 0.1T → 3T | 0.3445 | 0.3714 | +0.0269 |
| 1.5T → 3T | 0.2050 | 0.2448 | +0.0398 |
| **All 20 transitions** | **0.6218** | **0.3358** | **−0.2860** |

Put beside F3, this is the finding that should change what runs next. If a parameter-free
calibration captures most of the distance between the identity control and the U-Net, then the
leaderboard gap between 0.522 and 0.236 is largely calibration quality — and every architecture
comparison so far has been resolving differences in how well each model performs a histogram
match.

> Local nRMSE is `‖p−t‖₂/‖t‖₂` over the target's brain mask, which is **not** the platform's
> normalisation. Absolute values are not leaderboard-comparable; only the ratio is claimed. The
> three regressions are transitions whose scales already agree, where a two-subject fit overfits.

### F5 — Registration is clean, so it is not what is blurring the outputs

A negative result, included because it closes off a plausible explanation cheaply. The paired
prospective volumes share a grid (364×436×364 at 0.5 mm, identical affines), and residual
misalignment is small: brain-mask Dice 0.968–0.993, centre-of-mass offsets 0.09–1.41 mm.

Supervised L1 is not being trained against misaligned targets. The structured edge-and-vessel
error in Report II is a real loss of detail, not a registration artefact.

---

## 2. Routes

### R0 — Make the comparisons measurable before running them
**Do first · 2–3 days**

The A/B/C/D grid looks for differences of maybe 0.005 nRMSE. Right now the only readout is a
leaderboard submission per configuration, with no local held-out signal, no per-transition
breakdown, and early stopping that monitors training loss (with a comment in the source
acknowledging it cannot see overfitting). Four runs under those conditions produce four numbers
you cannot order.

- **R0.1 — Leave-one-subject-out harness.** Subjects `0006`, `0007`, `0009` give three folds.
  Report *per transition × per modality*, not a single average across transitions whose Δα
  differs by an order of magnitude. This is Report II's "evaluate per transition" item, made
  local and free.
- **R0.2 — Submit the calibration control.** One submission, no training: the F4 quantile map fit
  on all three paired subjects. It sets the real zero point for Task 3, the way `INPUT` did for
  the baselines. If it scores near the U-Net, every later comparison must be read against it.
- **R0.3 — Establish the seed-noise floor.** Run configuration A twice with different seeds. If
  |B−A| is smaller than the seed spread, the grid reports nothing.
- **R0.4 — Point early stopping at the held-out fold.** Once R0.1 exists,
  `early_stopping_monitor` has something real to watch.

### R1 — Conditioning, the highest-leverage part of the U-Net
**Do first · ~1 week**

Conditioning is worth 0.16 nRMSE in your own logs — unconditioned 0.402 against conditioned
0.236 at matched 10 epochs, a clean ablation since `UnconditionalUNet` is the same architecture
minus two embeddings. Every architecture swap so far has been worth roughly nothing. F2 and F3
say the current scheme uses only part of that headroom.

- **R1.1 — FiLM at every decoder scale.** Replace the single broadcast add at the bottleneck with
  per-scale scale-and-shift applied *after* each `InstanceNorm`. You have written this block
  twice already — `_ConditionalRefinement` in `conditional_swin_unetr.py` and the adaptive-norm
  unit in `conditional_vit.py` — including the zero-init trick, so the model starts as exactly
  the current one.
- **R1.2 — Condition on physics, not fifteen opaque indices.** Feed
  `(modality one-hot, log B₀_src, log B₀_tgt, Δα(src→tgt))` through a small MLP to produce the
  FiLM parameters. The twenty transitions then share structure instead of each learning its own
  vector from three subjects, and measured Δα enters the model directly rather than being
  hand-approximated as UniField does.
- **R1.3 — Make the source label carry weight.** Given F2, gate the *magnitude* of the correction
  on the transition; pair with R2.1 and scale the residual by a learned function of Δα. A
  transition with Δα ≤ 0 should be able to learn to leave the image nearly alone.
- **R1.4 — Control: delete the source embedding.** If F2 holds under training, removing it costs
  nothing and simplifies the model. If it costs something, F2 was measuring one checkpoint and
  the conclusion needs weakening. Either outcome is informative.

### R2 — Output parameterisation and loss
**Do first · 3–5 days**

- **R2.1 — Predict the residual, not the image.** The head is `Conv1×1 → Tanh`, so the network
  reconstructs the whole image from scratch through a saturating nonlinearity. The identity
  already scores SSIM 0.836 — free structure being thrown away and re-earned. Switch to
  `out = x + f(x)` with a zero-initialised head. Should matter most at 3T→7T and 5T→7T where
  Δα ≤ 0, and removes Tanh saturation risk at 0.1 T. Roughly a ten-line change with the best
  return per line here.
- **R2.2 — Spectral-profile loss instead of hand-set band weights.** Report II already shows
  UniField's fixed `w_b` would be harmful above 1.5 T (three of nine Δα entries are negative).
  Rather than deriving signed band weights, penalise `|log P_pred(f) − log P_tgt(f)|` over the
  0.02–0.20 fit band. That optimises the exact quantity Report I measured, gets the sign right by
  construction at every transition, and needs no per-transition tuning.
  `scripts/spectral_analysis.py` has the RAPSD machinery.
- **R2.3 — Sweep the loss against the actual scoring rule.** The challenge ranks per metric and
  sums ranks, so a 0.001 LPIPS gain can outweigh a 0.01 nRMSE loss. The current `L1 + 0.05·LPIPS`
  was not chosen against that rule. Sweep the LPIPS weight, add MS-SSIM, report the Pareto front
  over the three ranked metrics.
- **R2.4 — Take the free wins first.** Eight-way dihedral TTA, and weight averaging over the
  checkpoints already in `runs/task3_unet_pro/artifacts/`. Near-zero cost, reliably help nRMSE
  and SSIM, and neither appears anywhere in the log.

### R3 — SynthSeg as an anatomical prior
**Highest ceiling · 2–3 weeks**

The distinction that matters is whether SynthSeg must run at *inference* on a low-field source,
or only at *training* time on data you already hold.

> **Gate — run first, one day.** Does SynthSeg hold at 0.1 T? Segment the three subjects' 0.1 T
> volumes and score against SynthSeg on their paired 7 T, per region. R3.1 and R3.6 depend on the
> answer; R3.2–R3.5 do not. Two practical notes: only `synthseg_1.0.h5` is downloaded (not the
> 2.0 robust models), and TensorFlow is not installed in the `mri` env — this needs the separate
> environment used for `Evaluation/segment.py`.

- **R3.1 — Tissue posteriors as extra input channels.** *Gated.* Give the translator
  `(image, p_CSF, p_GM, p_WM)`. T1, T2 and T2* all vary with B₀ *and* differ between tissues, so
  the correct transform is tissue-dependent — precisely the argument the synthetic generator is
  built on. Right now the network must infer tissue identity from intensity, in the one regime
  where intensity is least reliable.
- **R3.2 — Auxiliary segmentation head.** *Ungated.* Add a head on the decoder predicting the
  SynthSeg labels of the target, supervised by a one-time offline pass. Training-time only, so no
  inference dependency and no gate risk. Forces decoder features to stay anatomically grounded
  rather than merely photometrically correct — the failure mode F3 and F4 point at.
- **R3.3 — A contrast-agnostic perceptual loss.** *Ungated, and the most interesting item here.*
  The LPIPS term currently uses AlexNet features trained on natural photographs. SynthSeg's
  encoder was trained specifically to be *invariant to MRI contrast and resolution* — exactly the
  invariance a field-translation perceptual metric should have. Distil SynthSeg into a small
  PyTorch 3D U-Net (its own generator supplies unlimited training data), freeze it, and use its
  features alongside or instead of AlexNet. R3.4 then gets its differentiable segmenter for free.
- **R3.4 — Claim the metrics nobody is competing for.** F1 says no Task 1 model beats the identity
  on Dice or volume consistency. With R3.3's differentiable segmenter, add a term on the fourteen
  scored regions' volumes and a soft-Dice term directly. Clearest rank-sum win available on
  Tasks 1 and 2.
- **R3.5 — Rebuild the synthetic generator on SynthSeg's own machinery.** *Ungated.* Subsumes most
  of Report II's generator to-do list. `estimate_priors.build_intensity_stats` estimates per-label
  GMM means and stds from real volumes — per-field priors over ~32 labels instead of three
  tissues, which removes the over-summing weight in Eq. 5 outright since hard labels are disjoint.
  `BrainGenerator` then adds what the four-step operator does not model at all: bias field, slice
  thickness and anisotropic resolution simulation, nonlinear deformation. Per-sample per-label
  sampling should also fix the 1.5 T diversity shortfall structurally rather than by tuning.
- **R3.6 — Supervise on all 80+ regions, evaluate on the scored 14.** *Partly gated.* The full
  parcellation is a much richer training signal and costs nothing extra to generate. Weight the
  loss toward the scored regions; keep the rest as auxiliary structure.

### R4 — Architecture: the route to argue against
**Deprioritise**

Three architecture swaps are in the log and none has beaten a plain conditional U-Net.
Swin UNETR 3D reaches 0.300 / 0.832 / 0.171. FPS-Former at 150 epochs reaches
0.248 / 0.879 / 0.150 — still 0.067 behind the U-Net on LPIPS, the metric a frequency-aware
architecture was supposed to win. The volumetric U-Net with the LeJEPA encoder ties the 2D one at
0.238 / 0.889 / 0.089.

Meanwhile conditioning alone is worth 0.16. The binding constraint is not capacity or attention
design. If spending here anyway, in descending order of expectation:

- **R4.1 — 2.5D input stack.** Feed three to five adjacent slices, predict the middle. The 3D
  model currently ties the 2D model at much higher cost, suggesting through-plane context is
  worth little — this is the cheap way to find out, one line in the dataset.
- **R4.2 — Per-transition low-rank adapters.** One shared backbone with twenty small adapters,
  addressing "the transitions are not interchangeable" directly and — worth checking against the
  rules — still a single parameter set for Task 3.
- **R4.3 — A restoration backbone rather than a reconstruction one.** NAFNet or Restormer, if a
  backbone gets swapped at all. Both are built for the degradation-inversion problem this
  actually is, unlike FPS-Former, whose premise (high frequencies present but undersampled in
  k-space) Report II already identifies as not holding here.

### R5 — The 2×2, once it can resolve anything
**After R0 · 1 week**

The grid is well posed and its comparisons are the right ones: `B−A` for the encoder, `C−A` for
the synthetic data, `D−C` against `B−A` for independence. The problem is purely that it has no
instrument. Run it after R0, report per transition, and resist adding a third factor — at four
cells the design is already at the edge of what three subjects can separate, and R0.3 will say
whether even that is optimistic.

One substantive addition: include the R0.2 calibration control as a fifth row. If synthetic
pretraining moves a model from below it to above it, that is a far more legible claim than a
fractional nRMSE change between two cells.

---

## 3. Suggested sequence

| When | What | Why |
|---|---|---|
| Days 1–2 | **R2.4, R2.1** — TTA, checkpoint averaging, residual head | Hours of work, no new infrastructure, and they change the number every later comparison is measured against |
| Week 1 | **R0** — LOSO harness, calibration-control submission, seed-noise floor | Nothing after this is interpretable without it; R0.2 may reframe the whole Task 3 result |
| Week 2 | **R1.1, R1.2** + the **R3 gate** | Highest-leverage model change, run against a working instrument; the gate answers in a day and decides half of R3 |
| Weeks 3–4 | **R3.3, R3.4, R2.2** — perceptual loss, Dice/volume terms, spectral-profile loss | R3.4 is the clearest unclaimed rank-sum win on Tasks 1 and 2. Add R3.1 here if the gate passed |
| Week 5+ | **R5, R3.5** — the 2×2 grid, generator rebuild | Both worth doing properly rather than early; R3.5 closes most of Report II's generator to-do list |

---

## Reproducing the findings

- **F1** — read directly off `Baseline Experiment Logs - Baselines.csv`.
- **F2, F3** — probe `runs/task3_unet_pro/artifacts/task3_unet_finetune_60.pt` on subject 0006,
  axial slice 180.
- **F4** — leave-one-subject-out over `0006/0007/0009`, 257-point quantile maps on stride-2
  subsampled volumes, foreground at 2% of the 99.5th percentile.
- **F5** — brain-mask centroids and Dice on the same volumes.

**Caveats.** F2 and F3 are measured on one checkpoint, one subject and one slice — strong enough
to act on, but re-run across subjects before writing up. F4's nRMSE convention is not the
platform's, so only the ratio transfers; R0.2 exists precisely to convert it into a
leaderboard-comparable number. Effort estimates assume the single GV100.
