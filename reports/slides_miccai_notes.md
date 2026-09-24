# Speaking notes — `slides_miccai.tex`

MICCAI 2026, MRIxFields Task 3. 9 slides, 5 minutes.
Order: task → spectrum → method → what changed → all trials → where the error is →
qualitative → takeaway.
**Keep this in sync with the deck: if a slide changes, change its block here.**

Build: from inside `reports/`, `~/anaconda3/envs/tex/bin/tectonic slides_miccai.tex`.
The qualitative figure is regenerated separately, from the repo root:
`~/anaconda3/envs/mri/bin/python scripts/make_slides_qualitative.py`.

---

### 1 · Title — 12 s

Who we are, what the task was, where we landed: **0.913652** SSIM.
Do not state a rank. We know we are in the top 9, not the exact order.

---

### 2 · Submit input as it is — 30 s

> **Point:** the baseline is unusually high, and that is the whole story.

- Every off-diagonal cell is a translation we submit: all 20 ordered field pairs, in
  each of the 3 modalities.
- Source and target are the same subject, registered.
- So we submitted the do-nothing control, `ŷ = x`: **0.836497**.
- Final model: 0.913652. That narrow band is all a model can win.

---

### 3 · Power Spectrum Analysis — 50 s

> **Point:** it is not one translation problem. It is two, pulling opposite ways.

- Fit a power law to the radially averaged spectrum of every volume, `P(f) ∝ f^-α`.
- The table is each source field's α **minus** the 7 T target's.
- **0.1 T is +1.0 to +1.5** — it holds about 3% of 7 T's fine detail. Never acquired, so
  it has to be generated.
- **3 T and 5 T are negative** — the input is *finer* than the target. The correct map is
  a slight blur.
- One metric is being asked to score both.

---

### 4 · Method — 35 s

> **Point:** what actually runs, left to right, in one picture.

- **In:** one axial slice, 4 co-registered channels — the contrast being predicted, plus
  the other three at the source field. Channel 0 duplicates the primary, which is why the
  old single-channel first conv transferred exactly.
- **Net:** 32-channel conditional U-Net, 32→512 over 4 levels. The one formula on the
  slide is the conditioning vector:
  `c = E_src(d_s) + E_tgt(d_t) + E_z(z) ∈ R^512`. It is added at the bottleneck and read
  by every decoder block's FiLM. Both embedding tables are 15×512 — one row per
  (modality, field) domain.
- **Out:** 30 slices, stacked back into the submitted slab.
- **Last stage costs no training:** mean of epochs 6–8, and each slice predicted under
  the four in-plane flips and averaged.
- Shapes are on the slide: in `(B, 4, 368, 448)` — 364×436 slices centre-cropped to
  368×448 — out `(364, 436, 30)`, the slab z ∈ [150, 180).

---

### 5 · Architectural — 35 s

> **Point:** every panel is the same U-Net. Red is the only difference. The number under
> each is its challenge SSIM; the number on each arrow is what that step bought.

- vanilla U-Net → **+ domain conditioning, +0.0278** → FiLM & residual, **−0.0048** →
  **+ MIM pretraining, +0.0103** → + the other contrasts, +0.0037 → + average & flip
  TTA, +0.0069.
- MIM: mask 50% of 16 px blocks, reconstruct, L1 on the holes. **The whole network**,
  not the encoder alone, on 1,939 unpaired volumes, then fine-tune on the three paired
  subjects.
- Careful with the last arrow: it reads **+0.0069, not +0.0042**, because two steps have
  no panel — SSIM in the loss and the axial-position token sit inside it.

---

### 6 · Trials — 50 s  *(the main slide)*

> **Point:** the whole log, dead ends included.

- Dashed red line: copying the input. Only the two ends are red — the no-network control
  and the shipped model.
- Biggest single step is domain conditioning, **+0.0278**.
- MIM pretraining, **+0.0103**.
- The three bracketed steps together, +0.0064.
- Last step is free: **checkpoint averaging and flip TTA, +0.0042, no training at all**.
- FiLM + residual, the one architectural change on the line, cost us 0.0048.
- Bottom right: Swin UNETR, a LeJEPA tubelet encoder, a DINOv3 ViT, and the rank-1
  team's flow-matching recipe reproduced in full. Same data. Every one below the U-Net.

---

### 7 · The split is where the error goes — 30 s

> **Point:** the spectral prediction is confirmed by where the model actually fails.

- 0.1 T is 40% of the transitions and **50%** of the error.
- Mapping toward high field is easier than away from it: 7 T easiest target (0.9676),
  3 T easiest source (0.9656).
- That asymmetry means the model adds detail well and removes it badly — exactly what
  the spectrum predicted.

---

### 8 · Two of the regimes — 38 s

> **Point:** show the two problems rather than assert them.

- **Top, 0.1 T → 7 T:** the detail is simply not in the source. The model puts it there.
  0.771 → 0.938.
- **Bottom, 3 T → 7 T:** the source is visibly sharper than the real 7 T. The model gives
  detail up, which is the right answer. 0.806 → 0.970.
- **Last column is the signed residual** (ours − real), same scale on both rows. Blue is
  too dark, red too bright.
  - At 0.1 T the error sits on the **folded grey matter at the brain's surface** — the
    detail that was never acquired. The model under-shoots it.
  - At 3 T the interior is nearly clean; what is left sits at the **brain's outer edge**.
- Worth saying out loud: asking a generative model to *lose* detail on purpose is the
  odd half of this task.

---

### 9 · Takeaway — 20 s

1. 0.1 T has to invent detail, 3 T and 5 T have to translate it. 0.1 T carried 50% of
   the error.
2. **Condition on the modality and the field pair: +0.0278.** Largest single step.
3. **Pretrain the whole network on 1,939 unpaired volumes: +0.0103.**
4. **Average three checkpoints and four flips: +0.0042**, no training.

Close on the repo link.

---

## If asked

- **Why no fancier backbone:** four of them, same data, every one below the 32-channel
  U-Net. Say "in our runs", not "in general" — the budgets were not matched.
- **Local vs leaderboard:** our local harness scored the wrong plane until 2026-09-11
  (sagittal instead of axial). That is why TTA and averaging were declined twice before
  being adopted. Fixed; harness and submission now agree to 1.8e-4.
- **Why flow matching was dropped:** we reproduced the rank-1 recipe through three
  stages. 0.886725, i.e. 0.027 below our regression U-Net.
- **Context signals:** neighbouring slices +0.0005, axial-position token +0.0006, both
  together −0.0010. They saturate at one.
- **The identity control:** `task3_input_validation`, Task 3 queue 9619636, 2026-08-28,
  accepted 180/180. Three other teams hit the same triple independently.
