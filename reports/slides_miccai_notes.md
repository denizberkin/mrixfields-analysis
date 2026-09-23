# Speaking notes — `slides_miccai.tex`

MICCAI 2026, MRIxFields Task 3. 8 slides, 5 minutes.
**Keep this in sync with the deck: if a slide changes, change its block here.**

Arc: the spectrum says the task is two problems → the error map agrees → so information,
not architecture, is what moved the score.

---

### 1 · Title — 15 s

Who we are, what the task was, where we landed: **0.913652** SSIM.

Do not state a rank. We know we are in the top 9, not the exact order.

---

### 2 · The task, and the number to beat — 35 s

> **Point:** the baseline is unusually high, and that is the whole story.

- 15 domains: 3 modalities × 5 field strengths. One of 20 ordered field pairs per case.
- Source and target are the same subject, registered.
- So we submitted the do-nothing control, `ŷ = x`: **0.836497**.
- So a model can only add **0.0772** on top of that, and it has to add it
  everywhere at once.

---

### 3 · Before modelling: what is actually missing? — 55 s

> **Point:** it is not one translation problem. It is two, and they pull in opposite
> directions.

- Fit a power law to the radially averaged spectrum of every volume, `P(f) ∝ f^-α`.
- Table is each source field's α **minus** the 7 T target's.
- **0.1 T is +1.0 to +1.5** — it holds about 3% of 7 T's fine detail. That detail was
  never acquired, so it has to be generated.
- **3 T and 5 T are negative** — the input is *finer* than the target. The correct map is
  a slight blur.
- One metric is being asked to score both.

---

### 4 · The split is where the error goes — 35 s

> **Point:** the spectral prediction is confirmed by where the model actually fails.

- 0.1 T is 40% of the transitions and **50%** of the error.
- Mapping toward high field is easier than away from it: 7 T easiest target, 3 T easiest
  source.
- That asymmetry means the model adds detail well and removes it badly — exactly what
  the spectrum predicted.

---

### 5 · Every trial, one chart — 70 s  *(the main slide, take your time)*

> **Point:** every gain came from giving the network more information. Nothing came from
> a better backbone.

- Dashed red line at the bottom: copying the input.
- Biggest single step is **domain conditioning, +0.0278** — just telling it which field
  pair it is solving.
- **MIM pretraining on 1,939 unpaired volumes, +0.0103.**
- Three smaller information steps together, +0.0064.
- Last step is free: **checkpoint averaging and flip TTA, +0.0042, no training at all** —
  bigger than anything since conditioning.
- Grey is architecture. FiLM + residual cost us 0.0048.
- Bottom right: Swin UNETR, a LeJEPA tubelet encoder, a DINOv3 ViT, and the rank-1 team's
  flow-matching recipe reproduced in full. Same data. Every one below the U-Net.

---

### 6 · Where the 0.0772 came from — 30 s

> **Point:** the decomposition, in one picture.

- A U-Net at all: +0.0334.
- **Information handed to it: +0.0444** — the largest share.
- Averaging, no training: +0.0042.
- Architecture changes: **−0.0048**, the only negative bar.

---

### 7 · The two regimes, seen — 40 s

> **Point:** show the two problems rather than assert them.

- **Top, 0.1 T → 7 T:** the cortical ribbon is simply not in the source. The model puts
  it there. 0.771 → 0.938.
- **Bottom, 3 T → 7 T:** the source is visibly sharper than the real 7 T. The model gives
  detail up, which is the right answer. 0.806 → 0.970.
- **Last column is the signed residual** (ours − real), same scale on both rows. Blue is
  too dark, red too bright.
  - At 0.1 T the error sits **on the cortical ribbon** — exactly the detail that was
    never acquired. The model under-shoots it.
  - At 3 T the interior is nearly clean and what is left sits **on the outer rim**.
  - Same conclusion as the bar chart on slide 4, one subject at a time.
- Worth saying out loud: asking a generative model to *lose* detail on purpose is the
  odd half of this task.

---

### 8 · Taking it away — 20 s

> **Point:** three things we would tell someone starting this challenge.

1. Measure the gap before choosing a model.
2. Pixel metrics are soft on blur — the finest detail carries ~5% of the variance the
   coarsest does.
3. Spend the last hour on averaging, not on a new architecture.

Close on the repo link.

---

## If asked

- **Local vs leaderboard:** our local harness scored the wrong plane until 2026-09-11
  (sagittal instead of axial). That is why TTA and averaging were declined twice before
  they were adopted. Fixed; harness and submission now agree to 1.8e-4.
- **Why no flow matching:** we reproduced the rank-1 recipe through three stages.
  0.886725, i.e. 0.027 below our regression U-Net.
- **Context signals:** neighbouring slices +0.0005, axial-position token +0.0006, both
  together −0.0010. They saturate at one.
- **Final model:** 32-channel conditional U-Net, 4-channel input (the predicted contrast
  plus the other three at the source field), MIM pretrained, average of epochs 6–8,
  4-flip TTA.
