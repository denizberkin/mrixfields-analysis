# docker.synapse.org/syn76236366/task3:v2 — pulled 2026-09-20

Pulled with `scripts/pull_synapse_docker.py` (Registry v2 API, no daemon).
Only the application layers (7–13) were extracted; layers 0–6 are the
`pytorch/pytorch:2.13.0-cuda12.6` base image.

| | |
|---|---|
| image digest | `sha256:7165fe78625729b778f37e3752221891a62d61014cb6876e5e60f4a22cf04c5a` |
| tags | `v2`, `latest` (same digest); `v1` is the earlier image |
| created | 2026-09-11T09:48:39+03:00 |
| entrypoint | `python3 /app/inference.py --input /input --output /output` |
| base | torch 2.13.0+cu126, python 3.12 |
| `app/weights/task3.pt` | 37,174,545 bytes, sha256 `be132195d8afc96494aa9c3a252065cadb97084547d421111b79dd392567063b` — weights-only copy of `runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt` (mean of fine-tune epochs 6, 7, 8) |
| leaderboard twin | validation submission **9780368** `mc_ssim_slice_avg_tta` (SSIM 0.913652), same weights + 4-flip TTA via `scripts/make_task3_submission.py --tta` |

`app/mrx/model.py` is a vendored copy of `experiment-pipeline/components/models/conditional_unet.py::ConditionalUNet`;
`app/inference.py` applies 4-flip TTA, slice conditioning, 368x448 centre crop, 1e-3 background mask, full 364-slice volume.
