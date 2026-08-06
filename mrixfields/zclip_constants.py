"""Shared z-clip constants for the (364, 436, 30) submission spec.

Both the submission packer (Submission/build_submission/build_submission.py)
and the GT pack zclip stage (tools/build_validation_ground_truth/build_gt_zclip.py)
import ``Z_CLIP_RANGE`` from here. This is the SINGLE source of truth.

If pred and GT ever used different z ranges, they'd both be shape (364, 436, 30)
so the evaluator's shape check would pass — but they'd represent different
physical z slices of the brain, and metrics would silently be computed on
misaligned voxels. Keeping the constant in one place prevents that drift.
"""

#: (z_start_inclusive, z_end_exclusive) — axial slice indices on the
#: 0.5 mm isotropic grid (364×436×364 dataset volume). 30 slices total,
#: covering physical z ≈ +3.0 mm to +17.5 mm in MNI space.
Z_CLIP_RANGE = (150, 180)
