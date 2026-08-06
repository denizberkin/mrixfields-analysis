#!/usr/bin/env python3
"""Generate MRIxFields2026 predictions and a formatted submission from YAML.

Run with the PyTorch environment, for example:

    /home/denizberkin/miniconda3/envs/mri/bin/python \
        scripts/generate_challenge_submission.py --manifest manifest.yaml

The manifest maps one or more checkpoints to modality/field-strength pairs.
Built-in inference supports the challenge baselines; ``inference_command``
adapts any other model. Task 3 enforces one unique checkpoint.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
OFFICIAL_ROOT = ROOT / "official" / "MRIxFields2026"
BASELINE_ROOT = OFFICIAL_ROOT / "Baseline"
SYNTHSEG_PYTHON = ROOT / ".conda" / "synthseg" / "bin" / "python"

FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")
SOURCE_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}
TASK_PAIRS = {
    "task1": {(source, "7T") for source in FIELDS if source != "7T"},
    "task2": {("0.1T", target) for target in FIELDS if target != "0.1T"},
    "task3": {(source, target) for source in FIELDS for target in FIELDS if source != target},
}
BASELINE_TASK_METHODS = {
    "task1": {"cut", "cyclegan"},
    "task2": {"cut", "cyclegan"},
    "task3": {"stargan_v2"},
}


@dataclass(frozen=True)
class Mapping:
    modality: str
    source: str
    target: str
    checkpoint: Path
    config: Path | None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.modality, self.source, self.target

    @property
    def pair(self) -> str:
        return f"{self.source}_to_{self.target}"


@dataclass
class RunSpec:
    name: str
    task: str
    method: str
    mode: str
    epoch_tag: str
    seed: int
    device: str
    inference_command: tuple[str, ...] | None
    mappings: list[Mapping]
    prediction_root: Path
    segmentation_root: Path
    submission_root: Path
    synthseg_dir: Path


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing .env file: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_path(env: dict[str, str], key: str) -> Path:
    value = env.get(key)
    if not value:
        raise ValueError(f"{key} is not defined in {ENV_FILE}")
    return Path(value).expanduser().resolve()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def derived_config(task: str, method: str, modality: str, source: str, target: str) -> Path:
    if task in {"task1", "task2"}:
        return BASELINE_ROOT / "configs" / task / method / f"{source}_to_{target}_{modality}.yaml"
    return BASELINE_ROOT / "configs" / "task3" / "stargan" / "any_to_any_all_modalities.yaml"


def parse_manifest(path: Path, env: dict[str, str]) -> RunSpec:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Manifest root must be a YAML mapping")

    required = ("name", "task", "method", "models")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Manifest is missing required keys: {missing}")

    name = str(raw["name"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("name may contain only letters, numbers, dot, underscore, and hyphen")
    task = str(raw["task"]).lower()
    method = str(raw["method"]).lower()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", method):
        raise ValueError("method may contain only letters, numbers, dot, underscore, and hyphen")
    command_raw = raw.get("inference_command")
    if command_raw is not None and (
        not isinstance(command_raw, list)
        or not command_raw
        or not all(isinstance(part, str) for part in command_raw)
    ):
        raise ValueError("inference_command must be a non-empty YAML list of strings")
    inference_command = tuple(command_raw) if command_raw else None
    mode = str(raw.get("mode", "pro_pretrained"))
    default_epoch = "epoch50" if task == "task3" else "epoch100"
    epoch_tag = str(raw.get("epoch_tag", default_epoch))
    seed = int(raw.get("seed", 0))
    device = str(raw.get("device", env.get("DEVICE", "cuda:0")))

    if task not in TASK_PAIRS:
        raise ValueError(f"Unknown task {task!r}; expected task1, task2, or task3")
    if inference_command is None and method not in BASELINE_TASK_METHODS[task]:
        raise ValueError(
            f"Method {method!r} has no built-in inference implementation for {task}; "
            "provide inference_command for a custom model"
        )

    paths = raw.get("paths", {}) or {}
    if not isinstance(paths, dict):
        raise ValueError("paths must be a YAML mapping")
    prediction_base = resolve_path(paths.get("predictions", env_path(env, "INFERENCE_DIR")), path.parent)
    segmentation_base = resolve_path(paths.get("segmentations", env_path(env, "PREDICTIONS_SEG_DIR")), path.parent)
    submission_base = resolve_path(paths.get("submissions", env_path(env, "SUBMISSION_DIR")), path.parent)
    synthseg_dir = resolve_path(paths.get("synthseg", env_path(env, "SYNTHSEG_DIR")), path.parent)

    models = raw["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty YAML list")

    mappings: list[Mapping] = []
    for model_index, model in enumerate(models):
        if not isinstance(model, dict) or "checkpoint" not in model or "mappings" not in model:
            raise ValueError(f"models[{model_index}] requires checkpoint and mappings")
        checkpoint = resolve_path(model["checkpoint"], path.parent)
        model_config = model.get("config")
        model_mappings = model["mappings"]
        if not isinstance(model_mappings, list) or not model_mappings:
            raise ValueError(f"models[{model_index}].mappings must be a non-empty list")

        for mapping_index, item in enumerate(model_mappings):
            if not isinstance(item, dict):
                raise ValueError(f"models[{model_index}].mappings[{mapping_index}] must be a mapping")
            try:
                modality = str(item["modality"])
                source = str(item["source"])
                target = str(item["target"])
            except KeyError as exc:
                raise ValueError(f"Mapping requires modality, source, and target: {item}") from exc
            if modality not in MODALITIES:
                raise ValueError(f"Invalid modality {modality!r}")
            if (source, target) not in TASK_PAIRS[task]:
                raise ValueError(f"Invalid pair {source}->{target} for {task}")
            config_value = item.get("config", model_config)
            config = resolve_path(config_value, path.parent) if config_value else None
            if config is None and inference_command is None:
                config = derived_config(task, method, modality, source, target)
            mappings.append(Mapping(modality, source, target, checkpoint, config))

    keys = [mapping.key for mapping in mappings]
    if len(keys) != len(set(keys)):
        raise ValueError("The manifest contains duplicate modality/source/target mappings")
    checkpoints = {mapping.checkpoint for mapping in mappings}
    if task == "task3" and len(checkpoints) != 1:
        raise ValueError("Task 3 must use exactly one unique checkpoint across every mapping")

    expected_keys = {
        (modality, source, target)
        for modality in MODALITIES
        for source, target in TASK_PAIRS[task]
    }
    selected_keys = set(keys)
    completeness = "complete" if selected_keys == expected_keys else "partial"
    print(f"Manifest selects {len(selected_keys)}/{len(expected_keys)} mappings ({completeness} {task} submission).")

    return RunSpec(
        name=name,
        task=task,
        method=method,
        mode=mode,
        epoch_tag=epoch_tag,
        seed=seed,
        device=device,
        inference_command=inference_command,
        mappings=mappings,
        prediction_root=prediction_base / name,
        segmentation_root=segmentation_base / name,
        submission_root=submission_base / name,
        synthseg_dir=synthseg_dir,
    )


def validate_inputs(spec: RunSpec, data_dir: Path) -> None:
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Official repository not found: {OFFICIAL_ROOT}")
    for mapping in spec.mappings:
        if not mapping.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {mapping.checkpoint}")
        if mapping.config is not None and not mapping.config.is_file():
            raise FileNotFoundError(f"Config not found: {mapping.config}")
        input_dir = data_dir / "Validating_prospective" / mapping.modality / mapping.source
        expected = {f"P_{mapping.modality}_{mapping.source}_{sid}.nii.gz" for sid in SOURCE_IDS[mapping.source]}
        actual = {path.name for path in input_dir.glob("*.nii.gz")}
        if actual != expected:
            raise RuntimeError(f"Validation inputs differ for {mapping.key}: expected={sorted(expected)}, actual={sorted(actual)}")


def output_dir(spec: RunSpec, mapping: Mapping) -> Path:
    task_name = f"{spec.task}_{mapping.source}_to_{mapping.target}_{mapping.modality}"
    return spec.prediction_root / task_name / spec.method / spec.mode / spec.epoch_tag


def expected_prediction_names(mapping: Mapping) -> set[str]:
    return {
        f"P_{mapping.modality}_{mapping.source}_{sid}.nii.gz"
        for sid in SOURCE_IDS[mapping.source]
    }


def format_inference_command(spec: RunSpec, mapping: Mapping, input_dir: Path, destination: Path) -> list[str]:
    values = {
        "checkpoint": str(mapping.checkpoint),
        "config": str(mapping.config or ""),
        "input_dir": str(input_dir),
        "output_dir": str(destination),
        "device": spec.device,
        "modality": mapping.modality,
        "source": mapping.source,
        "target": mapping.target,
        "task": spec.task,
        "method": spec.method,
        "seed": str(spec.seed),
    }
    try:
        return [part.format_map(values) for part in spec.inference_command or ()]
    except KeyError as exc:
        raise ValueError(f"Unknown inference_command placeholder: {exc.args[0]}") from exc


def run_custom_inference(
    spec: RunSpec,
    data_dir: Path,
    env: dict[str, str],
    overwrite: bool,
    dry_run: bool,
) -> None:
    child_env = os.environ.copy()
    child_env.update(env)
    for mapping in spec.mappings:
        destination = output_dir(spec, mapping)
        expected_names = expected_prediction_names(mapping)
        existing_names = {path.name for path in destination.glob("*.nii.gz")}
        if existing_names == expected_names and not overwrite:
            print(f"Skipping complete inference: {mapping.modality} {mapping.source}->{mapping.target}")
            continue

        input_dir = data_dir / "Validating_prospective" / mapping.modality / mapping.source
        print(f"Inference: {mapping.modality} {mapping.source}->{mapping.target} -> {destination}")
        command = format_inference_command(spec, mapping, input_dir, destination)
        if overwrite and destination.exists() and not dry_run:
            shutil.rmtree(destination)
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        run_command(command, child_env, dry_run)
        if not dry_run:
            actual_names = {path.name for path in destination.glob("*.nii.gz")}
            if actual_names != expected_names:
                raise RuntimeError(
                    f"Custom inference outputs differ for {mapping.key}: "
                    f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
                )


def target_domain(mapping: Mapping, config: dict[str, Any]) -> int | None:
    if config.get("method") != "stargan_v2":
        return None
    num_domains = int(config.get("model", {}).get("num_domains", 5))
    field_index = FIELDS.index(mapping.target)
    if num_domains == len(FIELDS):
        return field_index
    if num_domains == len(FIELDS) * len(MODALITIES):
        return MODALITIES.index(mapping.modality) * len(FIELDS) + field_index
    raise ValueError(f"Cannot derive target domain for num_domains={num_domains}: {mapping.config}")


def run_inference(
    spec: RunSpec,
    data_dir: Path,
    env: dict[str, str],
    overwrite: bool,
    dry_run: bool,
) -> None:
    if spec.inference_command is not None:
        run_custom_inference(spec, data_dir, env, overwrite, dry_run)
        return

    sys.path.insert(0, str(BASELINE_ROOT))
    sys.path.insert(0, str(BASELINE_ROOT / "scripts"))
    from mrixfields.data.utils import load_nifti, save_nifti  # noqa: PLC0415
    from inference import load_config, load_generator, predict_volume  # type: ignore  # noqa: PLC0415

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    if not dry_run and spec.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but PyTorch cannot access CUDA: {spec.device}")
    device = torch.device("cpu" if dry_run else spec.device)

    grouped: dict[tuple[Path, Path], list[Mapping]] = {}
    for mapping in spec.mappings:
        assert mapping.config is not None
        grouped.setdefault((mapping.checkpoint, mapping.config), []).append(mapping)

    for (checkpoint, config_path), mappings in grouped.items():
        config = load_config(str(config_path))
        if config.get("method") != spec.method:
            raise ValueError(
                f"Config method {config.get('method')!r} does not match manifest method "
                f"{spec.method!r}: {config_path}"
            )
        if dry_run:
            model = model_type = None
        else:
            model, model_type = load_generator(config, str(checkpoint), device)
            print(f"Loaded {checkpoint} for {len(mappings)} mapping(s).")

        for mapping in mappings:
            destination = output_dir(spec, mapping)
            expected_names = expected_prediction_names(mapping)
            existing_names = {path.name for path in destination.glob("*.nii.gz")}
            if existing_names == expected_names and not overwrite:
                print(f"Skipping complete inference: {mapping.modality} {mapping.source}->{mapping.target}")
                continue
            input_dir = data_dir / "Validating_prospective" / mapping.modality / mapping.source
            print(f"Inference: {mapping.modality} {mapping.source}->{mapping.target} -> {destination}")
            if dry_run:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            domain = target_domain(mapping, config)
            crop_size_raw = config["data"].get("crop_size")
            crop_size = tuple(crop_size_raw) if crop_size_raw else None
            slice_axis = int(config["data"].get("slice_axis", 2))
            model_config = config["model"]

            for nifti_path in sorted(input_dir.glob("*.nii.gz")):
                original_img = nib.load(str(nifti_path))
                original_affine = original_img.affine
                original_ornt = nib.io_orientation(original_affine)
                data, canonical_affine = load_nifti(nifti_path)
                canonical_ornt = nib.io_orientation(canonical_affine)
                prediction = predict_volume(
                    model,
                    model_type,
                    data,
                    crop_size=crop_size,
                    slice_axis=slice_axis,
                    device=device,
                    target_domain=domain,
                    latent_dim=model_config.get("latent_dim", 16),
                    style_dim=model_config.get("style_dim", 64),
                )
                orientation = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
                prediction = nib.orientations.apply_orientation(prediction, orientation)
                source = original_img.get_fdata(dtype=np.float32)
                prediction *= (source > 1e-6).astype(prediction.dtype)
                save_nifti(prediction, original_affine, destination / nifti_path.name, header=original_img.header)


def tensorflow_env(base_env: dict[str, str], synthseg_dir: Path) -> dict[str, str]:
    child = os.environ.copy()
    child.update(base_env)
    child["SYNTHSEG_DIR"] = str(synthseg_dir)
    baseline_path = str(BASELINE_ROOT)
    child["PYTHONPATH"] = baseline_path + (os.pathsep + child["PYTHONPATH"] if child.get("PYTHONPATH") else "")
    version = "python3.10"
    site_packages = SYNTHSEG_PYTHON.parent.parent / "lib" / version / "site-packages"
    cuda_libs = sorted((site_packages / "nvidia").glob("*/lib"))
    if cuda_libs:
        library_path = os.pathsep.join(map(str, cuda_libs))
        child["LD_LIBRARY_PATH"] = library_path + (
            os.pathsep + child["LD_LIBRARY_PATH"] if child.get("LD_LIBRARY_PATH") else ""
        )
    nvcc_bin = site_packages / "nvidia" / "cuda_nvcc" / "bin"
    if nvcc_bin.is_dir():
        child["PATH"] = f"{nvcc_bin}{os.pathsep}{child.get('PATH', '')}"
    return child


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def run_segmentation(spec: RunSpec, env: dict[str, str], overwrite: bool, dry_run: bool) -> None:
    if spec.task == "task3":
        print("Task 3 does not use segmentations; skipping SynthSeg.")
        return
    if not SYNTHSEG_PYTHON.is_file():
        raise FileNotFoundError(f"SynthSeg Python environment not found: {SYNTHSEG_PYTHON}")
    if not (spec.synthseg_dir / "models" / "synthseg_2.0.h5").is_file():
        raise FileNotFoundError(f"SynthSeg 2.0 weights not found below {spec.synthseg_dir}")
    child_env = tensorflow_env(env, spec.synthseg_dir)
    command = [
        str(SYNTHSEG_PYTHON),
        str(BASELINE_ROOT / "scripts" / "segment_predictions.py"),
        "--tasks", spec.task,
        f"--{spec.task}-method", spec.method,
        f"--{spec.task}-mode", spec.mode,
        f"--{spec.task}-epoch", spec.epoch_tag,
        "--predictions-dir", str(spec.prediction_root),
        "--predictions-seg-dir", str(spec.segmentation_root),
    ]
    if overwrite:
        command.append("--overwrite")
    run_command(command, child_env, dry_run)


def expected_packaged_files(spec: RunSpec) -> set[Path]:
    task_dir = spec.submission_root / spec.task
    expected: set[Path] = set()
    for mapping in spec.mappings:
        pair_dir = task_dir / mapping.modality / mapping.pair
        for sid in SOURCE_IDS[mapping.source]:
            expected.add(pair_dir / "pred" / f"P_{mapping.modality}_{mapping.target}_{sid}.nii.gz")
            if spec.task != "task3":
                expected.add(pair_dir / "seg" / f"P_{mapping.modality}_{mapping.target}_{sid}_seg.nii.gz")
    return expected


def package(spec: RunSpec, env: dict[str, str], dry_run: bool) -> Path:
    task_dir = spec.submission_root / spec.task
    if task_dir.exists() and not dry_run:
        shutil.rmtree(task_dir)
    child_env = tensorflow_env(env, spec.synthseg_dir)
    command = [
        str(SYNTHSEG_PYTHON),
        str(OFFICIAL_ROOT / "Submission" / "build_submission" / "build_submission.py"),
        "--tasks", spec.task,
        f"--{spec.task}-method", spec.method,
        f"--{spec.task}-mode", spec.mode,
        f"--{spec.task}-epoch", spec.epoch_tag,
        "--predictions-dir", str(spec.prediction_root),
        "--predictions-seg-dir", str(spec.segmentation_root),
        "--output-dir", str(spec.submission_root),
    ]
    run_command(command, child_env, dry_run)
    zip_path = spec.submission_root / f"{spec.task}.zip"
    if dry_run:
        return zip_path

    expected = expected_packaged_files(spec)
    actual = set(task_dir.rglob("*.nii.gz"))
    if actual != expected:
        missing = sorted(map(str, expected - actual))
        extra = sorted(map(str, actual - expected))
        raise RuntimeError(f"Packaged files differ from manifest. Missing={missing}; extra={extra}")
    for path in actual:
        if nib.load(str(path)).shape != (364, 436, 30):
            raise RuntimeError(f"Incorrect packaged shape: {path}")

    spec.submission_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for file_path in sorted(actual):
            archive.write(file_path, file_path.relative_to(task_dir.parent))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        if len([name for name in archive.namelist() if name.endswith(".nii.gz")]) != len(expected):
            raise RuntimeError("ZIP file count differs from manifest")
    return zip_path


def selected_stages(value: str) -> set[str]:
    return {"inference", "segment", "package"} if value == "all" else {value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="YAML checkpoint-to-mapping manifest")
    parser.add_argument(
        "--stage",
        choices=("all", "inference", "segment", "package"),
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--overwrite-inference", action="store_true")
    parser.add_argument("--overwrite-segmentations", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print actions without writing")
    args = parser.parse_args()

    env = load_dotenv(ENV_FILE)
    manifest_path = args.manifest.expanduser().resolve()
    spec = parse_manifest(manifest_path, env)
    data_dir = env_path(env, "DATA_DIR")
    validate_inputs(spec, data_dir)

    print(f"Run: {spec.name}")
    print(f"Task/method: {spec.task}/{spec.method}")
    print(f"Predictions: {spec.prediction_root}")
    print(f"Segmentations: {spec.segmentation_root}")
    print(f"Submission: {spec.submission_root}")

    stages = selected_stages(args.stage)
    if "inference" in stages:
        run_inference(spec, data_dir, env, args.overwrite_inference, args.dry_run)
    if "segment" in stages:
        run_segmentation(spec, env, args.overwrite_segmentations, args.dry_run)
    if "package" in stages:
        zip_path = package(spec, env, args.dry_run)
        print(f"Submission archive: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
