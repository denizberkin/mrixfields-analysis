"""Environment configuration loader for MRIxFields2026.

Reads settings from .env file at the repository root.
All path and device configuration is centralized here.

Relative paths in .env are resolved relative to the .env file location.
"""

import os
from pathlib import Path

_loaded = False
_env_dir: Path | None = None  # Directory containing .env file


def load_env():
    """Load .env file from repository root into os.environ.

    Only loads variables that are not already set in the environment,
    so explicit environment variables always take precedence.
    Silently skips if .env does not exist.
    """
    global _loaded, _env_dir
    if _loaded:
        return
    _loaded = True

    # Walk up from this file to find repo root (contains .env)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    _env_dir = env_path.parent

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


def _resolve_path(path: str) -> str:
    """Resolve relative paths relative to .env file location."""
    if path and _env_dir and not os.path.isabs(path):
        resolved = (_env_dir / path).resolve()
        return str(resolved)
    return path


def get_data_dir() -> str:
    """Get DATA_DIR from environment."""
    load_env()
    val = os.environ.get("DATA_DIR")
    if not val:
        raise RuntimeError(
            "DATA_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_preprocessed_dir() -> str | None:
    """Get PREPROCESSED_DIR from environment. Returns None if not set."""
    load_env()
    val = os.environ.get("PREPROCESSED_DIR")
    return _resolve_path(val) if val else None


def get_output_dir() -> str:
    """Get OUTPUT_DIR from environment."""
    load_env()
    val = os.environ.get("OUTPUT_DIR")
    if not val:
        raise RuntimeError(
            "OUTPUT_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_inference_dir() -> str:
    """Get INFERENCE_DIR from environment."""
    load_env()
    val = os.environ.get("INFERENCE_DIR")
    if not val:
        raise RuntimeError(
            "INFERENCE_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_predictions_seg_dir() -> str:
    """Get PREDICTIONS_SEG_DIR from environment."""
    load_env()
    val = os.environ.get("PREDICTIONS_SEG_DIR")
    if not val:
        raise RuntimeError(
            "PREDICTIONS_SEG_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_submission_dir() -> str:
    """Get SUBMISSION_DIR from environment."""
    load_env()
    val = os.environ.get("SUBMISSION_DIR")
    if not val:
        raise RuntimeError(
            "SUBMISSION_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_synthseg_dir() -> str:
    """Get SYNTHSEG_DIR from environment."""
    load_env()
    val = os.environ.get("SYNTHSEG_DIR")
    if not val:
        raise RuntimeError(
            "SYNTHSEG_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    resolved = _resolve_path(val)
    if not Path(resolved).exists():
        raise RuntimeError(
            f"SYNTHSEG_DIR does not exist: {resolved}\n"
            "Install SynthSeg: git clone https://github.com/BBillot/SynthSeg.git"
        )
    return resolved


def get_device() -> str:
    """Get DEVICE from environment."""
    load_env()
    return os.environ.get("DEVICE", "cuda:0")
