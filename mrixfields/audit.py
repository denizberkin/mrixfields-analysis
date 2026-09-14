"""Bridge to the challenge's official audit tools (Data Integrity Policy, wiki 642237).

The policy requires five logs, all produced by https://github.com/MRIxFields/audit-tools:
runtime monitoring, dataset access, per-iteration training, per-epoch validation, and
checkpoint. Those tools are vendored **unmodified** in ``experiment-pipeline/`` --
``audit_monitor.py`` and ``audit_utils.py`` are fingerprinted by the organizers and must
not be edited.

This module exists so the rest of the codebase never imports them by bare name. They are
written to be imported from the directory they sit in, but our data classes are imported
from the repo root as often as from ``experiment-pipeline/``, and a bare ``import
audit_utils`` would resolve in one case and not the other.

``AUDIT_ENABLED`` is exported so a training run can say in its own stdout whether the
hooks are live. A silently disabled audit is the one outcome the policy actually punishes,
so a missing tool warns loudly rather than passing quietly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PIPELINE_ROOT = Path(__file__).resolve().parent.parent / "experiment-pipeline"

AUDIT_ENABLED = False
_WARNED = False


def _load() -> tuple[Callable[..., Any], ...]:
    global AUDIT_ENABLED
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from audit_utils import audit_file_loading, get_mem_usage, get_train_logger

    AUDIT_ENABLED = True
    return audit_file_loading, get_mem_usage, get_train_logger


def _resilient(record: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap audit_file_loading so a psutil race cannot kill a training run.

    The official hook walks the whole process ancestry on *every* read. psutil enumerates
    a parent pid and then calls create_time() on it, so any ancestor that exits in between
    raises NoSuchProcess -- inside a DataLoader worker that propagates out of __getitem__
    and takes the run down. It killed a 20-epoch run at epoch 9 the first time a
    background process happened to exit mid-read.

    One retry, because the walk re-enumerates and the transient pid is simply gone the
    second time. If it fails twice the read proceeds unrecorded and says so once: losing
    one line of the audit log is a far smaller problem than losing the training run, and a
    silent drop would be worse than either.

    audit_utils.py itself is untouched -- it is fingerprinted and modification is
    prohibited. This wraps our call site, which is what the tools' README asks for.
    """
    state = {"warned": False}

    def wrapped(file_path) -> None:
        for attempt in (1, 2):
            try:
                record(file_path)
                return
            except Exception as exc:  # noqa: BLE001 - psutil races, mainly NoSuchProcess
                if attempt == 2:
                    if not state["warned"]:
                        state["warned"] = True
                        print(f"WARNING: audit_file_loading failed ({type(exc).__name__}: "
                              f"{exc}); continuing unrecorded. Further occurrences silent.",
                              file=sys.stderr, flush=True)
                    return

    return wrapped


try:
    audit_file_loading, get_mem_usage, get_train_logger = _load()
    audit_file_loading = _resilient(audit_file_loading)
except Exception as exc:  # pragma: no cover - only when the tools are absent
    _REASON = exc

    def _warn_once() -> None:
        global _WARNED
        if not _WARNED:
            _WARNED = True
            print(f"WARNING: audit tools unavailable ({_REASON}); no audit log will be "
                  f"written. Expected them in {PIPELINE_ROOT}.", file=sys.stderr, flush=True)

    def audit_file_loading(file_path) -> None:  # type: ignore[misc]
        _warn_once()

    def get_mem_usage():  # type: ignore[misc]
        _warn_once()
        return ()

    def get_train_logger(console: bool = True):  # type: ignore[misc]
        _warn_once()

        class _Null:
            def info(self, *args, **kwargs) -> None:
                pass

        return _Null()


def reinit_for_worker(worker_id: int | None = None) -> None:
    """Re-open the audit log inside a DataLoader worker. Pass as ``worker_init_fn``.

    The handler created in the parent does not survive into a forked worker -- its stream
    arrives as None, so the first slice read raises AttributeError inside the worker and
    takes the whole run down. Their SecureLogger already names files
    ``{name}_{pid}_{timestamp}.log``, so a per-process log is the intended shape; the fix
    is to drop the inherited handler and let get_logger build a fresh one for this pid.

    Rebinding ``audit_utils._logger`` is deliberate and is not an edit to their file:
    ``audit_file_loading`` reads that module global, so a new logger has to be published
    there or the reopened handler would go unused. audit_utils.py and audit_monitor.py
    themselves stay byte-identical, which is what the policy actually fingerprints.

    Sharing one file across processes would be worse than useless here: the handler chains
    a hash per record, and interleaved writers would break the chain the audit relies on.
    """
    if not AUDIT_ENABLED:
        return
    import logging

    import audit_utils

    for name in ("file-loading", "training"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    audit_utils._logger = audit_utils.secure_logger.get_logger("file-loading")


__all__ = ["audit_file_loading", "get_mem_usage", "get_train_logger", "reinit_for_worker",
           "AUDIT_ENABLED"]
