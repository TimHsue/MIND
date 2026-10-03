"""
Minimal distributed helpers used by the trimmed EDM pipeline.

The functions intentionally mirror the API surface consumed inside the
training/sampling scripts but avoid any external dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

try:
    import torch.distributed as _dist
except ImportError:  # pragma: no cover - defensive fallback
    _dist = None

_run_dir: Optional[Path] = None
_initialized = False


def _dist_available() -> bool:
    return _dist is not None and _dist.is_available()


def _dist_initialized() -> bool:
    return _dist_available() and _dist.is_initialized()


def init(device: str = 'cuda') -> None:
    """Initialize ``torch.distributed`` if rank/world-size environment variables are present."""
    global _initialized
    if _initialized or _dist_initialized():
        _initialized = True
        return

    if not _dist_available():
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return  # single-process execution

    backend = 'nccl' if device == 'cuda' and torch.cuda.is_available() else 'gloo'
    if backend == 'nccl':
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    _dist.init_process_group(backend=backend)
    from . import training_stats
    sync_device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if backend == 'nccl' else torch.device('cpu')
    training_stats.init_multiprocessing(rank=_dist.get_rank(), sync_device=sync_device)
    _initialized = True


def is_distributed() -> bool:
    return _dist_initialized()


def get_rank() -> int:
    if _dist_initialized():
        return _dist.get_rank()
    return 0


def get_world_size() -> int:
    if _dist_initialized():
        return _dist.get_world_size()
    return 1


def barrier() -> None:
    if _dist_initialized():
        _dist.barrier()


def print0(*args, **kwargs) -> None:
    """Print only on rank 0."""
    if get_rank() == 0:
        print(*args, **kwargs)


def set_run_dir(path: Optional[str]) -> None:
    """Store the run directory so ``update_progress``/``should_stop`` can operate."""
    global _run_dir
    _run_dir = Path(path) if path else None


def update_progress(cur_kimg: int, total_kimg: int) -> None:
    """Persist the current training progress for external monitors."""
    if get_rank() != 0 or _run_dir is None:
        return

    _run_dir.mkdir(parents=True, exist_ok=True)
    progress_file = _run_dir / "progress.txt"
    with progress_file.open("w", encoding="utf-8") as fp:
        fp.write(f"{cur_kimg}/{total_kimg}\n")


def should_stop() -> bool:
    """Check whether an abort file exists in the run directory."""
    if _run_dir is None:
        return False
    return (_run_dir / "_stop.txt").exists()


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Utility all_reduce wrapper that safely no-ops in single-process mode."""
    if _dist_initialized():
        _dist.all_reduce(tensor)
    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcast a picklable object from ``src`` to all ranks."""
    if not _dist_initialized():
        return obj
    objects = [obj]
    _dist.broadcast_object_list(objects, src=src)
    return objects[0]
