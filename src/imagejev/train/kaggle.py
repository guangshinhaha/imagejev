"""Helpers for running the training across Kaggle sessions.

A Kaggle GPU session is limited (about 9 hours) and its working directory disappears afterwards, so
a long run is a chain of sessions. Each session:

1. attaches the data as a Kaggle Dataset (built once with :func:`package_dataset`);
2. restores the previous session's checkpoint (:func:`restore_run`), attached as an input;
3. trains until ``time_budget_minutes`` runs out, saving ``last.pt``;
4. is saved as a notebook version, whose output the next session attaches.

See ``docs/kaggle.md``.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test-images", "test-tasks", "test-styles", "test-external")
RUN_FILES = ("last.pt", "best.pt", "train_log.csv", "eval_log.csv", "select_log.csv")


def _place(src: Path, dst: Path) -> None:
    """Hard-link when possible (no extra disk), otherwise copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def package_dataset(
    splits_dir: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path,
    dataset_id: str,
    title: str,
    splits: Sequence[str] = SPLITS,
) -> dict[str, Any]:
    """Lay out ``out_dir`` as a Kaggle dataset: ``splits/*.jsonl``, ``cache/*`` and metadata.

    ``dataset_id`` is ``<kaggle-username>/<slug>``. Upload with ``kaggle datasets create -p OUT``.
    It is created **private** by default; keep it that way, because the features and labels derive
    from sources with restrictive terms (see ``docs/data-licenses.md``).
    """
    splits_dir, cache_dir, out = Path(splits_dir), Path(cache_dir), Path(out_dir)
    if "/" not in dataset_id:
        raise ValueError("dataset_id must look like '<username>/<slug>'")
    n_files, n_bytes = 0, 0
    for name in splits:
        src = splits_dir / f"{name}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        _place(src, out / "splits" / src.name)
        n_files, n_bytes = n_files + 1, n_bytes + src.stat().st_size
    if not (cache_dir / "meta.json").exists():
        raise FileNotFoundError(f"{cache_dir} is not a feature cache (no meta.json)")
    for src in sorted(cache_dir.iterdir()):
        if src.suffix in {".npy", ".json"} and src.name != "failed.jsonl":
            _place(src, out / "cache" / src.name)
            n_files, n_bytes = n_files + 1, n_bytes + src.stat().st_size
    meta = {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"files": n_files, "bytes": n_bytes, "dataset_id": dataset_id}


def restore_run(previous: str | Path, run_dir: str | Path) -> bool:
    """Copy a previous session's checkpoint and logs into ``run_dir``.

    Returns True if something was restored. Never overwrites a ``last.pt`` that is newer than the
    one it would bring in, so re-running a cell is harmless.
    """
    previous, run_dir = Path(previous), Path(run_dir)
    src_last = previous / "last.pt"
    if not src_last.exists():
        return False
    dst_last = run_dir / "last.pt"
    if dst_last.exists() and dst_last.stat().st_mtime >= src_last.stat().st_mtime:
        return False
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (previous / name).exists():
            shutil.copy2(previous / name, run_dir / name)
    return True
