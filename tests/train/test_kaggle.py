import json
import os
import re
import time
from pathlib import Path

import pytest

from imagejev.train.kaggle import RUN_FILES, package_dataset, restore_run

ROOT = Path(__file__).resolve().parents[2]


# ---- the notebook itself ----------------------------------------------------------------------
def test_notebook_is_valid_and_every_python_cell_compiles():
    nb = json.loads((ROOT / "notebooks" / "kaggle_train.ipynb").read_text())
    assert nb["nbformat"] == 4 and nb["cells"]
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(codes) >= 5
    for i, cell in enumerate(codes):
        src = "".join(cell["source"])
        # shell lines (!cmd) and magics (%cd) are not Python; drop them before compiling
        py = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        compile(py, f"cell {i}", "exec")
    joined = "\n".join("".join(c["source"]) for c in codes)
    assert "imagejev.train.loop --config configs/kaggle_v0.yaml" in joined
    assert "restore_run" in joined and "/kaggle/input/imagejev-data" in joined


def test_kaggle_config_loads_and_paths_match_the_notebook():
    pytest.importorskip("yaml")
    pytest.importorskip("torch")
    from imagejev.train.loop import TrainConfig

    cfg = TrainConfig.from_yaml(ROOT / "configs" / "kaggle_v0.yaml")
    assert cfg.amp == "fp16" and cfg.time_budget_minutes and cfg.time_budget_minutes < 540
    nb = (ROOT / "notebooks" / "kaggle_train.ipynb").read_text()
    assert cfg.out_dir.replace("/kaggle/working/", "") in nb.replace("/kaggle/working/", "")
    assert cfg.cache_dir.startswith("/kaggle/input/imagejev-data")
    assert re.search(r"imagejev-data", nb)


# ---- packaging --------------------------------------------------------------------------------
def make_data(tmp_path):
    splits, cache = tmp_path / "splits", tmp_path / "cache"
    splits.mkdir()
    cache.mkdir()
    for name in ("train", "val", "test-images", "test-tasks", "test-styles", "test-external"):
        (splits / f"{name}.jsonl").write_text(f'{{"split": "{name}"}}\n')
    (cache / "meta.json").write_text('{"encoder_id": "x"}')
    (cache / "shard-00000.npy").write_bytes(b"npydata")
    (cache / "shard-00000.json").write_text('{"ids": ["a"]}')
    (cache / "failed.jsonl").write_text("not needed")
    (cache / "junk.txt").write_text("not needed")
    return splits, cache


def test_package_dataset_layout_and_metadata(tmp_path):
    splits, cache = make_data(tmp_path)
    info = package_dataset(splits, cache, tmp_path / "out", "me/imagejev-data", "imagejev data")
    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "splits").iterdir()) == sorted(f"{n}.jsonl" for n in (
        "train", "val", "test-images", "test-tasks", "test-styles", "test-external"))  # fmt: skip
    assert sorted(p.name for p in (out / "cache").iterdir()) == [
        "meta.json",
        "shard-00000.json",
        "shard-00000.npy",
    ]
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert (
        meta["id"] == "me/imagejev-data" and meta["title"] == "imagejev data" and meta["licenses"]
    )
    assert info["files"] == 9 and info["bytes"] > 0
    assert (out / "cache" / "shard-00000.npy").read_bytes() == b"npydata"
    package_dataset(
        splits, cache, out, "me/imagejev-data", "t"
    )  # repackaging over an old copy works


def test_package_dataset_validation(tmp_path):
    splits, cache = make_data(tmp_path)
    with pytest.raises(ValueError):
        package_dataset(splits, cache, tmp_path / "o", "no-slash", "t")
    (splits / "val.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        package_dataset(splits, cache, tmp_path / "o", "a/b", "t")
    (splits / "val.jsonl").write_text("{}")
    with pytest.raises(FileNotFoundError, match="feature cache"):
        package_dataset(splits, tmp_path / "empty", tmp_path / "o", "a/b", "t")


# ---- restoring a previous session -------------------------------------------------------------
def test_restore_run_copies_checkpoint_and_logs_but_never_regresses(tmp_path):
    prev, run = tmp_path / "prev", tmp_path / "run"
    prev.mkdir()
    for name in RUN_FILES:
        (prev / name).write_text(f"prev-{name}")
    assert restore_run(prev, run) is True
    assert all((run / n).read_text() == f"prev-{n}" for n in RUN_FILES)
    (run / "last.pt").write_text("newer progress")  # this session already trained further
    future = time.time() + 100
    os.utime(run / "last.pt", (future, future))
    assert restore_run(prev, run) is False and (run / "last.pt").read_text() == "newer progress"
    assert restore_run(tmp_path / "missing", tmp_path / "other") is False
    (prev / "best.pt").unlink()  # a previous run with no best checkpoint yet still restores
    assert restore_run(prev, tmp_path / "third") is True
