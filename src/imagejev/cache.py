"""On-disk cache of frozen vision features, built once and read many times during training.

Layout under ``root``::

    meta.json              encoder id, feature shape, dtype
    shard-00000.npy        float16 array (n, tokens, dim), written atomically
    shard-00000.json       {"ids": [...]}  <- a shard counts only once this file exists
    failed.jsonl           images that could not be read, with the reason

A shard is complete only when both files exist, so a crash mid-write leaves nothing half-trusted.
Restarting skips every id already in a complete shard. At 65 tokens x 768 dims in fp16 an image
takes about 100 KB, so 200k images fit in about 20 GB.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .errors import ImageLoadError
from .images import load_image

DTYPE = np.float16


class BatchEncoder(Protocol):
    encoder_id: str

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Features of shape ``(B, tokens, dim)``."""
        ...


def _atomic_write(path: Path, write: Callable[[Any], None], mode: str = "wb") -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open(mode) as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _shard_names(root: Path) -> list[str]:
    return sorted(p.stem for p in root.glob("shard-*.json"))


class FeatureCache:
    """Read access to a built cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta = self.root / "meta.json"
        if not meta.exists():
            raise FileNotFoundError(f"no feature cache at {self.root}")
        self.meta: dict[str, Any] = json.loads(meta.read_text())
        self._where: dict[str, tuple[str, int]] = {}
        for name in _shard_names(self.root):
            if not (self.root / f"{name}.npy").exists():
                continue
            for row, image_id in enumerate(
                json.loads((self.root / f"{name}.json").read_text())["ids"]
            ):
                self._where[image_id] = (name, row)
        self._open: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._where

    def ids(self) -> list[str]:
        return list(self._where)

    def _shard(self, name: str) -> np.ndarray:
        if name not in self._open:
            self._open[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self._open[name]

    def __getitem__(self, image_id: str) -> np.ndarray:
        name, row = self._where[image_id]
        return np.asarray(self._shard(name)[row])

    def get_many(self, image_ids: list[str]) -> np.ndarray:
        return np.stack([self[i] for i in image_ids])


class CacheWriter:
    """Accumulates features and flushes them as shards."""

    def __init__(self, root: str | Path, encoder_id: str, shard_size: int = 512):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder_id = encoder_id
        self.shard_size = shard_size
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["encoder_id"] != encoder_id:
                raise ValueError(
                    f"cache at {self.root} was built with {meta['encoder_id']!r}, "
                    f"not {encoder_id!r}"
                )
        self.done: set[str] = set()
        for name in _shard_names(self.root):
            if (self.root / f"{name}.npy").exists():
                self.done.update(json.loads((self.root / f"{name}.json").read_text())["ids"])
        for tmp in self.root.glob("*.tmp"):  # leftovers from an interrupted write
            tmp.unlink()
        names = _shard_names(self.root)
        self._next = int(names[-1].split("-")[1]) + 1 if names else 0
        self._ids: list[str] = []
        self._feats: list[np.ndarray] = []

    def add(self, image_ids: list[str], feats: np.ndarray) -> None:
        if len(image_ids) != len(feats):
            raise ValueError("ids and features differ in length")
        self._ids += image_ids
        self._feats.append(np.asarray(feats, dtype=DTYPE))
        while len(self._ids) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, n: int | None = None) -> None:
        if not self._ids:
            return
        n = n or len(self._ids)
        arr = np.concatenate(self._feats)
        ids, rest_ids, rest = self._ids[:n], self._ids[n:], arr[n:]
        arr = arr[:n]
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            meta = {"encoder_id": self.encoder_id, "shape": list(arr.shape[1:]), "dtype": "float16"}
            _atomic_write(meta_path, lambda f: f.write(json.dumps(meta, indent=2)), "w")
        name = f"shard-{self._next:05d}"
        _atomic_write(self.root / f"{name}.npy", lambda f: np.save(f, arr))
        # the manifest is written last: it is what makes the shard count
        _atomic_write(self.root / f"{name}.json", lambda f: f.write(json.dumps({"ids": ids})), "w")
        self._next += 1
        self.done.update(ids)
        self._ids, self._feats = rest_ids, ([rest] if len(rest_ids) else [])

    def close(self) -> None:
        self._flush()


def _batches(items: Iterator[tuple[str, Any]], size: int) -> Iterator[list[tuple[str, Any]]]:
    batch: list[tuple[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_cache(
    items: Iterable[tuple[str, Any]],
    encoder: BatchEncoder,
    root: str | Path,
    *,
    batch_size: int = 16,
    shard_size: int = 512,
    num_workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Encode every ``(image_id, source)`` not already cached. Safe to interrupt and rerun.

    ``source`` is anything ``load_image`` accepts. Unreadable images are skipped and listed in
    ``failed.jsonl``. Image decoding runs in a thread pool one batch ahead of the encoder.
    """
    writer = CacheWriter(root, encoder.encoder_id, shard_size)
    seen: set[str] = set()

    def fresh() -> Iterator[tuple[str, Any]]:
        for i, s in items:
            if i not in writer.done and i not in seen:  # also drops duplicates within the input
                seen.add(i)
                yield i, s

    todo = fresh()
    counts = {"encoded": 0, "skipped_existing": len(writer.done), "failed": 0}
    failed_path = Path(root) / "failed.jsonl"

    def load(
        batch: list[tuple[str, Any]],
    ) -> tuple[list[str], list[Image.Image], list[dict[str, str]]]:
        ids, imgs, bad = [], [], []
        for image_id, src in batch:
            try:
                imgs.append(load_image(src))
                ids.append(image_id)
            except (ImageLoadError, OSError) as e:
                bad.append({"id": image_id, "error": str(e)})
        return ids, imgs, bad

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        pending = []
        batches = _batches(iter(todo), batch_size)
        for batch in batches:  # prefetch up to num_workers batches ahead
            pending.append(pool.submit(load, batch))
            if len(pending) > num_workers:
                _consume(pending.pop(0), encoder, writer, counts, failed_path, progress)
        for fut in pending:
            _consume(fut, encoder, writer, counts, failed_path, progress)
    writer.close()
    return counts


def _consume(fut, encoder, writer, counts, failed_path, progress) -> None:
    ids, imgs, bad = fut.result()
    if bad:
        with failed_path.open("a") as f:
            f.writelines(json.dumps(b) + "\n" for b in bad)
        counts["failed"] += len(bad)
    if ids:
        feats = encoder.encode_images(imgs)
        if feats.shape[0] != len(ids):
            raise ValueError("encoder returned the wrong batch size")
        writer.add(ids, feats)
        counts["encoded"] += len(ids)
    if progress:
        progress(counts["encoded"], counts["failed"])


def items_from_facts(facts_jsonl: str | Path, image_root: str | Path) -> Iterator[tuple[str, Path]]:
    """``(image_id, path)`` pairs from an ``ImageFacts`` JSONL whose ``image_path`` is relative."""
    root = Path(image_root)
    with Path(facts_jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("image_path"):
                yield rec["image_id"], root / rec["image_path"]


if __name__ == "__main__":
    import argparse

    from .backends.siglip import SigLIPBackend

    ap = argparse.ArgumentParser(description="Build the frozen SigLIP 2 feature cache.")
    ap.add_argument("--source", action="append", required=True, metavar="FACTS.jsonl:IMAGE_ROOT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    def all_items() -> Iterator[tuple[str, Path]]:
        for spec in args.source:
            facts, root = spec.rsplit(":", 1)
            yield from items_from_facts(facts, root)

    backend = SigLIPBackend(device=args.device)
    done = {"n": 0}

    def show(enc: int, bad: int) -> None:
        if enc // 200 != done["n"] // 200:
            print(f"encoded {enc}  failed {bad}", flush=True)
        done["n"] = enc

    print(build_cache(all_items(), backend, args.out, batch_size=args.batch_size,
                      shard_size=args.shard_size, progress=show))  # fmt: skip
