"""Download and materialise the external test sets into images + ``ext:`` records.

Oxford-IIIT Pet (test split), CORD-v2 (test + validation) and ScreenSpot are fetched from the
Hugging Face Hub as parquet, their images written to disk as JPEG, and their typed questions built
by ``data/external.py``. Everything is evaluation-only: ``source="ext:..."`` sends it to
``test-external``. Network access goes through two small injectable functions so tests need none.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .external import iter_cord, iter_oxford_pets, iter_screenspot
from .records import ImageFacts, QuestionRecord

HUB = "https://huggingface.co"
DATASETS = {
    "pets": ("timm/oxford-iiit-pet", ("test",)),
    "cord": ("naver-clova-ix/cord-v2", ("test", "validation")),
    "screenspot": ("rootsautomation/ScreenSpot", ("test",)),
}
JPEG_QUALITY = 90

JsonFetcher = Callable[[str], Any]
FileFetcher = Callable[[str, Path], None]


def _http_json(url: str) -> Any:
    import httpx

    from ..backends.siglip import _use_system_trust_store

    _use_system_trust_store()
    r = httpx.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.json()


def _http_file(url: str, dest: Path) -> None:
    import httpx

    from ..backends.siglip import _use_system_trust_store

    _use_system_trust_store()
    tmp = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=1800) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def class_names(repo: str, feature: str, fetch_json: JsonFetcher = _http_json) -> list[str]:
    """Class label names of ``feature`` from the dataset card metadata."""
    info = fetch_json(f"{HUB}/api/datasets/{repo}")["cardData"]["dataset_info"]
    info = info[0] if isinstance(info, list) else info
    for f in info["features"]:
        if f["name"] == feature:
            names = f["dtype"]["class_label"]["names"]
            return (
                [names[str(i)] for i in range(len(names))]
                if isinstance(names, dict)
                else list(names)
            )
    raise KeyError(f"{repo} has no class-label feature {feature!r}")


def download_external(
    raw_dir: str | Path,
    fetch_json: JsonFetcher = _http_json,
    fetch_file: FileFetcher = _http_file,
) -> dict[str, list[Path]]:
    """Fetch each dataset's evaluation parquet files (skipping ones already on disk)."""
    raw = Path(raw_dir)
    out: dict[str, list[Path]] = {}
    for key, (repo, prefixes) in DATASETS.items():
        files = [
            f["path"]
            for f in fetch_json(f"{HUB}/api/datasets/{repo}/tree/main/data")
            if f["path"].endswith(".parquet") and Path(f["path"]).name.startswith(prefixes)
        ]
        if not files:
            raise RuntimeError(f"no parquet files for {repo} with prefixes {prefixes}")
        out[key] = []
        for rel in sorted(files):
            dest = raw / key / Path(rel).name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                fetch_file(f"{HUB}/datasets/{repo}/resolve/main/{rel}", dest)
            out[key].append(dest)
    return out


def _rows(path: Path, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=list(columns)):
        yield from batch.to_pylist()


def _save_jpeg(raw: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(raw)).convert("RGB").save(dest, quality=JPEG_QUALITY)


def _pick(rows: Sequence[Any], key: Callable[[Any], str], n: int | None, seed: int) -> list[Any]:
    """A deterministic subset: sorted by a hash of the key, so it does not depend on row order."""
    if n is None or n >= len(rows):
        return list(rows)
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{key(r)}".encode()).hexdigest())
    return ranked[:n]


def build_pets(path: Path, breeds: Sequence[str], out: Path, *, limit: int | None, seed: int):
    # Metadata first (cheap): the cat/dog label of every breed, and which images to keep.
    meta = list(_rows(path, ["image_id", "label", "label_cat_dog"]))
    species = {int(r["label"]): ("cat" if int(r["label_cat_dog"]) == 0 else "dog") for r in meta}
    missing = [breeds[i] for i in range(len(breeds)) if i not in species]
    if missing:
        raise ValueError(f"no image of these breeds in {path.name}, species unknown: {missing[:5]}")
    keep = {str(r["image_id"]) for r in _pick(meta, lambda r: str(r["image_id"]), limit, seed)}
    adapted = []
    for r in _rows(
        path, ["image", "label", "image_id"]
    ):  # then stream images, saving only the kept
        if str(r["image_id"]) not in keep:
            continue
        dest = out / "pets" / "images" / f"{r['image_id']}.jpg"
        _save_jpeg(r["image"]["bytes"], dest)
        adapted.append(
            {"image_id": r["image_id"], "label": r["label"], "image_path": str(dest.resolve())}
        )
    return _split(iter_oxford_pets(adapted, breeds, species, seed=seed))


def build_cord(paths: Sequence[Path], out: Path):
    records: list[QuestionRecord] = []
    facts: list[ImageFacts] = []
    for path in paths:
        split = path.name.split("-")[0]
        rows = []
        for i, r in enumerate(_rows(path, ["image", "ground_truth"])):
            dest = out / "cord" / "images" / f"{split}-{i}.jpg"
            _save_jpeg(r["image"]["bytes"], dest)
            rows.append(
                {
                    "id": f"{split}-{i}",
                    "ground_truth": r["ground_truth"],
                    "image_path": str(dest.resolve()),
                }
            )
        q, f = _split(iter_cord(rows, split=split))
        records += q
        facts += f
    return records, facts


def build_screenspot(paths: Sequence[Path], out: Path):
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for r in _rows(path, ["file_name", "data_source", "image"]):
            name = r["file_name"]
            dest = out / "screenspot" / "images" / (Path(name).stem + ".jpg")
            if name not in seen:  # a screenshot appears once per instruction; save it once
                seen.add(name)
                _save_jpeg(r["image"]["bytes"], dest)
            rows.append(
                {
                    "file_name": name,
                    "data_source": r["data_source"],
                    "image_path": str(dest.resolve()),
                }
            )
    return _split(iter_screenspot(rows))


def _split(items) -> tuple[list[QuestionRecord], list[ImageFacts]]:
    items = list(items)
    return (
        [i for i in items if isinstance(i, QuestionRecord)],
        [i for i in items if isinstance(i, ImageFacts)],
    )


def build_external(
    paths: Mapping[str, Sequence[Path]],
    out: str | Path,
    *,
    pets_breeds: Sequence[str],
    pets_images: int | None = 1200,
    seed: int = 0,
) -> tuple[list[QuestionRecord], list[ImageFacts]]:
    """Images + records for every dataset in ``paths`` (keys: pets, cord, screenspot)."""
    out = Path(out)
    records: list[QuestionRecord] = []
    facts: list[ImageFacts] = []
    parts = []
    if "pets" in paths:
        parts.append(build_pets(paths["pets"][0], pets_breeds, out, limit=pets_images, seed=seed))
    if "cord" in paths:
        parts.append(build_cord(paths["cord"], out))
    if "screenspot" in paths:
        parts.append(build_screenspot(paths["screenspot"], out))
    for q, f in parts:
        records += q
        facts += f
    return records, facts
