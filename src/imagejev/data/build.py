"""Assemble a benchmark dataset from every source, template it, and split it.

``python -m imagejev.data.build --out data/bench_v0 --coco-dir data/raw/coco ...`` runs:

1. component builders (each writes images + ``facts.jsonl`` + ``questions.jsonl`` under ``out``):
   synthetic web pages, synthetic documents, COCO photos, Rico screens, quality variants;
2. :func:`assemble`: absolute image paths, the templater (paraphrase, options, compositional,
   balance), then the split builder.

Evaluation-only use of third-party images is fine for an internal benchmark, but anything that ships
must follow ``docs/data-licenses.md`` (for COCO that means ``--coco-licenses permissive``).
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from . import docgen, quality, webgen
from .coco import PERMISSIVE_LICENSE_IDS, iter_coco, load_instances
from .records import ImageFacts, QuestionRecord, read_jsonl, write_jsonl
from .rico import iter_rico
from .splits import build_splits, check_splits, write_splits
from .templater import apply_templates


def build_coco(
    coco_dir: Path, out: Path, *, limit: int | None, licenses: frozenset[int] | None, seed: int
) -> tuple[list[QuestionRecord], list[ImageFacts]]:
    inst = load_instances(coco_dir / "instances_val2017.json")
    records: list[QuestionRecord] = []
    facts: list[ImageFacts] = []
    images = 0
    for o in iter_coco(
        inst, seed=seed, licenses=licenses, image_dir=str((coco_dir / "val2017").resolve())
    ):
        if isinstance(o, ImageFacts):
            if limit is not None and images >= limit:
                break
            images += 1
            facts.append(o)
        elif limit is None or images <= limit:
            records.append(o)
    keep = {f.image_id for f in facts}
    return [r for r in records if r.image_id in keep], facts


def build_rico(
    parquet: Path, out: Path, *, limit: int, seed: int
) -> tuple[list[QuestionRecord], list[ImageFacts]]:
    import pyarrow.parquet as pq

    cols = ["request_id", "activity", "is_keyboard_deployed", "screenshot"]
    rows = pq.read_table(parquet, columns=cols).slice(0, limit).to_pylist()
    img_dir = out / "rico" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(r["request_id"]): r for r in rows}
    facts_out: list[ImageFacts] = []
    records: list[QuestionRecord] = []
    for o in iter_rico(rows, split="test", seed=seed):
        if isinstance(o, ImageFacts):
            rid = o.image_id.rsplit(":", 1)[1]
            row = by_id[rid]
            path = img_dir / f"{rid}.jpg"
            if not path.exists():
                Image.open(io.BytesIO(row["screenshot"]["bytes"])).convert("RGB").save(
                    path, quality=90
                )
            o.image_path = str(path.resolve())
            facts_out.append(o)
        else:
            records.append(o)
    return records, facts_out


def build_quality(
    sources: Iterable[tuple[str, str, Path, str | None]], out: Path, *, seed: int
) -> tuple[list[QuestionRecord], list[ImageFacts]]:
    """Degraded copies of the given ``(image_id, domain, path, style)`` sources."""
    srcs = (
        quality.Source(iid, domain, Image.open(path).convert("RGB"), style)
        for iid, domain, path, style in sources
    )
    quality.generate(srcs, out / "quality", seed=seed, kinds_per_source=2)
    facts = list(read_jsonl(out / "quality" / "facts.jsonl", ImageFacts))
    for f in facts:  # make paths absolute so one resolver serves every component
        f.image_path = str((out / "quality" / f.image_path).resolve())
    return list(read_jsonl(out / "quality" / "questions.jsonl", QuestionRecord)), facts


def load_component(directory: Path) -> tuple[list[QuestionRecord], list[ImageFacts]]:
    """Read a generator's output directory, making image paths absolute."""
    facts = list(read_jsonl(directory / "facts.jsonl", ImageFacts))
    for f in facts:
        if f.image_path:
            f.image_path = str((directory / f.image_path).resolve())
    return list(read_jsonl(directory / "questions.jsonl", QuestionRecord)), facts


def assemble(
    records: Sequence[QuestionRecord],
    facts: Sequence[ImageFacts],
    out: Path,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Template, split and write ``splits/`` plus the combined ``facts_all.jsonl``."""
    templated = apply_templates(records, facts, seed=seed)
    result = build_splits(templated, facts, seed=seed)
    check_splits(result)
    report = write_splits(result, out / "splits")
    write_jsonl(facts, out / "facts_all.jsonl")
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-web", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=0)
    ap.add_argument(
        "--coco-dir", default=None, help="folder with instances_val2017.json + val2017/"
    )
    ap.add_argument("--coco-limit", type=int, default=None)
    ap.add_argument("--coco-licenses", choices=["all", "permissive"], default="all")
    ap.add_argument("--rico-parquet", default=None)
    ap.add_argument("--rico-limit", type=int, default=1500)
    ap.add_argument("--quality-sources", type=int, default=0, help="per domain")
    ap.add_argument("--reuse", action="store_true", help="keep web/docs output already on disk")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records: list[QuestionRecord] = []
    facts: list[ImageFacts] = []

    def add(r: list[QuestionRecord], f: list[ImageFacts], name: str) -> None:
        print(f"{name}: {len(f)} images, {len(r)} questions", flush=True)
        records.extend(r)
        facts.extend(f)

    if args.n_web:
        if not (args.reuse and (out / "web" / "questions.jsonl").exists()):
            webgen.generate(args.n_web, out / "web", seed=args.seed)
        add(*load_component(out / "web"), "web")
    if args.n_docs:
        if not (args.reuse and (out / "docs" / "questions.jsonl").exists()):
            docgen.generate(args.n_docs, out / "docs", seed=args.seed)
        add(*load_component(out / "docs"), "docs")
    if args.coco_dir:
        lic = PERMISSIVE_LICENSE_IDS if args.coco_licenses == "permissive" else None
        add(
            *build_coco(
                Path(args.coco_dir), out, limit=args.coco_limit, licenses=lic, seed=args.seed
            ),
            "coco",
        )
    if args.rico_parquet:
        add(
            *build_rico(Path(args.rico_parquet), out, limit=args.rico_limit, seed=args.seed), "rico"
        )
    if args.quality_sources:
        by_domain: dict[str, list[tuple[str, str, Path, str | None]]] = {}
        for f in facts:
            if f.image_path and f.source in ("coco", "docgen", "webgen"):
                by_domain.setdefault(f.domain, []).append(
                    (f.image_id, f.domain, Path(f.image_path), f.style)
                )
        picked = [s for items in by_domain.values() for s in items[: args.quality_sources]]
        add(*build_quality(picked, out, seed=args.seed), "quality")
    report = assemble(records, facts, out, seed=args.seed)
    print(
        json.dumps({k: report[k] for k in ("questions", "image_groups", "thin_families")}, indent=2)
    )


if __name__ == "__main__":
    main()
