"""Benchmark runner: one command runs any model over the splits and writes a report.

A *model* is anything with ``name`` and ``predict(image, questions, state=None)`` that returns the
public result dicts (``probs`` for choice/score, ``p_true`` for bool). ``imagejev.Model`` qualifies;
so do the baselines in ``bench.baselines``. Questions are grouped by image, so each image is
encoded once however many questions it carries.

Latency is measured separately from accuracy, on a sample of images:

* **cold**: image not cached, one question: image encoding + decision.
* **warm**: image already encoded, a different question: decision only. Only reported for models
  that expose ``encode()``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..calibration import fit_platt, fit_temperature
from ..data.records import ImageFacts, QuestionRecord, read_jsonl
from .metrics import Prediction, evaluate, from_bool, latency_summary, to_markdown


class Predictor(Protocol):
    name: str

    def predict(
        self, image: Any, questions: Mapping[str, Any], state: Any = None
    ) -> dict[str, dict[str, Any]]: ...


def option_labels(question: Mapping[str, Any]) -> list[str]:
    return list(question["criteria"] if question["type"] == "choice" else question["levels"])


def to_prediction(rec: QuestionRecord, out: Mapping[str, Any]) -> Prediction:
    """Turn one model result into a scored ``Prediction`` for the record's ground truth."""
    q = rec.question
    if q["type"] == "bool":
        return from_bool(rec.domain, float(out["p_true"]), bool(rec.answer), rec.task)
    labels = option_labels(q)
    p = np.array([out["probs"][label] for label in labels], dtype=np.float64)
    return Prediction(rec.domain, q["type"], p / p.sum(), labels.index(rec.answer), rec.task)


def group_by_image(records: Iterable[QuestionRecord]) -> dict[str, list[QuestionRecord]]:
    groups: dict[str, list[QuestionRecord]] = defaultdict(list)
    for r in records:
        groups[r.image_id].append(r)
    return dict(groups)


def subsample_images(
    records: Sequence[QuestionRecord], n: int | None, seed: int = 0
) -> list[QuestionRecord]:
    """Keep every question of at most ``n`` images, chosen by a hash of the image id.

    The choice depends only on the ids (not on order or on the model), so every model evaluated
    with the same ``n`` and ``seed`` sees exactly the same questions.
    """
    if n is None:
        return list(records)
    ids = sorted(
        {r.image_id for r in records},
        key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(),
    )
    keep = set(ids[:n])
    return [r for r in records if r.image_id in keep]


def run_split(
    model: Predictor,
    records: Iterable[QuestionRecord],
    image_for: Callable[[str], Any],
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[Prediction], dict[str, int]]:
    """Predict every record. An image that can't be loaded is skipped and counted, not fatal."""
    preds: list[Prediction] = []
    stats = {"questions": 0, "images": 0, "skipped_images": 0, "skipped_questions": 0}
    for n, (image_id, recs) in enumerate(group_by_image(records).items(), 1):
        try:
            image = image_for(image_id)
        except (OSError, KeyError, ValueError):
            stats["skipped_images"] += 1
            stats["skipped_questions"] += len(recs)
            continue
        out = model.predict(image, {f"q{i}": r.question for i, r in enumerate(recs)})
        preds += [to_prediction(r, out[f"q{i}"]) for i, r in enumerate(recs)]
        stats["images"] += 1
        stats["questions"] += len(recs)
        if progress:
            progress(n)
    return preds, stats


def measure_latency(
    model: Predictor,
    records: Iterable[QuestionRecord],
    image_for: Callable[[str], Any],
    *,
    n_images: int = 30,
    warmup: int = 2,
) -> dict[str, Any]:
    """Cold and warm latency in ms on up to ``n_images`` images that have 2+ questions."""
    groups = [(i, rs) for i, rs in group_by_image(records).items() if len(rs) >= 2]
    sample = groups[: n_images + warmup]
    can_warm = hasattr(model, "encode")
    cold, warm = [], []
    for k, (image_id, recs) in enumerate(sample):
        image = image_for(image_id)
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        t0 = time.perf_counter()
        model.predict(image, {"a": recs[0].question})
        t1 = time.perf_counter()
        if can_warm:
            handle = model.encode(image)  # cached by the call above
            t2 = time.perf_counter()
            model.predict(handle, {"b": recs[1].question})
            t3 = time.perf_counter()
        if k >= warmup:  # the first calls pay one-off costs (kernel compilation, text caches)
            cold.append((t1 - t0) * 1000)
            if can_warm:
                warm.append((t3 - t2) * 1000)
    out: dict[str, Any] = {"cold": latency_summary(cold) if cold else None}
    out["warm"] = latency_summary(warm) if warm else None
    return out


def fit_temperatures(
    model: Any,
    records: Iterable[QuestionRecord],
    image_for: Callable[[str], Any],
    *,
    min_examples: int = 20,
) -> dict[str, float]:
    """Fit one temperature per question type from raw logits on ``records`` (e.g. the val split).

    Types with fewer than ``min_examples`` examples keep their current temperature.
    """
    data: dict[str, tuple[list[Any], list[int]]] = {
        t: ([], []) for t in ("choice", "score", "bool")
    }
    for image_id, recs in group_by_image(records).items():
        try:
            image = image_for(image_id)
        except (OSError, KeyError, ValueError):
            continue
        logits = model.logits(image, {f"q{i}": r.question for i, r in enumerate(recs)})
        for i, r in enumerate(recs):
            t = r.question["type"]
            z = logits[f"q{i}"]
            data[t][0].append(z)
            data[t][1].append(
                int(bool(r.answer)) if t == "bool" else option_labels(r.question).index(r.answer)
            )
    fitted: dict[str, float] = {}
    for t, (zs, ys) in data.items():
        if len(ys) < min_examples:
            continue
        if t == "bool":  # a bias too: a bool score's zero need not mean 50%
            fitted["bool"], fitted["bool_bias"] = fit_platt(zs, ys)
        else:
            fitted[t] = fit_temperature(zs, ys, t)
    model.temperatures.update(fitted)
    return fitted


def load_model(name: str) -> Any:
    """``siglip2-zeroshot``, ``smolvlm-500m`` or ``smolvlm-2b``."""
    if name == "siglip2-zeroshot":
        from ..api import Model

        model = Model.load(name)
        model.name = name
        return model
    sizes = {
        "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "smolvlm-2b": "HuggingFaceTB/SmolVLM-Instruct",
    }
    if name in sizes:
        from .baselines.vlm import SmolVLM

        return SmolVLM(sizes[name])
    if (Path(name) / "config.json").exists():  # an exported imagejev model directory
        from ..api import Model

        model = Model.load(name)
        model.name = f"imagejev:{Path(name).name}"
        return model
    raise ValueError(
        f"unknown model {name!r}; try siglip2-zeroshot, {sorted(sizes)} or a model dir"
    )


def image_resolver(sources: Iterable[tuple[str | Path, str | Path]]) -> Callable[[str], Path]:
    """Map ``image_id`` to a file from ``(facts.jsonl, image_root)`` pairs."""
    paths: dict[str, Path] = {}
    for facts, root in sources:
        for f in read_jsonl(facts, ImageFacts):
            if f.image_path:
                paths[f.image_id] = Path(root) / f.image_path

    def resolve(image_id: str) -> Path:
        return paths[image_id]  # KeyError -> run_split counts the image as skipped

    return resolve


def run_benchmark(
    model: Predictor,
    split_files: Mapping[str, str | Path],
    image_for: Callable[[str], Any],
    out_dir: str | Path,
    *,
    fit_on: str | None = "val",
    latency_images: int = 30,
    max_images_per_split: int | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Run ``model`` over every split file and write ``results.json`` and ``report.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = {
        s: subsample_images(list(read_jsonl(p, QuestionRecord)), max_images_per_split)
        for s, p in split_files.items()
    }
    result: dict[str, Any] = {
        "model": model.name,
        "max_images_per_split": max_images_per_split,
        "splits": {},
    }
    if fit_on and fit_on in loaded and hasattr(model, "logits"):
        result["fitted_temperatures"] = fit_temperatures(model, loaded[fit_on], image_for)
    for split, records in loaded.items():
        cb = (lambda n, s=split: progress(s, n)) if progress else None
        preds, stats = run_split(model, records, image_for, progress=cb)
        result["splits"][split] = {"stats": stats, "rows": evaluate(preds) if preds else []}
    pool = next((loaded[s] for s in ("test-images", "val", "train") if loaded.get(s)), [])
    result["latency"] = (
        measure_latency(model, pool, image_for, n_images=latency_images) if pool else None
    )
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    (out / "report.md").write_text(render_report(result))
    return result


def render_report(result: Mapping[str, Any]) -> str:
    lines = [f"# {result['model']}", ""]
    if result.get("max_images_per_split"):
        lines += [
            f"Evaluated on a fixed subset of at most {result['max_images_per_split']} "
            "images per split.",
            "",
        ]
    if result.get("fitted_temperatures"):
        temps = ", ".join(f"{k}={v:.2f}" for k, v in result["fitted_temperatures"].items())
        lines += [f"Temperatures fitted on val: {temps}", ""]
    lat = result.get("latency")
    if lat:
        for kind in ("cold", "warm"):
            if lat.get(kind):
                v = lat[kind]
                lines.append(
                    f"- {kind} latency: p50 {v['p50_ms']:.1f} ms, "
                    f"p95 {v['p95_ms']:.1f} ms (n={v['n']})"
                )
        lines.append("")
    for split, body in result["splits"].items():
        s = body["stats"]
        lines.append(
            to_markdown(body["rows"], f"{split} ({s['questions']} questions, {s['images']} images)")
        )
        if s["skipped_images"]:
            lines.append(f"\n_{s['skipped_images']} images skipped (could not be loaded)._")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run a model over the benchmark splits.")
    ap.add_argument("--model", default="siglip2-zeroshot")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--images", action="append", required=True, metavar="FACTS.jsonl:IMAGE_ROOT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=None, help="default: every <split>.jsonl")
    ap.add_argument("--fit-on", default="val")
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="evaluate at most this many images per split (a fixed subset shared by all models)",
    )
    ap.add_argument(
        "--save-calibration",
        action="store_true",
        help="write the fitted temperatures into the model directory (for exported models)",
    )
    args = ap.parse_args()

    names = args.splits or [p.stem for p in sorted(Path(args.splits_dir).glob("*.jsonl"))]
    files = {n: Path(args.splits_dir) / f"{n}.jsonl" for n in names}
    resolver = image_resolver(tuple(s.rsplit(":", 1)) for s in args.images)  # type: ignore[arg-type]
    model = load_model(args.model)

    def show(split: str, n: int) -> None:
        if n % 200 == 0:
            print(f"{split}: {n} images", flush=True)

    run_benchmark(
        model,
        files,
        resolver,
        args.out,
        fit_on=args.fit_on,
        max_images_per_split=args.max_images,
        progress=show,
    )
    if args.save_calibration:
        path = model.save_temperatures(args.model)
        print(f"saved calibration to {path}")
    print(f"wrote {args.out}/report.md")
