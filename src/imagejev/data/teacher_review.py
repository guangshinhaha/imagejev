"""Spot-check teacher labels by hand and drop the criteria the teacher gets wrong.

``make_review_sheet`` draws a stratified sample of teacher records and writes, for a person:

* ``review.html``: image, question, options and the teacher's answer, 4 per row, printable;
* ``review.csv``: one row per sample with an empty ``human_ok`` column to fill with ``y`` or ``n``
  ("is the teacher's answer correct?"). Mark ``?`` if the question can't be judged from the image;
  those rows are ignored.

``score_review`` turns the filled CSV into per-criterion accuracy, ``keep_criteria`` applies the
80% bar from the spec (with a minimum number of judged examples), and ``filter_records`` writes
the teacher file with the failing criteria removed.
"""

from __future__ import annotations

import csv
import html
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..images import load_image
from .records import QuestionRecord
from .teacher import read_teacher_records

THUMB = 480
FIELDS = [
    "id",
    "image",
    "criterion",
    "question",
    "options",
    "teacher_answer",
    "confidence",
    "human_ok",
]


def sample_records(
    records: Sequence[QuestionRecord], n: int, seed: int = 0
) -> list[QuestionRecord]:
    """``n`` records spread as evenly as possible over criteria (round-robin by criterion)."""
    by_crit: dict[str, list[QuestionRecord]] = defaultdict(list)
    for r in records:
        by_crit[r.meta.get("criterion", r.task)].append(r)
    rng = random.Random(seed)
    for lst in by_crit.values():
        rng.shuffle(lst)
    order = sorted(by_crit)
    picked: list[QuestionRecord] = []
    i = 0
    while len(picked) < min(n, len(records)):
        lst = by_crit[order[i % len(order)]]
        if lst:
            picked.append(lst.pop())
        i += 1
        if i > 10 * (n + len(records)):
            break
    return picked


def _options(rec: QuestionRecord) -> str:
    q = rec.question
    if q["type"] == "bool":
        return "yes / no"
    labels = q["criteria"] if q["type"] == "choice" else q["levels"]
    return " | ".join(labels)


def make_review_sheet(
    teacher_jsonl: str | Path,
    image_paths: dict[str, str],
    out_dir: str | Path,
    n: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Write ``review.html`` + ``review.csv`` + thumbnails for a sample of ``n`` teacher answers."""
    recs = read_teacher_records(teacher_jsonl)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sample = sample_records(recs, n, seed)
    rows: list[dict[str, Any]] = []
    for k, r in enumerate(sample):
        thumb = f"images/{k:04d}.jpg"
        img = load_image(image_paths[r.image_id])
        img.thumbnail((THUMB, THUMB))
        img.save(out / thumb, quality=88)
        rows.append(
            {
                "id": k,
                "image": thumb,
                "criterion": r.meta.get("criterion", r.task),
                "question": r.question["instructions"],
                "options": _options(r),
                "teacher_answer": ("yes" if r.answer else "no")
                if r.question["type"] == "bool"
                else r.answer,
                "confidence": r.meta.get("confidence", ""),
                "human_ok": "",
            }
        )
    with (out / "review.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    def card(r: dict[str, Any]) -> str:
        e = html.escape
        return (
            f'<div class="c"><img src="{r["image"]}">'
            f'<div class="id">#{r["id"]} {e(r["criterion"])}</div>'
            f'<div class="q">{e(r["question"])}</div><div class="o">{e(r["options"])}</div>'
            f'<div class="a">teacher: <b>{e(str(r["teacher_answer"]))}</b></div></div>'
        )

    style = (
        "body{font:14px sans-serif;margin:16px}"
        ".g{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}"
        ".c{border:1px solid #ccc;padding:8px;break-inside:avoid}img{width:100%}.id{color:#666}"
        ".q{font-weight:600;margin:4px 0}.o{color:#555}.a{margin-top:4px}"
    )
    (out / "review.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Teacher review</title><style>{style}</style>"
        "<h2>Is the teacher's answer correct?</h2><p>Fill <code>human_ok</code> in review.csv with "
        "<b>y</b>, <b>n</b>, or <b>?</b> (cannot be judged from the image).</p>"
        f'<div class="g">{"".join(card(r) for r in rows)}</div>'
    )
    return {"samples": len(rows), "criteria": len({r["criterion"] for r in rows}), "dir": str(out)}


def score_review(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    """Per-criterion ``{n, correct, accuracy}`` from a filled CSV (``?`` and blanks skipped)."""
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "correct": 0})
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            verdict = row["human_ok"].strip().lower()
            if verdict not in ("y", "n"):
                continue
            s = stats[row["criterion"]]
            s["n"] += 1
            s["correct"] += verdict == "y"
    return {c: {**s, "accuracy": s["correct"] / s["n"]} for c, s in sorted(stats.items()) if s["n"]}


def keep_criteria(
    scores: dict[str, dict[str, Any]],
    all_criteria: Iterable[str],
    *,
    min_accuracy: float = 0.8,
    min_judged: int = 5,
) -> dict[str, Any]:
    """Keep a criterion only with enough judged examples *and* accuracy at least ``min_accuracy``.

    A criterion nobody judged is dropped, not assumed fine: the point of the check is that no
    teacher label is trusted without evidence.
    """
    keep, drop = [], {}
    for c in sorted(set(all_criteria)):
        s = scores.get(c)
        if s is None or s["n"] < min_judged:
            drop[c] = f"only {0 if s is None else s['n']} judged (need {min_judged})"
        elif s["accuracy"] < min_accuracy:
            drop[c] = f"accuracy {s['accuracy']:.2f} < {min_accuracy}"
        else:
            keep.append(c)
    return {"keep": keep, "drop": drop, "min_accuracy": min_accuracy, "min_judged": min_judged}


def filter_records(
    teacher_jsonl: str | Path, decision: dict[str, Any], out_path: str | Path
) -> dict[str, int]:
    """Write the teacher records of kept criteria to ``out_path``."""
    keep = set(decision["keep"])
    kept = dropped = 0
    with Path(out_path).open("w") as out:
        for line in Path(teacher_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("meta", {}).get("criterion") in keep:
                out.write(line + "\n")
                kept += 1
            else:
                dropped += 1
    return {"kept": kept, "dropped": dropped}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Teacher label spot-check.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("sheet", help="write the review sheet")
    mk.add_argument("--teacher", required=True)
    mk.add_argument("--facts", required=True)
    mk.add_argument("--out", required=True)
    mk.add_argument("--n", type=int, default=200)
    sc = sub.add_parser("score", help="score a filled sheet and filter the teacher file")
    sc.add_argument("--csv", required=True)
    sc.add_argument("--teacher", required=True)
    sc.add_argument("--out", required=True, help="filtered teacher jsonl")
    sc.add_argument("--min-accuracy", type=float, default=0.8)
    args = ap.parse_args()
    if args.cmd == "sheet":
        paths = {}
        with Path(args.facts).open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("image_path"):
                    paths[r["image_id"]] = r["image_path"]
        print(make_review_sheet(args.teacher, paths, args.out, args.n))
    else:
        scores = score_review(args.csv)
        recs = read_teacher_records(args.teacher)
        crits = {r.meta["criterion"] for r in recs}
        decision = keep_criteria(scores, crits, min_accuracy=args.min_accuracy)
        Path(args.out + ".decision.json").write_text(
            json.dumps({"scores": scores, **decision}, indent=2)
        )
        print(json.dumps(decision, indent=2))
        print(filter_records(args.teacher, decision, args.out))


if __name__ == "__main__":
    main()
