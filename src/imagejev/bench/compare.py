"""Render a side-by-side comparison of several ``results.json`` files as markdown.

``python -m imagejev.bench.compare --results base=a/results.json ours=b/results.json`` prints one
table per split. Each cell is ``accuracy / log loss / ECE``, so the README's numbers come straight
from the files instead of being typed by hand.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DOMAINS = ("photo", "document", "screenshot", "all")


def _row(result: Mapping[str, Any], split: str, domain: str) -> dict[str, Any] | None:
    body = result["splits"].get(split)
    if body is None:
        return None
    return next((r for r in body["rows"] if r["domain"] == domain and r["qtype"] == "all"), None)


def cell(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "-"
    return f"{row['accuracy']:.3f} / {row['log_loss']:.3f} / {row['ece']:.3f}"


def compare_table(results: Mapping[str, Mapping[str, Any]], split: str) -> str:
    """One markdown table for ``split``: rows are domains, columns are models."""
    names = list(results)
    first = next(iter(results.values()))
    if split not in first["splits"]:
        raise KeyError(f"{split!r} not in results")
    lines = [
        f"**{split}**",
        "",
        "| domain | n | " + " | ".join(names) + " |",
        "|---|---|" + "|".join("---" for _ in names) + "|",
    ]
    for d in DOMAINS:
        rows = [_row(r, split, d) for r in results.values()]
        if all(r is None for r in rows):
            continue
        n = next(r["n"] for r in rows if r is not None)
        lines.append(f"| {d} | {n} | " + " | ".join(cell(r) for r in rows) + " |")
    lines += ["", "_cells are accuracy / log loss / ECE_"]
    return "\n".join(lines)


def latency_line(results: Mapping[str, Mapping[str, Any]]) -> str:
    parts = []
    for name, r in results.items():
        lat = r.get("latency") or {}
        cold, warm = lat.get("cold"), lat.get("warm")
        if not cold:
            parts.append(f"{name}: n/a")
            continue
        text = f"{name}: cold {cold['p50_ms']:.0f} ms"
        if warm:
            text += f", warm {warm['p50_ms']:.1f} ms"
        parts.append(text)
    return "Latency p50 (one MacBook Pro, MPS): " + "; ".join(parts)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Compare benchmark results.")
    ap.add_argument("--results", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()
    results = {}
    for spec in args.results:
        name, path = spec.split("=", 1)
        results[name] = json.loads(Path(path).read_text())
    splits: Sequence[str] = args.splits or list(next(iter(results.values()))["splits"])
    for sp in splits:
        print(compare_table(results, sp), "\n")
    print(latency_line(results))


if __name__ == "__main__":
    main()
