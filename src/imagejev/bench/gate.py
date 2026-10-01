"""Release-criteria gate: does the candidate beat the SigLIP 2 zero-shot baseline?

Implements spec criteria 1 and 2 on the results the runner writes (``results.json``):

1. **Calibration and log loss.** In *every* domain the candidate has lower log loss and lower ECE
   than the baseline on the chosen split. The spec's final check uses ``test-tasks``; the stage-2
   gate uses ``val``.
2. **No domain below the baseline's accuracy** on any split both runs evaluated.

Domain roll-ups (all question types pooled) are compared, with the per-type rows shown for context.
Criteria 3 and 4 (the accuracy bar and latency; the shuffle test) are separate and not checked here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DOMAINS = ("photo", "document", "screenshot")


def _row(results: Mapping[str, Any], split: str, domain: str, qtype: str = "all") -> dict | None:
    body = results["splits"].get(split)
    if body is None:
        return None
    return next((r for r in body["rows"] if r["domain"] == domain and r["qtype"] == qtype), None)


@dataclass
class DomainCheck:
    domain: str
    candidate: dict[str, float]
    baseline: dict[str, float]
    log_loss_ok: bool
    ece_ok: bool

    @property
    def ok(self) -> bool:
        return self.log_loss_ok and self.ece_ok


@dataclass
class GateResult:
    split: str
    criterion1: list[DomainCheck] = field(default_factory=list)
    criterion2: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def criterion1_ok(self) -> bool:
        return bool(self.criterion1) and all(c.ok for c in self.criterion1) and not self.missing

    @property
    def criterion2_ok(self) -> bool:
        return bool(self.criterion2) and all(c["ok"] for c in self.criterion2)

    @property
    def passed(self) -> bool:
        return self.criterion1_ok and self.criterion2_ok

    def to_markdown(self, candidate: str = "candidate", baseline: str = "siglip2-zeroshot") -> str:
        v = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
        lines = [
            f"# Gate: {candidate} vs {baseline}",
            "",
            f"**{v(self.passed)}** (criterion 1: {v(self.criterion1_ok)}, criterion 2: "
            f"{v(self.criterion2_ok)})",
            "",
            f"## Criterion 1: lower log loss and ECE in every domain (`{self.split}`)",
            "",
            "| domain | log loss (cand / base) | ECE (cand / base) | result |",
            "|---|---|---|---|",
        ]
        for c in self.criterion1:
            lines.append(
                f"| {c.domain} | {c.candidate['log_loss']:.3f} / {c.baseline['log_loss']:.3f} "
                f"| {c.candidate['ece']:.3f} / {c.baseline['ece']:.3f} | {v(c.ok)} |"
            )
        for m in self.missing:
            lines.append(f"\n_Missing: {m}_")
        lines += [
            "",
            "## Criterion 2: no domain below the baseline's accuracy on any split",
            "",
            "| split | domain | accuracy (cand / base) | result |",
            "|---|---|---|---|",
        ]
        for c in self.criterion2:
            lines.append(
                f"| {c['split']} | {c['domain']} | {c['candidate']:.3f} / {c['baseline']:.3f} "
                f"| {v(c['ok'])} |"
            )
        return "\n".join(lines) + "\n"


def evaluate_gate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    split: str = "val",
    *,
    domains: Sequence[str] = DOMAINS,
    min_n: int = 30,
) -> GateResult:
    """Compare two ``results.json`` dicts. A domain with fewer than ``min_n`` questions in a split
    is reported as missing rather than judged on noise."""
    res = GateResult(split)
    for d in domains:
        c, b = _row(candidate, split, d), _row(baseline, split, d)
        if c is None or b is None or c["n"] < min_n or b["n"] < min_n:
            res.missing.append(f"{d} on {split} (needs >= {min_n} questions in both runs)")
            continue
        res.criterion1.append(
            DomainCheck(
                d,
                {"log_loss": c["log_loss"], "ece": c["ece"]},
                {"log_loss": b["log_loss"], "ece": b["ece"]},
                c["log_loss"] < b["log_loss"],
                c["ece"] < b["ece"],
            )
        )
    for sp in candidate["splits"]:
        for d in domains:
            c, b = _row(candidate, sp, d), _row(baseline, sp, d)
            if c is None or b is None or c["n"] < min_n or b["n"] < min_n:
                continue
            res.criterion2.append(
                {
                    "split": sp,
                    "domain": d,
                    "candidate": c["accuracy"],
                    "baseline": b["accuracy"],
                    "ok": c["accuracy"] >= b["accuracy"],
                }
            )
    return res


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Check release criteria 1-2 against a baseline.")
    ap.add_argument("--candidate", required=True, help="results.json of the candidate")
    ap.add_argument("--baseline", default="results/siglip2-zeroshot/results.json")
    ap.add_argument("--split", default="val", help="split for criterion 1 (spec: test-tasks)")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()
    cand = json.loads(Path(args.candidate).read_text())
    base = json.loads(Path(args.baseline).read_text())
    result = evaluate_gate(cand, base, args.split)
    md = result.to_markdown(cand.get("model", "candidate"), base.get("model", "baseline"))
    print(md)
    if args.out:
        Path(args.out).write_text(md)
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
