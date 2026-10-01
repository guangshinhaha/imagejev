"""Benchmark metrics, reported per domain x question type.

Every question, whatever its type, is scored as a classification over its options: ``choice`` and
``score`` use their option probabilities, and ``bool`` is the two-class case ``[p_false, p_true]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ..calibration import ece

EPS = 1e-12
COVERAGE = 0.8


@dataclass(frozen=True)
class Prediction:
    domain: str
    qtype: str  # "choice" | "score" | "bool"
    p: np.ndarray  # probabilities over the options (bool: [p_false, p_true])
    label: int  # index of the true option (bool: 0 = False, 1 = True)
    task: str = ""

    def __post_init__(self) -> None:
        p = np.asarray(self.p, dtype=np.float64)
        if p.ndim != 1 or len(p) < 2:
            raise ValueError("p must be a 1-D probability vector with at least 2 options")
        if not 0 <= self.label < len(p):
            raise ValueError(f"label {self.label} out of range for {len(p)} options")
        if abs(p.sum() - 1.0) > 1e-3 or (p < -1e-9).any():
            raise ValueError(f"p is not a probability vector: {p}")
        object.__setattr__(self, "p", p)

    @property
    def pred(self) -> int:
        return int(np.argmax(self.p))

    @property
    def confidence(self) -> float:
        return float(self.p.max())


def from_bool(domain: str, p_true: float, truth: bool, task: str = "") -> Prediction:
    return Prediction(domain, "bool", np.array([1.0 - p_true, p_true]), int(truth), task)


def accuracy(preds: Sequence[Prediction]) -> float:
    return float(np.mean([p.pred == p.label for p in preds]))


def within_one(preds: Sequence[Prediction]) -> float:
    """Share of score predictions at most one level from the truth."""
    return float(np.mean([abs(p.pred - p.label) <= 1 for p in preds]))


def log_loss(preds: Sequence[Prediction]) -> float:
    return float(np.mean([-math.log(max(float(p.p[p.label]), EPS)) for p in preds]))


def expected_calibration_error(preds: Sequence[Prediction], n_bins: int = 15) -> float:
    return ece([p.confidence for p in preds], [p.pred == p.label for p in preds], n_bins)


def accuracy_at_coverage(preds: Sequence[Prediction], coverage: float = COVERAGE) -> float:
    """Accuracy when answering only the ``coverage`` share of questions it is most confident about.

    Ties at the cutoff are broken by input order, so the result is deterministic. This is the
    number that shows whether low confidence really marks the answers to escalate.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    order = sorted(range(len(preds)), key=lambda i: -preds[i].confidence)  # stable
    keep = order[: max(1, math.ceil(coverage * len(preds)))]
    return float(np.mean([preds[i].pred == preds[i].label for i in keep]))


def latency_summary(times_ms: Sequence[float]) -> dict[str, float]:
    """p50 and p95 (linear interpolation) of a list of latencies in milliseconds."""
    if len(times_ms) == 0:
        raise ValueError("no latency samples")
    t = np.asarray(times_ms, dtype=np.float64)
    return {
        "n": len(t),
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
    }


def summarize(preds: Sequence[Prediction]) -> dict[str, float | int | None]:
    """All metrics for one group of predictions."""
    if not preds:
        raise ValueError("no predictions to summarize")
    is_score = all(p.qtype == "score" for p in preds)
    return {
        "n": len(preds),
        "accuracy": accuracy(preds),
        "within_one": within_one(preds) if is_score else None,
        "log_loss": log_loss(preds),
        "ece": expected_calibration_error(preds),
        f"acc@{int(COVERAGE * 100)}": accuracy_at_coverage(preds),
    }


def evaluate(preds: Iterable[Prediction]) -> list[dict[str, object]]:
    """Per-(domain, qtype) rows plus per-domain, per-type and overall roll-ups.

    Roll-ups pool the predictions (micro average), so large groups weigh more; read the per-group
    rows when domains must be compared fairly.
    """
    preds = list(preds)
    groups: dict[tuple[str, str], list[Prediction]] = {}
    for p in preds:
        groups.setdefault((p.domain, p.qtype), []).append(p)
    rows: list[dict[str, object]] = []

    def add(domain: str, qtype: str, members: list[Prediction]) -> None:
        rows.append({"domain": domain, "qtype": qtype, **summarize(members)})

    for (domain, qtype), members in sorted(groups.items()):
        add(domain, qtype, members)
    for domain in sorted({p.domain for p in preds}):
        add(domain, "all", [p for p in preds if p.domain == domain])
    for qtype in sorted({p.qtype for p in preds}):
        add("all", qtype, [p for p in preds if p.qtype == qtype])
    if preds:
        add("all", "all", preds)
    return rows


def to_markdown(rows: Sequence[dict[str, object]], title: str = "") -> str:
    cols = [("domain", "domain"), ("qtype", "type"), ("n", "n"), ("accuracy", "acc"),
            ("within_one", "within-1"), ("log_loss", "log loss"), ("ece", "ECE"),
            (f"acc@{int(COVERAGE * 100)}", f"acc@{int(COVERAGE * 100)}%")]  # fmt: skip

    def cell(v: object) -> str:
        if v is None:
            return "-"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = [f"### {title}"] if title else []
    lines.append("| " + " | ".join(h for _, h in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(lines)
