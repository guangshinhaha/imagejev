"""Calibration: expected calibration error and per-question-type temperature scaling."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

QUESTION_TYPES = ("choice", "score", "bool")
TEMPERATURES_FILE = "temperatures.json"
BOOL_BIAS = "bool_bias"  # stored beside the temperatures; may be any real number


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.shape != hit.shape:
        raise ValueError("confidences and correct must have the same length")
    if conf.size == 0:
        raise ValueError("ece needs at least one example")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # confidence == 1.0 belongs in the last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def _nll_choice(logits: list[np.ndarray], labels: np.ndarray, temperature: float) -> float:
    total = 0.0
    for z, y in zip(logits, labels, strict=True):
        s = z / temperature
        s = s - s.max()
        total -= s[y] - np.log(np.exp(s).sum())
    return total / len(labels)


def _nll_bool(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    s = logits / temperature
    # -log sigmoid(s) for y=1 and -log sigmoid(-s) for y=0, computed stably
    signed = np.where(labels == 1, s, -s)
    return float(np.mean(np.logaddexp(0.0, -signed)))


def fit_temperature(
    logits: Sequence[np.ndarray | float],
    labels: Sequence[int],
    kind: str,
    lo: float = 0.05,
    hi: float = 20.0,
) -> float:
    """Find the temperature minimising negative log-likelihood on a held-out set.

    ``kind`` is ``"bool"`` (logits are scalars, labels are 0/1) or ``"choice"`` / ``"score"``
    (logits are per-option arrays, labels are option indices). The loss is convex in 1/T, so a
    golden-section search over log T converges to the global optimum.
    """
    if kind not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {kind!r}")
    if len(logits) != len(labels) or len(labels) == 0:
        raise ValueError("need the same, non-zero number of logits and labels")
    y = np.asarray(labels, dtype=np.int64)
    if kind == "bool":
        z = np.asarray([float(np.ravel(v)[0]) for v in logits])
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("bool labels must be 0 or 1")

        def loss(t: float) -> float:
            return _nll_bool(z, y, t)
    else:
        arrs = [np.asarray(v, dtype=np.float64) for v in logits]
        if any(not 0 <= int(i) < len(a) for a, i in zip(arrs, y, strict=True)):
            raise ValueError("label index out of range")

        def loss(t: float) -> float:
            return _nll_choice(arrs, y, t)

    a, b = np.log(lo), np.log(hi)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(float(np.exp(c))), loss(float(np.exp(d)))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(float(np.exp(c)))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(float(np.exp(d)))
    return float(np.exp((a + b) / 2.0))


def fit_platt(
    logits: Sequence[float],
    labels: Sequence[int],
    *,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
    iters: int = 100,
) -> tuple[float, float]:
    """Fit ``p(true) = sigmoid(z / T + b)`` by Newton's method; returns ``(T, b)``.

    Temperature alone assumes a logit of 0 means 50%. Scores with no natural zero (a match score
    minus a generic caption's) need the bias too. The slope ``1/T`` is clamped to
    ``[min_scale, max_scale]``, so an uninformative or inverted score collapses to a near-constant
    prediction instead of silently flipping its meaning.
    """
    z = np.asarray([float(np.ravel(v)[0]) for v in logits])
    y = np.asarray(labels, dtype=np.float64)
    if len(z) != len(y) or len(y) == 0:
        raise ValueError("need the same, non-zero number of logits and labels")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("bool labels must be 0 or 1")
    # Standardise the score so Newton's method is well conditioned whatever its scale, start at the
    # smallest allowed slope, and backtrack so the loss can never go up.
    mu, sd = float(z.mean()), float(z.std()) or 1.0
    x = np.stack([(z - mu) / sd, np.ones_like(z)], axis=1)
    lo, hi = min_scale * sd, max_scale * sd  # slope bounds in standardised units

    def nll(w: np.ndarray) -> float:
        s = x @ w
        return float(np.mean(np.logaddexp(0.0, -np.where(y == 1.0, s, -s))))

    w = np.array([lo, 0.0])  # start at the slope floor: an inverted score never gets below it
    cur = nll(w)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (p - y) / len(y) + 1e-9 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x / len(y) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-8:
            cand = w - t * step
            cand[0] = np.clip(cand[0], lo, hi)
            new = nll(cand)
            if new <= cur + 1e-15:
                break
            t *= 0.5
        else:
            break
        done = cur - new < 1e-12
        w, cur = cand, new
        if done:
            break
    slope = w[0] / sd  # back to the original score's units
    return float(1.0 / slope), float(w[1] - slope * mu)


def save_temperatures(temperatures: dict[str, float], path: str | Path) -> Path:
    """Write temperatures to ``path`` (a file, or a model directory)."""
    p = Path(path)
    if p.suffix != ".json":
        p.mkdir(parents=True, exist_ok=True)
        p = p / TEMPERATURES_FILE
    unknown = set(temperatures) - set(QUESTION_TYPES) - {BOOL_BIAS}
    if unknown:
        raise ValueError(f"unknown question types: {sorted(unknown)}")
    p.write_text(json.dumps({k: float(v) for k, v in temperatures.items()}, indent=2) + "\n")
    return p


def load_temperatures(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if p.is_dir():
        p = p / TEMPERATURES_FILE
    data = json.loads(p.read_text())
    unknown = set(data) - set(QUESTION_TYPES) - {BOOL_BIAS}
    if unknown:
        raise ValueError(f"unknown question types in {p}: {sorted(unknown)}")
    if any(not v > 0 for k, v in data.items() if k != BOOL_BIAS):
        raise ValueError(f"temperatures in {p} must be positive")
    return {k: float(v) for k, v in data.items()}
