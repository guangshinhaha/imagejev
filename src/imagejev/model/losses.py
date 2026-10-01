"""Training losses, all proper scoring rules so calibration comes out of training.

* ``choice``: log loss (cross-entropy) over the options.
* ``bool``: binary log loss on the log-odds.
* ``score``: log loss **plus** the ranked probability score (RPS). Log loss alone treats predicting
  "unreadable" for a "clear" image the same as predicting "ok"; RPS compares cumulative
  distributions, so a miss by three levels costs more than a miss by one.

Targets may be soft (a teacher's probabilities over the options): log loss becomes
``-sum_i t_i log p_i`` and RPS uses the target's cumulative distribution. Bool targets are the
probability of "true" (0 or 1 for ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fusion import QTYPE_IDS, FusionBatch

RPS_WEIGHT = 1.0


@dataclass
class LossOutput:
    total: torch.Tensor  # scalar, weighted mean over questions
    per_question: torch.Tensor  # (Q,) before weighting
    by_type: dict[str, float]  # mean loss per question type present in the batch
    rps: float | None  # mean RPS over score questions, for logging


def to_grid(
    values: torch.Tensor, owner: torch.Tensor, slot: torch.Tensor, q: int, k: int, fill: float
) -> torch.Tensor:
    """Scatter per-option values ``(N,)`` into a ``(Q, K)`` grid; ``fill`` marks missing options."""
    grid = values.new_full((q, k), fill)
    grid[owner, slot] = values
    return grid


def rps(p: torch.Tensor, t: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
    """Ranked probability score per question from ``(Q, K)`` probabilities and targets.

    ``RPS = sum_{k=1}^{K-1} (CDF_p(k) - CDF_t(k))^2 / (K - 1)``. Padding carries zero probability,
    so both CDFs reach 1 at the last real option and padding adds nothing.
    """
    diff = p.cumsum(-1) - t.cumsum(-1)
    return (diff**2).sum(-1) / (n_options.to(p.dtype) - 1).clamp(min=1)


def fusion_loss(
    logits: torch.Tensor,
    batch: FusionBatch,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    rps_weight: float = RPS_WEIGHT,
) -> LossOutput:
    """Loss for one batch.

    ``logits`` and ``target`` are ``(N,)``, one entry per option. For ``choice``/``score`` questions
    the targets of a question's options sum to 1; for ``bool`` the single entry is P(true).
    ``weights`` is an optional per-question weight ``(Q,)``; the total is the weighted mean.
    """
    q = batch.qtype.shape[0]
    n_opt = batch.n_options()
    k = int(n_opt.max())
    neg = torch.finfo(logits.dtype).min
    grid = to_grid(logits, batch.owner, batch.slot, q, k, neg)
    tgt = to_grid(target.to(logits.dtype), batch.owner, batch.slot, q, k, 0.0)
    logp = F.log_softmax(grid, dim=-1)
    ce = -(tgt * logp.masked_fill(tgt == 0, 0.0)).sum(-1)  # log loss; 0 * -inf never appears

    p = logp.exp()
    score_rps = rps(p, tgt, n_opt)

    # bool questions have exactly one option: its logit is the log-odds of "true"
    bool_rows = batch.qtype == QTYPE_IDS["bool"]
    bce = torch.zeros(q, dtype=logits.dtype, device=logits.device)
    if bool_rows.any():
        first = grid[:, 0]
        bce = F.binary_cross_entropy_with_logits(first, tgt[:, 0], reduction="none")

    is_score = batch.qtype == QTYPE_IDS["score"]
    per_q = torch.where(bool_rows, bce, ce) + rps_weight * score_rps * is_score.to(logits.dtype)

    w = torch.ones_like(per_q) if weights is None else weights.to(per_q.dtype)
    total = (per_q * w).sum() / w.sum().clamp(min=1e-12)
    by_type = {
        name: float(per_q.detach()[batch.qtype == tid].mean())
        for name, tid in QTYPE_IDS.items()
        if bool((batch.qtype == tid).any())
    }
    mean_rps = float(score_rps.detach()[is_score].mean()) if bool(is_score.any()) else None
    return LossOutput(total, per_q, by_type, mean_rps)


def hard_targets(batch: FusionBatch, answers: list[int | bool]) -> torch.Tensor:
    """One-hot targets ``(N,)`` from the correct option index per question (bool: True/False)."""
    t = torch.zeros(batch.owner.shape[0])
    starts = torch.cumsum(batch.n_options(), 0) - batch.n_options()  # rows are grouped by question
    for qi, ans in enumerate(answers):
        if int(batch.qtype[qi]) == QTYPE_IDS["bool"]:
            t[starts[qi]] = float(bool(ans))
        else:
            t[starts[qi] + int(ans)] = 1.0
    return t
