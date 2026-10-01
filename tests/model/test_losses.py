import math

import pytest
import torch
import torch.nn.functional as F

from imagejev.model.fusion import QTYPE_IDS, FusionBatch
from imagejev.model.losses import fusion_loss, hard_targets, rps, to_grid


def batch_for(spec):
    """Only the layout fields matter to the loss. ``spec`` is a list of (qtype, n_options)."""
    owner, slot = [], []
    for q, (_, k) in enumerate(spec):
        owner += [q] * k
        slot += list(range(k))
    n = len(owner)
    return FusionBatch(
        opt_tokens=torch.zeros(n, 1, 1), opt_mask=torch.ones(n, 1, dtype=torch.bool),
        owner=torch.tensor(owner), slot=torch.tensor(slot),
        qtype=torch.tensor([QTYPE_IDS[t] for t, _ in spec]), image=torch.zeros(len(spec), 1, 1),
    )  # fmt: skip


def softmax(xs):
    e = [math.exp(x) for x in xs]
    return [v / sum(e) for v in e]


# ---- hand-computed values -----------------------------------------------------------------
def test_choice_log_loss_hand_computed():
    # logits [2, 0, 0] -> p0 = e^2 / (e^2 + 2) = 0.786986 ; loss = -ln p0
    out = fusion_loss(
        torch.tensor([2.0, 0.0, 0.0]), batch_for([("choice", 3)]), torch.tensor([1.0, 0, 0])
    )
    p0 = math.exp(2) / (math.exp(2) + 2)
    assert float(out.total) == pytest.approx(-math.log(p0)) == pytest.approx(0.23954, abs=1e-4)
    wrong = fusion_loss(
        torch.tensor([2.0, 0.0, 0.0]), batch_for([("choice", 3)]), torch.tensor([0, 1.0, 0])
    )
    assert float(wrong.total) == pytest.approx(-math.log(1 / (math.exp(2) + 2)))  # = 2.2395


def test_soft_targets_hand_computed():
    p = softmax([2.0, 0.0, 0.0])
    out = fusion_loss(
        torch.tensor([2.0, 0.0, 0.0]), batch_for([("choice", 3)]), torch.tensor([0.5, 0.5, 0.0])
    )
    assert float(out.total) == pytest.approx(-0.5 * math.log(p[0]) - 0.5 * math.log(p[1]))
    assert float(out.total) == pytest.approx(1.2395, abs=1e-3)


def test_soft_cross_entropy_is_minimised_by_matching_the_target():
    t = torch.tensor([0.7, 0.2, 0.1])
    b = batch_for([("choice", 3)])
    at_target = fusion_loss(t.log(), b, t).total  # logits = log t reproduces t exactly
    for logits in (
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, -2.0]),
        torch.tensor([-1.0, 1.0, 0.0]),
    ):
        assert float(fusion_loss(logits, b, t).total) > float(at_target)
    assert float(at_target) == pytest.approx(
        -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
    )


def test_bool_loss_hand_computed():
    b = batch_for([("bool", 1)])
    zero = fusion_loss(torch.tensor([0.0]), b, torch.tensor([1.0]))
    assert float(zero.total) == pytest.approx(math.log(2))
    wrong = fusion_loss(torch.tensor([2.0]), b, torch.tensor([0.0]))
    assert float(wrong.total) == pytest.approx(math.log(1 + math.exp(2)))  # softplus(2) = 2.1269
    right = fusion_loss(torch.tensor([2.0]), b, torch.tensor([1.0]))
    assert float(right.total) == pytest.approx(math.log(1 + math.exp(-2)))  # 0.1269


def test_rps_hand_computed():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    t = torch.tensor([[0.0, 0.0, 1.0]])
    # CDF_p = .5 .8 1.0 ; CDF_t = 0 0 1 ; diffs^2 = .25 + .64 + 0 ; / (K-1 = 2)
    assert float(rps(p, t, torch.tensor([3]))) == pytest.approx(0.445)
    assert float(rps(t, t, torch.tensor([3]))) == 0.0  # perfect prediction
    far = rps(torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 0, 1]]), torch.tensor([3]))
    near = rps(torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 1, 0]]), torch.tensor([3]))
    assert float(far) == pytest.approx(1.0) and float(near) == pytest.approx(
        0.5
    )  # distance matters


def test_rps_ignores_padding():
    p = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0]])
    t = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    assert float(rps(p, t, torch.tensor([3]))) == pytest.approx(0.445)


def test_score_loss_is_log_loss_plus_rps():
    logits = torch.tensor([0.5, 0.1, -0.3])
    b = batch_for([("score", 3)])
    t = torch.tensor([0.0, 0.0, 1.0])
    p = softmax(logits.tolist())
    expected_ce = -math.log(p[2])
    cdf = [p[0], p[0] + p[1], 1.0]
    expected_rps = ((cdf[0]) ** 2 + (cdf[1]) ** 2 + 0.0) / 2
    out = fusion_loss(logits, b, t)
    assert float(out.total) == pytest.approx(expected_ce + expected_rps, abs=1e-5)
    assert out.rps == pytest.approx(expected_rps, abs=1e-5)
    assert float(fusion_loss(logits, b, t, rps_weight=0.0).total) == pytest.approx(
        expected_ce, abs=1e-5
    )


def test_score_loss_prefers_a_near_miss_over_a_far_miss_unlike_plain_log_loss():
    b = batch_for([("score", 4)])
    t = torch.tensor([0.0, 0.0, 0.0, 1.0])  # truth: the top level
    # same probability (0.1) on the truth in both cases, but the rest of the mass sits near or far
    near = torch.tensor([0.1, 0.1, 0.7, 0.1]).log()
    far = torch.tensor([0.7, 0.1, 0.1, 0.1]).log()
    assert float(fusion_loss(near, b, t, rps_weight=0.0).total) == pytest.approx(
        float(fusion_loss(far, b, t, rps_weight=0.0).total), abs=1e-5
    )  # plain log loss cannot tell them apart
    assert float(fusion_loss(near, b, t).total) < float(fusion_loss(far, b, t).total) - 0.1


# ---- batching, weights, helpers ------------------------------------------------------------
def test_mixed_batch_matches_losses_computed_alone_and_by_type():
    spec = [("choice", 3), ("score", 4), ("bool", 1), ("choice", 2)]
    b = batch_for(spec)
    torch.manual_seed(0)
    logits = torch.randn(10)
    target = hard_targets(b, [1, 3, True, 0])
    out = fusion_loss(logits, b, target)
    assert out.per_question.shape == (4,)
    # question 0 alone
    solo = fusion_loss(logits[:3], batch_for([("choice", 3)]), target[:3])
    assert float(out.per_question[0]) == pytest.approx(float(solo.total), abs=1e-6)
    assert float(out.total) == pytest.approx(float(out.per_question.mean()), abs=1e-6)
    assert set(out.by_type) == {"choice", "score", "bool"}
    assert out.by_type["choice"] == pytest.approx(float(out.per_question[[0, 3]].mean()), abs=1e-6)


def test_choice_loss_matches_torch_cross_entropy_and_bool_matches_bce():
    torch.manual_seed(1)
    logits = torch.randn(5)
    out = fusion_loss(
        logits, batch_for([("choice", 5)]), hard_targets(batch_for([("choice", 5)]), [2])
    )
    assert float(out.total) == pytest.approx(
        float(F.cross_entropy(logits[None], torch.tensor([2]))), abs=1e-6
    )
    z = torch.randn(1)
    o = fusion_loss(z, batch_for([("bool", 1)]), torch.tensor([1.0]))
    assert float(o.total) == pytest.approx(
        float(F.binary_cross_entropy_with_logits(z, torch.tensor([1.0]))), abs=1e-6
    )


def test_weights_scale_questions_and_padding_does_not_leak():
    spec = [("choice", 2), ("choice", 6)]
    b = batch_for(spec)
    torch.manual_seed(2)
    logits = torch.randn(8)
    target = hard_targets(b, [0, 4])
    per_q = fusion_loss(logits, b, target).per_question
    w = torch.tensor([3.0, 1.0])
    weighted = fusion_loss(logits, b, target, weights=w).total
    assert float(weighted) == pytest.approx(float((per_q * w).sum() / w.sum()), abs=1e-6)
    solo = fusion_loss(logits[:2], batch_for([("choice", 2)]), target[:2])
    assert float(per_q[0]) == pytest.approx(
        float(solo.total), abs=1e-6
    )  # K=6 neighbour doesn't leak


def test_gradients_flow_and_are_finite_for_every_type():
    spec = [("choice", 3), ("score", 4), ("bool", 1)]
    b = batch_for(spec)
    logits = torch.randn(8, requires_grad=True)
    fusion_loss(logits, b, hard_targets(b, [1, 2, False])).total.backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad.abs() > 0).all()


def test_hard_targets_and_to_grid_layout():
    b = batch_for([("choice", 3), ("bool", 1), ("score", 2)])
    t = hard_targets(b, [2, True, 0])
    assert t.tolist() == [0, 0, 1, 1, 1, 0]
    g = to_grid(torch.arange(6.0), b.owner, b.slot, 3, 3, -1.0)
    assert g.tolist() == [[0, 1, 2], [3, -1, -1], [4, 5, -1]]
