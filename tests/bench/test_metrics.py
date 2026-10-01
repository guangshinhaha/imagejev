import math

import numpy as np
import pytest

from imagejev.bench.metrics import (
    Prediction,
    accuracy,
    accuracy_at_coverage,
    evaluate,
    expected_calibration_error,
    from_bool,
    latency_summary,
    log_loss,
    summarize,
    to_markdown,
    within_one,
)


def P(p, label, qtype="choice", domain="photo"):
    return Prediction(domain, qtype, np.array(p), label)


# three predictions with hand-computed answers
HAND = [P([0.7, 0.3], 0), P([0.4, 0.6], 0), P([0.9, 0.1], 0)]


def test_accuracy_hand_computed():
    assert accuracy(HAND) == pytest.approx(2 / 3)  # argmax = 0, 1, 0 against labels 0, 0, 0


def test_log_loss_hand_computed():
    # -(ln 0.7 + ln 0.4 + ln 0.9) / 3
    assert log_loss(HAND) == pytest.approx((0.356675 + 0.916291 + 0.105361) / 3, abs=1e-6)


def test_log_loss_is_finite_for_a_confident_wrong_answer():
    wrong = P([1.0, 0.0], 1)
    assert math.isfinite(log_loss([wrong])) and log_loss([wrong]) > 25


def test_perfect_and_uniform_log_loss():
    assert log_loss([P([1.0, 0.0], 0)]) == pytest.approx(0.0, abs=1e-9)
    assert log_loss([P([0.25] * 4, 2)]) == pytest.approx(math.log(4))


def test_ece_hand_computed():
    # confidences .7, .6, .9 land in three different bins; correct = yes, no, yes
    # ECE = (|1-.7| + |0-.6| + |1-.9|) / 3 = 1/3
    assert expected_calibration_error(HAND) == pytest.approx(1 / 3)


def test_ece_is_zero_when_confidence_matches_accuracy():
    # 10 predictions at 100% confidence, all correct
    assert expected_calibration_error([P([1.0, 0.0], 0)] * 10) == pytest.approx(0.0)
    # 10 at 80% confidence, exactly 8 correct
    preds = [P([0.8, 0.2], 0)] * 8 + [P([0.8, 0.2], 1)] * 2
    assert expected_calibration_error(preds) == pytest.approx(0.0)


def test_within_one_for_scores():
    # 5 levels: predicted 0,1,4,2 against truth 1,1,0,3 -> distances 1,0,4,1
    preds = [P(np.eye(5)[a], t, "score") for a, t in [(0, 1), (1, 1), (4, 0), (2, 3)]]
    assert within_one(preds) == pytest.approx(3 / 4)
    assert accuracy(preds) == pytest.approx(1 / 4)


def test_accuracy_at_80_coverage_hand_computed():
    # 5 questions; keep the 4 most confident. Confidences .99 .9 .8 .7 .6; correctness 1 1 0 1 0
    preds = [
        P([0.99, 0.01], 0), P([0.1, 0.9], 1), P([0.8, 0.2], 1), P([0.7, 0.3], 0), P([0.4, 0.6], 0),
    ]  # fmt: skip
    assert accuracy_at_coverage(preds, 0.8) == pytest.approx(3 / 4)  # drops the .6 (wrong) one
    assert accuracy_at_coverage(preds, 1.0) == pytest.approx(accuracy(preds))
    assert accuracy_at_coverage(preds, 0.2) == 1.0  # the single most confident one is right


def test_coverage_helps_when_confidence_is_informative():
    good = [P([0.95, 0.05], 0)] * 8 + [P([0.55, 0.45], 1)] * 2  # unsure ones are the wrong ones
    assert accuracy_at_coverage(good, 0.8) == 1.0 and accuracy(good) == 0.8


def test_coverage_ties_are_deterministic_and_validated():
    ties = [P([0.6, 0.4], 0), P([0.6, 0.4], 1)] * 3
    assert accuracy_at_coverage(ties, 0.5) == accuracy_at_coverage(ties, 0.5)
    with pytest.raises(ValueError):
        accuracy_at_coverage(ties, 0)


def test_latency_percentiles():
    out = latency_summary(list(range(1, 101)))
    assert out["p50_ms"] == pytest.approx(50.5) and out["p95_ms"] == pytest.approx(95.05)
    assert out["n"] == 100
    assert latency_summary([7.0]) == {"n": 1, "p50_ms": 7.0, "p95_ms": 7.0}
    with pytest.raises(ValueError):
        latency_summary([])


def test_bool_conversion_and_prediction_validation():
    p = from_bool("photo", 0.8, True)
    assert p.p.tolist() == pytest.approx([0.2, 0.8]) and p.label == 1 and p.pred == 1
    assert from_bool("photo", 0.3, True).pred == 0
    for bad in (
        lambda: P([0.5], 0),
        lambda: P([0.5, 0.4], 0),
        lambda: P([0.5, 0.5], 2),
        lambda: P([[0.5, 0.5]], 0),
    ):
        with pytest.raises(ValueError):
            bad()


def test_summarize_only_reports_within_one_for_score_groups():
    assert summarize(HAND)["within_one"] is None
    s = summarize([P(np.eye(3)[1], 1, "score")])
    assert s["within_one"] == 1.0 and s["n"] == 1 and "acc@80" in s
    with pytest.raises(ValueError):
        summarize([])


def test_evaluate_groups_by_domain_and_type_with_rollups():
    preds = [
        P([0.9, 0.1], 0, "choice", "photo"),
        P([0.1, 0.9], 0, "choice", "photo"),
        from_bool("document", 0.9, True),
        from_bool("document", 0.9, False),
        P(np.eye(3)[2], 2, "score", "document"),
    ]
    rows = {(r["domain"], r["qtype"]): r for r in evaluate(preds)}
    assert rows[("photo", "choice")]["n"] == 2 and rows[("photo", "choice")]["accuracy"] == 0.5
    assert rows[("document", "bool")]["accuracy"] == 0.5
    assert rows[("document", "score")]["within_one"] == 1.0
    assert rows[("document", "all")]["n"] == 3 and rows[("all", "bool")]["n"] == 2
    assert rows[("all", "all")]["n"] == 5
    # the overall accuracy is the pooled one: (1 + 0 + 1 + 0 + 1) / 5
    assert rows[("all", "all")]["accuracy"] == pytest.approx(3 / 5)
    assert evaluate([]) == []


def test_markdown_table():
    md = to_markdown(evaluate(HAND), "photo baseline")
    lines = md.splitlines()
    assert lines[0] == "### photo baseline" and lines[1].startswith("| domain | type | n | acc")
    assert "0.667" in md and "-" in md  # accuracy 2/3, and a '-' for the missing within-1
