import numpy as np
import pytest

from imagejev.api import Model, softmax
from imagejev.calibration import (
    ece,
    fit_platt,
    fit_temperature,
    load_temperatures,
    save_temperatures,
)


def test_ece_perfectly_calibrated_and_overconfident():
    assert ece([1.0, 1.0], [True, True]) == 0.0
    # always 90% confident but right half the time -> gap of 0.4
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        ece([], [])
    with pytest.raises(ValueError):
        ece([0.5], [True, False])


def _overconfident_choice(n=4000, k=4, boost=3.0, seed=0):
    rng = np.random.default_rng(seed)
    logits, labels = [], []
    for _ in range(n):
        true_logits = rng.normal(size=k)
        p = softmax(true_logits)
        labels.append(int(rng.choice(k, p=p)))
        logits.append(true_logits * boost)  # model is `boost` times too sharp
    return logits, labels


def _ece_of(logits, labels, t):
    conf, hit = [], []
    for z, y in zip(logits, labels, strict=True):
        p = softmax(z, t)
        conf.append(p.max())
        hit.append(int(np.argmax(p)) == y)
    return ece(conf, hit)


def test_fit_temperature_recovers_overconfidence_and_reduces_ece():
    logits, labels = _overconfident_choice()
    t = fit_temperature(logits, labels, "choice")
    assert t == pytest.approx(3.0, rel=0.15)
    assert _ece_of(logits, labels, t) < _ece_of(logits, labels, 1.0) / 3


def test_fit_temperature_bool():
    rng = np.random.default_rng(1)
    true = rng.normal(size=5000) * 2
    labels = (rng.random(5000) < 1 / (1 + np.exp(-true))).astype(int).tolist()
    t = fit_temperature((true * 4).tolist(), labels, "bool")
    assert t == pytest.approx(4.0, rel=0.2)


def test_fit_temperature_underconfident_gives_t_below_one():
    logits, labels = _overconfident_choice(boost=0.3)
    assert fit_temperature(logits, labels, "score") < 1.0


def test_fit_temperature_validates_input():
    with pytest.raises(ValueError):
        fit_temperature([np.zeros(3)], [5], "choice")
    with pytest.raises(ValueError):
        fit_temperature([0.0], [2], "bool")
    with pytest.raises(ValueError):
        fit_temperature([], [], "choice")
    with pytest.raises(ValueError):
        fit_temperature([np.zeros(2)], [0], "weird")


def test_temperatures_round_trip(tmp_path):
    temps = {"choice": 2.5, "score": 1.5, "bool": 0.8}
    p = save_temperatures(temps, tmp_path / "model")
    assert p.name == "temperatures.json"
    assert load_temperatures(tmp_path / "model") == temps
    assert load_temperatures(p) == temps


def test_temperature_file_validation(tmp_path):
    with pytest.raises(ValueError):
        save_temperatures({"bogus": 1.0}, tmp_path / "t.json")
    (tmp_path / "bad.json").write_text('{"choice": -1}')
    with pytest.raises(ValueError):
        load_temperatures(tmp_path / "bad.json")


def test_model_saves_its_temperatures(tmp_path):
    class B:
        encoder_id = "x"

    m = Model(B(), {"choice": 2.0})
    m.save_temperatures(tmp_path)
    assert load_temperatures(tmp_path)["choice"] == 2.0


def test_fit_platt_recovers_temperature_and_bias():
    rng = np.random.default_rng(3)
    true = rng.normal(size=8000) * 2
    labels = (rng.random(8000) < 1 / (1 + np.exp(-(true / 2.0 + 1.5)))).astype(int)
    # the model's score is the true logit scaled 3x and shifted by -4: no natural zero
    t, b = fit_platt((true * 3 - 4).tolist(), labels.tolist())
    # sigmoid(z/T + b) must reproduce sigmoid(true/2 + 1.5), where true = (z + 4) / 3
    assert t == pytest.approx(6.0, rel=0.15)
    assert b == pytest.approx(1.5 + 4 / 6.0, abs=0.2)


def test_platt_beats_temperature_alone_on_an_offset_score():
    rng = np.random.default_rng(4)
    z = rng.normal(size=4000)
    labels = (rng.random(4000) < 1 / (1 + np.exp(-(2 * z + 3)))).astype(int)  # positives dominate
    t_only = fit_temperature(z.tolist(), labels.tolist(), "bool")
    t, b = fit_platt(z.tolist(), labels.tolist())

    def nll(a, c):
        s = z * a + c
        return float(np.mean(np.logaddexp(0, -np.where(labels == 1, s, -s))))

    assert nll(1 / t, b) < nll(1 / t_only, 0.0) - 0.05


def test_platt_does_not_flip_an_inverted_or_useless_score():
    rng = np.random.default_rng(5)
    z = rng.normal(size=3000)
    inverted = (rng.random(3000) < 1 / (1 + np.exp(2 * z))).astype(int)  # high score -> false
    t, b = fit_platt(z.tolist(), inverted.tolist())
    assert 1 / t == pytest.approx(0.05)  # slope clamped to the floor: near-constant output
    noise = rng.integers(0, 2, 3000)
    t2, _ = fit_platt(z.tolist(), noise.tolist())
    assert 1 / t2 < 0.2


def test_platt_validates_input():
    with pytest.raises(ValueError):
        fit_platt([], [])
    with pytest.raises(ValueError):
        fit_platt([0.0, 1.0], [0, 2])
    with pytest.raises(ValueError):
        fit_platt([0.0], [0, 1])


def test_bias_round_trips_and_may_be_negative(tmp_path):
    temps = {"choice": 2.0, "bool": 1.3, "bool_bias": -0.7}
    p = save_temperatures(temps, tmp_path)
    assert load_temperatures(p) == temps
    (tmp_path / "bad.json").write_text('{"bool": -1.0}')
    with pytest.raises(ValueError):
        load_temperatures(tmp_path / "bad.json")


def test_model_applies_the_bool_bias():
    from imagejev.api import format_answer
    from imagejev.schema import parse_question

    q = parse_question("b", {"type": "bool", "instructions": "x"})
    neutral = format_answer(q, np.array([0.0]), 1.0, 0.0)["p_true"]
    shifted = format_answer(q, np.array([0.0]), 1.0, 2.0)["p_true"]
    assert neutral == pytest.approx(0.5) and shifted == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert (
        Model(type("B", (), {"encoder_id": "x"})(), {"bool_bias": 1.0}).temperatures["bool_bias"]
        == 1.0
    )
