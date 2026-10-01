import numpy as np
import pytest

from imagejev.api import Model, softmax
from imagejev.calibration import (
    ece,
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
