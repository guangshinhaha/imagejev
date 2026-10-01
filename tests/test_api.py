import numpy as np
import pytest
from PIL import Image

from imagejev.api import Model, format_answer, sigmoid, softmax
from imagejev.errors import ImageLoadError, QuestionSchemaError
from imagejev.schema import parse_question

CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "", "b": "", "c": ""}}
SCORE = {"type": "score", "instructions": "How?", "levels": ["lo", "mid", "hi"]}
BOOL = {"type": "bool", "instructions": "Is it?"}


class FakeBackend:
    encoder_id = "fake"

    def __init__(self):
        self.encodes = 0

    def encode_image(self, image):
        self.encodes += 1
        r = image.getpixel((0, 0))[0]
        return np.array([[r, 1.0], [0.0, 1.0]], dtype=np.float32)

    def logits(self, handle, state, question):
        r = float(handle.features[0, 0])
        if question.type == "bool":
            return np.array([r - 100.0])
        return np.arange(len(question.options), dtype=np.float64) * (r / 50.0)


def _img(r=200):
    return Image.new("RGB", (4, 4), (r, 0, 0))


def test_softmax_and_sigmoid():
    assert softmax(np.array([0.0, 0.0])).tolist() == [0.5, 0.5]
    assert sigmoid(0.0) == 0.5
    assert softmax(np.array([1.0, 0.0]), 10.0)[0] < softmax(np.array([1.0, 0.0]), 1.0)[0]


def test_output_shapes():
    out = Model(FakeBackend()).predict(_img(), {"c": CHOICE, "s": SCORE, "b": BOOL})
    assert set(out["c"]) == {"answer", "probs", "confidence"}
    assert set(out["s"]) == {"answer", "probs", "expected", "confidence"}
    assert set(out["b"]) == {"answer", "p_true", "confidence"}
    assert out["c"]["answer"] == "c"
    assert sum(out["c"]["probs"].values()) == pytest.approx(1.0)
    assert 0.0 <= out["s"]["expected"] <= 2.0
    assert out["b"]["answer"] is True
    assert out["b"]["confidence"] == pytest.approx(max(out["b"]["p_true"], 1 - out["b"]["p_true"]))


def test_bool_false_confidence():
    q = parse_question("b", BOOL)
    res = format_answer(q, np.array([-3.0]), 1.0)
    assert res["answer"] is False and res["confidence"] > 0.9


def test_predict_on_handle_equals_predict_on_image():
    backend = FakeBackend()
    m = Model(backend)
    qs = {"c": CHOICE, "s": SCORE, "b": BOOL}
    direct = m.predict(_img(), qs)
    handle = m.encode(_img())
    via_handle = m.predict(handle, qs)
    assert direct == via_handle
    assert backend.encodes == 1  # the handle skipped the vision encoder


def test_new_questions_on_cached_image_do_not_reencode():
    backend = FakeBackend()
    m = Model(backend)
    h = m.encode(_img())
    m.predict(h, {"c": CHOICE})
    m.predict(h, {"b": BOOL})
    assert backend.encodes == 1


def test_temperature_changes_confidence_not_answer():
    hot = Model(FakeBackend(), {"choice": 5.0}).predict(_img(), {"c": CHOICE})["c"]
    cold = Model(FakeBackend(), {"choice": 0.5}).predict(_img(), {"c": CHOICE})["c"]
    assert hot["answer"] == cold["answer"]
    assert hot["confidence"] < cold["confidence"]


def test_errors_surface_before_encoding():
    backend = FakeBackend()
    m = Model(backend)
    with pytest.raises(QuestionSchemaError):
        m.predict(_img(), {"x": {"type": "choice", "instructions": "x"}})
    assert backend.encodes == 0
    with pytest.raises(ImageLoadError):
        m.predict(b"junk", {"b": BOOL})


def test_unknown_model_name():
    with pytest.raises(ValueError):
        Model.load("nope")
