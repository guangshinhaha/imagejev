"""Runs the real SigLIP 2 checkpoint. Skipped unless torch+transformers are installed and the
checkpoint is reachable (``pytest -m slow`` to opt in)."""

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    from imagejev import Model

    try:
        return Model.load("siglip2-zeroshot", device="cpu")
    except OSError as e:  # checkpoint not reachable
        pytest.skip(f"checkpoint unavailable: {e}")


def _red_square():
    img = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(img).rectangle([64, 64, 320, 320], fill="red")
    return img


def test_feature_shape(model):
    h = model.encode(_red_square())
    assert h.features.shape == (65, 768)  # 1 global + 8x8 pooled patches


def test_all_question_types_on_real_model(model):
    qs = {
        "color": {
            "type": "choice",
            "instructions": "Dominant shape colour:",
            "criteria": {"red": "a red square", "blue": "a blue square", "green": "a green square"},
        },
        "size": {
            "type": "score",
            "instructions": "Image content:",
            "levels": ["empty", "small", "big"],
        },
        "has_square": {"type": "bool", "instructions": "A red square on a white background."},
    }
    out = model.predict(_red_square(), qs)
    assert out["color"]["answer"] == "red"
    assert abs(sum(out["color"]["probs"].values()) - 1.0) < 1e-6
    assert isinstance(out["has_square"]["answer"], bool)


def test_cached_handle_matches_raw_image(model):
    qs = {"b": {"type": "bool", "instructions": "A red square."}}
    img = _red_square()
    assert model.predict(img, qs) == model.predict(model.encode(img), qs)
