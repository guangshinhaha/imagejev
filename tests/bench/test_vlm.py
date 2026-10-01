import numpy as np
import pytest
from PIL import Image, ImageDraw

from imagejev.bench.baselines.vlm import LETTERS, build_prompt, option_letters
from imagejev.schema import parse_question


def q(spec):
    return parse_question("q", spec)


def test_letters():
    assert option_letters(3) == "ABC" and option_letters(26)[-1] == "Z"
    assert len(LETTERS) == 32 and len(set(LETTERS)) == 32
    for bad in (1, 33):
        with pytest.raises(ValueError):
            option_letters(bad)


def test_bool_prompt():
    p = build_prompt(q({"type": "bool", "instructions": "Is it red?"}))
    assert p == "Is it red?\nAnswer Yes or No."
    assert build_prompt(q({"type": "bool", "instructions": "Is it red?"}), "invoice").startswith(
        "Context: invoice\nIs it red?"
    )


def test_choice_prompt_lists_letters_labels_and_descriptions():
    p = build_prompt(
        q({"type": "choice", "instructions": "Which?", "criteria": {"cat": "a feline", "dog": ""}})
    )
    assert p.splitlines() == [
        "Which?", "Options:", "A. cat: a feline", "B. dog", "Answer with the letter of the correct option.",
    ]  # fmt: skip


def test_score_prompt_says_options_are_ordered():
    p = build_prompt(
        q({"type": "score", "instructions": "How big?", "levels": ["small", "mid", "big"]})
    )
    assert "ordered from lowest to highest" in p and "C. big" in p


# ---- the real model (slow): checks the claims the baseline relies on ---------------------------
pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def vlm():
    from imagejev.bench.baselines.vlm import SmolVLM

    try:
        return SmolVLM("HuggingFaceTB/SmolVLM-256M-Instruct")
    except OSError as e:
        pytest.skip(f"checkpoint unavailable: {e}")


def scene():
    img = Image.new("RGB", (480, 360), "white")
    ImageDraw.Draw(img).rectangle([80, 60, 360, 300], fill=(210, 20, 20))
    return img


@pytest.mark.slow
def test_cached_image_gives_identical_logits(vlm):
    qs = {
        "a": {"type": "bool", "instructions": "Is there a red square?"},
        "b": {"type": "choice", "instructions": "What colour is the shape?",
              "criteria": {"red": "", "green": "", "blue": ""}},
    }  # fmt: skip
    img = scene()
    vlm.clear_cache()
    direct = vlm.logits(img, qs)
    handle = vlm.encode(img)
    assert vlm.encode(handle) is handle and vlm.encode(img) is handle  # cached by content
    again = vlm.logits(handle, qs)
    for k in qs:
        assert np.array_equal(direct[k], again[k])


@pytest.mark.slow
def test_answers_are_sensible_and_output_shape_matches_model(vlm):
    out = vlm.predict(scene(), {
        "colour": {"type": "choice", "instructions": "What colour is the shape?",
                   "criteria": {"red": "", "green": "", "blue": ""}},
        "square": {"type": "bool", "instructions": "Is there a red shape on a white background?"},
    })  # fmt: skip
    assert out["colour"]["answer"] == "red" and set(out["colour"]) == {
        "answer",
        "probs",
        "confidence",
    }
    assert sum(out["colour"]["probs"].values()) == pytest.approx(1.0)
    assert (
        set(out["square"]) == {"answer", "p_true", "confidence"} and out["square"]["p_true"] > 0.5
    )
