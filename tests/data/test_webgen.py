import collections
import json

import pytest

from imagejev.data.records import ImageFacts, QuestionRecord, read_jsonl
from imagejev.data.webgen import (
    INPUT_LEVELS,
    TEMPLATES,
    PageSpec,
    input_level,
    labels_from_measure,
    make_themes,
    questions_for,
    render_html,
    sample_spec,
)

THEMES = make_themes()


def test_themes_are_deterministic_and_distinct():
    assert make_themes() == THEMES
    assert len({t.id for t in THEMES}) == 12
    assert len({t.accent for t in THEMES}) > 6
    assert any(t.dark for t in THEMES) and any(not t.dark for t in THEMES)


def test_sample_spec_deterministic_and_covers_templates():
    assert sample_spec(3, 0, THEMES) == sample_spec(3, 0, THEMES)
    assert sample_spec(3, 0, THEMES) != sample_spec(3, 1, THEMES)
    specs = [sample_spec(i, 0, THEMES) for i in range(400)]
    assert {s.template for s in specs} == set(TEMPLATES)
    assert all(s.captcha is False for s in specs if s.template not in ("login", "checkout"))
    assert all(not s.cart_empty for s in specs if s.template != "shop")
    rates = {
        k: sum(getattr(s, k) for s in specs) / len(specs) for k in ("modal", "error", "cookie")
    }
    assert all(0.25 < r < 0.5 for r in rates.values())


def spec(**kw):
    base = dict(
        index=1, template="login", theme="t00", viewport="desktop", modal=False, error=False,
        spinner=False, cookie=False, captcha=False, cta_below_fold=False, cart_empty=False,
        brand="Acme",
    )  # fmt: skip
    return PageSpec(**{**base, **kw})


@pytest.mark.parametrize(
    ("flag", "marker"),
    [("modal", 'data-gt="modal"'), ("error", 'data-gt="error"'), ("spinner", 'data-gt="spinner"'),
     ("cookie", 'data-gt="cookie"'), ("captcha", 'data-gt="captcha"')],
)  # fmt: skip
def test_html_has_marker_exactly_when_flag_set(flag, marker):
    theme = THEMES[0]
    assert marker in render_html(spec(**{flag: True}), theme)
    assert marker not in render_html(spec(), theme)


def test_every_template_renders_a_cta_and_cart_state():
    for t in TEMPLATES:
        html = render_html(spec(template=t), THEMES[1])
        assert 'data-gt="cta"' in html
    assert 'data-gt="cart-empty"' in render_html(spec(template="shop", cart_empty=True), THEMES[0])
    assert 'data-gt="cart-empty"' not in render_html(spec(template="shop"), THEMES[0])


def measure(**present):
    m = {k: {"inView": True, "clear": True} for k in present if present[k]}
    m["cta"] = {"inView": True, "clear": True}
    m["__inputs"] = 2
    return m


def test_labels_follow_measurement_and_reject_contradictions():
    ok = labels_from_measure(spec(modal=True), measure(modal=True))
    assert ok and ok["modal_open"] and ok["cta_clickable"] and ok["input_count"] == 2
    # spec says modal, DOM shows none
    assert labels_from_measure(spec(modal=True), measure()) is None
    # DOM shows an error banner the spec didn't ask for
    assert labels_from_measure(spec(), measure(error=True)) is None
    # CTA missing entirely
    assert labels_from_measure(spec(), {"__inputs": 0}) is None


def test_cta_clickable_requires_unoccluded_and_below_fold_must_be_offscreen():
    m = measure()
    m["cta"] = {"inView": True, "clear": False}  # covered by a modal backdrop
    assert labels_from_measure(spec(), m)["cta_clickable"] is False
    m["cta"] = {"inView": False, "clear": False}
    assert labels_from_measure(spec(cta_below_fold=True), m)["cta_clickable"] is False
    m["cta"] = {"inView": True, "clear": True}
    assert labels_from_measure(spec(cta_below_fold=True), m) is None  # spacer failed


def test_input_level_bins():
    assert [input_level(n) for n in (0, 1, 2, 3, 4, 9)] == [
        "none", "one", "two or three", "two or three", "four or more", "four or more",
    ]  # fmt: skip


def test_questions_are_valid_and_match_labels():
    s = spec(template="shop", modal=True, cart_empty=True)
    labels = labels_from_measure(s, measure(modal=True))
    qs = questions_for(s, labels, seed=0)
    by_task = {q.task: q for q in qs}
    assert by_task["web.page_type"].answer == "shop"
    assert by_task["web.cart_empty"].answer is True
    assert by_task["web.input_count"].question["levels"] == INPUT_LEVELS
    for q in qs:
        assert q.style == "shop:t00" and q.domain == "screenshot"
        if q.task == "web.modal_open":
            assert q.answer is True
    assert qs == questions_for(s, labels, seed=0)
    assert "web.cart_empty" not in {
        q.task for q in questions_for(spec(), labels_from_measure(spec(), measure()), 0)
    }


def test_bool_questions_are_roughly_balanced_across_many_pages():
    answers = collections.Counter()
    for i in range(300):
        s = sample_spec(i, 0, THEMES)
        m = measure(
            modal=s.modal, error=s.error, spinner=s.spinner, cookie=s.cookie, captcha=s.captcha
        )
        if s.cta_below_fold:
            m["cta"] = {"inView": False, "clear": False}
        labels = labels_from_measure(s, m)
        for q in questions_for(s, labels, 0):
            if q.task.startswith("web.") and q.question["type"] == "bool" and q.task not in (
                "web.cta_clickable", "web.cart_empty",
            ):  # fmt: skip
                answers[q.answer] += 1
    share = answers[True] / sum(answers.values())
    assert 0.35 < share < 0.65, share


@pytest.mark.slow
def test_generate_with_real_browser(tmp_path):
    pytest.importorskip("playwright")
    from PIL import Image

    from imagejev.data.webgen import generate

    try:
        counts = generate(16, tmp_path, seed=0)
    except Exception as e:  # no Chrome available
        pytest.skip(f"browser unavailable: {e}")
    assert counts["dropped"] == 0 and counts["pages"] == 16
    facts = list(read_jsonl(tmp_path / "facts.jsonl", ImageFacts))
    qs = list(read_jsonl(tmp_path / "questions.jsonl", QuestionRecord))
    assert len(facts) == 16 and len(qs) == counts["questions"]
    for f in facts:
        img = Image.open(tmp_path / f.image_path)
        assert img.size in {(1280, 720), (390, 844)}
        assert f.facts["viewport"] == ("desktop" if img.size[0] == 1280 else "mobile")
    json.dumps([f.facts for f in facts])
