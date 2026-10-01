import warnings

import pytest

from imagejev.errors import QuestionSchemaError
from imagejev.schema import (
    MAX_OPTIONS,
    parse_question,
    parse_questions,
    state_to_text,
    truncate_state,
)


def test_choice_from_dict_keeps_order_and_descriptions():
    q = parse_question(
        "t",
        {"type": "choice", "instructions": "Which?", "criteria": {"a": "alpha", "b": "beta"}},
    )
    assert q.labels == ("a", "b")
    assert q.options[0].text() == "a: alpha"


def test_choice_from_list():
    q = parse_question("t", {"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    assert q.options[1].text() == "b"


def test_score_levels_ordered():
    q = parse_question("t", {"type": "score", "instructions": "x", "levels": ["lo", "mid", "hi"]})
    assert q.labels == ("lo", "mid", "hi")


def test_bool_needs_instructions():
    assert parse_question("t", {"type": "bool", "instructions": "Is it?"}).options == ()
    with pytest.raises(QuestionSchemaError):
        parse_question("t", {"type": "bool"})


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "nope", "instructions": "x"},
        {"instructions": "x"},
        {"type": "choice", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": {}},
        {"type": "choice", "instructions": "x", "criteria": {"only": "one"}},
        {"type": "score", "instructions": "x", "levels": ["one"]},
        {"type": "score", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": {"a": "", "": "empty"}},
        {"type": "score", "instructions": "x", "levels": ["a", "a"]},
        {"type": "choice", "instructions": "x", "criteria": 5},
        "not a dict",
    ],
)
def test_malformed_questions_raise(spec):
    with pytest.raises(QuestionSchemaError):
        parse_question("t", spec)


def test_option_limit():
    ok = {str(i): "" for i in range(MAX_OPTIONS)}
    parse_question("t", {"type": "choice", "instructions": "x", "criteria": ok})
    too_many = {str(i): "" for i in range(MAX_OPTIONS + 1)}
    with pytest.raises(QuestionSchemaError, match="maximum"):
        parse_question("t", {"type": "choice", "instructions": "x", "criteria": too_many})
    with pytest.raises(QuestionSchemaError, match="maximum"):
        parse_question("t", {"type": "score", "instructions": "x", "levels": list(too_many)})


def test_parse_questions_requires_nonempty_dict():
    with pytest.raises(QuestionSchemaError):
        parse_questions({})
    qs = parse_questions({"a": {"type": "bool", "instructions": "x"}})
    assert [q.id for q in qs] == ["a"]


def test_state_to_text():
    assert state_to_text(None) == ""
    assert state_to_text("hi") == "hi"
    assert state_to_text({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    with pytest.raises(QuestionSchemaError):
        state_to_text({"x": object()})


def test_truncate_state_warns_only_when_cut():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert truncate_state("short text", max_tokens=10) == "short text"
    long = " ".join(["word"] * 300)
    with pytest.warns(UserWarning, match="truncated"):
        cut = truncate_state(long, max_tokens=256)
    assert cut.count("word") == 256


class FakeTok:
    def encode(self, text, **kw):
        return list(range(len(text.split())))

    def decode(self, ids, **kw):
        return " ".join(["w"] * len(ids))


def test_truncate_state_with_tokenizer():
    with pytest.warns(UserWarning):
        cut = truncate_state("a b c d e", max_tokens=3, tokenizer=FakeTok())
    assert cut == "w w w"
