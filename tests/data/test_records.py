import pytest

from imagejev.data.records import ImageFacts, QuestionRecord, read_jsonl, write_jsonl

Q = {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}}


def test_jsonl_round_trip(tmp_path):
    recs = [
        QuestionRecord("i1", "photo", "s", "t", Q, "a", soft={"a": 0.7, "b": 0.3}),
        QuestionRecord("i2", "document", "s", "t2", {"type": "bool", "instructions": "y"}, True),
    ]
    p = tmp_path / "q.jsonl"
    assert write_jsonl(recs, p) == 2
    assert list(read_jsonl(p, QuestionRecord)) == recs
    facts = [ImageFacts("i1", "photo", "s", {"k": 1}, style="t1")]
    write_jsonl(facts, tmp_path / "f.jsonl")
    assert list(read_jsonl(tmp_path / "f.jsonl", ImageFacts)) == facts


def test_soft_label_validation():
    with pytest.raises(ValueError):
        QuestionRecord("i", "photo", "s", "t", Q, "a", soft={"a": 0.5, "b": 0.2})
    with pytest.raises(ValueError):
        QuestionRecord("i", "photo", "s", "t", Q, "a", soft={"a": 1.0})


def test_unknown_domain():
    with pytest.raises(ValueError):
        ImageFacts("i", "video", "s", {})
