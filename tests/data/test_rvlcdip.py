import random

import pytest

from imagejev.data.records import ImageFacts, QuestionRecord
from imagejev.data.rvlcdip import (
    CONFUSABLE_GROUPS,
    NAMES,
    class_name,
    iter_rvl_cdip,
    pick_options,
)


def test_class_name_accepts_index_or_name_and_rejects_bad():
    assert (
        class_name(0) == "letter" and class_name(11) == "invoice" and class_name("memo") == "memo"
    )
    for bad in (16, -1, "poster"):
        with pytest.raises(ValueError):
            class_name(bad)


def test_pick_options_contains_answer_and_prefers_confusable():
    rng = random.Random(0)
    opts = pick_options("invoice", 4, rng)
    assert "invoice" in opts and len(set(opts)) == 4
    # invoice's confusable group has exactly 3 mates, so a 4-way question uses all of them
    assert set(opts) == {"invoice", "form", "questionnaire", "budget"}
    assert sorted(pick_options("memo", 16, rng)) == sorted(NAMES)
    with pytest.raises(ValueError):
        pick_options("memo", 1, rng)


def test_groups_are_valid_class_names():
    assert all(c in NAMES for g in CONFUSABLE_GROUPS for c in g)


def test_iter_emits_facts_and_valid_questions():
    rows = [{"id": i, "label": i % 16} for i in range(64)]
    out = list(iter_rvl_cdip(rows, split="val", seed=2))
    facts = [o for o in out if isinstance(o, ImageFacts)]
    qs = [o for o in out if isinstance(o, QuestionRecord)]
    assert len(facts) == len(qs) == 64
    for q in qs:
        assert q.answer in q.question["criteria"]
        assert q.meta["n_options"] in (4, 8, 16)
        assert q.question["criteria"][q.answer]  # options carry descriptions
    assert {q.meta["n_options"] for q in qs} == {4, 8, 16}
    assert out == list(iter_rvl_cdip(rows, split="val", seed=2))
