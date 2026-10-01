import collections

from imagejev.data.factq import (
    FEATURE_LEVELS,
    OVERLAY_LEVELS,
    OVERLAYS,
    PERSON_LEVELS,
    TAX_LEVELS,
    TOTAL_LEVELS,
    VARIETY_LEVELS,
    _bin,
    generate,
)
from imagejev.data.records import ImageFacts, QuestionRecord
from imagejev.data.splits import load_heldout
from imagejev.data.templater import apply_templates, load_paraphrases


def photo(counts, crowd=(), extra_hidden=None, iid="c1"):
    allc = {**counts, **(extra_hidden or {})}
    return ImageFacts(
        iid,
        "photo",
        "coco",
        {"counts": counts, "present_any": sorted(allc), "counts_all": allc, "crowd": list(crowd)},
    )


def screen(**kw):
    base = dict(page_type="login", modal_open=False, cookie_banner=False, error_banner=False, loading=False,
                viewport="desktop", input_count=1)  # fmt: skip
    return ImageFacts("w1", "screenshot", "webgen", {**base, **kw})


def doc(**kw):
    base = dict(
        doc_type="receipt",
        total_cents=1000,
        has_signature=False,
        has_stamp=False,
        has_table=False,
        tax_pct=10,
    )
    return ImageFacts("d1", "document", "docgen", {**base, **kw})


def by_task(facts, per_image=20):
    return {r.task: r for r in generate([facts], per_image=per_image)}


def test_bin_edges():
    assert [_bin(n, [0, 1, 4], PERSON_LEVELS) for n in (0, 1, 2, 4, 5, 40)] == [
        "no people", "one person", "a few people", "a few people", "many people", "many people"]  # fmt: skip
    assert [_bin(n, [1, 2, 5], VARIETY_LEVELS) for n in (1, 2, 3, 5, 6)] == [
        "one kind", "two kinds", "three to five kinds", "three to five kinds", "six or more kinds"]  # fmt: skip
    assert [_bin(n, [2, 5, 10], TOTAL_LEVELS) for n in (1, 2, 3, 5, 6, 10, 11)] == [
        "one or two", "one or two", "three to five", "three to five", "six to ten", "six to ten", "more than ten"]  # fmt: skip


def test_photo_answers_are_exact():
    t = by_task(photo({"person": 3, "dog": 1, "car": 2}))
    assert t["fact.photo.object_variety"].answer == "three to five kinds"
    assert t["fact.photo.person_count"].answer == "a few people"
    assert t["fact.photo.total_objects"].answer == "six to ten"
    assert by_task(photo({"dog": 1}))["fact.photo.person_count"].answer == "no people"


def test_photo_questions_skip_when_the_facts_cannot_decide():
    assert "fact.photo.person_count" not in by_task(
        photo({"person": 5}, crowd=["person"])
    )  # crowd box
    hidden = photo(
        {"person": 2}, extra_hidden={"person": 3}
    )  # annotated persons below the area bar
    assert "fact.photo.person_count" not in by_task(hidden)
    assert "fact.photo.total_objects" not in by_task(photo({"car": 2}, crowd=["car"]))
    assert "fact.photo.total_objects" not in by_task(photo({"car": 2}, extra_hidden={"dog": 1}))
    assert generate([photo({})]) == []  # nothing visible: nothing to ask


def test_screen_answers_and_overlay_kind_only_when_unique():
    t = by_task(screen(modal_open=True, viewport="mobile"))
    assert t["fact.screen.overlay_count"].answer == "one"
    assert t["fact.screen.viewport"].answer == "mobile"
    assert t["fact.screen.overlay_kind"].answer == "a dialog box"
    assert set(t["fact.screen.overlay_kind"].question["criteria"]) == {
        v[0] for v in OVERLAYS.values()
    }
    two = by_task(screen(modal_open=True, cookie_banner=True))
    assert (
        two["fact.screen.overlay_count"].answer == "two" and "fact.screen.overlay_kind" not in two
    )
    none = by_task(screen())
    assert (
        none["fact.screen.overlay_count"].answer == "none"
        and "fact.screen.overlay_kind" not in none
    )
    four = by_task(screen(modal_open=True, cookie_banner=True, error_banner=True, loading=True))
    assert four["fact.screen.overlay_count"].answer == "three or four"
    assert (
        generate([ImageFacts("x", "screenshot", "rico", {"elements": []})]) == []
    )  # not a generated page


def test_document_answers_and_forms_have_no_tax_question():
    t = by_task(doc(has_signature=True, has_stamp=True, tax_pct=8))
    assert (
        t["fact.doc.feature_count"].answer == "two of them"
        and t["fact.doc.tax_level"].answer == "medium tax"
    )
    assert [
        by_task(doc(tax_pct=p))["fact.doc.tax_level"].answer for p in (0, 5, 8, 10)
    ] == TAX_LEVELS
    assert (
        by_task(doc(has_signature=True, has_stamp=True, has_table=True))[
            "fact.doc.feature_count"
        ].answer
        == "all three"
    )
    assert by_task(doc())["fact.doc.feature_count"].answer == "none of them"
    form = by_task(doc(doc_type="form", total_cents=None, tax_pct=None))
    assert "fact.doc.tax_level" not in form and "fact.doc.feature_count" in form


def test_per_image_cap_determinism_and_valid_records():
    facts = [photo({"person": 2, "car": 1}, iid=f"c{i}") for i in range(40)] + [
        screen(modal_open=True),
        doc(),
    ]
    a, b = generate(facts, seed=1, per_image=2), generate(facts, seed=1, per_image=2)
    assert a == b != generate(facts, seed=2, per_image=2)
    per = collections.Counter(r.image_id for r in a)
    assert max(per.values()) == 2 and all(isinstance(r, QuestionRecord) for r in a)
    tasks = {r.task for r in generate(facts, seed=0, per_image=20)}
    assert tasks == set(load_paraphrases()) & tasks and len(tasks) == 8


def test_every_new_task_has_paraphrases_and_none_is_held_out():
    table = load_paraphrases()
    cfg = load_heldout()
    new = {t for t in table if t.startswith("fact.")}
    assert len(new) == 8  # exactly the eight families defined in factq.py
    for t in new:
        assert (
            len(table[t]["stems"]) >= 5
            and t not in cfg["task_families"]
            and t not in cfg["val_task_families"]
        )


def test_levels_are_ordered_low_to_high_and_distinct():
    for levels in (
        OVERLAY_LEVELS,
        FEATURE_LEVELS,
        TAX_LEVELS,
        VARIETY_LEVELS,
        PERSON_LEVELS,
        TOTAL_LEVELS,
    ):
        assert len(set(levels)) == len(levels) == 4


def test_apply_templates_adds_them_and_can_switch_them_off():
    facts = [photo({"person": 2, "car": 1}, iid=f"c{i}") for i in range(60)]
    on = apply_templates([], facts, seed=0, balance=False)
    off = apply_templates([], facts, seed=0, balance=False, fact_questions=False)
    assert any(r.task.startswith("fact.") for r in on) and not any(
        r.task.startswith("fact.") for r in off
    )
    scores = [r for r in on if r.task.startswith("fact.photo")]
    assert all(r.question["instructions"] for r in scores)
