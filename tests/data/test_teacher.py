import collections
import json

import pytest

from imagejev.data.records import ImageFacts, QuestionRecord, write_jsonl
from imagejev.data.splits import build_splits, check_splits
from imagejev.data.teacher import (
    BY_ID,
    CRITERIA,
    build_jobs,
    label_images,
    read_teacher_records,
    train_images,
)
from imagejev.data.templater import apply_templates, rewrite
from imagejev.schema import parse_question


class FakeTeacher:
    name = "fake-teacher"

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)

    def probs(self, image, question):
        self.calls += 1
        if image in self.fail_on:
            raise OSError("cannot read")
        if question.type == "bool":
            return [0.8]
        weights = [i + 1.0 for i in range(len(question.options))]
        return weights  # deliberately unnormalised: the labeller must normalise


def images(n, domain="photo"):
    return [(f"{domain}{i}", f"{domain}{i}.jpg", domain) for i in range(n)]


# ---- the criteria library --------------------------------------------------------------------
def test_every_criterion_is_a_valid_question_and_ids_are_unique():
    assert len(CRITERIA) >= 20 and len({c.task for c in CRITERIA}) == len(CRITERIA)
    for c in CRITERIA:
        q = parse_question(c.task, c.question)
        assert c.domain in ("photo", "screenshot") and c.task == f"teacher.{c.domain}.{c.id}"
        if q.type != "bool":
            assert 2 <= len(q.options) <= 12
    assert set(BY_ID) == {c.task for c in CRITERIA}
    kinds = {(c.domain, c.question["type"]) for c in CRITERIA}
    assert {"choice", "score", "bool"} <= {t for _, t in kinds}


# ---- jobs --------------------------------------------------------------------------------
def test_jobs_respect_domain_per_image_and_are_deterministic():
    imgs = images(40, "photo") + images(40, "screenshot")
    jobs = build_jobs(imgs, per_image=3, seed=1)
    assert len(jobs) == 240 and jobs == build_jobs(imgs, per_image=3, seed=1) != build_jobs(
        imgs, per_image=3, seed=2
    )
    assert all(j.criterion.domain == j.domain for j in jobs)
    per_image = collections.Counter(j.image_id for j in jobs)
    assert set(per_image.values()) == {3}
    assert all(
        len({j.criterion.id for j in jobs if j.image_id == i}) == 3 for i in per_image
    )  # distinct


def test_criteria_are_spread_roughly_evenly():
    jobs = build_jobs(images(600, "photo"), per_image=3, seed=0)
    counts = collections.Counter(j.criterion.id for j in jobs)
    n_photo = sum(1 for c in CRITERIA if c.domain == "photo")
    assert len(counts) == n_photo
    expected = 600 * 3 / n_photo
    assert all(0.6 * expected < v < 1.4 * expected for v in counts.values()), counts


def test_unknown_domain_gets_no_jobs_and_small_pools_are_capped():
    assert build_jobs([("x", "x.jpg", "document")], per_image=3) == []
    one = [c for c in CRITERIA if c.domain == "photo"][:2]
    assert len(build_jobs(images(1), per_image=5, criteria=one)) == 2


# ---- labelling ---------------------------------------------------------------------------
def test_records_are_valid_normalised_and_tagged_with_the_teacher(tmp_path):
    out = tmp_path / "t.jsonl"
    jobs = build_jobs(images(30, "photo") + images(30, "screenshot"), per_image=3)
    counts = label_images(jobs, FakeTeacher(), out)
    assert counts["labelled"] == len(jobs) and counts["failed"] == 0 and not counts["stopped_early"]
    recs = read_teacher_records(out)  # re-validated by QuestionRecord's constructor
    assert len(recs) == len(jobs)
    for r in recs:
        q = parse_question(r.task, r.question)
        floor = (
            0.5 if q.type == "bool" else 1 / len(q.options)
        )  # confidence is at least chance level
        assert r.source == "teacher" and r.meta["teacher"] == "fake-teacher"
        assert floor - 1e-6 <= r.meta["confidence"] <= 1.0
        if q.type == "bool":
            assert r.soft == {"true": 0.8} and r.answer is True
        else:
            assert sum(r.soft.values()) == pytest.approx(1.0) and set(r.soft) == set(q.labels)
            assert r.answer == q.labels[-1]  # the largest weight is the last option
            assert r.meta["confidence"] == pytest.approx(max(r.soft.values()), abs=1e-4)


def test_labelling_is_resumable_and_never_repeats_work(tmp_path):
    out = tmp_path / "t.jsonl"
    jobs = build_jobs(images(20), per_image=2)
    first = FakeTeacher()
    # a budget that is already spent labels nothing, then a normal run does everything once
    assert label_images(jobs, first, out, time_budget_s=1e-9)["stopped_early"] == 1
    n = label_images(jobs[:15], first, out)["labelled"]
    second = FakeTeacher()
    c = label_images(jobs, second, out)
    assert (
        c["skipped_existing"] == 15
        and c["labelled"] == len(jobs) - 15
        and second.calls == len(jobs) - 15
    )
    keys = [f"{r.image_id}|{r.task}" for r in read_teacher_records(out)]
    assert len(keys) == len(set(keys)) == len(jobs) and n == 15


def test_unreadable_images_are_counted_and_skipped(tmp_path):
    jobs = build_jobs(images(10), per_image=1)
    bad = {jobs[2].image, jobs[5].image}
    c = label_images(jobs, FakeTeacher(fail_on=bad), tmp_path / "t.jsonl")
    assert c["failed"] == 2 and c["labelled"] == 8


# ---- keeping soft labels out of evaluation -------------------------------------------------
def soft_choice(image_id="img0", answer="a"):
    q = {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": "", "c": "", "d": ""}}
    return QuestionRecord(image_id, "photo", "teacher", "teacher.photo.x", q, answer,
                          soft={"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1})  # fmt: skip


def test_bool_soft_label_validation():
    q = {"type": "bool", "instructions": "x"}
    ok = QuestionRecord("i", "photo", "teacher", "t", q, True, soft={"true": 0.7})
    assert ok.soft == {"true": 0.7}
    for bad in ({"true": 1.4}, {"yes": 0.5}, {"true": 0.5, "false": 0.5}):
        with pytest.raises(ValueError):
            QuestionRecord("i", "photo", "teacher", "t", q, True, soft=bad)


def test_the_templater_keeps_soft_labels_and_only_reorders_options():
    rec = soft_choice()
    orders = set()
    for seed in range(40):
        out = rewrite(rec, seed, p_none=1.0)  # p_none=1 would normally drop the true option
        assert (
            out.soft == rec.soft
            and out.answer == "a"
            and set(out.question["criteria"]) == set("abcd")
        )
        orders.add(tuple(out.question["criteria"]))
    assert len(orders) > 5  # still shuffled


def test_teacher_records_survive_only_in_the_train_split():
    recs, facts = [], []
    for i in range(400):
        recs.append(soft_choice(f"img{i}"))
        facts.append(ImageFacts(f"img{i}", "photo", "coco", {}))
    res = build_splits(recs, facts, seed=0)
    check_splits(res)
    kept = {s: len(r) for s, r in res.splits.items() if r}
    assert set(kept) == {"train"} and kept["train"] > 250
    assert res.dropped["teacher-labelled question outside train"] == 400 - kept["train"]


def test_check_splits_rejects_a_teacher_label_in_an_evaluation_split():
    from imagejev.data.splits import SPLITS, SplitResult

    empty = {s: [] for s in SPLITS}
    with pytest.raises(AssertionError, match="teacher-labelled"):
        check_splits(SplitResult({**empty, "val": [soft_choice()]}, {"img0": "val"}))


def test_pipeline_keeps_teacher_soft_labels_in_training_targets():
    torch = pytest.importorskip("torch")  # noqa: F841
    from imagejev.train.data import target_vector

    rec = soft_choice()
    q = parse_question("t", rec.question)
    assert target_vector(rec, q) == [0.5, 0.3, 0.1, 0.1]
    bq = parse_question("b", {"type": "bool", "instructions": "x"})
    soft_bool = QuestionRecord(
        "i",
        "photo",
        "teacher",
        "t",
        {"type": "bool", "instructions": "x"},
        True,
        soft={"true": 0.65},
    )
    assert target_vector(soft_bool, bq) == [0.65]
    assert apply_templates([rec], [], seed=0, balance=False)[0].soft == rec.soft


# ---- picking images -----------------------------------------------------------------------
def test_train_images_only_real_sources_in_the_train_split(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()

    def q(i, dom, src):
        return QuestionRecord(i, dom, src, "t.x", {"type": "bool", "instructions": "x"}, True)

    write_jsonl([q(f"c{i}", "photo", "coco") for i in range(30)]
                + [q(f"r{i}", "screenshot", "rico") for i in range(30)]
                + [q(f"w{i}", "screenshot", "webgen") for i in range(30)], splits / "train.jsonl")  # fmt: skip
    facts = [ImageFacts(i, d, s, {}, image_path=f"/img/{i}.jpg") for i, d, s in
             [(f"c{i}", "photo", "coco") for i in range(30)] + [(f"r{i}", "screenshot", "rico") for i in range(30)]
             + [(f"w{i}", "screenshot", "webgen") for i in range(30)]]  # fmt: skip
    write_jsonl(facts, tmp_path / "facts.jsonl")
    got = train_images(splits, tmp_path / "facts.jsonl", per_domain=10, seed=1)
    assert len(got) == 20 and {d for _, _, d in got} == {"photo", "screenshot"}
    assert not any(
        i.startswith("w") for i, _, _ in got
    )  # synthetic pages already have exact labels
    assert all(p == f"/img/{i}.jpg" for i, p, _ in got)
    assert got == train_images(splits, tmp_path / "facts.jsonl", per_domain=10, seed=1)
    assert json.dumps(got)
