import csv
import json

import pytest
from PIL import Image

from imagejev.data.teacher import build_jobs, label_images
from imagejev.data.teacher_review import (
    FIELDS,
    filter_records,
    keep_criteria,
    make_review_sheet,
    sample_records,
    score_review,
)


class FakeTeacher:
    name = "fake"

    def probs(self, image, question):
        return [0.7] if question.type == "bool" else [1.0] * len(question.options)


@pytest.fixture
def labelled(tmp_path):
    paths = {}
    imgs = []
    for i in range(80):
        dom = "photo" if i % 2 == 0 else "screenshot"
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (900, 600), (i * 3 % 255, 20, 90)).save(p)
        paths[f"{dom}{i}"] = str(p)
        imgs.append((f"{dom}{i}", str(p), dom))
    out = tmp_path / "teacher.jsonl"
    label_images(build_jobs(imgs, per_image=3), FakeTeacher(), out)
    return out, paths


def test_sample_is_spread_over_criteria_and_deterministic(labelled):
    from imagejev.data.teacher import read_teacher_records

    recs = read_teacher_records(labelled[0])
    s = sample_records(recs, 50, seed=1)
    assert len(s) == 50 and len({id(r) for r in s}) == 50
    per = {}
    for r in s:
        per[r.meta["criterion"]] = per.get(r.meta["criterion"], 0) + 1
    assert max(per.values()) - min(per.values()) <= 1 and len(per) == 25  # every criterion appears
    assert [id(r) for r in sample_records(recs, 50, seed=1)] != []
    assert [r.image_id for r in sample_records(recs, 50, seed=1)] == [r.image_id for r in s]
    assert len(sample_records(recs, 10**6)) == len(recs)  # asking for more returns everything


def test_review_sheet_has_images_csv_and_html(labelled, tmp_path):
    out = tmp_path / "review"
    info = make_review_sheet(labelled[0], labelled[1], out, n=40, seed=0)
    assert info["samples"] == 40 and info["criteria"] >= 20
    rows = list(csv.DictReader((out / "review.csv").open()))
    assert len(rows) == 40 and list(rows[0]) == FIELDS and all(r["human_ok"] == "" for r in rows)
    for r in rows:
        assert max(Image.open(out / r["image"]).size) <= 480  # thumbnails
    page = (out / "review.html").read_text()
    assert page.count('class="c"') == 40 and "human_ok" in page and "&" not in page.split("<h2>")[0]
    assert {r["teacher_answer"] for r in rows if "yes / no" == r["options"]} <= {"yes", "no"}


def fill(csv_path, verdict_for):
    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        r["human_ok"] = verdict_for(r)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_scoring_and_the_80_percent_bar(labelled, tmp_path):
    out = tmp_path / "review"
    make_review_sheet(labelled[0], labelled[1], out, n=250, seed=0)
    bad = "has_tab_bar"
    fill(
        out / "review.csv",
        lambda r: "n" if r["criterion"] == bad else ("?" if r["id"] == "0" else "y"),
    )
    scores = score_review(out / "review.csv")
    assert scores[bad]["accuracy"] == 0.0 and scores["theme"]["accuracy"] == 1.0
    n_rows = len(list(csv.DictReader((out / "review.csv").open())))
    assert n_rows == 240  # 80 images x 3 criteria: asking for 250 returns everything
    assert sum(s["n"] for s in scores.values()) == n_rows - 1  # the single '?' is ignored
    from imagejev.data.teacher import CRITERIA

    decision = keep_criteria(scores, [c.id for c in CRITERIA])
    assert (
        bad in decision["drop"] and "0.00" in decision["drop"][bad] and "theme" in decision["keep"]
    )
    n_kept = filter_records(labelled[0], decision, tmp_path / "kept.jsonl")
    kept = [json.loads(line) for line in (tmp_path / "kept.jsonl").read_text().splitlines()]
    assert n_kept["kept"] == len(kept) and n_kept["dropped"] > 0
    assert all(r["meta"]["criterion"] != bad for r in kept)


def test_unjudged_or_barely_judged_criteria_are_dropped_not_trusted():
    scores = {"a": {"n": 20, "correct": 17, "accuracy": 0.85}, "b": {"n": 3, "correct": 3, "accuracy": 1.0},
              "c": {"n": 10, "correct": 7, "accuracy": 0.7}}  # fmt: skip
    d = keep_criteria(scores, ["a", "b", "c", "never_judged"], min_accuracy=0.8, min_judged=5)
    assert d["keep"] == ["a"]
    assert (
        "only 3 judged" in d["drop"]["b"]
        and "0.70" in d["drop"]["c"]
        and "only 0 judged" in d["drop"]["never_judged"]
    )
    assert keep_criteria(scores, ["a"], min_accuracy=0.9)["keep"] == []


def test_score_review_ignores_blank_and_unknown_verdicts(tmp_path):
    p = tmp_path / "r.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, v in enumerate(["y", "Y", "n", "", "?", "maybe", " y "]):
            w.writerow(
                {
                    "id": i,
                    "image": "",
                    "criterion": "c",
                    "question": "",
                    "options": "",
                    "teacher_answer": "",
                    "confidence": "",
                    "human_ok": v,
                }
            )
    assert score_review(p) == {"c": {"n": 4, "correct": 3, "accuracy": 0.75}}
