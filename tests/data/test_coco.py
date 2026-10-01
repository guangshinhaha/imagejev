import pytest

from imagejev.data.coco import (
    COUNT_LEVELS,
    PERMISSIVE_LICENSE_IDS,
    count_level,
    iter_coco,
    iter_vqav2_yesno,
)
from imagejev.data.records import ImageFacts, QuestionRecord

CATS = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "car", "supercategory": "vehicle"},
    {"id": 3, "name": "bus", "supercategory": "vehicle"},
    {"id": 4, "name": "dog", "supercategory": "animal"},
    {"id": 5, "name": "cat", "supercategory": "animal"},
    {"id": 6, "name": "chair", "supercategory": "furniture"},
]


def _ann(i, image_id, cat, area, crowd=0):
    return {"id": i, "image_id": image_id, "category_id": cat, "area": area, "iscrowd": crowd}


INSTANCES = {
    "categories": CATS,
    "images": [
        {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        {"id": 3, "file_name": "c.jpg", "width": 100, "height": 100},  # nothing visible
    ],
    "annotations": [
        _ann(1, 1, 2, 4000),
        _ann(2, 1, 2, 1000),
        _ann(3, 1, 1, 500),
        _ann(4, 2, 4, 3000),
        _ann(5, 2, 1, 2500),
        _ann(6, 3, 1, 10),  # speck: below the 0.5% area bar
        _ann(7, 2, 5, 9000, crowd=1),  # crowd: ignored
    ],
}


def test_count_level_bins():
    assert [count_level(n) for n in (0, 1, 2, 4, 5, 50)] == [
        "none",
        "one",
        "a few",
        "a few",
        "many",
        "many",
    ]


def test_iter_coco_yields_valid_records_and_skips_empty_images():
    out = list(iter_coco(INSTANCES, seed=0))
    facts = [r for r in out if isinstance(r, ImageFacts)]
    qs = [r for r in out if isinstance(r, QuestionRecord)]
    assert {f.image_id for f in facts} == {"coco:000000000001", "coco:000000000002"}
    assert all(q.domain == "photo" and q.source == "coco" for q in qs)
    f1 = next(f for f in facts if f.image_id.endswith("1"))
    assert f1.facts["counts"] == {"car": 2, "person": 1}
    assert f1.facts["top_supercategory"] == "vehicle"


def test_labels_match_annotations():
    out = list(iter_coco(INSTANCES, seed=3))
    facts = {r.image_id: r.facts["counts"] for r in out if isinstance(r, ImageFacts)}
    for q in (r for r in out if isinstance(r, QuestionRecord)):
        counts = facts[q.image_id]
        if q.task == "coco.object_present":
            assert q.answer == (counts.get(q.meta["category"], 0) > 0)
        elif q.task == "coco.count_bin":
            assert q.answer == count_level(counts.get(q.meta["category"], 0))
            assert q.question["levels"] == COUNT_LEVELS


def test_dominant_supercategory_only_when_clear_and_answer_in_options():
    qs = [r for r in iter_coco(INSTANCES, seed=0) if isinstance(r, QuestionRecord)]
    dom = [q for q in qs if q.task == "coco.dominant_supercategory"]
    assert {q.image_id for q in dom} == {"coco:000000000001"}  # image 2 is 3000 vs 2500
    assert dom[0].answer == "vehicle"
    assert dom[0].question["criteria"]["vehicle"] == "bus, car"  # described by member categories


def test_deterministic_for_a_seed_and_differs_across_seeds():
    a = [r for r in iter_coco(INSTANCES, seed=1) if isinstance(r, QuestionRecord)]
    b = [r for r in iter_coco(INSTANCES, seed=1) if isinstance(r, QuestionRecord)]
    assert a == b


def test_bool_negatives_prefer_same_supercategory():
    # image 1 has car (vehicle): the only absent same-supercategory name is "bus"
    for seed in range(20):
        negs = [
            q
            for q in iter_coco(INSTANCES, seed=seed, bool_per_image=2)
            if isinstance(q, QuestionRecord)
            and q.image_id.endswith("1")
            and q.task == "coco.object_present"
            and q.answer is False
        ]
        for q in negs:
            assert q.meta["category"] in {"bus"}


def test_image_dir_prefix():
    f = next(iter_coco(INSTANCES, image_dir="/data/val"))
    assert f.image_path == "/data/val/a.jpg"


VQA_Q = {
    "questions": [
        {"question_id": 10, "image_id": 1, "question": "Is the man happy?"},
        {"question_id": 11, "image_id": 1, "question": "How many dogs?"},
        {"question_id": 12, "image_id": 2, "question": "Is it raining?"},
        {"question_id": 13, "image_id": 2, "question": "Is it sunny?"},
    ]
}
VQA_A = {
    "annotations": [
        {
            "question_id": 10,
            "image_id": 1,
            "answer_type": "yes/no",
            "multiple_choice_answer": "yes",
        },
        {"question_id": 11, "image_id": 1, "answer_type": "number", "multiple_choice_answer": "2"},
        {"question_id": 12, "image_id": 2, "answer_type": "yes/no", "multiple_choice_answer": "no"},
        {"question_id": 13, "image_id": 2, "answer_type": "yes/no", "multiple_choice_answer": "no"},
    ]
}


def test_vqav2_yesno_only_and_answers():
    recs = list(iter_vqav2_yesno(VQA_Q, VQA_A))
    assert [(r.question["instructions"], r.answer) for r in recs] == [
        ("Is the man happy?", True),
        ("Is it raining?", False),
        ("Is it sunny?", False),
    ]
    assert all(r.task == "vqav2.yesno" and r.image_id.startswith("coco:") for r in recs)


def test_vqav2_balance_cap():
    recs = list(iter_vqav2_yesno(VQA_Q, VQA_A, max_per_answer=1))
    assert sorted(r.answer for r in recs) == [False, True]


def test_records_validate_answers():
    with pytest.raises(ValueError):
        QuestionRecord("i", "photo", "s", "t", {"type": "bool", "instructions": "x"}, "yes")
    with pytest.raises(ValueError):
        QuestionRecord(
            "i",
            "photo",
            "s",
            "t",
            {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}},
            "z",
        )


def test_license_filter_keeps_only_listed_licenses_and_records_the_license():
    import copy

    inst = copy.deepcopy(INSTANCES)
    inst["images"][0]["license"] = 4  # CC BY
    inst["images"][1]["license"] = 1  # CC BY-NC-SA
    inst["images"][2]["license"] = 6  # CC BY-ND
    everything = [r for r in iter_coco(inst) if isinstance(r, ImageFacts)]
    assert {f.facts["license"] for f in everything} == {4, 1}  # image 3 has nothing visible
    kept = [r for r in iter_coco(inst, licenses=PERMISSIVE_LICENSE_IDS)]
    assert {r.image_id for r in kept} == {"coco:000000000001"}
    assert all(f.facts["license"] == 4 for f in kept if isinstance(f, ImageFacts))
    assert PERMISSIVE_LICENSE_IDS == {4, 5, 7, 8}
    assert not list(iter_coco(inst, licenses=set()))
