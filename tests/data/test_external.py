import collections
import json

import pytest

from imagejev.data.external import (
    PLATFORMS,
    cord_facts,
    iter_cord,
    iter_oxford_pets,
    iter_screenspot,
    normalize_platform,
    pretty,
)
from imagejev.data.records import ImageFacts, QuestionRecord
from imagejev.data.splits import build_splits, check_splits

BREEDS = ["abyssinian", "bengal", "birman", "american_bulldog", "beagle", "boxer", "pug", "samoyed"]
SPECIES = {0: "cat", 1: "cat", 2: "cat", 3: "dog", 4: "dog", 5: "dog", 6: "dog", 7: "dog"}


def split(out):
    out = list(out)
    return (
        [o for o in out if isinstance(o, ImageFacts)],
        [o for o in out if isinstance(o, QuestionRecord)],
    )


# ---- Oxford-IIIT Pet ----------------------------------------------------------------------
def test_pretty_breed_names():
    assert (
        pretty("american_pit_bull_terrier") == "American Pit Bull Terrier"
        and pretty("pug") == "Pug"
    )


def test_pets_questions_are_exact_and_distractors_are_hard():
    rows = [{"image_id": f"img{i}", "label": i % 8} for i in range(80)]
    facts, qs = split(iter_oxford_pets(rows, BREEDS, SPECIES, seed=1))
    assert len(facts) == 80 and all(
        f.source == "ext:oxford-iiit-pet" and f.domain == "photo" for f in facts
    )
    by_img = collections.defaultdict(dict)
    for q in qs:
        by_img[q.image_id][q.task] = q
    for iid, tasks in by_img.items():
        label = int(iid.split("img")[1]) % 8
        assert tasks["ext.pets.species"].answer == SPECIES[label]
        breed = tasks["ext.pets.breed"]
        assert breed.answer == pretty(BREEDS[label]) and breed.answer in breed.question["criteria"]
        opts = [o for o in breed.question["criteria"] if o != breed.answer]
        same = [o for o in opts if SPECIES[[pretty(b) for b in BREEDS].index(o)] == SPECIES[label]]
        n_same_available = sum(
            1 for i, sp in SPECIES.items() if sp == SPECIES[label] and i != label
        )
        assert len(same) >= min(len(opts), n_same_available)  # same-species breeds come first
        asked = tasks["ext.pets.is_breed"]
        named = (
            asked.question["instructions"].removeprefix("Is the pet in this photo a ").rstrip("?")
        )
        assert asked.answer == (named == pretty(BREEDS[label]))
    truth = collections.Counter(t["ext.pets.is_breed"].answer for t in by_img.values())
    assert 25 < truth[True] < 55
    assert [q for q in qs] == [q for q in split(iter_oxford_pets(rows, BREEDS, SPECIES, seed=1))[1]]


# ---- CORD-v2 ------------------------------------------------------------------------------
def gt(parse):
    return json.dumps({"gt_parse": parse, "meta": {}})


def test_cord_facts_handle_dict_and_list_menus_and_ambiguous_payment():
    single = cord_facts(
        gt(
            {
                "menu": {"nm": "TICKET", "cnt": "2"},
                "total": {"total_price": "60.000", "creditcardprice": "60.000"},
            }
        )
    )
    assert single == {
        "n_items": 1,
        "has_tax": False,
        "has_service_charge": False,
        "paid_cash": False,
    }
    many = cord_facts(gt({"menu": [{"nm": "a"}, {"nm": "b"}, {"nm": "c"}],
                          "sub_total": {"tax_price": "5", "service_price": "2"},
                          "total": {"cashprice": "50"}}))  # fmt: skip
    assert many == {"n_items": 3, "has_tax": True, "has_service_charge": True, "paid_cash": True}
    both = cord_facts(
        gt({"menu": [{"nm": "a"}], "total": {"cashprice": "1", "creditcardprice": "1"}})
    )
    neither = cord_facts(gt({"menu": [{"nm": "a"}], "total": {}}))
    assert both["paid_cash"] is None and neither["paid_cash"] is None
    assert cord_facts({"gt_parse": {}})["n_items"] == 0  # accepts parsed dicts too


def test_iter_cord_questions_and_skips():
    rows = [
        {"id": "1", "ground_truth": gt({"menu": [{"nm": "a"}] * 5, "sub_total": {"tax_price": "1"}, "total": {"cashprice": "9"}})},
        {"id": "2", "ground_truth": gt({"total": {"creditcardprice": "9"}})},
        {"id": "3", "ground_truth": gt({"menu": {"nm": "x"}, "total": {"cashprice": "1", "creditcardprice": "1"}})},
    ]  # fmt: skip
    _, qs = split(iter_cord(rows))
    by = collections.defaultdict(dict)
    for q in qs:
        by[q.image_id][q.task] = q
    assert by["cord:test:1"]["ext.cord.item_count"].answer == "four or five"
    assert by["cord:test:1"]["ext.cord.has_tax"].answer is True
    assert by["cord:test:1"]["ext.cord.paid_cash"].answer is True
    assert "ext.cord.item_count" not in by["cord:test:2"]  # no menu: unknown, not "zero items"
    assert by["cord:test:2"]["ext.cord.paid_cash"].answer is False
    assert "ext.cord.paid_cash" not in by["cord:test:3"]  # both methods shown: ambiguous


# ---- ScreenSpot ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "want"),
    [("ios", "ios"), ("android", "android"), ("macos", "macos"), ("windows", "windows"),
     ("tool", "web"), ("shop", "web"), ("gitlab", "web"), ("forum", "web"), ("web", "web"), ("linux", None)],
)  # fmt: skip
def test_normalize_platform_covers_every_value_seen_in_the_real_data(raw, want):
    assert normalize_platform(raw) == want and set(PLATFORMS) == {
        "ios",
        "android",
        "macos",
        "windows",
        "web",
    }


def test_screenspot_one_question_per_screenshot_and_skips_bad_labels():
    rows = [
        {"file_name": "a.png", "data_source": "shop"}, {"file_name": "a.png", "data_source": "shop"},
        {"file_name": "b.png", "data_source": "ios"},
        {"file_name": "c.png", "data_source": "ios"}, {"file_name": "c.png", "data_source": "android"},
        {"file_name": "d.png", "data_source": "unheard-of"},
    ]  # fmt: skip
    facts, qs = split(iter_screenspot(rows))
    assert [q.image_id for q in qs] == ["screenspot:test:a.png", "screenspot:test:b.png"]
    assert [q.answer for q in qs] == ["web", "ios"] and len(facts) == 2
    assert all(q.source == "ext:screenspot" and q.domain == "screenshot" for q in qs)


# ---- routing -------------------------------------------------------------------------------
def test_external_records_land_only_in_test_external():
    rows = [{"image_id": f"i{i}", "label": i % 8} for i in range(200)]
    facts, qs = split(iter_oxford_pets(rows, BREEDS, SPECIES))
    res = build_splits(qs, facts, seed=0)
    check_splits(res)
    assert len(res.splits["test-external"]) == len(qs)
    assert all(
        not res.splits[s] for s in ("train", "val", "test-images", "test-tasks", "test-styles")
    )
