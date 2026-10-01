from collections import Counter

from imagejev.data.records import ImageFacts, QuestionRecord
from imagejev.data.rico import detect_elements, iter_rico, visible_node_classes


def chunk(nodes):
    """Build a columnar chunk like the Hub config: parallel lists, one entry per node."""
    return {
        "klass": [n["klass"] for n in nodes],
        "ancestors": [n.get("ancestors", ["android.view.View", "java.lang.Object"]) for n in nodes],
        "bounds": [n.get("bounds", [0, 0, 100, 50]) for n in nodes],
        "visible_to_user": [n.get("visible", True) for n in nodes],
        "visibility": [n.get("visibility", "visible") for n in nodes],
    }


def activity(*nodes):
    return {"root": {"klass": "DecorView"}, "children": [chunk(list(nodes))]}


def test_detects_visible_stock_widgets_and_superclasses():
    a = activity(
        {"klass": "android.widget.Spinner"},
        {"klass": "com.foo.FancySwitch", "ancestors": ["android.support.v7.widget.SwitchCompat"]},
    )
    assert detect_elements(a) == {"dropdown", "switch"}


def test_invisible_and_empty_nodes_are_ignored():
    a = activity(
        {"klass": "android.widget.Spinner", "visible": False},
        {"klass": "android.widget.Switch", "visibility": "gone"},
        {"klass": "android.widget.SeekBar", "bounds": [10, 10, 10, 60]},
    )
    assert detect_elements(a) == set()
    assert list(visible_node_classes(a)) == []


def test_view_switcher_is_not_a_switch():
    a = activity({"klass": "android.widget.ViewSwitcher"}, {"klass": "android.widget.TextSwitcher"})
    assert detect_elements(a) == set()


def row(rid, nodes, keyboard=False):
    return {"request_id": rid, "activity": activity(*nodes), "is_keyboard_deployed": keyboard}


ROWS = [
    row("1", [{"klass": "android.widget.Spinner"}, {"klass": "android.widget.AdView"}]),
    row("2", [{"klass": "android.widget.LinearLayout"}]),
    row("3", [{"klass": "android.widget.EditText"}], keyboard=True),
]


def split(out):
    out = list(out)  # a generator can only be iterated once
    return (
        [o for o in out if isinstance(o, ImageFacts)],
        [o for o in out if isinstance(o, QuestionRecord)],
    )


def test_default_asks_positives_only_for_verified_elements():
    facts, qs = split(iter_rico(ROWS, split="t"))
    pos = [q for q in qs if q.task == "rico.element_present" and q.answer]
    assert {q.meta["element"] for q in pos} == {"dropdown"}
    f1 = next(f for f in facts if f.image_id == "rico:t:1")
    assert f1.facts["elements"] == ["dropdown"]
    assert f1.facts["elements_unverified"] == ["ad_banner"]


def test_negatives_never_ask_about_present_or_untrusted_elements():
    _, qs = split(iter_rico(ROWS, split="t", bool_per_screen=6))
    for q in (q for q in qs if q.task == "rico.element_present" and not q.answer):
        assert q.meta["element"] in {
            "web_view",
            "map",
            "video",
            "switch",
            "dropdown",
            "date_picker",
            "seek_bar",
        }
        if q.image_id == "rico:t:1":
            assert q.meta["element"] != "dropdown"


def test_include_unverified_allows_other_positives():
    _, qs = split(iter_rico(ROWS, split="t", include_unverified=True, bool_per_screen=4, seed=1))
    assert {q.meta["element"] for q in qs if q.task == "rico.element_present" and q.answer} >= {
        "dropdown"
    }


def test_keyboard_questions_keep_all_open_and_subsample_closed():
    many = [row(str(i), [{"klass": "android.widget.LinearLayout"}]) for i in range(400)]
    many.append(row("open", [{"klass": "android.widget.LinearLayout"}], keyboard=True))
    _, qs = split(iter_rico(many, split="t", keyboard_closed_rate=0.1))
    kb = Counter(q.answer for q in qs if q.task == "rico.keyboard_open")
    assert kb[True] == 1
    assert 10 <= kb[False] <= 80


def test_records_are_valid_and_deterministic():
    a = list(iter_rico(ROWS, split="t", seed=5))
    b = list(iter_rico(ROWS, split="t", seed=5))
    assert a == b
    assert all(
        q.domain == "screenshot" and q.source == "rico" for q in a if isinstance(q, QuestionRecord)
    )
