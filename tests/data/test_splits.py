import collections
import json

import pytest

from imagejev.data.docgen import DOC_TYPES, VARIANTS
from imagejev.data.records import ImageFacts, QuestionRecord
from imagejev.data.splits import (
    SPLITS,
    SplitResult,
    build_splits,
    check_splits,
    group_id,
    is_heldout_style,
    load_heldout,
    write_splits,
)
from imagejev.data.templater import load_paraphrases
from imagejev.data.webgen import TEMPLATES, make_themes

CFG = load_heldout()
BOOL = {"type": "bool", "instructions": "x"}


def rec(image_id, task, *, domain="photo", source="s", style=None, meta=None, answer=True):
    return QuestionRecord(
        image_id, domain, source, task, BOOL, answer, style=style, meta=meta or {}
    )


# ---- the committed config ------------------------------------------------------------------
def test_heldout_tasks_exist_and_cover_every_domain_and_type():
    table = load_paraphrases()
    assert set(CFG["task_families"]) <= set(table)
    # (domain, type) of each held-out task, from the task names and what the generators emit
    kinds = {
        "coco.dominant_supercategory": ("photo", "choice"),
        "comp.photo.more_of": ("photo", "bool"),
        "doc.merchant_type": ("document", "choice"),
        "doc.item_count": ("document", "score"),
        "comp.doc.total_between": ("document", "bool"),
        "web.cart_empty": ("screenshot", "bool"),
        "web.input_count": ("screenshot", "score"),
        "comp.screen.form_and_captcha": ("screenshot", "bool"),
        "quality.jpeg": ("all", "score"),
    }
    assert set(kinds) == set(CFG["task_families"])
    domains = {d for d, _ in kinds.values()}
    types = {t for _, t in kinds.values()}
    assert {"photo", "document", "screenshot"} <= domains and types == {"choice", "score", "bool"}
    assert all(CFG["task_families"].values())  # every entry says why


def test_validation_families_exist_are_disjoint_from_test_families_and_cover_every_domain():
    table = load_paraphrases()
    val = set(CFG["val_task_families"])
    assert val <= set(table) and not val & set(CFG["task_families"])
    assert all(CFG["val_task_families"].values())
    domains = {
        "photo": "comp.photo.or",
        "document": "doc.has_table",
        "screenshot": "web.cookie_banner",
    }
    assert set(domains.values()) <= val  # one per domain, so each domain has a validation signal
    assert "quality.noise" in val and "doc.stamp_text" in val  # a score family and a choice family


def test_a_family_in_both_sets_is_rejected():
    bad = {**CFG, "val_task_families": {sorted(CFG["task_families"])[0]: "oops"}}
    with pytest.raises(ValueError, match="both"):
        build_splits([rec("a", "x.y")], config=bad)


def test_style_patterns_hit_something_but_not_most_styles():
    styles = [f"{t}:{th.id}" for t in TEMPLATES for th in make_themes()]
    styles += [f"{d}:{v}" for d in DOC_TYPES for v in VARIANTS]
    hit = [s for s in styles if is_heldout_style(s, CFG["style_patterns"])]
    assert hit and len(hit) / len(styles) < 0.3
    assert not is_heldout_style(None, CFG["style_patterns"])
    assert any(s.endswith(":t10") for s in hit) and "receipt:c" in hit and "receipt:a" not in hit


def test_group_fractions_are_sane():
    assert set(CFG["group_fractions"]) == {"train", "val", "val-tasks", "test-images", "test-tasks"}
    assert abs(sum(CFG["group_fractions"].values()) - 1.0) < 1e-9


# ---- building ------------------------------------------------------------------------------
def make_world(n=3000):
    held = sorted(CFG["task_families"])
    records, facts = [], []
    for i in range(n):
        iid = f"coco:{i:06d}"
        records += [rec(iid, "coco.object_present"), rec(iid, "vqav2.yesno", source="vqav2")]
        records.append(rec(iid, held[0]))  # held-out task on every photo
        records.append(rec(iid, sorted(CFG["val_task_families"])[0]))  # validation family
        records.append(rec(f"{iid}+blur", "quality.blur", meta={"source_id": iid}))
        facts.append(ImageFacts(f"{iid}+blur", "photo", "quality", {"source_id": iid}))
        w = f"web:{i:07d}"
        theme = f"t{i % 12:02d}"
        records += [
            rec(w, "web.modal_open", domain="screenshot", source="webgen", style=f"login:{theme}"),
            rec(w, "web.cart_empty", domain="screenshot", source="webgen", style=f"login:{theme}"),
        ]
        d = f"doc:{i:07d}"
        records.append(
            rec(d, "doc.has_signature", domain="document", source="docgen",
                style=f"{DOC_TYPES[i % 3]}:{VARIANTS[i % 3]}")
        )  # fmt: skip
        if i % 10 == 0:
            records.append(rec(f"ext:{i}", "ext.task", source="ext:somebench"))
    return records, facts


@pytest.fixture(scope="module")
def world():
    records, facts = make_world()
    return records, facts, build_splits(records, facts, seed=0)


def test_all_rules_hold_on_a_large_world(world):
    _, _, res = world
    check_splits(res)  # raises on any violation


def test_no_image_group_appears_in_two_splits(world):
    _, _, res = world
    where = {}
    for s, recs in res.splits.items():
        for r in recs:
            assert where.setdefault(group_id(r), s) == s


def test_derived_images_follow_their_source(world):
    _, _, res = world
    where = {}
    for s, recs in res.splits.items():
        for r in recs:
            where[r.image_id] = s
    pairs = [(k, v) for k, v in where.items() if k.endswith("+blur")]
    assert pairs
    for derived, split in pairs:
        assert where.get(derived.split("+")[0], split) == split


def test_cocos_and_vqav2_questions_on_one_image_share_a_split(world):
    _, _, res = world
    by_image = collections.defaultdict(set)
    for s, recs in res.splits.items():
        for r in recs:
            by_image[r.image_id].add(s)
    assert all(len(v) == 1 for v in by_image.values())


def test_heldout_tasks_only_in_test_tasks_and_test_tasks_only_heldout(world):
    _, _, res = world
    held = set(CFG["task_families"])
    for s, recs in res.splits.items():
        tasks = {r.task for r in recs}
        if s == "test-tasks":
            assert tasks and tasks <= held
        elif s != "test-external":
            assert not tasks & held, s
    assert res.splits["test-tasks"], "test-tasks must not be empty"


def test_styles_and_external_routing(world):
    _, _, res = world
    patterns = CFG["style_patterns"]
    for r in res.splits["test-styles"]:
        assert is_heldout_style(r.style, patterns)
    for s in ("train", "val", "test-images", "test-tasks"):
        assert not any(is_heldout_style(r.style, patterns) for r in res.splits[s])
    assert res.splits["test-external"]
    assert all(r.source.startswith("ext:") for r in res.splits["test-external"])
    assert not any(r.source.startswith("ext:") for s in SPLITS[:-1] for r in res.splits[s])


def test_fractions_roughly_match_config(world):
    records, _, res = world
    coco_groups = {g for g, s in res.group_split.items() if g.startswith("coco:")}
    counts = collections.Counter(res.group_split[g] for g in coco_groups)
    for name, want in CFG["group_fractions"].items():
        got = counts[name] / len(coco_groups)
        assert abs(got - want) < 0.03, (name, got, want)


def test_every_record_is_kept_or_counted_as_dropped(world):
    records, _, res = world
    kept = sum(len(v) for v in res.splits.values())
    assert kept + sum(res.dropped.values()) == len(records)
    assert res.dropped["held-out task question on another split's image"] > 0


def test_deterministic_and_seed_sensitive():
    records, facts = make_world(400)
    a = build_splits(records, facts, seed=1)
    b = build_splits(records, facts, seed=1)
    c = build_splits(records, facts, seed=2)
    assert a.group_split == b.group_split
    assert a.group_split != c.group_split


def test_adding_data_does_not_move_existing_images():
    records, facts = make_world(300)
    small = build_splits(records[:600], facts[:100], seed=0).group_split
    big = build_splits(records, facts, seed=0).group_split
    assert all(big[g] == s for g, s in small.items())  # hash-based: stable as the set grows


# ---- the checker catches leaks ---------------------------------------------------------------
def test_check_splits_catches_each_kind_of_leak():
    held = sorted(CFG["task_families"])[0]
    ok = rec("a", "coco.object_present")
    with pytest.raises(AssertionError, match="held-out task"):
        check_splits(
            SplitResult({**{s: [] for s in SPLITS}, "train": [rec("a", held)]}, {"a": "train"})
        )
    with pytest.raises(AssertionError, match="seen task"):
        check_splits(
            SplitResult({**{s: [] for s in SPLITS}, "test-tasks": [ok]}, {"a": "test-tasks"})
        )
    with pytest.raises(AssertionError, match="is in"):
        res = SplitResult(
            {**{s: [] for s in SPLITS}, "train": [ok], "val": [rec("a", "x.y")]}, {"a": "train"}
        )
        check_splits(res)
    with pytest.raises(AssertionError, match="style leak"):
        r = rec("b", "web.modal_open", domain="screenshot", style="login:t10")
        check_splits(SplitResult({**{s: [] for s in SPLITS}, "train": [r]}, {"b": "train"}))
    with pytest.raises(AssertionError):
        r = rec("c", "ext.t", source="ext:x")
        check_splits(SplitResult({**{s: [] for s in SPLITS}, "train": [r]}, {"c": "train"}))


def test_write_splits_files_and_report(tmp_path):
    records, facts = make_world(200)
    res = build_splits(records, facts)
    rep = write_splits(res, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {f"{s}.jsonl" for s in SPLITS} | {
        "splits_report.json"
    }
    assert json.loads((tmp_path / "splits_report.json").read_text()) == rep
    assert sum(rep["questions"].values()) + sum(rep["dropped"].values()) == len(records)


def test_report_flags_thin_heldout_families():
    records, facts = make_world(120)
    res = build_splits(records, facts, seed=0)
    thin = res.thin_families(min_size=100)
    # only one held-out family exists in this toy world, and the others have no examples at all
    assert set(thin) == set(CFG["task_families"])
    assert all(n < 100 for n in thin.values())
    assert res.report()["thin_families"] == thin
    big = build_splits(*make_world(3000), seed=0)
    first = sorted(CFG["task_families"])[0]
    assert first not in big.thin_families(min_size=100)
    assert big.report(min_family_size=100)["test_tasks_per_family"][first] >= 100


def test_val_tasks_holds_only_validation_families_and_they_appear_nowhere_else(world):
    _, _, res = world
    val_fams = set(CFG["val_task_families"])
    tasks = {r.task for r in res.splits["val-tasks"]}
    assert tasks and tasks <= val_fams
    for s in SPLITS:
        if s != "val-tasks":
            assert not {r.task for r in res.splits[s]} & val_fams, s
    assert not {r.task for r in res.splits["test-tasks"]} & val_fams


def test_selection_split_shares_no_images_with_anything_else(world):
    _, _, res = world
    groups = {g for g, s in res.group_split.items() if s == "val-tasks"}
    assert groups
    for s in SPLITS:
        if s != "val-tasks":
            assert not groups & {group_id(r) for r in res.splits[s]}


def test_check_splits_catches_validation_family_leaks():
    val_fam = sorted(CFG["val_task_families"])[0]
    empty = {s: [] for s in SPLITS}
    with pytest.raises(AssertionError, match="validation task"):
        check_splits(SplitResult({**empty, "train": [rec("a", val_fam)]}, {"a": "train"}))
    with pytest.raises(AssertionError, match="seen task"):
        check_splits(
            SplitResult(
                {**empty, "val-tasks": [rec("a", "coco.object_present")]}, {"a": "val-tasks"}
            )
        )
    held = sorted(CFG["task_families"])[0]
    with pytest.raises(AssertionError, match="seen task"):
        check_splits(
            SplitResult({**empty, "val-tasks": [rec("a", held)]}, {"a": "val-tasks"})
        )  # test family
    with pytest.raises(AssertionError, match="seen task"):
        check_splits(SplitResult({**empty, "test-tasks": [rec("a", val_fam)]}, {"a": "test-tasks"}))


def test_report_flags_thin_validation_families(world):
    _, _, res = world
    rep = res.report(min_family_size=10**9)  # everything is thin at this bar
    assert set(rep["thin_val_families"]) == set(CFG["val_task_families"])
    assert "val_tasks_per_family" in rep
