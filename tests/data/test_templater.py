import collections
import random
import re

from imagejev.data.coco import iter_coco
from imagejev.data.docgen import labels_from_measure as doc_labels
from imagejev.data.docgen import money
from imagejev.data.docgen import questions_for as doc_questions
from imagejev.data.docgen import sample_spec as doc_spec
from imagejev.data.quality import KINDS
from imagejev.data.records import ImageFacts, QuestionRecord
from imagejev.data.rvlcdip import iter_rvl_cdip
from imagejev.data.templater import (
    NONE_LABEL,
    PAGE_PHRASE,
    apply_templates,
    balance_bools,
    compositional,
    load_paraphrases,
    render,
    rewrite,
)
from imagejev.data.webgen import (
    BOOL_TASKS,
    make_themes,
)
from imagejev.data.webgen import labels_from_measure as web_labels
from imagejev.data.webgen import questions_for as web_questions
from imagejev.data.webgen import sample_spec as web_spec


# ---- paraphrases -------------------------------------------------------------------------
def test_every_task_has_at_least_20_distinct_phrasings_with_valid_slots():
    table = load_paraphrases()
    prefixes = table["_prefixes"]
    assert len(prefixes) == 5 and "" in prefixes
    for task, entry in table.items():
        if task.startswith("_"):
            continue
        used = set()
        for stem in entry["stems"]:
            used |= set(re.findall(r"\{(\w+)\}", stem))
        assert used <= set(entry["slots"]), task
        slots = {k: "X" for k in entry["slots"]}
        phrasings = {p + s.format(**slots) for p in prefixes for s in entry["stems"]}
        assert len(phrasings) >= 20, task
        assert all("{" not in x and "}" not in x for x in phrasings)


def test_render_unknown_task_returns_none_and_is_deterministic():
    assert render("vqav2.yesno", {}, random.Random(0)) is None
    a = render("web.modal_open", {}, random.Random(5))
    assert a == render("web.modal_open", {}, random.Random(5))
    seen = {render("web.modal_open", {}, random.Random(s)) for s in range(200)}
    assert len(seen) >= 20


def test_all_emitted_tasks_are_covered_by_paraphrases():
    table = load_paraphrases()
    tasks: set[str] = set()
    themes = make_themes()
    for i in range(60):
        s = web_spec(i, 0, themes)
        m = {
            k: {"inView": True, "clear": True}
            for k in ("modal", "error", "spinner", "cookie", "captcha")
            if getattr(s, k)
        }
        m["cta"] = {"inView": not s.cta_below_fold, "clear": not s.cta_below_fold}
        m["__inputs"] = 1
        lab = web_labels(s, m)
        tasks |= {q.task for q in web_questions(s, lab, 0)}
        d = doc_spec(i, 0)
        dm = {"__size": [1, 1]}
        if d.signature:
            dm["signature"] = {"inside": True, "text": ""}
        if d.stamp:
            dm["stamp"] = {"inside": True, "text": d.stamp}
        if d.table:
            dm["table"] = {"inside": True, "text": ""}
        if d.doc_type != "form":
            dm["total"] = {"inside": True, "text": money(d.total)}
        tasks |= {q.task for q in doc_questions(d, doc_labels(d, dm), 0)}
    tasks |= {f"quality.{k}" for k in KINDS}
    tasks |= {t for t in BOOL_TASKS.values() and [v[0] for v in BOOL_TASKS.values()]}
    tasks |= {"rvl.doc_type", "rico.keyboard_open", "rico.element_present"}
    tasks |= {"coco.object_present", "coco.count_bin", "coco.dominant_supercategory"}
    assert tasks - set(table) == set(), tasks - set(table)


# ---- rewrite -----------------------------------------------------------------------------
def choice(task="rvl.doc_type", n=8, answer="a0"):
    return QuestionRecord(
        "img", "document", "s", task,
        {"type": "choice", "instructions": "orig", "criteria": {f"a{i}": f"desc{i}" for i in range(n)}},
        answer,
    )  # fmt: skip


def test_rewrite_paraphrases_and_keeps_a_valid_answer():
    outs = [rewrite(choice(), seed=s, p_none=0.0) for s in range(50)]
    assert len({o.question["instructions"] for o in outs}) > 10
    for o in outs:
        assert o.answer in o.question["criteria"] and o.answer == "a0"
        assert 3 <= len(o.question["criteria"]) <= 8
        assert o.question["criteria"]["a0"] == "desc0"
    assert len({tuple(o.question["criteria"]) for o in outs}) > 10  # shuffled orders


def test_none_of_these_removes_the_true_option():
    outs = [rewrite(choice(), seed=s, p_none=1.0) for s in range(30)]
    for o in outs:
        assert o.answer == NONE_LABEL and "a0" not in o.question["criteria"]
        assert NONE_LABEL in o.question["criteria"]
    # never for tasks that already have their own catch-all, and never with <3 options
    assert rewrite(choice(task="doc.stamp_text"), 0, p_none=1.0).answer == "a0"
    assert rewrite(choice(n=2), 0, p_none=1.0).answer == "a0"


def test_subset_prefers_confusable_distractors():
    rec = QuestionRecord(
        "img", "document", "s", "rvl.doc_type",
        {"type": "choice", "instructions": "x",
         "criteria": {n: "" for n in ["invoice", "form", "questionnaire", "budget", "letter", "memo", "resume", "email"]}},
        "invoice",
    )  # fmt: skip
    for s in range(60):
        o = rewrite(rec, s, p_none=0.0)
        labels = set(o.question["criteria"])
        if len(labels) < 8:
            mates = {"form", "questionnaire", "budget"}
            assert labels & mates == (labels - {"invoice"}) & mates
            assert len(labels & mates) == min(len(labels) - 1, 3)


def test_slots_come_from_meta_and_unknown_tasks_are_untouched():
    coco = QuestionRecord(
        "i", "photo", "coco", "coco.object_present",
        {"type": "bool", "instructions": "orig"}, True, meta={"category": "umbrella"},
    )  # fmt: skip
    assert "an umbrella" in rewrite(coco, 1).question["instructions"]
    rico = QuestionRecord(
        "i", "screenshot", "rico", "rico.element_present",
        {"type": "bool", "instructions": "orig"}, True, meta={"element": "dropdown"},
    )  # fmt: skip
    assert "a dropdown menu" in rewrite(rico, 1).question["instructions"]
    vqa = QuestionRecord(
        "i",
        "photo",
        "vqav2",
        "vqav2.yesno",
        {"type": "bool", "instructions": "Is it raining?"},
        False,
    )
    assert rewrite(vqa, 1) == vqa
    thr = QuestionRecord(
        "i", "document", "docgen", "doc.total_over",
        {"type": "bool", "instructions": "orig"}, True, meta={"threshold": "$25.00"},
    )  # fmt: skip
    assert "$25.00" in rewrite(thr, 1).question["instructions"]


# ---- compositional: answers recomputed independently ---------------------------------------
PHOTO = ImageFacts(
    "p1", "photo", "coco",
    {"counts": {"dog": 2, "person": 1, "car": 3, "bus": 1},
     "present_any": ["bird", "bus", "car", "dog", "person"],  # a tiny bird is annotated but not visible
     "counts_all": {"bus": 1, "car": 3, "dog": 2, "person": 1, "bird": 1},
     "crowd": ["car"]},
)  # fmt: skip
VOCAB = ["bird", "bus", "car", "cat", "cow", "dog", "person", "sheep", "train"]


def photo_truth(task, slots, f=PHOTO):
    counts = f.facts["counts"]
    strip = lambda s: s.split(" ", 1)[1] if s[:2] in ("a ", "an") or s.startswith("an ") else s  # noqa: E731
    if task == "comp.photo.and_not":
        a, b = strip(slots["a"]), strip(slots["b"])
        return a in counts and b not in counts
    if task == "comp.photo.or":
        return strip(slots["a"]) in counts or strip(slots["b"]) in counts
    if task == "comp.photo.more_than":
        return counts[slots["cat"]] > int(slots["n"])
    if task == "comp.photo.more_of":
        return counts[slots["a"]] > counts[slots["b"]]
    raise AssertionError(task)


def test_photo_compositional_answers_are_exact_and_avoid_ambiguous_objects():
    seen = collections.Counter()
    for seed in range(300):
        for q in compositional([PHOTO], seed=seed, per_image=4, vocab=VOCAB):
            seen[q.task] += 1
            text = q.question["instructions"]
            # the tiny annotated bird and the crowd-labelled cars must never be asserted about
            assert not re.search(r"\bbird\b", text), text
            if q.task in ("comp.photo.more_than", "comp.photo.more_of"):
                assert not re.search(r"\bcar\b", text), text
    assert set(seen) == {
        "comp.photo.and_not",
        "comp.photo.or",
        "comp.photo.more_than",
        "comp.photo.more_of",
    }


def test_photo_answers_match_recomputation_via_slots():
    import imagejev.data.templater as T

    for seed in range(100):
        rng = random.Random(seed)
        for task, slots, ans in T._photo(PHOTO, VOCAB, rng):
            assert ans == photo_truth(task, slots), (task, slots)


DOC = ImageFacts(
    "d1", "document", "docgen",
    {"doc_type": "receipt", "merchant_type": "cafe", "total_cents": 4200, "has_signature": True,
     "has_stamp": False, "stamp_text": None, "has_table": False, "n_items": 3},
)  # fmt: skip


def test_document_compositional_answers():
    import imagejev.data.templater as T

    seen = set()
    for seed in range(200):
        for task, slots, ans in T._document(DOC, random.Random(seed)):
            seen.add(task)
            d = DOC.facts
            if task == "comp.doc.merchant_over":
                merchant = slots["merchant"].split(" ", 1)[1]
                thr = round(float(slots["thr"].strip("$").replace(",", "")) * 100)
                assert ans == (merchant == "cafe" and d["total_cents"] > thr)
            elif task == "comp.doc.signed_stamped":
                assert ans is False
            elif task == "comp.doc.signature_or_table":
                assert ans is True
            elif task == "comp.doc.total_between":
                lo = round(float(slots["lo"].strip("$").replace(",", "")) * 100)
                hi = round(float(slots["hi"].strip("$").replace(",", "")) * 100)
                assert ans == (lo <= 4200 <= hi)
            elif task == "comp.doc.invoice_not_paid":
                assert ans is False  # it is a receipt
    assert seen >= {"comp.doc.merchant_over", "comp.doc.signed_stamped", "comp.doc.total_between"}


def test_form_has_no_total_question():
    form = ImageFacts(
        "d2",
        "document",
        "docgen",
        {**DOC.facts, "doc_type": "form", "merchant_type": None, "total_cents": None},
    )
    tasks = {q.task for s in range(80) for q in compositional([form], seed=s, per_image=6)}
    assert "comp.doc.total_between" not in tasks


def test_screen_compositional_answers():
    f = ImageFacts(
        "w1", "screenshot", "webgen",
        {"page_type": "login", "modal_open": True, "error_banner": True, "loading": False, "cookie_banner": True,
         "captcha": False, "cta_clickable": False, "input_count": 2, "cart_empty": None, "viewport": "desktop"},
    )  # fmt: skip
    got = {}
    for s in range(100):
        for q in compositional([f], seed=s, per_image=5):
            got.setdefault(q.task, set()).add(
                (q.question["instructions"].split(". ")[-1], q.answer)
            )
    assert {a for _, a in got["comp.screen.modal_and_cookie"]} == {True}
    assert {a for _, a in got["comp.screen.cta_and_no_modal"]} == {False}
    assert {a for _, a in got["comp.screen.form_and_captcha"]} == {False}
    assert {a for _, a in got["comp.screen.loading_or_error"]} == {True}
    # page_with_error: true only when the asked page type is the real one
    answers = collections.defaultdict(set)
    for s in range(300):
        for q in compositional([f], seed=s, per_image=5):
            if q.task == "comp.screen.page_with_error":
                m = re.search(
                    r"\b(login|landing|shop|blog|dashboard|checkout)\b", q.question["instructions"]
                )
                answers[m.group(1)].add(q.answer)
    assert answers["login"] == {True} and all(
        v == {False} for k, v in answers.items() if k != "login"
    )
    assert set(PAGE_PHRASE.values()) >= set(answers)


def test_compositional_skips_unknown_domains_and_rico_facts():
    rico = ImageFacts(
        "r1", "screenshot", "rico", {"elements": ["dropdown"], "keyboard_open": False}
    )
    assert compositional([rico], seed=0) == []


# ---- balance & pipeline --------------------------------------------------------------------
def boolrec(i, task, ans):
    return QuestionRecord(f"i{i}", "photo", "s", task, {"type": "bool", "instructions": "x"}, ans)


def test_balance_bools_per_task_and_leaves_degenerate_tasks_alone():
    recs = [boolrec(i, "t1", i % 10 == 0) for i in range(500)]  # 10% true
    recs += [boolrec(1000 + i, "t2", i % 2 == 0) for i in range(100)]  # already balanced
    recs += [boolrec(2000 + i, "t3", False) for i in range(40)]  # single class
    out = balance_bools(recs, seed=1)
    by = collections.defaultdict(collections.Counter)
    for r in out:
        by[r.task][r.answer] += 1
    assert 0.45 <= by["t1"][True] / sum(by["t1"].values()) <= 0.55
    assert by["t2"] == {True: 50, False: 50} and by["t3"] == {False: 40}
    assert balance_bools(recs, seed=1) == out  # deterministic
    assert sum(by["t1"].values()) == 100  # kept every minority example


COCO = {
    "categories": [{"id": i, "name": n, "supercategory": s} for i, (n, s) in enumerate(
        [("person", "person"), ("dog", "animal"), ("cat", "animal"), ("car", "vehicle"), ("bus", "vehicle")], 1)],
    "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100} for i in range(1, 41)],
    "annotations": [
        {"id": 1000 * i + k, "image_id": i, "category_id": 1 + (i + k) % 5, "area": 1500 + 300 * k, "iscrowd": 0}
        for i in range(1, 41) for k in range(1 + i % 3)
    ],
}  # fmt: skip


def test_end_to_end_all_domains_balanced_and_valid():
    records, facts = [], []
    for o in iter_coco(COCO, seed=0):
        (facts if isinstance(o, ImageFacts) else records).append(o)
    rvl = list(iter_rvl_cdip([{"id": i, "label": i % 16} for i in range(40)]))
    records += [r for r in rvl if isinstance(r, QuestionRecord)]
    facts += [r for r in rvl if isinstance(r, ImageFacts)]
    themes = make_themes()
    for i in range(40):
        s = web_spec(i, 0, themes)
        m = {
            k: {"inView": True, "clear": True}
            for k in ("modal", "error", "spinner", "cookie", "captcha")
            if getattr(s, k)
        }
        m["cta"] = {"inView": not s.cta_below_fold, "clear": not s.cta_below_fold}
        m["__inputs"] = 2
        lab = web_labels(s, m)
        facts.append(ImageFacts(s.image_id, "screenshot", "webgen", lab, style=s.style))
        records += web_questions(s, lab, 0)
        d = doc_spec(i, 0)
        dm = {"__size": [1, 1]}
        for key, flag in (("signature", d.signature), ("stamp", d.stamp), ("table", d.table)):
            if flag:
                dm[key] = {"inside": True, "text": d.stamp or ""}
        if d.doc_type != "form":
            dm["total"] = {"inside": True, "text": money(d.total)}
        dl = doc_labels(d, dm)
        facts.append(ImageFacts(d.image_id, "document", "docgen", dl, style=d.style))
        records += doc_questions(d, dl, 0)
    out = apply_templates(records, facts, seed=3)
    assert {r.domain for r in out if r.task.startswith("comp.")} == {
        "photo",
        "document",
        "screenshot",
    }
    by = collections.defaultdict(collections.Counter)
    for r in out:
        if r.question["type"] == "bool":
            by[r.task][r.answer] += 1
    for task, c in by.items():
        if len(c) == 2 and sum(c.values()) >= 20:
            assert 0.40 <= c[True] / sum(c.values()) <= 0.60, (task, c)
    assert out == apply_templates(records, facts, seed=3)
    assert all(isinstance(r, QuestionRecord) for r in out)
