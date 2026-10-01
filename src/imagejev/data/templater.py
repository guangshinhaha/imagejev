"""Question templater: paraphrase, vary options, add compositional questions, balance bools.

Input is what the adapters and generators emit (``QuestionRecord`` + ``ImageFacts``); output is the
training question set. Everything is deterministic for a given ``seed``.

1. **Paraphrase.** ``paraphrases.json`` holds stems per task and a few prefixes: 5 x 5 = 25
   phrasings each. Tasks not listed (e.g. VQAv2, whose questions are human-written) are untouched.
2. **Vary choices.** Option order is shuffled; large option sets are cut to a subset that keeps
   the answer and prefers confusable classes as distractors; some questions drop the true option
   and make "none of these" the answer.
3. **Compositional questions.** Built from ``ImageFacts`` with exact answers, e.g. "a restaurant
   receipt over $50". A question is only asked when the facts decide it unambiguously.
4. **Balance.** Each bool task is downsampled to 50/50.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from functools import cache
from importlib import resources
from typing import Any

from .docgen import MERCHANT_TYPES, money
from .records import ImageFacts, QuestionRecord
from .rico import ELEMENTS
from .rvlcdip import CONFUSABLE_GROUPS as RVL_GROUPS

NONE_LABEL = "none of these"
NONE_DESCRIPTION = "the correct answer is not listed"

# Choice tasks where "none of these" is a natural, fair answer (no catch-all class of their own).
NONE_OK = frozenset(
    {
        "rvl.doc_type",
        "web.page_type",
        "doc.type",
        "doc.merchant_type",
        "coco.dominant_supercategory",
    }
)
CONFUSABLE: dict[str, list[set[str]]] = {
    "rvl.doc_type": RVL_GROUPS,
    "web.page_type": [{"login", "checkout"}, {"landing", "article"}, {"shop", "dashboard"}],
    "doc.type": [{"receipt", "invoice"}],
    "doc.merchant_type": [
        {"restaurant", "cafe"},
        {"grocery", "pharmacy"},
        {"hardware", "gas station"},
    ],
}
PAGE_PHRASE = {
    "login": "login",
    "landing": "landing",
    "shop": "shop",
    "article": "blog",
    "dashboard": "dashboard",
    "checkout": "checkout",
}


@cache
def load_paraphrases() -> dict[str, Any]:
    text = resources.files("imagejev.data").joinpath("paraphrases.json").read_text()
    return json.loads(text)


def _rng(seed: int, *parts: str) -> random.Random:
    h = hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _article(name: str) -> str:
    return f"an {name}" if name[0] in "aeiou" else f"a {name}"


def render(task: str, slots: dict[str, str], rng: random.Random) -> str | None:
    """A random phrasing of ``task`` with ``slots`` filled in, or ``None`` if it has no stems."""
    table = load_paraphrases()
    entry = table.get(task)
    if entry is None:
        return None
    text = rng.choice(entry["stems"]).format(**slots)
    return rng.choice(table["_prefixes"]) + text


def _slots_for(rec: QuestionRecord) -> dict[str, str] | None:
    """Slot values for tasks whose wording depends on the record, else ``{}``; None if unknown."""
    m = rec.meta
    if rec.task == "coco.object_present":
        return {"x": _article(m["category"])}
    if rec.task == "coco.count_bin":
        return {"cat": m["category"]}
    if rec.task == "rico.element_present":
        return {"x": ELEMENTS[m["element"]][1]}
    if rec.task == "doc.total_over":
        return {"thr": m["threshold"]}
    return {}


def _vary_choice(rec: QuestionRecord, rng: random.Random, p_none: float) -> QuestionRecord:
    q = dict(rec.question)
    options: dict[str, str] = dict(q["criteria"])
    labels = list(options)
    answer = rec.answer
    groups = CONFUSABLE.get(rec.task, [])
    if rec.task in NONE_OK and len(labels) >= 3 and rng.random() < p_none:
        labels.remove(answer)
        options = {k: options[k] for k in labels}
        options[NONE_LABEL] = NONE_DESCRIPTION
        answer = NONE_LABEL
    elif len(labels) > 3 and rng.random() < 0.5:
        k = rng.randint(3, len(labels) - 1)
        mates = [c for g in groups if answer in g for c in g if c != answer and c in labels]
        rest = [c for c in labels if c != answer and c not in mates]
        rng.shuffle(mates)
        rng.shuffle(rest)
        labels = [answer, *(mates + rest)[: k - 1]]
        options = {k_: options[k_] for k_ in labels}
    order = list(options)
    rng.shuffle(order)
    q["criteria"] = {k: options[k] for k in order}
    return replace(rec, question=q, answer=answer, soft=None)


def rewrite(rec: QuestionRecord, seed: int, p_none: float = 0.1) -> QuestionRecord:
    """Paraphrase one record and vary its options."""
    rng = _rng(seed, rec.image_id, rec.task, json_key(rec.question))
    slots = _slots_for(rec)
    out = rec
    text = render(rec.task, slots or {}, rng)
    if text is not None:
        out = replace(out, question={**out.question, "instructions": text})
    if out.question["type"] == "choice":
        out = _vary_choice(out, rng, p_none)
    return out


def json_key(q: dict[str, Any]) -> str:
    return json.dumps(q, sort_keys=True)


# ---------------------------------------------------------------------------------------------
# Compositional questions (exact, from facts)
# ---------------------------------------------------------------------------------------------
Comp = tuple[str, dict[str, str], bool]  # (task, slots, answer)


def _photo(f: ImageFacts, vocab: list[str], rng: random.Random) -> list[Comp]:
    counts: dict[str, int] = f.facts["counts"]
    any_ = set(f.facts["present_any"])
    crowd = set(f.facts["crowd"])
    exact = f.facts["counts_all"]
    visible = sorted(counts)
    absent = sorted(set(vocab) - any_)
    # a category is "decidable" for presence if it is clearly visible or not annotated at all
    out: list[Comp] = []
    if visible:
        a = rng.choice(visible)
        others = [c for c in visible if c != a]
        b = rng.choice(absent) if absent and (rng.random() < 0.5 or not others) else None
        b = b or (rng.choice(others) if others else None)
        if b:
            out.append(
                ("comp.photo.and_not", {"a": _article(a), "b": _article(b)}, b not in counts)
            )
        pool = [c for c in visible if c not in crowd and exact.get(c) == counts[c]]
        if pool:
            c = rng.choice(pool)
            n = rng.choice([1, 2, 3, 5])
            out.append(("comp.photo.more_than", {"n": str(n), "cat": c}, counts[c] > n))
        if len(pool) >= 2:
            x, y = rng.sample(pool, 2)
            if counts[x] != counts[y]:
                out.append(("comp.photo.more_of", {"a": x, "b": y}, counts[x] > counts[y]))
    # "A or B": half the time both are absent (false), otherwise one visible and one absent (true)
    if len(absent) >= 2 and (not visible or rng.random() < 0.5):
        x, y = rng.sample(absent, 2)
        out.append(("comp.photo.or", {"a": _article(x), "b": _article(y)}, False))
    elif visible and absent:
        pair = [rng.choice(visible), rng.choice(absent)]
        rng.shuffle(pair)
        out.append(("comp.photo.or", {"a": _article(pair[0]), "b": _article(pair[1])}, True))
    return out


def _document(f: ImageFacts, rng: random.Random) -> list[Comp]:
    d = f.facts
    out: list[Comp] = []
    thr = rng.choice([1000, 2500, 5000, 10000, 20000])
    merchant = (
        d["merchant_type"]
        if d["merchant_type"] and rng.random() < 0.6
        else rng.choice(sorted(MERCHANT_TYPES))
    )
    is_match = d["doc_type"] == "receipt" and d["merchant_type"] == merchant
    out.append(
        (
            "comp.doc.merchant_over",
            {"merchant": _article(merchant), "thr": money(thr)},
            bool(is_match and d["total_cents"] > thr),
        )
    )
    out.append(("comp.doc.signed_stamped", {}, bool(d["has_signature"] and d["has_stamp"])))
    out.append(("comp.doc.signature_or_table", {}, bool(d["has_signature"] or d["has_table"])))
    if d["total_cents"] is not None:
        lo = rng.choice([500, 1500, 4000, 9000])
        hi = lo + rng.choice([3000, 8000, 20000, 60000])
        out.append(
            (
                "comp.doc.total_between",
                {"lo": money(lo), "hi": money(hi)},
                lo <= d["total_cents"] <= hi,
            )
        )
    if d["doc_type"] != "form" or rng.random() < 0.3:
        out.append(
            (
                "comp.doc.invoice_not_paid",
                {},
                bool(d["doc_type"] == "invoice" and d["stamp_text"] != "PAID"),
            )
        )
    return out


def _screen(f: ImageFacts, rng: random.Random) -> list[Comp]:
    d = f.facts
    if "page_type" not in d:  # only synthetic web pages carry these facts
        return []
    return [
        (
            "comp.screen.page_with_error",
            {
                "page": PAGE_PHRASE[
                    rng.choice(sorted(PAGE_PHRASE)) if rng.random() < 0.4 else d["page_type"]
                ]
            },
            None,  # decided below, once the asked page type is known
        ),
        ("comp.screen.modal_and_cookie", {}, bool(d["modal_open"] and d["cookie_banner"])),
        ("comp.screen.cta_and_no_modal", {}, bool(d["cta_clickable"] and not d["modal_open"])),
        (
            "comp.screen.form_and_captcha",
            {},
            bool(d["input_count"] >= 2 and d["captcha"]),
        ),
        ("comp.screen.loading_or_error", {}, bool(d["loading"] or d["error_banner"])),
    ]


def compositional(
    facts: Iterable[ImageFacts],
    *,
    seed: int = 0,
    per_image: int = 2,
    vocab: list[str] | None = None,
) -> list[QuestionRecord]:
    facts = list(facts)
    if vocab is None:
        names: set[str] = set()
        for f in facts:
            if f.domain == "photo" and "present_any" in f.facts:
                names.update(f.facts["present_any"])
        vocab = sorted(names)
    out: list[QuestionRecord] = []
    for f in facts:
        rng = _rng(seed, "comp", f.image_id)
        if f.domain == "photo" and "present_any" in f.facts:
            cands = _photo(f, vocab, rng)
        elif f.domain == "document" and "total_cents" in f.facts:  # docgen; RVL has only a type
            cands = _document(f, rng)
        elif f.domain == "screenshot":
            cands = _screen(f, rng)
        else:
            continue
        resolved: list[Comp] = []
        for task, slots, ans in cands:
            if task == "comp.screen.page_with_error":
                asked = next(k for k, v in PAGE_PHRASE.items() if v == slots["page"])
                ans = bool(f.facts["page_type"] == asked and f.facts["error_banner"])
            resolved.append((task, slots, bool(ans)))
        rng.shuffle(resolved)
        for task, slots, ans in resolved[:per_image]:
            text = render(task, slots, rng)
            assert text is not None, task
            out.append(
                QuestionRecord(
                    f.image_id,
                    f.domain,
                    f.source,
                    task,
                    {"type": "bool", "instructions": text},
                    ans,
                    style=f.style,
                )
            )
    return out


# ---------------------------------------------------------------------------------------------
# Balance and the full pipeline
# ---------------------------------------------------------------------------------------------
def balance_bools(
    records: list[QuestionRecord], *, seed: int = 0, tolerance: float = 0.04
) -> list[QuestionRecord]:
    """Downsample the majority answer of every bool task so true/false is within 50% +- tolerance.

    A task with only one answer can't be balanced and is returned unchanged.
    """
    by_task: dict[str, dict[bool, list[int]]] = defaultdict(lambda: {True: [], False: []})
    for i, r in enumerate(records):
        if r.question["type"] == "bool":
            by_task[r.task][bool(r.answer)].append(i)
    drop: set[int] = set()
    rng = random.Random(seed)
    for groups in by_task.values():
        t, f = groups[True], groups[False]
        if not t or not f:
            continue
        share = len(t) / (len(t) + len(f))
        if abs(share - 0.5) <= tolerance:
            continue
        major, minor = (t, f) if len(t) > len(f) else (f, t)
        keep = set(rng.sample(major, len(minor)))
        drop.update(set(major) - keep)
    return [r for i, r in enumerate(records) if i not in drop]


def apply_templates(
    records: Iterable[QuestionRecord],
    facts: Iterable[ImageFacts],
    *,
    seed: int = 0,
    p_none: float = 0.1,
    comp_per_image: int = 2,
    balance: bool = True,
) -> list[QuestionRecord]:
    """The full templating pipeline; see the module docstring."""
    out = [rewrite(r, seed, p_none) for r in records]
    out += compositional(facts, seed=seed, per_image=comp_per_image)
    return balance_bools(out, seed=seed) if balance else out
