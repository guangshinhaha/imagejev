"""More task families from stored facts, so the model sees many ways to ask about each domain.

The pilot model failed on held-out task families, and the likeliest cause is how few families each
(domain, question type) cell had in training: with one score family per domain it can learn "score
means blur" instead of "do what the question says". These add families whose answers are decided by
facts we already hold, so every label is exact and no new images (or image features) are needed.

Each family yields the same thing as the other generators, a ``QuestionRecord`` with an exact
answer. A family whose truth the facts cannot give is skipped for that image, never guessed.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from typing import Any

from .records import ImageFacts, QuestionRecord


def _rng(seed: int, image_id: str) -> random.Random:
    h = hashlib.sha256(f"facts|{seed}|{image_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _bin(n: int, edges: list[int], levels: list[str]) -> str:
    """``levels[i]`` where ``n`` first falls at or below ``edges[i]``; the last level is open."""
    for edge, level in zip(edges, levels, strict=False):
        if n <= edge:
            return level
    return levels[-1]


# ---- photos (COCO facts) -------------------------------------------------------------------
VARIETY_LEVELS = ["one kind", "two kinds", "three to five kinds", "six or more kinds"]
PERSON_LEVELS = ["no people", "one person", "a few people", "many people"]
TOTAL_LEVELS = ["one or two", "three to five", "six to ten", "more than ten"]


def _photo(f: ImageFacts) -> list[tuple[str, dict[str, Any], str]]:
    d = f.facts
    if "counts" not in d or not d["counts"]:
        return []
    counts: dict[str, int] = d["counts"]
    crowd = set(d.get("crowd", []))
    exact = d.get("counts_all", {})
    out: list[tuple[str, dict[str, Any], str]] = []
    out.append(
        (
            "fact.photo.object_variety",
            {"type": "score", "levels": VARIETY_LEVELS,
             "instructions": "How many different kinds of object are in the photo?"},
            _bin(len(counts), [1, 2, 5], VARIETY_LEVELS),
        )
    )  # fmt: skip
    # people: only when no hidden or crowd-labelled person could change the count
    persons = counts.get("person", 0)
    if "person" not in crowd and exact.get("person", 0) == persons:
        out.append(
            (
                "fact.photo.person_count",
                {"type": "score", "instructions": "How many people are in the photo?",
                 "levels": PERSON_LEVELS},
                _bin(persons, [0, 1, 4], PERSON_LEVELS),
            )
        )  # fmt: skip
    if (
        not crowd
        and all(exact.get(k) == v for k, v in counts.items())
        and set(exact) == set(counts)
    ):
        out.append(
            (
                "fact.photo.total_objects",
                {"type": "score", "levels": TOTAL_LEVELS,
                 "instructions": "Roughly how many objects are in the photo in all?"},
                _bin(sum(counts.values()), [2, 5, 10], TOTAL_LEVELS),
            )
        )  # fmt: skip
    return out


# ---- screenshots (generated pages) ---------------------------------------------------------
OVERLAY_LEVELS = ["none", "one", "two", "three or four"]
OVERLAYS = {
    "modal_open": ("a dialog box", "a popup dialog covering part of the page"),
    "cookie_banner": ("a cookie banner", "a consent notice at the edge of the page"),
    "error_banner": ("an error banner", "a red error message across the top"),
    "loading": ("a loading spinner", "a spinner over the page"),
}


def _screen(f: ImageFacts) -> list[tuple[str, dict[str, Any], str]]:
    d = f.facts
    if "page_type" not in d:
        return []
    shown = [k for k in OVERLAYS if d.get(k)]
    out: list[tuple[str, dict[str, Any], str]] = [
        (
            "fact.screen.overlay_count",
            {"type": "score", "instructions": "How many popups, banners or overlays are showing?",
             "levels": OVERLAY_LEVELS},
            _bin(len(shown), [0, 1, 2], OVERLAY_LEVELS),
        ),
        (
            "fact.screen.viewport",
            {"type": "choice", "instructions": "What kind of screen is this page shown on?",
             "criteria": {"desktop": "a wide computer window",
                          "mobile": "a narrow phone-sized screen"}},
            str(d["viewport"]),
        ),
    ]  # fmt: skip
    if len(shown) == 1:
        out.append(
            (
                "fact.screen.overlay_kind",
                {"type": "choice", "instructions": "Which overlay is showing on the page?",
                 "criteria": {OVERLAYS[k][0]: OVERLAYS[k][1] for k in OVERLAYS}},
                OVERLAYS[shown[0]][0],
            )
        )  # fmt: skip
    return out


# ---- documents (generated) -----------------------------------------------------------------
TAX_LEVELS = ["no tax", "low tax", "medium tax", "high tax"]
FEATURE_LEVELS = ["none of them", "one of them", "two of them", "all three"]


def _document(f: ImageFacts) -> list[tuple[str, dict[str, Any], str]]:
    d = f.facts
    if "total_cents" not in d:
        return []
    n_feat = int(bool(d["has_signature"])) + int(bool(d["has_stamp"])) + int(bool(d["has_table"]))
    out: list[tuple[str, dict[str, Any], str]] = [
        (
            "fact.doc.feature_count",
            {"type": "score",
             "instructions": "How many of these does the page have: a signature, a stamp, a table?",
             "levels": FEATURE_LEVELS},
            FEATURE_LEVELS[n_feat],
        )
    ]  # fmt: skip
    if d.get("tax_pct") is not None:
        out.append(
            (
                "fact.doc.tax_level",
                {"type": "score", "instructions": "How much tax is charged on this document?",
                 "levels": TAX_LEVELS},
                _bin(int(d["tax_pct"]), [0, 5, 8], TAX_LEVELS),
            )
        )  # fmt: skip
    return out


_BY_DOMAIN = {"photo": _photo, "screenshot": _screen, "document": _document}


def generate(
    facts: Iterable[ImageFacts], *, seed: int = 0, per_image: int = 2
) -> list[QuestionRecord]:
    """Up to ``per_image`` of the applicable fact-based questions for each image."""
    out: list[QuestionRecord] = []
    for f in facts:
        make = _BY_DOMAIN.get(f.domain)
        if make is None:
            continue
        cands = make(f)
        _rng(seed, f.image_id).shuffle(cands)
        for task, question, answer in cands[:per_image]:
            out.append(
                QuestionRecord(
                    f.image_id, f.domain, f.source, task, question, answer, style=f.style
                )
            )
    return out
