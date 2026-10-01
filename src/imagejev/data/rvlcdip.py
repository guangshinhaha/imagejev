"""RVL-CDIP -> document-type choice questions (16 classes)."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .records import ImageFacts, QuestionRecord

SOURCE = "rvl_cdip"

# Class order matches the dataset's integer labels.
CLASSES = [
    ("letter", "a formal letter addressed to a person or company"),
    ("form", "a structured form with fields to fill in"),
    ("email", "a printed email with sender, recipient and subject lines"),
    ("handwritten", "a page of handwritten text"),
    ("advertisement", "a promotional advertisement for a product or service"),
    ("scientific report", "a technical or scientific report"),
    ("scientific publication", "a published research paper with abstract and references"),
    ("specification", "a technical specification or standards document"),
    ("file folder", "a file folder or cover sheet with a label tab"),
    ("news article", "a news or magazine article"),
    ("budget", "a budget or financial table"),
    ("invoice", "an invoice or bill for goods or services"),
    ("presentation", "a slide from a presentation"),
    ("questionnaire", "a survey or questionnaire with questions to answer"),
    ("resume", "a resume or CV"),
    ("memo", "an internal memo with To / From / Date / Subject lines"),
]
NAMES = [n for n, _ in CLASSES]
DESCRIPTIONS = dict(CLASSES)

# Classes people (and models) mix up most; used to pick hard distractors.
CONFUSABLE_GROUPS = [
    {"letter", "memo", "email"},
    {"scientific report", "scientific publication", "specification"},
    {"form", "questionnaire", "budget", "invoice"},
    {"advertisement", "news article", "presentation"},
    {"resume", "handwritten", "file folder"},
]


def class_name(label: int | str) -> str:
    if isinstance(label, str):
        if label not in DESCRIPTIONS:
            raise ValueError(f"unknown RVL-CDIP class {label!r}")
        return label
    if not 0 <= int(label) < len(NAMES):
        raise ValueError(f"RVL-CDIP label out of range: {label}")
    return NAMES[int(label)]


def _rng(seed: int, image_id: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def pick_options(answer: str, k: int, rng: random.Random) -> list[str]:
    """``answer`` plus ``k - 1`` distractors, confusable classes first, in random order."""
    if not 2 <= k <= len(NAMES):
        raise ValueError(f"k must be between 2 and {len(NAMES)}")
    mates = sorted({c for g in CONFUSABLE_GROUPS if answer in g for c in g} - {answer})
    rest = sorted(set(NAMES) - set(mates) - {answer})
    rng.shuffle(mates)
    rng.shuffle(rest)
    options = [answer, *(mates + rest)[: k - 1]]
    rng.shuffle(options)
    return options


def iter_rvl_cdip(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    seed: int = 0,
    option_counts: tuple[int, ...] = (4, 8, 16),
    image_dir: str | None = None,
) -> Iterator[ImageFacts | QuestionRecord]:
    """Yield facts and one ``rvl.doc_type`` question per row.

    Rows need ``id`` and ``label`` (integer index or class name). The number of options per
    question is drawn from ``option_counts`` so the model sees 4-way, 8-way and 16-way variants.
    """
    for row in rows:
        iid = f"rvl:{split}:{row['id']}"
        answer = class_name(row["label"])
        path = f"{image_dir}/{row['id']}.png" if image_dir else None
        yield ImageFacts(iid, "document", SOURCE, {"doc_type": answer}, image_path=path)
        rng = _rng(seed, iid)
        options = pick_options(answer, rng.choice(option_counts), rng)
        yield QuestionRecord(
            iid,
            "document",
            SOURCE,
            "rvl.doc_type",
            {
                "type": "choice",
                "instructions": "What type of document is this?",
                "criteria": {o: DESCRIPTIONS[o] for o in options},
            },
            answer,
            meta={"n_options": len(options)},
        )
