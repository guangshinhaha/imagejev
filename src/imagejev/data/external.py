"""Adapters for the external zero-shot test sets (never trained on), one per domain.

* **Oxford-IIIT Pet** (photos, CC BY-SA 4.0): species, breed and "is this breed X" questions.
* **CORD-v2** (documents, CC BY 4.0): item count, tax line, service charge and payment method on
  real receipts. Totals are not used: the receipts are Indonesian and priced in rupiah, so dollar
  bins would be wrong.
* **ScreenSpot** (screenshots, Apache 2.0): which platform a screenshot comes from.

All records use ``source="ext:<name>"`` so the split builder routes them to ``test-external``.
Why these three and what was ruled out: ``docs/data-licenses.md`` and the spec (section 4.5).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from .docgen import ITEM_LEVELS, item_level
from .records import ImageFacts, QuestionRecord

PETS = "ext:oxford-iiit-pet"
CORD = "ext:cord-v2"
SCREENSPOT = "ext:screenspot"


def _rng(seed: int, *parts: str) -> random.Random:
    h = hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---- Oxford-IIIT Pet ------------------------------------------------------------------------
def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def iter_oxford_pets(
    rows: Iterable[Mapping[str, Any]],
    breeds: Sequence[str],
    species_of_breed: Mapping[int, str],
    *,
    split: str = "test",
    seed: int = 0,
    option_counts: tuple[int, ...] = (4, 8),
) -> Iterator[ImageFacts | QuestionRecord]:
    """Rows need ``image_id``, ``label`` (index into ``breeds``) and optionally ``image_path``.

    ``species_of_breed`` maps a breed index to ``"cat"`` or ``"dog"``; build it from the
    ``label_cat_dog`` column. Distractors for the breed question come from the same species first.
    """
    by_species: dict[str, list[int]] = {}
    for idx, sp in species_of_breed.items():
        by_species.setdefault(sp, []).append(idx)
    for row in rows:
        iid = f"pets:{split}:{row['image_id']}"
        label = int(row["label"])
        species = species_of_breed[label]
        breed = pretty(breeds[label])
        yield ImageFacts(
            iid,
            "photo",
            PETS,
            {"breed": breed, "species": species},
            image_path=row.get("image_path"),
        )
        rng = _rng(seed, iid)

        def rec(task: str, q: dict[str, Any], answer: Any, _iid: str = iid) -> QuestionRecord:
            return QuestionRecord(_iid, "photo", PETS, task, q, answer)

        yield rec(
            "ext.pets.species",
            {
                "type": "choice",
                "instructions": "Is the pet in this photo a cat or a dog?",
                "criteria": {"cat": "a cat", "dog": "a dog"},
            },
            species,
        )
        same = [i for i in by_species[species] if i != label]
        other = [i for sp, ids in by_species.items() if sp != species for i in ids]
        k = rng.choice(option_counts)
        rng.shuffle(same)
        rng.shuffle(other)
        picks = [label, *(same + other)[: k - 1]]
        rng.shuffle(picks)
        yield rec(
            "ext.pets.breed",
            {
                "type": "choice",
                "instructions": "Which breed is the pet in this photo?",
                "criteria": {pretty(breeds[i]): "" for i in picks},
            },
            breed,
        )
        asked = label if rng.random() < 0.5 else rng.choice(same)
        yield rec(
            "ext.pets.is_breed",
            {
                "type": "bool",
                "instructions": f"Is the pet in this photo a {pretty(breeds[asked])}?",
            },
            asked == label,
        )


# ---- CORD-v2 --------------------------------------------------------------------------------
def _as_list(menu: Any) -> list[Any]:
    if menu is None:
        return []
    return list(menu) if isinstance(menu, list) else [menu]


def cord_facts(ground_truth: str | Mapping[str, Any]) -> dict[str, Any]:
    """Exact receipt facts from a CORD ``ground_truth`` JSON (string or parsed)."""
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    parse = gt["gt_parse"]
    items = _as_list(parse.get("menu"))
    sub = parse.get("sub_total") or {}
    total = parse.get("total") or {}
    cash, card = "cashprice" in total, "creditcardprice" in total
    return {
        "n_items": len(items),
        "has_tax": "tax_price" in sub,
        "has_service_charge": "service_price" in sub,
        "paid_cash": cash if cash != card else None,  # only when exactly one method is shown
    }


def iter_cord(
    rows: Iterable[Mapping[str, Any]], *, split: str = "test"
) -> Iterator[ImageFacts | QuestionRecord]:
    """Rows need ``id``, ``ground_truth`` and optionally ``image_path``."""
    for row in rows:
        iid = f"cord:{split}:{row['id']}"
        f = cord_facts(row["ground_truth"])
        yield ImageFacts(iid, "document", CORD, f, image_path=row.get("image_path"))

        def rec(task: str, q: dict[str, Any], answer: Any, _iid: str = iid) -> QuestionRecord:
            return QuestionRecord(_iid, "document", CORD, task, q, answer)

        def boolq(task: str, text: str, answer: bool) -> QuestionRecord:
            return rec(task, {"type": "bool", "instructions": text}, answer)

        if f["n_items"] > 0:
            yield rec(
                "ext.cord.item_count",
                {
                    "type": "score",
                    "instructions": "How many line items are listed?",
                    "levels": ITEM_LEVELS,
                },
                item_level(f["n_items"]),
            )
        yield boolq("ext.cord.has_tax", "Does the receipt show a tax line?", f["has_tax"])
        yield boolq(
            "ext.cord.has_service_charge",
            "Does the receipt include a service charge?",
            f["has_service_charge"],
        )
        if f["paid_cash"] is not None:
            yield boolq("ext.cord.paid_cash", "Was this receipt paid in cash?", f["paid_cash"])


# ---- ScreenSpot -----------------------------------------------------------------------------
PLATFORMS = {
    "ios": "an iPhone or iPad app screen",
    "android": "an Android app screen",
    "macos": "a macOS desktop application",
    "windows": "a Windows desktop application",
    "web": "a web page in a browser",
}
_WEB_WORDS = ("web", "gitlab", "shop", "forum", "tool")


def normalize_platform(data_source: str) -> str | None:
    """Map a ScreenSpot ``data_source`` to one of ``PLATFORMS``, or ``None`` if unrecognised."""
    s = data_source.strip().lower()
    if "ios" in s or "iphone" in s or "ipad" in s:
        return "ios"
    if "android" in s:
        return "android"
    if "mac" in s:
        return "macos"
    if "windows" in s:
        return "windows"
    return "web" if any(w in s for w in _WEB_WORDS) else None


def iter_screenspot(
    rows: Iterable[Mapping[str, Any]], *, split: str = "test"
) -> Iterator[ImageFacts | QuestionRecord]:
    """One platform question per *distinct screenshot* (it has several instructions in the data).

    Rows need ``file_name``, ``data_source`` and optionally ``image_path``. A screenshot with
    conflicting platform labels, or an unrecognised one, is skipped.
    """
    labels: dict[str, set[str | None]] = {}
    first: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        labels.setdefault(row["file_name"], set()).add(normalize_platform(row["data_source"]))
        first.setdefault(row["file_name"], row)
    for name, found in labels.items():
        if len(found) != 1 or None in found:
            continue
        (platform,) = found
        iid = f"screenspot:{split}:{name}"
        yield ImageFacts(iid, "screenshot", SCREENSPOT, {"platform": platform},
                         image_path=first[name].get("image_path"))  # fmt: skip
        yield QuestionRecord(
            iid,
            "screenshot",
            SCREENSPOT,
            "ext.screenspot.platform",
            {
                "type": "choice",
                "instructions": "Which kind of interface is shown in this screenshot?",
                "criteria": dict(PLATFORMS),
            },
            platform,
        )
