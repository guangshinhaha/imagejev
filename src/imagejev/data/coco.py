"""COCO instances -> photo questions, and VQAv2 yes/no -> bool questions."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .records import ImageFacts, QuestionRecord

SOURCE = "coco"
COUNT_LEVELS = ["none", "one", "a few", "many"]


def count_level(n: int) -> str:
    if n <= 0:
        return COUNT_LEVELS[0]
    if n == 1:
        return COUNT_LEVELS[1]
    return COUNT_LEVELS[2] if n <= 4 else COUNT_LEVELS[3]


def _rng(seed: int, image_id: str, salt: str = "") -> random.Random:
    h = hashlib.sha256(f"{seed}:{image_id}:{salt}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _article(name: str) -> str:
    return "an" if name[0] in "aeiou" else "a"


def coco_image_id(raw_id: int) -> str:
    return f"coco:{int(raw_id):012d}"


def load_instances(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def iter_coco(
    instances: dict[str, Any],
    *,
    seed: int = 0,
    min_area_frac: float = 0.005,
    min_dominance: float = 0.6,
    bool_per_image: int = 2,
    count_per_image: int = 1,
    image_dir: str | None = None,
) -> Iterator[ImageFacts | QuestionRecord]:
    """Yield an ``ImageFacts`` and several ``QuestionRecord``s per annotated image.

    Only non-crowd instances covering at least ``min_area_frac`` of the image count as visible,
    so labels aren't decided by specks. Images with no visible objects are skipped. The
    dominant-supercategory question is only asked when one supercategory covers at least
    ``min_dominance`` of the visible object area, so its label is unambiguous.
    """
    cats = {c["id"]: c for c in instances["categories"]}
    names = sorted(c["name"] for c in cats.values())
    supers = sorted({c["supercategory"] for c in cats.values()})
    by_super: dict[str, list[str]] = defaultdict(list)
    super_of: dict[str, str] = {}
    for c in cats.values():
        by_super[c["supercategory"]].append(c["name"])
        super_of[c["name"]] = c["supercategory"]
    anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in instances["annotations"]:
        anns[a["image_id"]].append(a)

    # COCO supercategory names ("outdoor", "accessory") mean little alone, so describe each one
    # by a few of its member categories; the answer is then decidable from the text.
    super_desc = {s: ", ".join(sorted(by_super[s])[:6]) for s in supers}

    for img in sorted(instances["images"], key=lambda i: i["id"]):
        area = img["width"] * img["height"]
        vis = [
            a
            for a in anns.get(img["id"], [])
            if not a.get("iscrowd", 0) and a["area"] >= min_area_frac * area
        ]
        if not vis:
            continue
        iid = coco_image_id(img["id"])
        counts: dict[str, int] = defaultdict(int)
        super_area: dict[str, float] = defaultdict(float)
        for a in vis:
            c = cats[a["category_id"]]
            counts[c["name"]] += 1
            super_area[c["supercategory"]] += a["area"]
        path = f"{image_dir}/{img['file_name']}" if image_dir else img.get("file_name")
        top_super = max(super_area, key=super_area.__getitem__)
        facts = {
            "counts": dict(sorted(counts.items())),
            "top_supercategory": top_super,
            "top_supercategory_share": super_area[top_super] / sum(super_area.values()),
        }
        yield ImageFacts(iid, "photo", SOURCE, facts, image_path=path)
        rng = _rng(seed, iid)
        present = sorted(counts)
        absent = [n for n in names if n not in counts]

        def rec(
            task: str, question: dict[str, Any], answer: Any, _iid: str = iid, **meta: Any
        ) -> QuestionRecord:
            return QuestionRecord(_iid, "photo", SOURCE, task, question, answer, meta=meta)

        # bool: object present, half positives and half hard negatives (same supercategory first)
        n_pos = min(len(present), (bool_per_image + 1) // 2)
        for name in rng.sample(present, n_pos):
            yield rec(
                "coco.object_present",
                {"type": "bool", "instructions": f"Is there {_article(name)} {name} in the image?"},
                True,
                category=name,
            )
        for _ in range(bool_per_image - n_pos):
            pool = [
                n for s in {super_of[p] for p in present} for n in by_super[s] if n in absent
            ] or absent
            if not pool:
                break
            name = rng.choice(sorted(pool))
            yield rec(
                "coco.object_present",
                {"type": "bool", "instructions": f"Is there {_article(name)} {name} in the image?"},
                False,
                category=name,
            )

        # score: how many of a category (one present category, plus sometimes an absent one)
        picks = rng.sample(present, min(len(present), count_per_image))
        if absent and rng.random() < 0.25:
            picks[-1:] = [rng.choice(absent)] if picks else []
        for name in picks:
            yield rec(
                "coco.count_bin",
                {
                    "type": "score",
                    "instructions": f"How many {name} instances are in the image?",
                    "levels": COUNT_LEVELS,
                },
                count_level(counts.get(name, 0)),
                category=name,
            )

        # choice: which kind of object takes up the most of the image (only when clear-cut)
        if facts["top_supercategory_share"] >= min_dominance:
            others = [s for s in supers if s != top_super]
            options = [top_super, *rng.sample(others, min(3, len(others)))]
            rng.shuffle(options)
            yield rec(
                "coco.dominant_supercategory",
                {
                    "type": "choice",
                    "instructions": "Which kind of object takes up the most of the image?",
                    "criteria": {o: super_desc[o] for o in options},
                },
                top_super,
            )


def iter_vqav2_yesno(
    questions: dict[str, Any],
    annotations: dict[str, Any],
    *,
    max_per_answer: int | None = None,
) -> Iterator[QuestionRecord]:
    """VQAv2 yes/no questions as bool records with the human consensus answer.

    ``max_per_answer`` caps yes and no separately, keeping the set balanced.
    """
    qtext = {q["question_id"]: q for q in questions["questions"]}
    taken = {"yes": 0, "no": 0}
    for a in annotations["annotations"]:
        if a.get("answer_type") != "yes/no":
            continue
        ans = a["multiple_choice_answer"].strip().lower()
        if ans not in taken:
            continue
        if max_per_answer is not None and taken[ans] >= max_per_answer:
            continue
        q = qtext[a["question_id"]]
        taken[ans] += 1
        yield QuestionRecord(
            coco_image_id(a["image_id"]),
            "photo",
            "vqav2",
            "vqav2.yesno",
            {"type": "bool", "instructions": q["question"].strip()},
            ans == "yes",
            meta={"question_id": a["question_id"]},
        )
