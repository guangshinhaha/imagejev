"""Shared record formats for all data sources.

Adapters and generators emit two kinds of JSONL records:

* ``ImageFacts``: known-true attributes of an image. The templater (#13) builds compositional
  questions from these.
* ``QuestionRecord``: one typed question about one image with its known answer, plus optional
  soft labels from a teacher model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..schema import parse_question

DOMAINS = ("photo", "document", "screenshot")


@dataclass
class ImageFacts:
    image_id: str
    domain: str
    source: str
    facts: dict[str, Any]
    image_path: str | None = None
    style: str | None = None  # synthetic template/theme id, used for held-out styles

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}")


@dataclass
class QuestionRecord:
    image_id: str
    domain: str
    source: str
    task: str  # task family, e.g. "coco.object_present"; unit of held-out-task splits
    question: dict[str, Any]  # same shape as the public API question spec
    answer: str | bool
    # teacher probabilities: over the option labels, or {"true": p} for a bool question
    soft: dict[str, float] | None = None
    style: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}")
        q = parse_question(self.task, self.question)
        if q.type == "bool":
            if not isinstance(self.answer, bool):
                raise ValueError(f"{self.task}: bool answer must be True/False")
        elif self.answer not in q.labels:
            raise ValueError(f"{self.task}: answer {self.answer!r} not in options {q.labels}")
        if self.soft is not None:
            if q.type == "bool":
                if set(self.soft) != {"true"} or not 0.0 <= self.soft["true"] <= 1.0:
                    raise ValueError(
                        f"{self.task}: a bool soft label is {{'true': p}} with p in [0, 1]"
                    )
            else:
                if set(self.soft) != set(q.labels):
                    raise ValueError(f"{self.task}: soft labels must cover exactly the options")
                if abs(sum(self.soft.values()) - 1.0) > 1e-3:
                    raise ValueError(f"{self.task}: soft labels must sum to 1")


def write_jsonl(records: Iterable[Any], path: str | Path) -> int:
    n = 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: type) -> Iterator[Any]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield cls(**json.loads(line))
