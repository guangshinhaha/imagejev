"""Turn sampled questions into model inputs: text features, cached image features, targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..data.records import QuestionRecord
from ..model.fusion import QTYPE_IDS, FusionBatch
from ..model.text import TextEncoder
from ..schema import Question, parse_question


@dataclass
class Prepared:
    batch: FusionBatch
    targets: torch.Tensor  # (N,) one entry per option; bool: P(true)
    questions: list[Question]  # parsed, in the same order as ``batch.qtype``
    records: list[QuestionRecord]


def option_texts(q: Question) -> list[str | None]:
    """Second segment of the sentence pair for each option; ``bool`` is instructions only."""
    return [None] if q.type == "bool" else [o.text() for o in q.options]


def target_vector(rec: QuestionRecord, q: Question) -> list[float]:
    if q.type == "bool":
        return [1.0 if rec.answer else 0.0]
    if rec.soft is not None:
        return [float(rec.soft[label]) for label in q.labels]
    t = [0.0] * len(q.options)
    t[q.labels.index(rec.answer)] = 1.0  # type: ignore[arg-type]
    return t


class BatchBuilder:
    """Builds a ``Prepared`` batch. ``features`` maps ``image_id`` to ``(65, d_image)`` features
    (a ``FeatureCache`` or any mapping)."""

    def __init__(self, features: Mapping[str, Any], text: TextEncoder, device: torch.device | str):
        self.features = features
        self.text = text
        self.device = torch.device(device)

    def _image(self, image_id: str) -> np.ndarray:
        return np.asarray(self.features[image_id])

    def build(self, records: Sequence[QuestionRecord]) -> Prepared:
        questions = [parse_question(r.task, r.question) for r in records]
        firsts: list[str] = []
        seconds: list[str | None] = []
        owner: list[int] = []
        slot: list[int] = []
        targets: list[float] = []
        for qi, (rec, q) in enumerate(zip(records, questions, strict=True)):
            segs = option_texts(q)
            firsts += [q.instructions] * len(segs)
            seconds += segs
            owner += [qi] * len(segs)
            slot += list(range(len(segs)))
            targets += target_vector(rec, q)
        enc = self.text.encode_pairs(firsts, seconds)
        image = torch.from_numpy(np.stack([self._image(r.image_id) for r in records])).float()
        batch = FusionBatch(
            opt_tokens=enc.tokens,
            opt_mask=enc.mask,
            opt_focus=enc.focus,
            owner=torch.tensor(owner, device=self.device),
            slot=torch.tensor(slot, device=self.device),
            qtype=torch.tensor([QTYPE_IDS[q.type] for q in questions], device=self.device),
            image=image.to(self.device),
        )
        return Prepared(batch, torch.tensor(targets, device=self.device), questions, list(records))
