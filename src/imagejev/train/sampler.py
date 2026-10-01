"""Balanced batch sampler: equal domains and question types, a fixed teacher share.

A batch holds ``images_per_batch * questions_per_image`` questions. Picking images first and hoping
the question types come out even does not work: the mix would depend on which images were drawn. So
each batch gets **quotas**:

* the teacher-labelled pool takes ``teacher_fraction`` of the questions (rounded);
* within each pool, the quota is split equally across the non-empty ``(domain, question type)``
  cells, so domains and types are equal by construction (any remainder goes to random cells).

The batch is then filled one image at a time: pick a cell that still has quota, pick a random
question in it, and top the image up with up to ``questions_per_image - 1`` more of its questions
whose cells still have quota. Images are not repeated within a batch. That keeps the saving that
motivated image-first sampling (one cached image feature serves several questions) without giving up
the exact mix.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..data.records import QuestionRecord

Cell = tuple[str, str]  # (domain, question type)


@dataclass(frozen=True)
class SamplerConfig:
    images_per_batch: int = 32
    questions_per_image: int = 4
    teacher_fraction: float = 0.15
    seed: int = 0

    @property
    def batch_questions(self) -> int:
        return self.images_per_batch * self.questions_per_image


def cell_of(rec: QuestionRecord) -> Cell:
    return rec.domain, rec.question["type"]


class BalancedSampler:
    def __init__(self, records: Sequence[QuestionRecord], config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()
        if not records:
            raise ValueError("no records to sample from")
        if not 0 <= self.config.teacher_fraction < 1:
            raise ValueError("teacher_fraction must be in [0, 1)")
        self.records = list(records)
        self.pool_of = [r.soft is not None for r in self.records]  # True = teacher pool
        self.cells: dict[tuple[bool, Cell], list[int]] = defaultdict(list)
        self.by_image: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            self.cells[(self.pool_of[i], cell_of(r))].append(i)
            self.by_image[r.image_id].append(i)
        self.has_teacher = any(self.pool_of)
        self.has_regular = not all(self.pool_of)

    # -- quotas -----------------------------------------------------------------------------
    def _quotas(self, rng: random.Random) -> dict[tuple[bool, Cell], int]:
        total = self.config.batch_questions
        teacher_q = round(self.config.teacher_fraction * total) if self.has_teacher else 0
        if not self.has_regular:
            teacher_q = total
        pools = {True: teacher_q, False: total - teacher_q}
        quotas: dict[tuple[bool, Cell], int] = {}
        for pool, amount in pools.items():
            cells = sorted(c for (p, c) in self.cells if p == pool)
            if not cells or amount == 0:
                continue
            base, extra = divmod(amount, len(cells))
            for c in cells:
                quotas[(pool, c)] = base
            for c in rng.sample(cells, extra):
                quotas[(pool, c)] += 1
        return quotas

    # -- sampling ---------------------------------------------------------------------------
    def sample_batch(self, rng: random.Random) -> list[list[QuestionRecord]]:
        """One batch, a list of per-image question lists."""
        quota = self._quotas(rng)
        k = self.config.questions_per_image
        used_images: set[str] = set()
        batch: list[list[QuestionRecord]] = []

        def key(i: int) -> tuple[bool, Cell]:
            return self.pool_of[i], cell_of(self.records[i])

        while any(v > 0 for v in quota.values()):
            open_cells = [c for c, v in quota.items() if v > 0]
            cell = rng.choice(sorted(open_cells))
            candidates = [
                i for i in self.cells[cell] if self.records[i].image_id not in used_images
            ]
            if not candidates:  # every image in this cell is already in the batch: allow repeats
                candidates = self.cells[cell]
            first = rng.choice(candidates)
            image_id = self.records[first].image_id
            chosen = [first]
            quota[cell] -= 1
            extras = [i for i in self.by_image[image_id] if i != first and quota.get(key(i), 0) > 0]
            rng.shuffle(extras)
            for i in extras:
                if len(chosen) == k:
                    break
                if quota.get(key(i), 0) > 0:
                    chosen.append(i)
                    quota[key(i)] -= 1
            if image_id in used_images:  # a forced repeat joins the existing group
                next(g for g in batch if g[0].image_id == image_id).extend(
                    self.records[i] for i in chosen
                )
            else:
                used_images.add(image_id)
                batch.append([self.records[i] for i in chosen])
        return batch

    def __iter__(self) -> Iterator[list[list[QuestionRecord]]]:
        rng = random.Random(self.config.seed)
        while True:
            yield self.sample_batch(rng)
