"""Split builder: train / val / test-images / test-tasks / test-styles / test-external.

Rules, in order, applied to *image groups* (a group is a source image and everything derived from
it, e.g. ``coco:1`` and ``coco:1+blur``):

1. A group from an external dataset (``source`` starts with ``ext:``) -> ``test-external``.
2. A group whose synthetic style matches ``style_patterns`` -> ``test-styles``.
3. Every other group is hashed into exactly one of train / val / test-images / test-tasks.

Every group lands in exactly one split, so no image ever appears in two. On top of that:

* Held-out **test task families** (``heldout.json``) are only kept in ``test-tasks``, and
  **validation task families** only in ``val-tasks``. Their questions on images assigned elsewhere
  are dropped (and counted), never moved, so they can't leak into training and no image is split.
* ``test-tasks`` keeps *only* test families and ``val-tasks`` *only* validation families, so each
  measures criteria never seen in training. The two sets of families are disjoint: model selection
  uses ``val-tasks`` and never sees the test families.
* The other splits never contain a held-out family of either kind.

``check_splits`` re-verifies all of this on a finished assignment.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from .records import ImageFacts, QuestionRecord, write_jsonl

SPLITS = (
    "train",
    "val",
    "val-tasks",
    "test-images",
    "test-tasks",
    "test-styles",
    "test-external",
)
HASHED = ("train", "val", "val-tasks", "test-images", "test-tasks")


@cache
def load_heldout() -> dict[str, Any]:
    return json.loads(resources.files("imagejev.data").joinpath("heldout.json").read_text())


def group_id(rec: QuestionRecord | ImageFacts) -> str:
    """The source image a record belongs to (derived images map back to their source)."""
    if isinstance(rec, QuestionRecord):
        sid = rec.meta.get("source_id")
    else:
        sid = rec.facts.get("source_id")
    return str(sid) if sid else rec.image_id


def is_heldout_style(style: str | None, patterns: Iterable[str]) -> bool:
    return style is not None and any(fnmatch.fnmatchcase(style, p) for p in patterns)


def _bucket(seed: int, group: str) -> float:
    h = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(h[:12], 16) / 16**12


def assign_group(
    group: str,
    *,
    seed: int,
    external: bool,
    heldout_style: bool,
    fractions: dict[str, float],
) -> str:
    if external:
        return "test-external"
    if heldout_style:
        return "test-styles"
    x = _bucket(seed, group) * sum(fractions.values())
    acc = 0.0
    for name in HASHED:
        acc += fractions[name]
        if x < acc:
            return name
    return HASHED[-1]


@dataclass
class SplitResult:
    splits: dict[str, list[QuestionRecord]]
    group_split: dict[str, str]
    dropped: Counter = field(default_factory=Counter)
    heldout_tasks: tuple[str, ...] = ()
    val_tasks: tuple[str, ...] = ()

    def thin_families(self, min_size: int = 100) -> dict[str, int]:
        """Test-held-out families with fewer than ``min_size`` test-tasks questions.

        Each family's test size is only ``test-tasks`` share (8%) of its prevalence, so a family
        that is rare in the data can be too small to measure anything. Generate more data, or pick
        a more common family, before trusting a held-out-task result.
        """
        n = Counter(r.task for r in self.splits["test-tasks"])
        return {t: n[t] for t in self.heldout_tasks if n[t] < min_size}

    def thin_val_families(self, min_size: int = 100) -> dict[str, int]:
        """The same check for the validation families in ``val-tasks``."""
        n = Counter(r.task for r in self.splits["val-tasks"])
        return {t: n[t] for t in self.val_tasks if n[t] < min_size}

    def report(self, min_family_size: int = 100) -> dict[str, Any]:
        table: dict[str, Counter] = {s: Counter() for s in SPLITS}
        for s, recs in self.splits.items():
            for r in recs:
                table[s][f"{r.domain}/{r.question['type']}"] += 1
        groups = Counter(self.group_split.values())
        return {
            "questions": {s: sum(c.values()) for s, c in table.items()},
            "by_domain_and_type": {s: dict(sorted(c.items())) for s, c in table.items()},
            "image_groups": {s: groups.get(s, 0) for s in SPLITS},
            "dropped": dict(self.dropped),
            "test_tasks_per_family": dict(
                sorted(Counter(r.task for r in self.splits["test-tasks"]).items())
            ),
            "thin_families": self.thin_families(min_family_size),
            "val_tasks_per_family": dict(
                sorted(Counter(r.task for r in self.splits["val-tasks"]).items())
            ),
            "thin_val_families": self.thin_val_families(min_family_size),
        }


def build_splits(
    records: Iterable[QuestionRecord],
    facts: Iterable[ImageFacts] = (),
    *,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> SplitResult:
    cfg = config or load_heldout()
    heldout_tasks = set(cfg["task_families"])
    val_tasks = set(cfg.get("val_task_families", {}))
    if heldout_tasks & val_tasks:
        raise ValueError(
            f"families in both the test and validation sets: {heldout_tasks & val_tasks}"
        )
    patterns = cfg["style_patterns"]
    ext_prefix = cfg["external_source_prefix"]
    fractions = cfg["group_fractions"]

    # A group's style and origin come from any of its records or facts.
    style_of: dict[str, str | None] = {}
    external: dict[str, bool] = defaultdict(bool)
    records = list(records)
    for item in [*records, *facts]:
        g = group_id(item)
        if item.style and g not in style_of:
            style_of[g] = item.style
        if item.source.startswith(ext_prefix):
            external[g] = True
    group_split: dict[str, str] = {}
    for item in [*records, *facts]:
        g = group_id(item)
        if g not in group_split:
            group_split[g] = assign_group(
                g,
                seed=seed,
                external=external[g],
                heldout_style=is_heldout_style(style_of.get(g), patterns),
                fractions=fractions,
            )
    splits: dict[str, list[QuestionRecord]] = {s: [] for s in SPLITS}
    dropped: Counter = Counter()
    for r in records:
        s = group_split[group_id(r)]
        if s in ("test-tasks", "val-tasks"):
            own = heldout_tasks if s == "test-tasks" else val_tasks
            if r.task in own:
                splits[s].append(r)
            else:
                dropped[f"seen-task question on a {s} image"] += 1
        elif (r.task in heldout_tasks or r.task in val_tasks) and s != "test-external":
            dropped["held-out task question on another split's image"] += 1
        else:
            splits[s].append(r)
    return SplitResult(
        splits, group_split, dropped, tuple(sorted(heldout_tasks)), tuple(sorted(val_tasks))
    )


def check_splits(result: SplitResult, config: dict[str, Any] | None = None) -> None:
    """Raise ``AssertionError`` if any split rule is broken."""
    cfg = config or load_heldout()
    heldout_tasks = set(cfg["task_families"])
    val_tasks = set(cfg.get("val_task_families", {}))
    seen: dict[str, str] = {}
    for s, recs in result.splits.items():
        for r in recs:
            g = group_id(r)
            assert seen.setdefault(g, s) == s, f"image group {g} is in {seen[g]} and {s}"
            assert result.group_split[g] == s, f"{g} assigned {result.group_split[g]} but in {s}"
            if s == "test-tasks":
                assert r.task in heldout_tasks, f"seen task {r.task} in test-tasks"
            elif s == "val-tasks":
                assert r.task in val_tasks, f"seen task {r.task} in val-tasks"
            elif s != "test-external":
                assert r.task not in heldout_tasks, f"held-out task {r.task} in {s}"
                assert r.task not in val_tasks, f"validation task {r.task} in {s}"
            if s in ("train", "val", "val-tasks", "test-images", "test-tasks"):
                assert not is_heldout_style(r.style, cfg["style_patterns"]), f"style leak in {s}"
            is_ext = r.source.startswith(cfg["external_source_prefix"])
            assert is_ext == (s == "test-external"), f"{r.source} in {s}"


def write_splits(result: SplitResult, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    for s, recs in result.splits.items():
        write_jsonl(recs, out / f"{s}.jsonl")
    rep = result.report()
    (out / "splits_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep
