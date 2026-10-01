"""Teacher-labelled slice: judgement questions on real images, answered by a bigger VLM.

Exact labels (COCO annotations, rendered pages) can't cover judgement-style or open-world criteria
("does this look posed?", "is there a tab bar on this screen?"), so a larger VLM supplies
**soft labels**: its probabilities over the options, read from one forward pass (no generation),
exactly the way the small-VLM baseline is read.

Rules that keep these labels from corrupting evaluation:

* They are used for **training only**. The split builder drops any teacher-labelled question that
  lands outside ``train``, and :func:`label_images` is meant to be fed train-split images.
* A teacher's answer is not ground truth, so every record carries the teacher's name and its
  confidence, and a person should spot-check a sample before trusting a criterion (issue #34).
* The labeller is resumable and appends one JSON line per answer, so an overnight run can be
  stopped and continued.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..schema import Question, parse_question
from .records import QuestionRecord

SOURCE = "teacher"


@dataclass(frozen=True)
class Criterion:
    id: str
    domain: str  # "photo" or "screenshot"
    question: dict[str, Any]

    @property
    def task(self) -> str:
        return f"teacher.{self.domain}.{self.id}"


def _choice(id_: str, domain: str, text: str, options: dict[str, str]) -> Criterion:
    return Criterion(id_, domain, {"type": "choice", "instructions": text, "criteria": options})


def _score(id_: str, domain: str, text: str, levels: list[str]) -> Criterion:
    return Criterion(id_, domain, {"type": "score", "instructions": text, "levels": levels})


def _bool(id_: str, domain: str, text: str) -> Criterion:
    return Criterion(id_, domain, {"type": "bool", "instructions": text})


CRITERIA: list[Criterion] = [
    # ---- real photos: subjective or open-world, so no annotation decides them ----
    _choice(
        "scene_setting",
        "photo",
        "Where does this photo seem to be taken?",
        {
            "indoors": "inside a building or room",
            "outdoors": "outside, in the open air",
            "unclear": "it cannot be told",
        },
    ),  # fmt: skip
    _choice(
        "time_of_day",
        "photo",
        "What time of day does the photo show?",
        {
            "daytime": "daylight",
            "night": "dark, at night",
            "indoor lighting": "lit by artificial light indoors",
            "unclear": "it cannot be told",
        },
    ),  # fmt: skip
    _choice(
        "shot_type",
        "photo",
        "How is the photo framed?",
        {
            "close-up": "the subject fills the frame",
            "medium shot": "the subject with some surroundings",
            "wide shot": "a broad view of the whole scene",
        },
    ),  # fmt: skip
    _choice(
        "mood",
        "photo",
        "What mood does the photo give?",
        {
            "cheerful": "happy and bright",
            "calm": "peaceful and quiet",
            "tense": "stressful or dramatic",
            "neutral": "no particular mood",
        },
    ),  # fmt: skip
    _choice(
        "natural_or_built",
        "photo",
        "What does the scene mostly consist of?",
        {
            "natural": "plants, animals, sky, water",
            "man-made": "buildings, vehicles, objects",
            "mixed": "both in similar amounts",
        },
    ),  # fmt: skip
    _choice(
        "dominant_colour",
        "photo",
        "Which colour dominates the photo?",
        {
            "red": "",
            "green": "",
            "blue": "",
            "yellow": "",
            "brown": "",
            "gray": "",
            "white": "",
            "black": "",
        },
    ),  # fmt: skip
    _score(
        "photo_quality",
        "photo",
        "How good a photograph is this overall?",
        ["poor", "fair", "good", "excellent"],
    ),  # fmt: skip
    _score("busyness", "photo", "How busy is the scene?", ["empty", "sparse", "busy", "crowded"]),
    _bool("readable_text", "photo", "Is there any readable text in the photo?"),
    _bool("looks_posed", "photo", "Does the photo look posed rather than candid?"),
    _bool("subject_centered", "photo", "Is the main subject in the middle of the frame?"),
    _bool("looks_professional", "photo", "Does this look like a professional photograph?"),
    _bool("cluttered", "photo", "Is the scene cluttered?"),
    # ---- real app screens: UI judgements Rico's class names cannot give reliably ----
    _bool("has_tab_bar", "screenshot", "Does this screen have a tab bar?"),
    _bool("has_bottom_navigation", "screenshot", "Does this screen have a bottom navigation bar?"),
    _bool("has_search_bar", "screenshot", "Is there a search bar on this screen?"),
    _bool("shows_list", "screenshot", "Does the screen show a list of items?"),
    _bool("shows_map", "screenshot", "Does the screen show a map?"),
    _bool("has_form_fields", "screenshot", "Does the screen have fields the user can type into?"),
    _bool("shows_ad", "screenshot", "Is an advertisement visible on this screen?"),
    _bool("shows_dialog", "screenshot", "Is a dialog or popup open on this screen?"),
    _bool("image_heavy", "screenshot", "Is this screen mostly made of images?"),
    _choice(
        "screen_purpose",
        "screenshot",
        "What is this screen for?",
        {
            "login or sign-up": "entering credentials or creating an account",
            "settings": "changing preferences",
            "feed or list": "browsing a list or feed of items",
            "article or detail": "reading or viewing one item",
            "media player": "playing audio or video",
            "map or navigation": "finding places or directions",
            "shopping": "browsing or buying products",
            "messaging": "chatting",
            "other": "none of the above",
        },
    ),  # fmt: skip
    _choice(
        "theme",
        "screenshot",
        "Is the screen's colour scheme light or dark?",
        {"light": "mostly light backgrounds", "dark": "mostly dark backgrounds"},
    ),  # fmt: skip
    _score(
        "density",
        "screenshot",
        "How much is on the screen?",
        ["sparse", "moderate", "dense", "very dense"],
    ),  # fmt: skip
]
BY_ID = {c.task: c for c in CRITERIA}


class Teacher(Protocol):
    name: str

    def probs(self, image: Any, question: Question) -> list[float]:
        """Probabilities over the options (sum 1); for bool, ``[p_true]``."""
        ...


@dataclass(frozen=True)
class Job:
    image_id: str
    image: Any  # a path, or anything the teacher accepts
    domain: str
    criterion: Criterion

    @property
    def key(self) -> str:
        return f"{self.image_id}|{self.criterion.task}"


def build_jobs(
    images: Iterable[tuple[str, Any, str]],
    *,
    per_image: int = 3,
    seed: int = 0,
    criteria: Sequence[Criterion] = CRITERIA,
) -> list[Job]:
    """``per_image`` criteria for each ``(image_id, image, domain)``, spread evenly.

    Each image's criteria are a deterministic hash-ordered pick of those that fit its domain, so
    every criterion gets roughly the same number of images.
    """
    by_domain: dict[str, list[Criterion]] = {}
    for c in criteria:
        by_domain.setdefault(c.domain, []).append(c)
    jobs: list[Job] = []
    for image_id, image, domain in images:
        pool = by_domain.get(domain, [])
        h = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
        rng = random.Random(int(h[:16], 16))
        for c in rng.sample(pool, min(per_image, len(pool))):
            jobs.append(Job(image_id, image, domain, c))
    return jobs


def _record(job: Job, teacher: Teacher, probs: list[float]) -> QuestionRecord:
    q = parse_question(job.criterion.task, job.criterion.question)
    if q.type == "bool":
        p = min(1.0, max(0.0, float(probs[0])))
        soft = {"true": p}
        answer: Any = p >= 0.5
        confidence = max(p, 1 - p)
    else:
        total = sum(probs)
        norm = [p / total for p in probs]
        soft = dict(zip(q.labels, norm, strict=True))
        best = max(range(len(norm)), key=norm.__getitem__)
        answer, confidence = q.labels[best], norm[best]
    return QuestionRecord(
        job.image_id,
        job.domain,
        SOURCE,
        job.criterion.task,
        job.criterion.question,
        answer,
        soft=soft,
        meta={
            "teacher": teacher.name,
            "confidence": round(confidence, 4),
            "criterion": job.criterion.id,
        },
    )


def label_images(
    jobs: Sequence[Job],
    teacher: Teacher,
    out_path: str | Path,
    *,
    time_budget_s: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    load_image: Callable[[Any], Any] | None = None,
) -> dict[str, int]:
    """Label every job not already in ``out_path`` and append the records (one JSON per line).

    Resumable: rerunning skips finished jobs. Stops cleanly after ``time_budget_s`` and reports
    ``stopped_early``. A job whose image fails to load is skipped and counted.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(f"{r['image_id']}|{r['task']}")
    todo = [j for j in jobs if j.key not in done]
    deadline = time.monotonic() + time_budget_s if time_budget_s else None
    counts = {
        "labelled": 0,
        "skipped_existing": len(jobs) - len(todo),
        "failed": 0,
        "stopped_early": 0,
    }
    with out.open("a") as f:
        for n, job in enumerate(todo, 1):
            if deadline is not None and time.monotonic() >= deadline:
                counts["stopped_early"] = 1
                break
            try:
                image = load_image(job.image) if load_image else job.image
                q = parse_question(job.criterion.task, job.criterion.question)
                rec = _record(job, teacher, teacher.probs(image, q))
            except (OSError, ValueError):
                counts["failed"] += 1
                continue
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
            f.flush()
            counts["labelled"] += 1
            if progress:
                progress(n, len(todo))
    return counts


def read_teacher_records(path: str | Path) -> list[QuestionRecord]:
    return [
        QuestionRecord(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


class HFVLMTeacher:
    """A transformers image-text model read through next-token logits (no generation).

    ``choice``/``score`` options are lettered and the letter-token logits become the option
    probabilities; ``bool`` is Yes against No. Bare and space-prefixed tokens are summed. One
    forward pass per question.
    """

    def __init__(self, checkpoint: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str | None = None,
                 max_pixels: int = 448 * 448):  # fmt: skip
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from ..backends.siglip import _use_system_trust_store, pick_device
        from ..bench.baselines.vlm import LETTERS, NO, YES

        _use_system_trust_store()
        self._torch = torch
        self.device = pick_device(device)
        self.name = checkpoint.rsplit("/", 1)[-1]
        try:  # Qwen-style processors take max_pixels to bound the image tokens; others reject it
            self.processor = AutoProcessor.from_pretrained(checkpoint, max_pixels=max_pixels)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(checkpoint)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = (
            AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=dtype)
            .to(self.device)
            .eval()
        )
        tok = self.processor.tokenizer
        self._ids = {
            w: [tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w)]
            for w in (*YES, *NO, *LETTERS)
        }

    def probs(self, image: Any, question: Question) -> list[float]:
        from ..bench.baselines.vlm import NO, YES, build_prompt, option_letters
        from ..images import load_image

        torch = self._torch
        img = load_image(image)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": build_prompt(question)}],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        enc = self.processor(text=[prompt], images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()

        def lse(words: tuple[str, ...]) -> float:
            return float(torch.logsumexp(logits[[i for w in words for i in self._ids[w]]], dim=0))

        if question.type == "bool":
            z = lse(YES) - lse(NO)
            return [float(torch.sigmoid(torch.tensor(z)))]
        scores = torch.tensor([lse((letter,)) for letter in option_letters(len(question.options))])
        return torch.softmax(scores, 0).tolist()


def train_images(
    splits_dir: str | Path,
    facts_path: str | Path,
    *,
    per_domain: int,
    seed: int = 0,
    sources: Sequence[str] = ("coco", "rico"),
) -> list[tuple[str, str, str]]:
    """``(image_id, path, domain)`` for real images in the **train** split, ``per_domain`` each.

    Only images the training split already contains are labelled, so the soft labels can never end
    up on an evaluation image. Synthetic sources are skipped: their labels are exact already.
    """
    wanted = set(sources)
    train_ids: dict[str, str] = {}
    with (Path(splits_dir) / "train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["source"] in wanted:
                train_ids[r["image_id"]] = r["domain"]
    paths: dict[str, str] = {}
    with Path(facts_path).open() as f:
        for line in f:
            r = json.loads(line)
            if r["image_id"] in train_ids and r.get("image_path"):
                paths[r["image_id"]] = r["image_path"]
    out: list[tuple[str, str, str]] = []
    for domain in sorted(set(train_ids.values())):
        ids = sorted(
            (i for i, d in train_ids.items() if d == domain and i in paths),
            key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(),
        )
        out += [(i, paths[i], domain) for i in ids[:per_domain]]
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Label real train-split images with a teacher VLM.")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-per-domain", type=int, default=500)
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--checkpoint", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--time-budget-minutes", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    images = train_images(
        args.splits_dir, args.facts, per_domain=args.images_per_domain, seed=args.seed
    )
    jobs = build_jobs(images, per_image=args.per_image, seed=args.seed)
    print(f"{len(images)} images, {len(jobs)} questions to label", flush=True)
    teacher = HFVLMTeacher(args.checkpoint)
    budget = args.time_budget_minutes * 60 if args.time_budget_minutes else None
    counts = label_images(
        jobs, teacher, args.out, time_budget_s=budget,
        progress=lambda n, total: print(f"{n}/{total}", flush=True) if n % 25 == 0 else None,
    )  # fmt: skip
    print(counts)
    if counts["stopped_early"]:
        print("time budget reached; run the same command again to continue")


if __name__ == "__main__":
    main()
