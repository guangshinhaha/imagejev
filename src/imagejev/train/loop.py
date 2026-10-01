"""Training loop: AdamW (separate rates for fusion and LoRA), cosine schedule, resumable.

Everything needed to continue after a killed session is in ``last.pt``: weights, optimizer and
scheduler state, the step, the best score so far, and every random-number generator. Resuming
reproduces the uninterrupted run exactly (tested). CSV logs record training loss per step and
evaluation metrics by domain x question type; the checkpoint with the best selection score is kept
as ``best.pt``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..bench.metrics import Prediction, evaluate, from_bool
from ..data.records import QuestionRecord
from ..model.fusion import FusionModel, correction_of
from ..model.losses import fusion_loss
from ..model.text import TextEncoder
from .data import BatchBuilder
from .sampler import BalancedSampler, SamplerConfig


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "auto"
    amp: str = "none"  # "none" | "fp16" | "bf16"; autocast is used on CUDA only
    steps: int = 3000
    eval_every: int = 250
    ckpt_every: int = 250
    images_per_batch: int = 16
    questions_per_image: int = 4
    teacher_fraction: float = 0.15
    lr_fusion: float = 3e-4
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    rps_weight: float = 1.0
    # Shrinks the learned correction toward zero (i.e. toward the zero-shot prior): the loss gains
    # correction_l2 * mean(correction^2). Unseen task families keep the prior unless the evidence
    # for a change is strong; 0 disables it.
    correction_l2: float = 0.0
    lora_rank: int = 16
    lora_alpha: float = 32.0
    dropout: float = 0.0
    d_model: int = 512
    n_blocks: int = 4
    heads: int = 8
    backbone: str = "answerdotai/ModernBERT-base"
    use_prior: bool = True  # start from SigLIP's zero-shot logits (see model/prior.py)
    prior_checkpoint: str = "google/siglip2-base-patch16-384"
    train_file: str = ""
    eval_files: dict[str, str] = field(default_factory=dict)
    select_on: str = "val"
    max_eval_questions: int | None = None
    time_budget_minutes: float | None = None  # stop and save after this long (session limits)
    cache_dir: str = ""
    out_dir: str = "runs/run"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def lr_factor(step: int, warmup: int, total: int, min_frac: float) -> float:
    """Linear warmup to 1, then cosine decay to ``min_frac``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def macro_log_loss(rows: Sequence[Mapping[str, Any]], min_n: int = 10) -> float:
    """Mean log loss over (domain, type) cells; roll-up rows and tiny cells are ignored."""
    cells = [
        r["log_loss"]
        for r in rows
        if r["domain"] != "all" and r["qtype"] != "all" and r["n"] >= min_n
    ]
    if not cells:
        raise ValueError("no evaluation cell has enough examples")
    return float(sum(cells) / len(cells))


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _truncate_csv(path: Path, max_step: int) -> None:
    """Drop rows logged after the checkpoint we are resuming from."""
    if not path.exists():
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if int(r["step"]) <= max_step]
    if len(keep) != len(rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)


class _Csv:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.path, self.fields = path, list(fields)
        self.fresh = not path.exists() or path.stat().st_size == 0

    def write(self, row: Mapping[str, Any]) -> None:
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if self.fresh:
                w.writeheader()
                self.fresh = False
            w.writerow({k: row.get(k, "") for k in self.fields})


EVAL_FIELDS = [
    "step",
    "split",
    "domain",
    "qtype",
    "n",
    "accuracy",
    "within_one",
    "log_loss",
    "ece",
    "acc@80",
]
TRAIN_FIELDS = [
    "step",
    "loss",
    "lr",
    "loss_choice",
    "loss_score",
    "loss_bool",
    "rps",
    "corr_rms",
    "questions",
    "seconds",
]


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        text: TextEncoder,
        fusion: FusionModel,
        features: Mapping[str, Any],
        train_records: Sequence[QuestionRecord],
        eval_sets: Mapping[str, Sequence[QuestionRecord]],
        prior: Any = None,
    ):
        self.cfg = cfg
        self.device = cfg.resolve_device()
        self.text = text.to(self.device)
        self.fusion = fusion.to(self.device)
        if getattr(fusion, "use_prior", False) and prior is None:
            raise ValueError("the fusion model uses a prior but no prior embedder was given")
        self.builder = BatchBuilder(features, self.text, self.device, prior=prior)
        if prior is not None:  # embed every string once, up front
            from ..model.prior import warm
            from ..schema import parse_question

            every = list(train_records) + [r for rs in eval_sets.values() for r in rs]
            warm(prior, [parse_question(r.task, r.question) for r in every])
        self.sampler = BalancedSampler(
            train_records,
            SamplerConfig(
                cfg.images_per_batch, cfg.questions_per_image, cfg.teacher_fraction, cfg.seed
            ),
        )
        self.eval_sets = {k: list(v) for k, v in eval_sets.items()}
        if cfg.select_on not in self.eval_sets:
            raise ValueError(
                f"select_on={cfg.select_on!r} is not one of the eval sets {list(self.eval_sets)}"
            )
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        groups = [
            {"params": list(self.fusion.parameters()), "lr": cfg.lr_fusion},
            {"params": list(self.text.lora_parameters()), "lr": cfg.lr_lora},
        ]
        self.opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        base = [cfg.lr_fusion, cfg.lr_lora]
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            [lambda s, b=1.0: lr_factor(s, cfg.warmup_steps, cfg.steps, cfg.min_lr_frac)]
            * len(base),
        )
        # fp16 autocast needs CUDA; bf16 also works on CPU, which is how the wiring is tested
        self.use_amp = cfg.amp != "none" and (
            self.device.type == "cuda" or (self.device.type == "cpu" and cfg.amp == "bf16")
        )
        self.amp_dtype = torch.float16 if cfg.amp == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and cfg.amp == "fp16")
        self.rng = random.Random(cfg.seed)
        self.step = 0
        self.best = math.inf
        self.train_csv = _Csv(self.out / "train_log.csv", TRAIN_FIELDS)
        self.eval_csv = _Csv(self.out / "eval_log.csv", EVAL_FIELDS)
        self.select_csv = _Csv(
            self.out / "select_log.csv", ["step", "split", "macro_log_loss", "best"]
        )

    # -- checkpoints ------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "best": self.best,
            "fusion": self.fusion.state_dict(),
            "lora": self.text.lora_state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": {
                "sampler": self.rng.getstate(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "config": dataclasses.asdict(self.cfg),
        }

    def save(self, name: str = "last.pt") -> Path:
        path = self.out / name
        _atomic_save(self.state(), path)
        return path

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.fusion.load_state_dict(ck["fusion"])
        self.text.load_lora_state_dict(ck["lora"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step, self.best = int(ck["step"]), float(ck["best"])
        r = ck["rng"]
        self.rng.setstate(r["sampler"])
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"].cpu())
        self.text.clear_cache()
        for name in ("train_log.csv", "eval_log.csv", "select_log.csv"):
            _truncate_csv(self.out / name, self.step)

    # -- evaluation -------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, records: Sequence[QuestionRecord], chunk: int = 96) -> list[Prediction]:
        self.fusion.eval()
        self.text.eval()
        self.text.clear_cache()  # LoRA weights just changed
        preds: list[Prediction] = []
        for i in range(0, len(records), chunk):
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                prep = self.builder.build(records[i : i + chunk])
                logits = self.fusion(prep.batch).float().cpu()
            start = 0
            for rec, q in zip(prep.records, prep.questions, strict=True):
                n = 1 if q.type == "bool" else len(q.options)
                z = logits[start : start + n]
                start += n
                if q.type == "bool":
                    preds.append(
                        from_bool(
                            rec.domain, float(torch.sigmoid(z[0])), bool(rec.answer), rec.task
                        )
                    )
                else:
                    p = torch.softmax(z.double(), 0).numpy()
                    preds.append(
                        Prediction(rec.domain, q.type, p, q.labels.index(rec.answer), rec.task)
                    )  # type: ignore[arg-type]
        self.text.clear_cache()
        self.fusion.train()
        self.text.train()
        return preds

    def run_eval(self) -> float:
        score = math.nan
        for split, records in self.eval_sets.items():
            if self.cfg.max_eval_questions:
                records = records[: self.cfg.max_eval_questions]
            rows = evaluate(self.predict(records))
            for r in rows:
                self.eval_csv.write({"step": self.step, "split": split, **r})
            if split == self.cfg.select_on:
                score = macro_log_loss(rows)
                improved = score < self.best
                if improved:
                    self.best = score
                    self.save("best.pt")
                self.select_csv.write(
                    {
                        "step": self.step,
                        "split": split,
                        "macro_log_loss": score,
                        "best": int(improved),
                    }
                )
        return score

    # -- training ---------------------------------------------------------------------------
    def train_step(self) -> dict[str, float]:
        t0 = time.perf_counter()
        groups = self.sampler.sample_batch(self.rng)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            prep = self.builder.build([r for g in groups for r in g])
            logits = self.fusion(prep.batch)
        out = fusion_loss(logits.float(), prep.batch, prep.targets, rps_weight=self.cfg.rps_weight)
        total = out.total
        corr_rms = 0.0
        if getattr(self.fusion, "use_prior", False):  # always logged, so runs can be compared
            corr = correction_of(self.fusion, prep.batch, logits.float())
            corr_rms = float(corr.detach().pow(2).mean().sqrt())
            if self.cfg.correction_l2 > 0:
                total = total + self.cfg.correction_l2 * corr.pow(2).mean()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.opt.param_groups for p in g["params"]], self.cfg.grad_clip
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.step += 1
        return {
            "step": self.step,
            "loss": float(out.total.detach()),
            "lr": self.opt.param_groups[0]["lr"],
            **{f"loss_{k}": v for k, v in out.by_type.items()},
            "rps": out.rps if out.rps is not None else "",
            "corr_rms": corr_rms,
            "questions": len(prep.records),
            "seconds": time.perf_counter() - t0,
        }

    def run(self, on_step: Callable[[int], None] | None = None, resume: bool = True) -> bool:
        """Train to ``cfg.steps``; returns False if it stopped early on the time budget.

        On a budget stop the latest state is saved to ``last.pt``, so running again (in a new
        session) continues exactly where this one stopped.
        """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        last = self.out / "last.pt"
        if resume and last.exists():
            self.load(last)
        budget = self.cfg.time_budget_minutes
        deadline = time.monotonic() + budget * 60 if budget else None
        self.fusion.train()
        self.text.train()
        while self.step < self.cfg.steps:
            row = self.train_step()
            self.train_csv.write(row)
            if on_step:
                on_step(self.step)
            if self.step % self.cfg.eval_every == 0 or self.step == self.cfg.steps:
                self.run_eval()
            if self.step % self.cfg.ckpt_every == 0 or self.step == self.cfg.steps:
                self.save()
            if deadline is not None and time.monotonic() >= deadline and self.step < self.cfg.steps:
                self.save()
                return False
        return True


def build_trainer(cfg: TrainConfig) -> Trainer:
    """Assemble the real thing: ModernBERT + LoRA, the fusion model, the feature cache and data."""
    from ..cache import FeatureCache
    from ..data.records import read_jsonl

    features = FeatureCache(cfg.cache_dir)

    def load(path: str) -> list[QuestionRecord]:
        recs = list(read_jsonl(path, QuestionRecord))
        kept = [r for r in recs if r.image_id in features]
        if len(kept) < len(recs):
            n_drop = len(recs) - len(kept)
            print(f"{path}: dropped {n_drop} of {len(recs)} questions with no cached image")
        if not kept:
            raise ValueError(f"no question in {path} has cached image features")
        return kept

    train = load(cfg.train_file)
    evals = {name: load(path) for name, path in cfg.eval_files.items()}
    text = TextEncoder.from_pretrained(
        cfg.backbone, rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.dropout
    )
    prior = None
    if cfg.use_prior:
        from ..backends.siglip import SigLIPBackend
        from ..model.prior import SigLIPTextEmbedder

        prior = SigLIPTextEmbedder(
            SigLIPBackend(cfg.prior_checkpoint, device=str(cfg.resolve_device()))
        )
    fusion = FusionModel(
        d_text=text.hidden,
        d_image=int(features.meta["shape"][1]),
        d=cfg.d_model,
        heads=cfg.heads,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        use_prior=cfg.use_prior,
    )
    return Trainer(cfg, text, fusion, features, train, evals, prior=prior)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the imagejev fusion model.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    ap.add_argument("--steps", type=int, default=None, help="override the config's step count")
    ap.add_argument("--time-budget-minutes", type=float, default=None)
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if args.steps:
        cfg.steps = args.steps
    if args.time_budget_minutes:
        cfg.time_budget_minutes = args.time_budget_minutes
    trainer = build_trainer(cfg)
    n_train = sum(p.numel() for p in trainer.fusion.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for p in trainer.text.lora_parameters())
    print(
        f"device {trainer.device}; trainable: fusion {n_train / 1e6:.1f}M, LoRA {n_lora / 1e6:.1f}M"
    )
    finished = trainer.run(
        resume=not args.fresh,
        on_step=lambda s: print(f"step {s}", flush=True) if s % 10 == 0 else None,
    )
    if finished:
        print(
            f"done at step {trainer.step}; best {cfg.select_on} macro log loss {trainer.best:.4f}"
        )
    else:
        print(f"time budget reached at step {trainer.step}/{cfg.steps}; run again to resume")


if __name__ == "__main__":
    main()
