"""Text side of the model: ModernBERT with LoRA adapters, plus an inference-time encoding cache.

Each answer option is encoded together with the question's instructions as a sentence pair
(``[CLS] instructions [SEP] option label: description [SEP]``), so its token features already know
what is being asked. ``bool`` questions are a single "option" holding only the instructions. The
optional ``state`` text is encoded once per call.

LoRA is applied to the attention projections only (``attn.Wqkv`` and ``attn.Wo``; ``mlp.Wo`` shares
the name, so matching is on the full path). Everything else in the backbone is frozen.

The cache keeps encodings of (instructions, option) pairs, which repeat across images. It is only
valid while the LoRA weights are fixed, so it is bypassed whenever gradients are enabled or the
module is in training mode.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"
LORA_TARGETS = ("attn.Wqkv", "attn.Wo")


class LoRALinear(nn.Module):
    """``y = W x + (alpha / r) * B A x`` with ``W`` frozen. ``B`` starts at zero, so the wrapped
    layer is initially identical to the original."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(x) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + update * self.scale


def inject_lora(
    module: nn.Module,
    targets: Sequence[str] = LORA_TARGETS,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> int:
    """Wrap each ``nn.Linear`` whose dotted path ends with one of ``targets``; returns the count."""
    paths = [
        name
        for name, m in module.named_modules()
        if isinstance(m, nn.Linear) and any(name == t or name.endswith("." + t) for t in targets)
    ]
    for path in paths:
        parent_path, _, leaf = path.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha, dropout))
    return len(paths)


class Tokenizer(Protocol):
    def __call__(self, text: Any, text_pair: Any = None, **kwargs: Any) -> Any: ...


@dataclass
class Encoded:
    """Token features for a batch of sequences.

    ``mask`` is True for real tokens. ``focus`` marks the tokens that identify the option: its own
    segment for a pair, every real token for a single segment. Pooling over ``focus`` keeps the
    instructions (identical for every option of a question) from drowning out the option itself.
    """

    tokens: torch.Tensor  # (B, L, d)
    mask: torch.Tensor  # (B, L) bool
    focus: torch.Tensor  # (B, L) bool, a subset of ``mask``


def _key(first: str, second: str | None) -> str:
    """Cache key; a missing second segment differs from an empty one."""
    tail = "\x01" if second is None else second
    return hashlib.sha256(f"{first}\x00{tail}".encode()).hexdigest()


class TextEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        *,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        max_len: int = 96,
        cache_size: int = 20000,
        targets: Sequence[str] = LORA_TARGETS,
    ):
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_lora = inject_lora(backbone, targets, rank, alpha, dropout)
        if self.n_lora == 0:
            raise ValueError(f"no linear layers matched the LoRA targets {tuple(targets)}")
        self.hidden = int(backbone.config.hidden_size)
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_BACKBONE, **kw: Any) -> TextEncoder:
        from transformers import AutoModel, AutoTokenizer

        from ..backends.siglip import _use_system_trust_store

        _use_system_trust_store()
        return cls(AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name), **kw)

    # -- parameters -------------------------------------------------------------------------
    def lora_parameters(self) -> Iterator[nn.Parameter]:
        for m in self.modules():
            if isinstance(m, LoRALinear):
                yield m.lora_a
                yield m.lora_b

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [k for k in unexpected] + [k for k in missing if "lora_" in k]
        if bad:
            raise ValueError(f"LoRA state does not match this encoder: {bad[:3]}")

    # -- encoding ---------------------------------------------------------------------------
    def _cacheable(self) -> bool:
        return not self.training and not torch.is_grad_enabled()

    def _run_group(self, firsts: Sequence[str], seconds: Sequence[str] | None) -> Encoded:
        """One forward pass over rows that are all pairs or all single segments."""
        enc = self.tokenizer(
            list(firsts),
            list(seconds) if seconds is not None else None,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device).bool()
        out = self.backbone(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        if seconds is None:
            return Encoded(out, mask, mask)
        focus = torch.zeros_like(mask)
        for r in range(len(firsts)):
            seg = torch.tensor([sid == 1 for sid in enc.sequence_ids(r)], device=device)
            focus[r, : len(seg)] = seg
        empty = ~focus.any(
            -1, keepdim=True
        )  # an option that truncated away: fall back to all tokens
        return Encoded(out, mask, torch.where(empty, mask, focus & mask))

    def _run(self, firsts: Sequence[str], seconds: Sequence[str | None]) -> Encoded:
        """Encode rows. Pairs and single segments go through separate passes, so a row's features
        never depend on what else is in the batch (a mixed batch would give a single segment a
        trailing empty segment)."""
        paired = [i for i, s in enumerate(seconds) if s is not None]
        single = [i for i, s in enumerate(seconds) if s is None]
        parts: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for rows, use_pair in ((paired, True), (single, False)):
            if not rows:
                continue
            res = self._run_group(
                [firsts[i] for i in rows],
                [seconds[i] for i in rows] if use_pair else None,  # type: ignore[misc]
            )
            for k, i in enumerate(rows):
                n = int(res.mask[k].sum())
                parts[i] = (res.tokens[k, :n], res.mask[k, :n], res.focus[k, :n])
        length = max(t.shape[0] for t, _, _ in parts.values())
        first = next(iter(parts.values()))[0]
        tokens = first.new_zeros(len(firsts), length, self.hidden)
        mask = torch.zeros(len(firsts), length, dtype=torch.bool, device=first.device)
        focus = torch.zeros_like(mask)
        for i, (t, m, f) in parts.items():
            tokens[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = m
            focus[i, : t.shape[0]] = f
        return Encoded(tokens, mask, focus)

    def encode_pairs(self, firsts: Sequence[str], seconds: Sequence[str | None]) -> Encoded:
        """Encode ``(instructions, option)`` pairs; ``option=None`` means instructions only."""
        if len(firsts) != len(seconds):
            raise ValueError("firsts and seconds must have the same length")
        if not firsts:
            raise ValueError("nothing to encode")
        if not self._cacheable():
            return self._run(firsts, seconds)
        keys = [_key(a, b) for a, b in zip(firsts, seconds, strict=True)]
        found = {i: self._cache[k] for i, k in enumerate(keys) if k in self._cache}
        missing = [i for i in range(len(keys)) if i not in found]
        self.cache_hits += len(found)
        self.cache_misses += len(missing)
        if missing:
            res = self._run([firsts[i] for i in missing], [seconds[i] for i in missing])
            for row, i in enumerate(missing):
                n = int(res.mask[row].sum())
                found[i] = self._cache[keys[i]] = (
                    res.tokens[row, :n].detach(),
                    res.focus[row, :n].clone(),
                )
        for k in keys:
            if k in self._cache:
                self._cache.move_to_end(k)
        while len(self._cache) > self._cache_size:  # evict only after this batch is assembled
            self._cache.popitem(last=False)
        seqs = [found[i] for i in range(len(keys))]
        length = max(t.shape[0] for t, _ in seqs)
        tokens = seqs[0][0].new_zeros(len(seqs), length, self.hidden)
        mask = torch.zeros(len(seqs), length, dtype=torch.bool, device=tokens.device)
        focus = torch.zeros_like(mask)
        for r, (t, f) in enumerate(seqs):
            tokens[r, : t.shape[0]] = t
            mask[r, : t.shape[0]] = True
            focus[r, : t.shape[0]] = f
        return Encoded(tokens, mask, focus)

    def encode_state(self, text: str) -> Encoded | None:
        """Features for the optional state text (batch of one), or ``None`` if there is none."""
        return self.encode_pairs([text], [None]) if text.strip() else None

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = self.cache_misses = 0
