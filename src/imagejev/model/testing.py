"""Tiny random models and a fake tokenizer, so tests and smoke runs need no downloads."""

from __future__ import annotations

import hashlib

import torch

from .fusion import FusionModel
from .text import TextEncoder

VOCAB = 600


class FakeEncoding(dict):
    """Dict of tensors that also answers ``sequence_ids`` like a fast tokenizer's output."""

    def __init__(self, data, seqs):
        super().__init__(data)
        self._seqs = seqs

    def sequence_ids(self, i):
        return self._seqs[i]


class FakeTokenizer:
    """Whitespace tokenizer: words hash to ids in [10, VOCAB); [CLS]=1, [SEP]=2, pad=0."""

    def _ids(self, text: str) -> list[int]:
        return [
            10 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (VOCAB - 10) for w in text.split()
        ]

    def __call__(
        self,
        text,
        text_pair=None,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    ):
        pairs = text_pair if text_pair is not None else [None] * len(text)
        rows, seqs = [], []
        for a, b in zip(text, pairs, strict=True):
            ia = self._ids(a)
            ids = [1, *ia, 2]
            sid: list[int | None] = [None, *([0] * len(ia)), None]
            if b is not None:
                ib = self._ids(b)
                ids += [*ib, 2]
                sid += [*([1] * len(ib)), None]
            rows.append(ids[:max_length])
            seqs.append(sid[:max_length])
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return FakeEncoding({"input_ids": input_ids, "attention_mask": mask}, seqs)


def tiny_backbone(layers: int = 2, hidden: int = 32, seed: int = 0):
    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        intermediate_size=2 * hidden,
        vocab_size=VOCAB,
        max_position_embeddings=256,
        global_attn_every_n_layers=2,
        local_attention=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    torch.manual_seed(seed)
    return ModernBertModel(cfg)


def tiny_text_encoder(rank: int = 4, seed: int = 0, hidden: int = 32) -> TextEncoder:
    return TextEncoder(tiny_backbone(hidden=hidden, seed=seed), FakeTokenizer(), rank=rank)


def tiny_fusion(d_text: int = 32, d_image: int = 24, d: int = 32, seed: int = 0) -> FusionModel:
    torch.manual_seed(seed)
    return FusionModel(d_text=d_text, d_image=d_image, d=d, heads=4, n_blocks=2, adapter_hidden=48)
