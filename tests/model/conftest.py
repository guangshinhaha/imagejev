"""Shared fixtures: a tiny random ModernBERT and a fake tokenizer, so model tests run offline."""

import hashlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

VOCAB = 600


class FakeTokenizer:
    """Whitespace tokenizer: words hash to ids in [10, VOCAB); [CLS]=1, [SEP]=2, pad=0."""

    def _ids(self, text):
        return [
            10 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (VOCAB - 10) for w in text.split()
        ]

    def __call__(self, text, text_pair=None, padding=True, truncation=True, max_length=96,
                 return_tensors="pt"):  # fmt: skip
        pairs = text_pair if text_pair is not None else [None] * len(text)
        rows = []
        for a, b in zip(text, pairs, strict=True):
            ids = [1, *self._ids(a), 2] + ([*self._ids(b), 2] if b is not None else [])
            rows.append(ids[:max_length])
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def tiny_backbone(layers=2, hidden=64):
    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        intermediate_size=2 * hidden, vocab_size=VOCAB, max_position_embeddings=256,
        global_attn_every_n_layers=2, local_attention=16,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
    )  # fmt: skip
    torch.manual_seed(0)
    return ModernBertModel(cfg).eval()


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def backbone():
    return tiny_backbone()


@pytest.fixture
def make_backbone():
    return tiny_backbone
