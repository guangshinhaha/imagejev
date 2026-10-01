import pytest
import torch
from torch import nn

from imagejev.model.text import LoRALinear, TextEncoder, inject_lora


def test_lora_starts_as_the_identity_and_trains_only_its_own_weights():
    base = nn.Linear(8, 6)
    x = torch.randn(4, 8)
    expected = base(x)
    lora = LoRALinear(base, rank=4, alpha=8)
    assert torch.allclose(lora(x), expected)  # B starts at zero
    assert [p.requires_grad for p in base.parameters()] == [False, False]
    lora(x).sum().backward()
    assert lora.lora_a.grad is not None and lora.lora_b.grad is not None
    assert base.weight.grad is None
    with torch.no_grad():
        lora.lora_b.normal_()
    assert not torch.allclose(lora(x), expected)


def test_lora_scale_and_rank_validation():
    assert LoRALinear(nn.Linear(4, 4), rank=2, alpha=8).scale == 4.0
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), rank=0)


def test_inject_targets_attention_only_not_the_mlp_wo(backbone):
    n = inject_lora(backbone, rank=4)
    assert n == 4  # two layers x (Wqkv, Wo)
    wrapped = [name for name, m in backbone.named_modules() if isinstance(m, LoRALinear)]
    assert all(".attn." in name for name in wrapped) and len(wrapped) == 4
    assert not any("mlp" in name for name in wrapped)
    assert inject_lora(backbone, targets=("nonexistent",)) == 0


def test_encoder_freezes_backbone_and_counts_lora_parameters(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer, rank=4)
    trainable = [p for p in enc.parameters() if p.requires_grad]
    lora = list(enc.lora_parameters())
    assert len(trainable) == len(lora) == 8 and set(map(id, trainable)) == set(map(id, lora))
    # per layer: Wqkv (192x64) -> 4*64 + 192*4 ; Wo (64x64) -> 4*64 + 64*4
    per_layer = (4 * 64 + 192 * 4) + (4 * 64 + 64 * 4)
    assert sum(p.numel() for p in lora) == 2 * per_layer
    assert enc.hidden == 64 and enc.n_lora == 4


def test_no_matching_layers_is_an_error(backbone, tokenizer):
    with pytest.raises(ValueError, match="no linear layers"):
        TextEncoder(backbone, tokenizer, targets=("nope",))


def test_pairs_and_single_segments_have_different_lengths(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer).eval()
    with torch.no_grad():
        pair = enc.encode_pairs(["is it red"], ["red: a colour"])
        solo = enc.encode_pairs(["is it red"], [None])
    assert pair.tokens.shape[1] > solo.tokens.shape[1] and pair.mask.all() and solo.mask.all()
    assert pair.tokens.shape[2] == 64


def test_padding_mask_marks_real_tokens(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer).eval()
    with torch.no_grad():
        out = enc.encode_pairs(["a", "a b c d e"], ["x", "y"])
    assert out.mask[0].sum() < out.mask[1].sum() and out.mask[1].all()
    assert (out.tokens[0][~out.mask[0]] == 0).all()  # padding positions are zeroed


def test_cache_returns_identical_tensors_and_counts_hits(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer).eval()
    with torch.no_grad():
        first = enc.encode_pairs(["q one", "q two"], ["a: x", "b: y"])
        assert (enc.cache_hits, enc.cache_misses) == (0, 2)
        again = enc.encode_pairs(["q one", "q two"], ["a: x", "b: y"])
        assert (enc.cache_hits, enc.cache_misses) == (2, 2)
        mixed = enc.encode_pairs(["q two", "q new"], ["b: y", "c: z"])
    assert torch.equal(first.tokens, again.tokens) and torch.equal(first.mask, again.mask)
    assert (enc.cache_hits, enc.cache_misses) == (3, 3)
    # a cached row matches a fresh, uncached encoding of the same pair
    enc.clear_cache()
    with torch.no_grad():
        fresh = enc.encode_pairs(["q two"], ["b: y"])
    n = int(fresh.mask.sum())
    assert torch.allclose(mixed.tokens[0, :n], fresh.tokens[0, :n], atol=1e-5)


def test_none_and_empty_second_segment_are_different_cache_entries(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer).eval()
    with torch.no_grad():
        enc.encode_pairs(["same"], [None])
        enc.encode_pairs(["same"], [""])
    assert enc.cache_misses == 2


def test_cache_is_bypassed_when_training_or_with_gradients(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer, rank=4)
    enc.train()
    with torch.no_grad():
        enc.encode_pairs(["q"], ["a"])
    assert enc.cache_misses == 0 and not enc._cache  # training mode: never cached
    enc.eval()
    out = enc.encode_pairs(["q"], ["a"])  # grad enabled: not cached either
    assert out.tokens.requires_grad and not enc._cache
    out.tokens.sum().backward()
    assert all(p.grad is not None for p in enc.lora_parameters())


def test_stale_cache_cannot_survive_a_weight_change(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer, rank=4).eval()
    with torch.no_grad():
        before = enc.encode_pairs(["q"], ["a"]).tokens.clone()
        for p in enc.lora_parameters():
            p.add_(0.5)
        stale = enc.encode_pairs(["q"], ["a"]).tokens
    assert torch.equal(before, stale)  # still cached: callers must clear after changing weights
    enc.clear_cache()
    with torch.no_grad():
        fresh = enc.encode_pairs(["q"], ["a"]).tokens
    assert not torch.allclose(before, fresh)


def test_cache_eviction(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer, cache_size=2).eval()
    with torch.no_grad():
        out = enc.encode_pairs(["a", "b", "c"], [None] * 3)  # more new entries than the cache holds
        again = enc.encode_pairs(["c", "a"], [None] * 2)
    assert len(enc._cache) == 2 and out.tokens.shape[0] == 3 and again.tokens.shape[0] == 2
    assert torch.allclose(out.tokens[2, : again.tokens.shape[1]], again.tokens[0], atol=1e-5)


def test_state_encoding(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer).eval()
    with torch.no_grad():
        assert enc.encode_state("   ") is None
        s = enc.encode_state("merchant returned the item")
    assert s.tokens.shape[0] == 1 and s.mask.all()


def test_lora_state_round_trip(backbone, tokenizer, make_backbone):
    a = TextEncoder(backbone, tokenizer, rank=4)
    with torch.no_grad():
        for p in a.lora_parameters():
            p.normal_()
    state = a.lora_state_dict()
    assert state and all("lora_" in k for k in state)
    b = TextEncoder(make_backbone(), tokenizer, rank=4)
    b.load_lora_state_dict(state)
    assert all(
        torch.equal(x, y) for x, y in zip(a.lora_parameters(), b.lora_parameters(), strict=True)
    )
    c = TextEncoder(make_backbone(), tokenizer, rank=8)
    with pytest.raises(RuntimeError):
        c.load_lora_state_dict(state)  # wrong rank: shapes differ


def test_input_validation(backbone, tokenizer):
    enc = TextEncoder(backbone, tokenizer).eval()
    with pytest.raises(ValueError):
        enc.encode_pairs(["a"], [None, None])
    with pytest.raises(ValueError):
        enc.encode_pairs([], [])


@pytest.mark.slow
def test_real_modernbert_lora_budget_and_cache_consistency():
    try:
        enc = TextEncoder.from_pretrained(rank=16).eval()
    except OSError as e:
        pytest.skip(f"checkpoint unavailable: {e}")
    assert enc.n_lora == 44 and enc.hidden == 768  # 22 layers x (Wqkv, Wo)
    # per layer: Wqkv 768->2304 and Wo 768->768 at rank 16
    assert sum(p.numel() for p in enc.lora_parameters()) == 22 * (
        16 * 768 + 2304 * 16 + 16 * 768 + 768 * 16
    )
    with torch.no_grad():
        a = enc.encode_pairs(["Is there a dog in the image?"], ["dog: an animal"])
        b = enc.encode_pairs(["Is there a dog in the image?"], ["dog: an animal"])
    assert a.tokens.shape[2] == 768 and torch.equal(a.tokens, b.tokens) and enc.cache_hits == 1
