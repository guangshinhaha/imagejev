import pytest
import torch

from imagejev.model.fusion import (
    N_IMAGE_TOKENS,
    QTYPE_IDS,
    FusionBatch,
    FusionModel,
    trainable_parameter_count,
)

D_TEXT, D_IMG, D = 16, 24, 32


def small_model(**kw):
    torch.manual_seed(0)
    args = dict(d_text=D_TEXT, d_image=D_IMG, d=D, heads=4, n_blocks=2, adapter_hidden=48)
    return FusionModel(**{**args, **kw}).eval()


def make_batch(spec, seed=0, max_len=7, with_state=True):
    """``spec`` is a list of (qtype, n_options). Returns a batch and the owner of each row."""
    g = torch.Generator().manual_seed(seed)
    owner, slot, lens = [], [], []
    for q, (_, k) in enumerate(spec):
        for s in range(k):
            owner.append(q)
            slot.append(s)
            lens.append(int(torch.randint(3, max_len + 1, (1,), generator=g)))
    n = len(owner)
    tokens = torch.randn(n, max_len, D_TEXT, generator=g)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    for i, ln in enumerate(lens):
        mask[i, :ln] = True
    tokens = tokens * mask.unsqueeze(-1)
    state = torch.randn(len(spec), 5, D_TEXT, generator=g) if with_state else None
    state_mask = torch.ones(len(spec), 5, dtype=torch.bool) if with_state else None
    return FusionBatch(
        opt_tokens=tokens,
        opt_mask=mask,
        owner=torch.tensor(owner),
        slot=torch.tensor(slot),
        qtype=torch.tensor([QTYPE_IDS[t] for t, _ in spec]),
        image=torch.randn(len(spec), N_IMAGE_TOKENS, D_IMG, generator=g),
        state_tokens=state,
        state_mask=state_mask,
    )


def permute_options(b, perm, keep_slots):
    """Reorder option rows; ``keep_slots`` keeps each row's original slot (rank) with it."""
    slot = b.slot[perm] if keep_slots else b.slot.clone()
    return FusionBatch(b.opt_tokens[perm], b.opt_mask[perm], b.owner[perm], slot, b.qtype, b.image,
                       b.state_tokens, b.state_mask)  # fmt: skip


def shuffle_within_questions(b, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.arange(len(b.owner))
    for q in range(len(b.qtype)):
        rows = (b.owner == q).nonzero().squeeze(-1)
        perm[rows] = rows[torch.randperm(len(rows), generator=g)]
    return perm


def reassign_slots(b):
    """Slots follow row order inside each question (what a fresh, shuffled input would have)."""
    slot = torch.zeros_like(b.slot)
    for q in range(len(b.qtype)):
        rows = (b.owner == q).nonzero().squeeze(-1)
        slot[rows] = torch.arange(len(rows))
    b.slot = slot
    return b


def test_forward_shape_finite_and_mixed_types():
    m = small_model()
    b = make_batch([("choice", 4), ("score", 3), ("bool", 1), ("choice", 6)])
    out = m(b)
    assert out.shape == (14,) and torch.isfinite(out).all()


def test_choice_and_bool_ignore_option_order_exactly():
    m = small_model()
    b = make_batch([("choice", 5), ("bool", 1), ("choice", 3), ("choice", 8)])
    base = m(b)
    for seed in range(5):
        perm = shuffle_within_questions(b, seed)
        shuffled = reassign_slots(permute_options(b, perm, keep_slots=False))
        out = m(shuffled)
        assert torch.allclose(out, base[perm], atol=1e-5), seed
        # the probabilities per question, after undoing the shuffle, are identical
        for q in range(4):
            rows = (b.owner == q).nonzero().squeeze(-1)
            p0 = base[rows].softmax(0)
            new_rows = (shuffled.owner == q).nonzero().squeeze(-1)
            p1 = out[new_rows].softmax(0)
            assert torch.allclose(p0[(perm[new_rows] - rows[0])], p1, atol=1e-5)


def test_score_levels_are_equivariant_when_each_keeps_its_rank():
    m = small_model()
    b = make_batch([("score", 4), ("score", 3)])
    base = m(b)
    perm = shuffle_within_questions(b, 3)
    out = m(permute_options(b, perm, keep_slots=True))
    assert torch.allclose(out, base[perm], atol=1e-5)


def test_score_rank_matters_so_reordered_levels_change_the_answer():
    m = small_model()
    b = make_batch([("score", 4)])
    perm = torch.tensor([3, 2, 1, 0])  # same options, ranks reversed
    out = m(reassign_slots(permute_options(b, perm, keep_slots=False)))
    assert not torch.allclose(out, m(b)[perm], atol=1e-3)
    # while the same reversal on a choice question changes nothing
    c = make_batch([("choice", 4)])
    out_c = m(reassign_slots(permute_options(c, perm, keep_slots=False)))
    assert torch.allclose(out_c, m(c)[perm], atol=1e-5)


def test_questions_in_a_batch_do_not_leak_into_each_other():
    m = small_model()
    full = make_batch([("choice", 4), ("choice", 5), ("score", 3)], seed=1)
    alone_rows = (full.owner == 0).nonzero().squeeze(-1)
    alone = FusionBatch(full.opt_tokens[alone_rows], full.opt_mask[alone_rows], full.owner[alone_rows],
                        full.slot[alone_rows], full.qtype[:1], full.image[:1],
                        full.state_tokens[:1], full.state_mask[:1])  # fmt: skip
    assert torch.allclose(m(full)[alone_rows], m(alone), atol=1e-5)


def test_extra_padding_does_not_change_outputs():
    m = small_model()
    b = make_batch([("choice", 4), ("bool", 1)], max_len=6)
    pad = 5
    wide = FusionBatch(
        torch.cat([b.opt_tokens, torch.randn(len(b.owner), pad, D_TEXT)], dim=1),  # garbage in pad slots
        torch.cat([b.opt_mask, torch.zeros(len(b.owner), pad, dtype=torch.bool)], dim=1),
        b.owner, b.slot, b.qtype, b.image, b.state_tokens, b.state_mask,
    )  # fmt: skip
    assert torch.allclose(m(b), m(wide), atol=1e-5)


def test_image_and_state_actually_matter_and_state_is_optional():
    m = small_model()
    b = make_batch([("choice", 4)])
    base = m(b)
    b2 = make_batch([("choice", 4)])
    b2.image = torch.randn_like(b.image)
    assert not torch.allclose(m(b2), base, atol=1e-3)
    b3 = make_batch([("choice", 4)])
    b3.state_tokens = torch.randn_like(b.state_tokens)
    assert not torch.allclose(m(b3), base, atol=1e-3)
    no_state = make_batch([("choice", 4)], with_state=False)
    assert torch.isfinite(m(no_state)).all()
    # same image and options as `no_state`, plus state tokens that are all masked out. (Building it
    # with make_batch would draw the image from a different random stream and compare unlike inputs.)
    masked = make_batch([("choice", 4)], with_state=False)
    masked.state_tokens = torch.randn(1, 5, D_TEXT)
    masked.state_mask = torch.zeros(1, 5, dtype=torch.bool)
    assert torch.allclose(m(masked), m(no_state), atol=1e-5)


def test_gradients_reach_every_parameter_that_is_used():
    m = small_model().train()
    b = make_batch([("choice", 4), ("score", 3), ("bool", 1)])
    m(b).sum().backward()
    missing = [
        n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert missing == [], missing


def test_unused_question_type_heads_get_no_gradient_but_no_error():
    m = small_model().train()
    m(make_batch([("choice", 3)])).sum().backward()
    assert m.heads[QTYPE_IDS["choice"]][0].weight.grad is not None
    assert m.heads[QTYPE_IDS["bool"]][0].weight.grad is None


def test_parameter_budget_at_real_dimensions():
    n = trainable_parameter_count(FusionModel())
    assert 22e6 < n < 28e6, n  # about 23.7M: 4 blocks of d=512, two adapters, a mixer and heads


def test_batch_helpers():
    b = make_batch([("choice", 4), ("bool", 1)])
    assert b.n_options().tolist() == [4, 1]


@pytest.mark.parametrize("bad_blocks", [0])
def test_zero_blocks_still_runs(bad_blocks):
    m = small_model(n_blocks=bad_blocks)
    assert torch.isfinite(m(make_batch([("choice", 3)]))).all()
