import collections
import random

import pytest

from imagejev.data.records import QuestionRecord
from imagejev.train.sampler import BalancedSampler, SamplerConfig, cell_of

DOMAINS = ("photo", "document", "screenshot")
SPECS = {
    "choice": {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}},
    "score": {"type": "score", "instructions": "x", "levels": ["lo", "hi"]},
    "bool": {"type": "bool", "instructions": "x"},
}
ANSWER = {"choice": "a", "score": "lo", "bool": True}


def make_records(images_per_domain=60, per_image=5, skew=True, teacher_every=0, seed=0):
    """Deliberately lopsided: photos have far more bool questions than anything else."""
    rng = random.Random(seed)
    recs = []
    for d_i, domain in enumerate(DOMAINS):
        for i in range(images_per_domain * (3 if (skew and d_i == 0) else 1)):
            for q in range(per_image):
                t = "bool" if (skew and domain == "photo" and q < 4) else rng.choice(list(SPECS))
                rec = QuestionRecord(f"{domain}{i}", domain, "s", f"t.{t}", SPECS[t], ANSWER[t])
                if teacher_every and rng.random() < 1 / teacher_every and t != "bool":
                    labels = list(SPECS[t].get("criteria") or SPECS[t]["levels"])
                    rec.soft = {labels[0]: 0.7, labels[1]: 0.3}
                recs.append(rec)
    return recs


def flatten(batch):
    return [r for group in batch for r in group]


def test_every_batch_has_exact_equal_domain_and_type_quotas():
    s = BalancedSampler(
        make_records(), SamplerConfig(images_per_batch=18, questions_per_image=4, seed=1)
    )
    it = iter(s)
    for _ in range(50):
        qs = flatten(next(it))
        assert len(qs) == 72
        counts = collections.Counter(cell_of(r) for r in qs)
        assert len(counts) == 9 and set(counts.values()) == {8}  # 72 questions / 9 cells, exactly
        assert collections.Counter(r.domain for r in qs) == {d: 24 for d in DOMAINS}


def test_mix_is_balanced_even_when_the_data_is_wildly_skewed():
    recs = make_records()
    raw = collections.Counter(cell_of(r) for r in recs)
    assert max(raw.values()) > 10 * min(raw.values())  # the pool really is lopsided
    s = BalancedSampler(recs, SamplerConfig(images_per_batch=16, questions_per_image=4, seed=2))
    seen = collections.Counter()
    it = iter(s)
    for _ in range(100):
        seen.update(cell_of(r) for r in flatten(next(it)))
    shares = [v / sum(seen.values()) for v in seen.values()]
    assert all(abs(x - 1 / 9) < 0.01 for x in shares), shares


def test_teacher_share_is_exact_and_each_pool_is_balanced_internally():
    s = BalancedSampler(
        make_records(teacher_every=3),
        SamplerConfig(images_per_batch=25, questions_per_image=4, seed=3),
    )
    it = iter(s)
    for _ in range(30):
        qs = flatten(next(it))
        teacher = [r for r in qs if r.soft is not None]
        assert len(qs) == 100 and len(teacher) == 15  # 15% of 100
        t_cells = collections.Counter(cell_of(r) for r in teacher)
        assert max(t_cells.values()) - min(t_cells.values()) <= 1  # equal across teacher cells


def test_no_teacher_pool_means_no_teacher_questions_and_full_batches():
    s = BalancedSampler(
        make_records(), SamplerConfig(images_per_batch=10, questions_per_image=4, seed=4)
    )
    qs = flatten(next(iter(s)))
    assert len(qs) == 40 and all(r.soft is None for r in qs)


def test_questions_come_from_their_image_and_groups_are_capped_and_distinct():
    s = BalancedSampler(
        make_records(), SamplerConfig(images_per_batch=12, questions_per_image=4, seed=5)
    )
    it = iter(s)
    multi = 0
    for _ in range(40):
        batch = next(it)
        ids = [g[0].image_id for g in batch]
        assert len(ids) == len(set(ids))  # no image twice in a batch
        for g in batch:
            assert 1 <= len(g) <= 4 and len({r.image_id for r in g}) == 1
            multi += len(g) > 1
    assert multi > 0  # images really are shared across several questions


def test_deterministic_for_a_seed_and_varies_across_seeds():
    recs = make_records()

    def first(seed):
        return [id(r) for r in flatten(next(iter(BalancedSampler(recs, SamplerConfig(seed=seed)))))]

    assert first(7) == first(7) and first(7) != first(8)


def test_missing_cells_are_renormalised_and_tiny_pools_still_fill():
    recs = [
        r
        for r in make_records(5, 6, skew=False)
        if r.domain != "screenshot" and r.question["type"] != "score"
    ]
    s = BalancedSampler(recs, SamplerConfig(images_per_batch=8, questions_per_image=4, seed=6))
    qs = flatten(next(iter(s)))
    assert len(qs) == 32
    counts = collections.Counter(cell_of(r) for r in qs)
    assert len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 1
    tiny = [
        r for r in make_records(1, 2, skew=False)
    ]  # fewer images than slots: repeats are allowed
    assert (
        len(flatten(next(iter(BalancedSampler(tiny, SamplerConfig(images_per_batch=8, seed=1))))))
        == 32
    )


def test_only_teacher_records_are_sampled_when_nothing_else_exists():
    recs = [r for r in make_records(teacher_every=1) if r.soft is not None]
    assert recs
    qs = flatten(next(iter(BalancedSampler(recs, SamplerConfig(images_per_batch=6, seed=2)))))
    assert len(qs) == 24 and all(r.soft is not None for r in qs)


def test_validation():
    with pytest.raises(ValueError):
        BalancedSampler([])
    with pytest.raises(ValueError):
        BalancedSampler(make_records(2, 2), SamplerConfig(teacher_fraction=1.0))
