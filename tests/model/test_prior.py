import hashlib

import numpy as np
import pytest
import torch

from imagejev.backends.trained import TrainedBackend, export_model, load_trained
from imagejev.images import CachedEncoder
from imagejev.model.fusion import QTYPE_IDS, FusionBatch, FusionModel
from imagejev.model.prior import (
    GENERIC_CAPTION,
    prior_from_embeddings,
    prior_texts,
    warm,
)
from imagejev.model.testing import tiny_fusion, tiny_text_encoder
from imagejev.schema import parse_question
from imagejev.train.data import BatchBuilder

D_IMG = 24
CHOICE = {
    "type": "choice",
    "instructions": "Which colour?",
    "criteria": {"red": "warm", "green": "", "blue": "cool"},
}
SCORE = {"type": "score", "instructions": "How bright?", "levels": ["dark", "mid", "bright"]}
BOOL = {"type": "bool", "instructions": "Is it red?"}


class FakeEmbedder:
    """Deterministic unit vectors per string, in the same space as the image's global token."""

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            v = rng.normal(size=D_IMG).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    @property
    def scale_bias(self):
        return 10.0, -2.0


def feats(seed=0):
    return np.random.default_rng(seed).normal(size=(65, D_IMG)).astype(np.float16)


# ---- the formula --------------------------------------------------------------------------
def test_prior_texts_match_the_zero_shot_baselines_strings():
    q = parse_question("q", CHOICE)
    assert prior_texts(q) == [
        "Which colour? red: warm",
        "Which colour? green",
        "Which colour? blue: cool",
    ]
    assert prior_texts(parse_question("b", BOOL)) == ["Is it red?", GENERIC_CAPTION]
    assert prior_texts(
        parse_question("c", {"type": "choice", "instructions": "", "criteria": {"a": "", "b": ""}})
    ) == ["a", "b"]


def test_prior_values_are_scaled_cosine_similarity_and_bool_is_a_difference():
    e = FakeEmbedder().embed(["x", "y", "z"])
    g = np.random.default_rng(1).normal(size=D_IMG).astype(np.float32) * 7  # length must not matter
    p = prior_from_embeddings(e, g, 10.0, -2.0, "choice")
    expected = e @ (g / np.linalg.norm(g)) * 10.0 - 2.0
    assert np.allclose(p, expected, atol=1e-5)
    b = prior_from_embeddings(e[:2], g, 10.0, -2.0, "bool")
    assert b.shape == (1,) and b[0] == pytest.approx(
        expected[0] - expected[1], abs=1e-5
    )  # the bias cancels


def test_warm_embeds_every_distinct_string_once():
    emb = FakeEmbedder()
    qs = [parse_question("c", CHOICE), parse_question("c2", CHOICE), parse_question("b", BOOL)]
    n = warm(emb, qs, chunk=2)
    assert (
        n == len({t for q in qs for t in prior_texts(q)}) == 5
    )  # 3 options + instruction + generic
    assert sum(len(c) for c in emb.calls) == 5 and all(len(c) <= 2 for c in emb.calls)


# ---- the model ----------------------------------------------------------------------------
def prior_batch(spec, seed=0):
    n_per = [k for _, k in spec]
    n = sum(n_per)
    g = torch.Generator().manual_seed(seed)
    owner, slot = [], []
    for q, k in enumerate(n_per):
        owner += [q] * k
        slot += list(range(k))
    return FusionBatch(
        opt_tokens=torch.randn(n, 5, 32, generator=g), opt_mask=torch.ones(n, 5, dtype=torch.bool),
        owner=torch.tensor(owner), slot=torch.tensor(slot),
        qtype=torch.tensor([QTYPE_IDS[t] for t, _ in spec]),
        image=torch.randn(len(spec), 65, D_IMG, generator=g),
        prior=torch.randn(n, generator=g) * 3,
    )  # fmt: skip


def small(use_prior):
    torch.manual_seed(0)
    return FusionModel(
        d_text=32, d_image=D_IMG, d=32, heads=4, n_blocks=2, adapter_hidden=48, use_prior=use_prior
    ).eval()


def test_a_fresh_prior_model_reproduces_the_zero_shot_logits_exactly():
    b = prior_batch([("choice", 4), ("score", 3), ("bool", 1)])
    out = small(True)(b)
    assert torch.allclose(out, b.prior, atol=1e-6)  # head output is zero, gain is 1
    assert not torch.allclose(
        small(False)(prior_batch([("choice", 4), ("score", 3), ("bool", 1)])), b.prior, atol=1e-3
    )


def test_a_prior_model_refuses_a_batch_without_a_prior():
    b = prior_batch([("choice", 3)])
    b.prior = None
    with pytest.raises(ValueError, match="no prior"):
        small(True)(b)
    assert torch.isfinite(small(False)(b)).all()  # the old behaviour is untouched


def test_the_correction_and_the_gain_both_train_and_can_override_a_misleading_prior():
    m = small(True).train()
    b = prior_batch([("choice", 4)] * 6)
    target = torch.zeros(24)
    target[::4] = 1.0  # the right answer is always the first option
    b.prior = -10 * target  # a prior that confidently prefers every option but the right one
    opt = torch.optim.Adam(m.parameters(), lr=2e-2)
    from imagejev.model.losses import fusion_loss

    first = None
    for _ in range(150):
        out = fusion_loss(m(b), b, target)
        first = first or float(out.total.detach())
        opt.zero_grad()
        out.total.backward()
        opt.step()
    assert float(out.total.detach()) < 0.3 * first  # learned to ignore or invert the prior
    assert not torch.allclose(m.prior_gain, torch.ones(3))  # the gain moved
    assert m.heads[0][-1].weight.abs().sum() > 0  # the zero-initialised correction is now non-zero


def test_non_prior_models_have_no_gain_parameter():
    assert not hasattr(small(False), "prior_gain") and "prior_gain" in small(True).state_dict()


# ---- the batch builder and the serving backend must agree ---------------------------------
def make_records():
    from imagejev.data.records import QuestionRecord

    return [
        QuestionRecord("img0", "photo", "s", "t.choice", CHOICE, "red"),
        QuestionRecord("img0", "photo", "s", "t.score", SCORE, "mid"),
        QuestionRecord("img1", "photo", "s", "t.bool", BOOL, True),
    ]


def test_builder_prior_matches_the_formula_row_by_row():
    f = {"img0": feats(0), "img1": feats(1)}
    emb = FakeEmbedder()
    prep = BatchBuilder(f, tiny_text_encoder(), "cpu", prior=emb).build(make_records())
    assert prep.batch.prior.shape == (3 + 3 + 1,)
    scale, bias = emb.scale_bias
    want = np.concatenate([
        prior_from_embeddings(emb.embed(prior_texts(parse_question("q", CHOICE))), f["img0"][0], scale, bias, "choice"),
        prior_from_embeddings(emb.embed(prior_texts(parse_question("q", SCORE))), f["img0"][0], scale, bias, "score"),
        prior_from_embeddings(emb.embed(prior_texts(parse_question("q", BOOL))), f["img1"][0], scale, bias, "bool"),
    ])  # fmt: skip
    assert np.allclose(prep.batch.prior.numpy(), want, atol=1e-5)
    assert BatchBuilder(f, tiny_text_encoder(), "cpu").build(make_records()).batch.prior is None


class FakeVision:
    encoder_id = "fake:v1"

    def encode_image(self, image):
        return feats(int(np.asarray(image)[0, 0, 0]))


def test_serving_backend_computes_the_same_logits_as_training_would():
    from PIL import Image

    text, fusion = tiny_text_encoder(seed=3), tiny_fusion(seed=3)
    fusion = FusionModel(
        d_text=32, d_image=D_IMG, d=32, heads=4, n_blocks=2, adapter_hidden=48, use_prior=True
    )
    with torch.no_grad():
        for p in fusion.heads.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # a non-trivial correction
    emb = FakeEmbedder()
    backend = TrainedBackend(text, fusion, FakeVision(), "cpu", prior=emb)
    img = Image.new("RGB", (4, 4), (5, 0, 0))
    handle = CachedEncoder(backend).encode(img)
    recs = make_records()[:2]
    prep = BatchBuilder({"img0": np.asarray(handle.features)}, text.eval(), "cpu", prior=emb).build(
        recs
    )
    with torch.no_grad():
        train_path = fusion.eval()(prep.batch).numpy()
    serve = np.concatenate([backend.logits(handle, "", q) for q in prep.questions])
    assert np.allclose(train_path, serve, atol=1e-5)


def test_export_and_load_keep_the_prior(tmp_path):
    text = tiny_text_encoder(seed=0)
    fusion = FusionModel(
        d_text=32, d_image=D_IMG, d=32, heads=4, n_blocks=2, adapter_hidden=48, use_prior=True
    )
    cfg = {
        "backbone": "tiny",
        "lora_rank": 4,
        "lora_alpha": 32.0,
        "d_model": 32,
        "heads": 4,
        "n_blocks": 2,
    }
    torch.save(
        {
            "fusion": fusion.state_dict(),
            "lora": text.lora_state_dict(),
            "config": cfg,
            "step": 1,
            "best": 1.0,
        },
        tmp_path / "ck.pt",
    )
    d = export_model(tmp_path / "ck.pt", tmp_path / "m", vision_checkpoint="fake")
    assert __import__("json").loads((d / "config.json").read_text())["fusion"]["use_prior"] is True
    backend, _ = load_trained(
        d, device="cpu", text=tiny_text_encoder(seed=0), vision=FakeVision(), prior=FakeEmbedder()
    )
    assert backend.fusion.use_prior and torch.equal(backend.fusion.prior_gain, fusion.prior_gain)
