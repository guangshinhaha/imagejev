import csv
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from imagejev.data.records import QuestionRecord  # noqa: E402
from imagejev.model.testing import tiny_fusion, tiny_text_encoder  # noqa: E402
from imagejev.schema import parse_question  # noqa: E402
from imagejev.train.data import BatchBuilder, option_texts, target_vector  # noqa: E402
from imagejev.train.loop import TrainConfig, Trainer, lr_factor, macro_log_loss  # noqa: E402

LEVELS = ["lo", "mid", "hi"]
CHOICE = {
    "type": "choice",
    "instructions": "Which class?",
    "criteria": {"a": "first", "b": "second", "c": "third"},
}
SCORE = {"type": "score", "instructions": "How high?", "levels": LEVELS}
BOOL = {"type": "bool", "instructions": "Is it class zero?"}


def toy_world(n=60, seed=0):
    """The answer to every question is encoded in the image features, so it is learnable.

    The signal is on every image token. Putting it on the single global token only is *not* learned
    by a tiny random model in a few hundred steps (cross-attention has to discover 1 informative
    token among 65), while patch-only and all-token signals reach 100%; see the PR for #29.
    """
    rng = np.random.default_rng(seed)
    feats, recs = {}, []
    for i in range(n):
        c = i % 3
        f = rng.normal(0, 0.3, (65, 24)).astype(np.float32)
        f[:, c] += 3.0
        feats[f"img{i}"] = f
        domain = "photo" if i % 2 == 0 else "document"
        recs += [
            QuestionRecord(f"img{i}", domain, "s", "t.choice", CHOICE, "abc"[c]),
            QuestionRecord(f"img{i}", domain, "s", "t.score", SCORE, LEVELS[c]),
            QuestionRecord(f"img{i}", domain, "s", "t.bool", BOOL, c == 0),
        ]
    return feats, recs


def make_trainer(tmp_path, steps=60, seed=0, **over):
    feats, recs = toy_world()
    cfg = TrainConfig(
        seed=seed, device="cpu", steps=steps, eval_every=20, ckpt_every=1, images_per_batch=6,
        questions_per_image=3, teacher_fraction=0.0, lr_fusion=3e-3, lr_lora=3e-3, warmup_steps=5,
        d_model=32, n_blocks=2, lora_rank=4, train_file="", select_on="val", out_dir=str(tmp_path),
        **over,
    )  # fmt: skip
    return Trainer(cfg, tiny_text_encoder(), tiny_fusion(), feats, recs[:120], {"val": recs[120:]})


# ---- pure helpers -------------------------------------------------------------------------
def test_lr_schedule_warmup_then_cosine():
    vals = [lr_factor(s, warmup=10, total=110, min_frac=0.1) for s in range(0, 111, 10)]
    assert vals[0] == pytest.approx(0.1) and vals[1] == pytest.approx(1.0)  # (0+1)/10, then peak
    assert all(a >= b for a, b in zip(vals[1:], vals[2:], strict=False))  # monotone decay
    assert vals[-1] == pytest.approx(0.1)
    assert lr_factor(60, 10, 110, 0.1) == pytest.approx(0.55)  # halfway down the cosine


def test_macro_log_loss_ignores_rollups_and_tiny_cells():
    rows = [
        {"domain": "photo", "qtype": "bool", "n": 50, "log_loss": 0.6},
        {"domain": "photo", "qtype": "choice", "n": 50, "log_loss": 1.0},
        {"domain": "photo", "qtype": "score", "n": 3, "log_loss": 9.0},  # too small
        {"domain": "all", "qtype": "all", "n": 103, "log_loss": 5.0},
        {"domain": "photo", "qtype": "all", "n": 103, "log_loss": 5.0},
    ]
    assert macro_log_loss(rows) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        macro_log_loss([rows[3]])


def test_config_from_yaml_and_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("steps: 12\nlr_fusion: 0.001\neval_files:\n  val: a.jsonl\n")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.steps == 12 and cfg.lr_fusion == 0.001 and cfg.eval_files == {"val": "a.jsonl"}
    p.write_text("stepz: 12\n")
    with pytest.raises(ValueError, match="stepz"):
        TrainConfig.from_yaml(p)


def test_target_vectors_and_option_texts():
    q = parse_question("c", CHOICE)
    assert target_vector(QuestionRecord("i", "photo", "s", "t", CHOICE, "b"), q) == [0.0, 1.0, 0.0]
    soft = QuestionRecord("i", "photo", "s", "t", CHOICE, "a", soft={"a": 0.6, "b": 0.3, "c": 0.1})
    assert target_vector(soft, q) == [0.6, 0.3, 0.1]
    b = parse_question("b", BOOL)
    assert target_vector(QuestionRecord("i", "photo", "s", "t", BOOL, True), b) == [1.0]
    assert option_texts(b) == [None] and option_texts(q) == ["a: first", "b: second", "c: third"]


def test_batch_builder_shapes_and_alignment():
    feats, recs = toy_world(6)
    builder = BatchBuilder(feats, tiny_text_encoder(), "cpu")
    prep = builder.build(recs[:6])  # two images x (choice, score, bool)
    b = prep.batch
    n = 3 + 3 + 1 + 3 + 3 + 1
    assert b.opt_tokens.shape[0] == n == prep.targets.shape[0] and b.image.shape == (6, 65, 24)
    assert b.qtype.tolist() == [0, 1, 2, 0, 1, 2] and b.owner.tolist()[:7] == [0, 0, 0, 1, 1, 1, 2]
    assert torch.equal(b.image[0], torch.from_numpy(feats["img0"]).float())
    assert prep.targets[:3].tolist() == [1.0, 0.0, 0.0] and prep.targets[6].item() == 1.0


# ---- the loop -----------------------------------------------------------------------------
def test_training_learns_the_toy_task_and_writes_logs(tmp_path):
    t = make_trainer(tmp_path, steps=150)
    before = macro_log_loss(
        __import__("imagejev.bench.metrics", fromlist=["evaluate"]).evaluate(
            t.predict(t.eval_sets["val"])
        )
    )
    t.run()
    after_rows = [
        r for r in csv.DictReader((tmp_path / "eval_log.csv").open()) if int(r["step"]) == 150
    ]
    after = np.mean(
        [float(r["log_loss"]) for r in after_rows if r["domain"] != "all" and r["qtype"] != "all"]
    )
    assert after < 0.6 * before and after < 0.5, (before, after)
    acc = {r["qtype"]: float(r["accuracy"]) for r in after_rows if r["domain"] == "all"}
    assert acc["bool"] > 0.85 and acc["choice"] > 0.8 and acc["score"] > 0.8, acc
    train = list(csv.DictReader((tmp_path / "train_log.csv").open()))
    assert [int(r["step"]) for r in train] == list(range(1, 151))
    assert float(train[-1]["loss"]) < float(train[0]["loss"])
    sel = list(csv.DictReader((tmp_path / "select_log.csv").open()))
    assert [int(r["step"]) for r in sel] == [20, 40, 60, 80, 100, 120, 140, 150]
    assert any(r["best"] == "1" for r in sel)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()


def params_of(t):
    return [
        p.detach().clone() for p in list(t.fusion.parameters()) + list(t.text.lora_parameters())
    ]


@pytest.fixture
def single_thread():
    """Multi-threaded CPU matmuls reduce in a nondeterministic order: two identical uninterrupted
    runs differ by ~1e-3 with 8 threads and are bit-identical with 1. Exact-resume is only
    provable single-threaded."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_a_killed_run_resumes_to_exactly_the_uninterrupted_result(tmp_path, single_thread):
    straight = make_trainer(tmp_path / "straight", steps=14)
    straight.run()

    class Crash(Exception):
        pass

    def crash_at_8(step):
        if step == 8:
            raise Crash

    first = make_trainer(tmp_path / "killed", steps=14)
    with pytest.raises(Crash):
        first.run(on_step=crash_at_8)  # dies after step 8's update, before its checkpoint
    assert first.step == 8
    resumed = make_trainer(tmp_path / "killed", steps=14)  # a fresh process: new models, same dir
    resumed.run()
    assert resumed.step == 14
    for a, b in zip(params_of(straight), params_of(resumed), strict=True):
        assert torch.equal(a, b), float((a - b).abs().max())  # not merely close: identical
    a = [float(r["loss"]) for r in csv.DictReader((tmp_path / "straight" / "train_log.csv").open())]
    b = [float(r["loss"]) for r in csv.DictReader((tmp_path / "killed" / "train_log.csv").open())]
    assert len(a) == len(b) == 14 and a == b  # no duplicated or missing steps, same losses


def test_resume_picks_up_the_best_score_and_does_not_restart(tmp_path):
    t = make_trainer(tmp_path, steps=20)
    t.run()
    assert math.isfinite(t.best) and t.step == 20
    again = make_trainer(tmp_path, steps=20)
    again.run()  # nothing left to do
    assert again.step == 20 and again.best == pytest.approx(t.best)
    longer = make_trainer(tmp_path, steps=30)
    longer.run()
    assert longer.step == 30


def test_selection_split_must_exist(tmp_path):
    feats, recs = toy_world(6)
    cfg = TrainConfig(device="cpu", select_on="nope", out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="select_on"):
        Trainer(cfg, tiny_text_encoder(), tiny_fusion(), feats, recs, {"val": recs})


def test_amp_is_ignored_off_cuda(tmp_path):
    t = make_trainer(tmp_path, steps=2, amp="fp16")
    assert t.use_amp is False
    t.run()
    assert t.step == 2


def test_time_budget_stops_saves_and_chained_sessions_equal_one_run(tmp_path, single_thread):
    straight = make_trainer(tmp_path / "straight", steps=12)
    assert straight.run() is True

    out = tmp_path / "sessions"
    sessions, finished = 0, False
    while not finished:
        t = make_trainer(out, steps=12, time_budget_minutes=1e-9)  # a budget that is already spent
        finished = t.run()  # a fresh "session": new process state, same output directory
        sessions += 1
        assert t.step >= sessions or finished  # every session makes progress
        assert (out / "last.pt").exists()
        assert sessions < 40
    assert sessions == 12 and t.step == 12  # one step per session with such a budget
    for a, b in zip(params_of(straight), params_of(t), strict=True):
        assert torch.equal(a, b)
    losses = [float(r["loss"]) for r in csv.DictReader((out / "train_log.csv").open())]
    assert len(losses) == 12 and losses == [
        float(r["loss"]) for r in csv.DictReader((tmp_path / "straight" / "train_log.csv").open())
    ]


def test_a_generous_budget_does_not_interrupt(tmp_path):
    t = make_trainer(tmp_path, steps=6, time_budget_minutes=60)
    assert t.run() is True and t.step == 6


def test_bf16_autocast_wiring_runs_on_cpu(tmp_path):
    t = make_trainer(tmp_path, steps=3, amp="bf16")
    assert t.use_amp is True
    assert t.run() is True
    losses = [float(r["loss"]) for r in csv.DictReader((tmp_path / "train_log.csv").open())]
    assert len(losses) == 3 and all(math.isfinite(x) for x in losses)
    preds = t.predict(t.eval_sets["val"][:6])
    assert len(preds) == 6 and all(abs(p.p.sum() - 1) < 1e-3 for p in preds)
