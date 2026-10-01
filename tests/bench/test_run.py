import json

import numpy as np
import pytest
from PIL import Image

from imagejev.api import Model
from imagejev.bench.run import (
    fit_temperatures,
    group_by_image,
    image_resolver,
    measure_latency,
    option_labels,
    render_report,
    run_benchmark,
    run_split,
    to_prediction,
)
from imagejev.data.records import ImageFacts, QuestionRecord, write_jsonl

BOOL = {"type": "bool", "instructions": "Is it red?"}
CHOICE = {
    "type": "choice",
    "instructions": "Colour?",
    "criteria": {"red": "", "green": "", "blue": ""},
}
SCORE = {"type": "score", "instructions": "How bright?", "levels": ["dark", "mid", "bright"]}
COLORS = {"red": (200, 0, 0), "green": (0, 200, 0), "blue": (0, 0, 200)}


class ColorBackend:
    """Reads the dominant channel of the image; right most of the time, so metrics are non-trivial."""

    encoder_id = "color"

    def __init__(self):
        self.encodes = 0

    def encode_image(self, image):
        self.encodes += 1
        return np.array([image.resize((1, 1)).getpixel((0, 0))], dtype=np.float32)  # (1, 3)

    def logits(self, handle, state, question):
        r, g, b = (float(x) for x in handle.features[0])
        if question.type == "bool":
            return np.array([(r - max(g, b)) / 20])
        if question.type == "score":
            lum = (r + g + b) / 3
            return -np.abs(np.array([0.0, 1.0, 2.0]) - lum / 100.0) * 3
        by = {"red": r, "green": g, "blue": b}
        return np.array([by[o.label] / 20 for o in question.options])


def model():
    m = Model(ColorBackend())
    m.name = "color"
    return m


def make_records(n_per_color=8):
    recs, imgs = [], {}
    for color, rgb in COLORS.items():
        for i in range(n_per_color):
            iid = f"{color}{i}"
            img = Image.new("RGB", (4, 4), rgb)
            img.putpixel((1, 1), (i, i, i))  # distinct pixels, or the content-keyed cache dedupes
            imgs[iid] = img
            recs += [
                QuestionRecord(iid, "photo", "s", "t.choice", CHOICE, color),
                QuestionRecord(iid, "photo", "s", "t.red", BOOL, color == "red"),
                QuestionRecord(iid, "photo", "s", "t.score", SCORE, "mid"),
            ]
    return recs, imgs


def test_option_labels_and_to_prediction():
    assert option_labels(CHOICE) == ["red", "green", "blue"] and option_labels(SCORE) == [
        "dark",
        "mid",
        "bright",
    ]
    rec = QuestionRecord("i", "photo", "s", "t", CHOICE, "green")
    p = to_prediction(rec, {"probs": {"red": 0.2, "green": 0.7, "blue": 0.1}})
    assert (
        p.label == 1
        and p.pred == 1
        and p.qtype == "choice"
        and p.p.tolist() == pytest.approx([0.2, 0.7, 0.1])
    )
    b = to_prediction(QuestionRecord("i", "photo", "s", "t", BOOL, False), {"p_true": 0.9})
    assert b.label == 0 and b.pred == 1 and b.qtype == "bool"


def test_run_split_scores_every_question_and_encodes_each_image_once():
    recs, imgs = make_records()
    m = model()
    preds, stats = run_split(m, recs, imgs.__getitem__)
    assert stats == {"questions": 72, "images": 24, "skipped_images": 0, "skipped_questions": 0}
    assert m.backend.encodes == 24  # three questions per image, one encode
    assert len(preds) == 72
    choice = [p for p in preds if p.qtype == "choice"]
    assert np.mean([p.pred == p.label for p in choice]) == 1.0


def test_unloadable_images_are_counted_not_fatal():
    recs, imgs = make_records(2)
    del imgs["red0"]
    preds, stats = run_split(model(), recs, imgs.__getitem__)
    assert stats["skipped_images"] == 1 and stats["skipped_questions"] == 3 and len(preds) == 15


def test_grouping_preserves_order():
    recs, _ = make_records(2)
    assert list(group_by_image(recs))[:3] == ["red0", "red1", "green0"]


def test_fit_temperatures_updates_model_and_reports():
    recs, imgs = make_records(12)
    m = model()
    before = dict(m.temperatures)
    fitted = fit_temperatures(m, recs, imgs.__getitem__, min_examples=20)
    assert set(fitted) == {"choice", "bool", "bool_bias", "score"}
    assert all(m.temperatures[k] == fitted[k] for k in fitted) and m.temperatures != before
    m2 = model()
    assert fit_temperatures(m2, recs[:9], imgs.__getitem__, min_examples=20) == {}  # too few
    assert m2.temperatures == before


def test_latency_has_cold_and_warm_and_warm_skips_the_encoder():
    recs, imgs = make_records(10)
    m = model()
    lat = measure_latency(m, recs, imgs.__getitem__, n_images=8, warmup=2)
    assert lat["cold"]["n"] == 8 and lat["warm"]["n"] == 8
    assert lat["cold"]["p95_ms"] >= lat["cold"]["p50_ms"] > 0

    class NoEncode:
        name = "plain"

        def predict(self, image, questions, state=None):
            return {k: {"p_true": 0.5} for k in questions}

    plain = measure_latency(NoEncode(), recs[:30], imgs.__getitem__, n_images=3, warmup=0)
    assert plain["cold"]["n"] == 3 and plain["warm"] is None


def test_image_resolver_and_end_to_end_report(tmp_path):
    recs, imgs = make_records(12)
    root = tmp_path / "imgs"
    root.mkdir()
    facts = []
    for iid, im in imgs.items():
        im.save(root / f"{iid}.png")
        facts.append(ImageFacts(iid, "photo", "s", {}, image_path=f"{iid}.png"))
    write_jsonl(facts, tmp_path / "facts.jsonl")
    resolve = image_resolver([(tmp_path / "facts.jsonl", root)])
    assert resolve("red0") == root / "red0.png"
    with pytest.raises(KeyError):
        resolve("nope")

    split_files = {}
    for name, part in {"val": recs[:60], "test-images": recs[60:]}.items():
        write_jsonl(part, tmp_path / f"{name}.jsonl")
        split_files[name] = tmp_path / f"{name}.jsonl"
    m = model()
    result = run_benchmark(m, split_files, lambda i: Image.open(resolve(i)), tmp_path / "out",
                           fit_on="val", latency_images=4)  # fmt: skip
    assert set(result["splits"]) == {"val", "test-images"}
    assert result["fitted_temperatures"] and result["latency"]["cold"]
    rep = (tmp_path / "out" / "report.md").read_text()
    assert (
        rep.startswith("# color")
        and "### test-images" in rep
        and "Temperatures fitted on val" in rep
    )
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["model"] == "color" and saved["splits"]["val"]["stats"]["questions"] == 60
    assert render_report(saved) == rep


def test_subsample_is_deterministic_keeps_whole_images_and_ignores_order():
    from imagejev.bench.run import subsample_images

    recs, _ = make_records(12)
    sub = subsample_images(recs, 10, seed=3)
    ids = {r.image_id for r in sub}
    assert len(ids) == 10 and all(
        len([r for r in sub if r.image_id == i]) == 3 for i in ids
    )  # all questions kept
    assert subsample_images(list(reversed(recs)), 10, seed=3) == [
        r for r in reversed(recs) if r.image_id in ids
    ]
    assert {r.image_id for r in subsample_images(recs, 10, seed=4)} != ids
    assert subsample_images(recs, None) == recs and len(subsample_images(recs, 10**6)) == len(recs)


def test_run_benchmark_applies_the_cap_and_reports_it(tmp_path):
    recs, imgs = make_records(12)
    write_jsonl(recs, tmp_path / "val.jsonl")
    res = run_benchmark(model(), {"val": tmp_path / "val.jsonl"}, imgs.__getitem__, tmp_path / "o",
                        fit_on=None, latency_images=2, max_images_per_split=7)  # fmt: skip
    assert res["splits"]["val"]["stats"]["images"] == 7 and res["max_images_per_split"] == 7
    assert "at most 7 images" in (tmp_path / "o" / "report.md").read_text()
