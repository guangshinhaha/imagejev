import json
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imagejev.data.quality import (
    KINDS,
    Source,
    degrade,
    generate,
    is_good_source,
    iter_quality,
    mean_luma,
    noise_level,
    sharpness,
)
from imagejev.data.records import ImageFacts, QuestionRecord, read_jsonl


def textured(size=(512, 384), seed=0, base=128):
    """A photo-like test image: smooth blobs plus hard edges and fine lines."""
    rng = np.random.default_rng(seed)
    small = rng.normal(base, 40, (size[1] // 16, size[0] // 16, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(small).resize(size, Image.BICUBIC)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 40)
        d.rectangle([x, y, x + 40, y + 20], outline=tuple(int(v) for v in rng.integers(0, 255, 3)))
        d.line([x, y, x + 55, y + 35], fill=(20, 20, 20), width=1)
    return img


def page(size=(600, 800)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for i in range(30):
        d.text(
            (40, 30 + i * 22),
            "The quick brown fox jumps over the lazy dog 0123456789",
            fill="black",
        )
    return img


def level_ranges(img, kind, metric, seeds=range(8)):
    spec = KINDS[kind]
    return [
        (min(v), max(v))
        for v in (
            [metric(degrade(img, kind, lvl, random.Random(s))[0]) for s in seeds]
            for lvl in range(len(spec.levels))
        )
    ]


def blur_metric(img):
    return sharpness(img, 128)  # at 512 px heavy blurs are indistinguishable; see sharpness()


@pytest.mark.parametrize("source", [textured, page], ids=["photo", "page"])
def test_blur_levels_are_strictly_ordered_on_measurements(source):
    ranges = level_ranges(source(), "blur", blur_metric)
    # level 0 is the blurriest, so sharpness must rise with every level, with no overlap
    for (_, prev_max), (next_min, _) in zip(ranges, ranges[1:], strict=False):
        assert prev_max < next_min, ranges


@pytest.mark.parametrize(
    ("kind", "metric"), [("noise", noise_level), ("brightness", mean_luma)], ids=["noise", "bright"]
)
def test_noise_and_brightness_levels_are_strictly_ordered(kind, metric):
    ranges = level_ranges(textured(), kind, metric)
    for (_, prev_max), (next_min, _) in zip(ranges, ranges[1:], strict=False):
        assert prev_max < next_min, ranges


def test_jpeg_levels_increase_distortion_and_tilt_expands_canvas():
    img = textured()
    diffs = []
    for lvl in range(4):
        out, params = degrade(img, "jpeg", lvl, random.Random(1))
        diffs.append(float(np.abs(np.asarray(out, float) - np.asarray(img, float)).mean()))
        assert (lvl == 0) == (params == {})
    assert diffs[0] == 0.0 and diffs[1] < diffs[2] < diffs[3]
    seen = set()
    for s in range(20):
        out, p = degrade(page(), "tilt", 2, random.Random(s))
        assert 5.0 <= abs(p["degrees"]) <= 8.0 and out.size[0] > 600
        seen.add(p["degrees"] > 0)
    assert seen == {True, False}


def test_untouched_level_returns_an_unmodified_copy():
    img = textured()
    out, params = degrade(img, "blur", 3, random.Random(0))
    assert params == {} and out is not img
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_is_good_source_by_domain():
    assert is_good_source(textured(), "photo")
    assert not is_good_source(Image.new("RGB", (400, 400), (120, 120, 120)))  # flat
    assert not is_good_source(textured(size=(160, 160)))  # too small
    assert not is_good_source(textured(base=235))  # washed-out photo
    assert is_good_source(page(), "document")  # white page is normal for documents
    assert not is_good_source(page(), "photo")


def test_iter_quality_labels_match_parameters_and_respect_domains():
    srcs = [
        Source("p1", "photo", textured(seed=1)),
        Source("p2", "photo", textured(seed=2)),
        Source("d1", "document", page(), style="invoice:a"),
        Source("flat", "photo", Image.new("RGB", (400, 400), (120, 120, 120))),
    ]
    out = list(iter_quality(srcs, seed=0, kinds_per_source=3))
    assert {f.facts["source_id"] for _, f, _ in out} == {"p1", "p2", "d1"}  # flat one skipped
    for _img, f, q in out:
        kind = f.facts["kind"]
        assert q.task == f"quality.{kind}" and q.answer == f.facts["level"]
        spec = KINDS[kind]
        lvl = spec.levels.index(q.answer)
        band = spec.bands[lvl]
        if band is None:
            assert not q.meta.get("radius_frac") and "sigma" not in q.meta
        elif kind == "tilt":
            assert band[0] <= abs(q.meta["degrees"]) <= band[1]
        else:
            assert any(band[0] <= v <= band[1] for k, v in q.meta.items() if k != "source_id")
        assert f.domain in spec.domains
        assert q.meta["source_id"] == f.facts["source_id"]
    assert {f.facts["kind"] for _, f, _ in out if f.facts["source_id"] == "d1"} <= {
        "blur", "noise", "jpeg", "tilt",
    }  # fmt: skip
    assert out == [] or [q.answer for *_, q in out] == [
        q.answer for *_, q in iter_quality(srcs, seed=0, kinds_per_source=3)
    ]
    assert next(q for *_, q in out if q.style).style == "invoice:a"


def test_generate_writes_images_and_records(tmp_path):
    srcs = [Source(f"p{i}", "photo", textured(seed=i)) for i in range(6)]
    counts = generate(srcs, tmp_path, seed=3, kinds_per_source=2)
    assert counts["images"] == 12
    facts = list(read_jsonl(tmp_path / "facts.jsonl", ImageFacts))
    qs = list(read_jsonl(tmp_path / "questions.jsonl", QuestionRecord))
    assert len(facts) == len(qs) == 12
    for f, q in zip(facts, qs, strict=True):
        assert (tmp_path / f.image_path).exists()
        assert f.image_path.endswith(".png") == (q.task == "quality.jpeg")
    json.dumps([q.meta for q in qs])
