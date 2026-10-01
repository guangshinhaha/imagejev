import json

import numpy as np
import pytest
from PIL import Image

from imagejev.cache import CacheWriter, FeatureCache, build_cache, items_from_facts


class FakeEncoder:
    """Features are a deterministic function of the image's red channel, so reruns must agree."""

    encoder_id = "fake:v1"

    def __init__(self, fail_after_batches=None):
        self.batches = 0
        self.fail_after = fail_after_batches

    def encode_images(self, images):
        if self.fail_after is not None and self.batches >= self.fail_after:
            raise KeyboardInterrupt("simulated crash")
        self.batches += 1
        out = np.zeros((len(images), 5, 8), dtype=np.float32)
        for k, im in enumerate(images):
            out[k] = im.getpixel((0, 0))[0] + np.arange(40).reshape(5, 8) / 100
        return out


def make_items(n, tmp_path=None):
    return [(f"img{i:03d}", Image.new("RGB", (8, 8), (i, 0, 0))) for i in range(n)]


def read_all(root):
    c = FeatureCache(root)
    return {i: c[i] for i in c.ids()}


def test_build_and_read_back(tmp_path):
    counts = build_cache(make_items(50), FakeEncoder(), tmp_path, batch_size=8, shard_size=16)
    assert counts == {"encoded": 50, "skipped_existing": 0, "failed": 0}
    c = FeatureCache(tmp_path)
    assert len(c) == 50 and "img007" in c and "nope" not in c
    f = c["img007"]
    assert f.dtype == np.float16 and f.shape == (5, 8)
    assert np.allclose(f, 7 + np.arange(40).reshape(5, 8) / 100, atol=1e-2)
    assert c.get_many(["img001", "img002"]).shape == (2, 5, 8)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta == {"encoder_id": "fake:v1", "shape": [5, 8], "dtype": "float16"}
    shards = sorted(tmp_path.glob("shard-*.npy"))
    assert len(shards) == 4  # 16 + 16 + 16 + 2
    assert [np.load(s, mmap_mode="r").shape[0] for s in shards] == [16, 16, 16, 2]


def test_interrupt_and_resume_gives_identical_result(tmp_path):
    clean = tmp_path / "clean"
    build_cache(make_items(70), FakeEncoder(), clean, batch_size=8, shard_size=16)
    crashed = tmp_path / "crashed"
    with pytest.raises(KeyboardInterrupt):
        build_cache(make_items(70), FakeEncoder(fail_after_batches=5), crashed, batch_size=8,
                    shard_size=16)  # fmt: skip
    partial = FeatureCache(crashed)
    assert 0 < len(partial) < 70  # progress was kept
    enc = FakeEncoder()
    counts = build_cache(make_items(70), enc, crashed, batch_size=8, shard_size=16)
    assert counts["skipped_existing"] == len(partial)
    assert counts["encoded"] == 70 - len(partial)
    a, b = read_all(clean), read_all(crashed)
    assert a.keys() == b.keys() and len(b) == 70
    assert all(np.array_equal(a[k], b[k]) for k in a)
    # nothing was stored twice
    all_ids = [
        i for p in sorted(crashed.glob("shard-*.json")) for i in json.loads(p.read_text())["ids"]
    ]
    assert len(all_ids) == len(set(all_ids)) == 70


def test_half_written_shards_are_ignored(tmp_path):
    build_cache(make_items(20), FakeEncoder(), tmp_path, batch_size=4, shard_size=8)
    (tmp_path / "shard-00099.npy").write_bytes(b"garbage")  # npy written, manifest missing
    (tmp_path / "shard-00100.npy.tmp").write_bytes(b"partial")
    c = FeatureCache(tmp_path)
    assert len(c) == 20
    w = CacheWriter(tmp_path, "fake:v1", 8)  # opening a writer clears stale tmp files
    assert not list(tmp_path.glob("*.tmp")) and len(w.done) == 20


def test_rerun_with_everything_cached_encodes_nothing(tmp_path):
    build_cache(make_items(30), FakeEncoder(), tmp_path, batch_size=8, shard_size=16)
    enc = FakeEncoder()
    counts = build_cache(make_items(30), enc, tmp_path, batch_size=8, shard_size=16)
    assert counts["encoded"] == 0 and enc.batches == 0 and counts["skipped_existing"] == 30


def test_encoder_mismatch_is_refused(tmp_path):
    build_cache(make_items(4), FakeEncoder(), tmp_path)

    class Other(FakeEncoder):
        encoder_id = "other"

    with pytest.raises(ValueError, match="built with"):
        build_cache(make_items(4), Other(), tmp_path)


def test_unreadable_images_are_recorded_and_skipped(tmp_path):
    items = make_items(6)
    items.insert(2, ("broken", b"not an image"))
    items.insert(5, ("missing", tmp_path / "nope.png"))
    counts = build_cache(items, FakeEncoder(), tmp_path / "c", batch_size=3, shard_size=4)
    assert counts["encoded"] == 6 and counts["failed"] == 2
    failed = [
        json.loads(line) for line in (tmp_path / "c" / "failed.jsonl").read_text().splitlines()
    ]
    assert {f["id"] for f in failed} == {"broken", "missing"}
    assert "broken" not in FeatureCache(tmp_path / "c")


def test_duplicate_ids_in_the_input_are_encoded_once(tmp_path):
    items = make_items(5) + make_items(5)
    counts = build_cache(items, FakeEncoder(), tmp_path, batch_size=2, shard_size=4)
    assert counts["encoded"] == 5 and len(FeatureCache(tmp_path)) == 5


@pytest.mark.parametrize("workers", [1, 4])
def test_result_does_not_depend_on_worker_count(tmp_path, workers):
    build_cache(make_items(40), FakeEncoder(), tmp_path / str(workers), batch_size=5, shard_size=16,
                num_workers=workers)  # fmt: skip
    got = read_all(tmp_path / str(workers))
    assert len(got) == 40 and np.allclose(got["img010"][0, 0], 10, atol=1e-2)


def test_missing_cache_and_item_helpers(tmp_path):
    with pytest.raises(FileNotFoundError):
        FeatureCache(tmp_path / "none")
    root = tmp_path / "imgs"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(root / "images" / "a.png")
    facts = tmp_path / "facts.jsonl"
    facts.write_text(
        json.dumps({"image_id": "a", "image_path": "images/a.png"}) + "\n"
        + json.dumps({"image_id": "b", "image_path": None}) + "\n\n"
    )  # fmt: skip
    assert list(items_from_facts(facts, root)) == [("a", root / "images" / "a.png")]
    c = build_cache(items_from_facts(facts, root), FakeEncoder(), tmp_path / "c")
    assert c["encoded"] == 1


def test_wrong_batch_size_from_encoder_is_an_error(tmp_path):
    class Bad(FakeEncoder):
        def encode_images(self, images):
            return super().encode_images(images)[:-1]

    with pytest.raises(ValueError, match="wrong batch size"):
        build_cache(make_items(4), Bad(), tmp_path, batch_size=4)
