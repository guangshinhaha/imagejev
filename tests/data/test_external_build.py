import io
import json
from pathlib import Path

import pytest
from PIL import Image

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from imagejev.data.external_build import (  # noqa: E402
    build_external,
    class_names,
    download_external,
)
from imagejev.data.splits import build_splits, check_splits  # noqa: E402

BREEDS = ["abyssinian", "bengal", "beagle", "boxer"]  # two cats, two dogs


def png(color=(120, 30, 30), size=(40, 30)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, format="PNG")
    return {"bytes": b.getvalue(), "path": "x.png"}


def write_pets(path, n=24):
    rows = [
        {
            "image": png((i * 9 % 255, 10, 10)),
            "label": i % 4,
            "image_id": f"pet_{i}",
            "label_cat_dog": 0 if i % 4 < 2 else 1,
        }
        for i in range(n)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def gt(parse):
    return json.dumps({"gt_parse": parse})


def write_cord(path, n=6):
    rows = [
        {
            "image": png(),
            "ground_truth": gt({"menu": [{"nm": "a"}] * (1 + i % 4), "total": {"cashprice": "9"}}),
        }
        for i in range(n)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_screenspot(path):
    rows = []
    for name, src in (("a.png", "ios"), ("a.png", "ios"), ("b.png", "shop"), ("c.png", "macos")):
        rows.append({"file_name": name, "data_source": src, "image": png(size=(60, 40))})
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def parquet_paths(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_pets(raw / "pets.parquet")
    write_cord(raw / "test-00000.parquet")
    write_cord(raw / "validation-00000.parquet", n=4)
    write_screenspot(raw / "ss.parquet")
    return {
        "pets": [raw / "pets.parquet"],
        "cord": [raw / "test-00000.parquet", raw / "validation-00000.parquet"],
        "screenspot": [raw / "ss.parquet"],
    }


def test_build_external_writes_images_and_ext_records(tmp_path, parquet_paths):
    recs, facts = build_external(
        parquet_paths, tmp_path / "out", pets_breeds=BREEDS, pets_images=10, seed=0
    )
    assert recs and facts
    assert all(r.source.startswith("ext:") for r in recs) and all(
        f.source.startswith("ext:") for f in facts
    )
    by_domain = {
        d: {f.source for f in facts if f.domain == d} for d in ("photo", "document", "screenshot")
    }
    assert by_domain == {
        "photo": {"ext:oxford-iiit-pet"},
        "document": {"ext:cord-v2"},
        "screenshot": {"ext:screenspot"},
    }
    for f in facts:
        assert (
            f.image_path
            and Path(f.image_path).is_absolute()
            and Image.open(f.image_path).mode == "RGB"
        )
    assert sum(f.source == "ext:oxford-iiit-pet" for f in facts) == 10  # the requested subset size
    assert sum(f.source == "ext:cord-v2" for f in facts) == 10  # 6 test + 4 validation receipts
    assert sum(f.source == "ext:screenspot" for f in facts) == 3  # distinct screenshots, not rows


def test_subset_is_deterministic_and_independent_of_row_order(tmp_path, parquet_paths):
    a, _ = build_external(
        {"pets": parquet_paths["pets"]}, tmp_path / "a", pets_breeds=BREEDS, pets_images=8, seed=1
    )
    b, _ = build_external(
        {"pets": parquet_paths["pets"]}, tmp_path / "b", pets_breeds=BREEDS, pets_images=8, seed=1
    )
    c, _ = build_external(
        {"pets": parquet_paths["pets"]}, tmp_path / "c", pets_breeds=BREEDS, pets_images=8, seed=2
    )
    ids = lambda rs: sorted({r.image_id for r in rs})  # noqa: E731
    assert ids(a) == ids(b) and ids(a) != ids(c)


def test_species_comes_from_the_data_and_a_missing_breed_is_an_error(tmp_path, parquet_paths):
    recs, _ = build_external(
        {"pets": parquet_paths["pets"]}, tmp_path / "o", pets_breeds=BREEDS, pets_images=None
    )
    species_q = next(
        r for r in recs if r.task == "ext.pets.species" and r.image_id.endswith("pet_0")
    )
    assert species_q.answer == "cat"  # label 0 (abyssinian) is labelled cat in the parquet
    with pytest.raises(ValueError, match="species unknown"):
        build_external(
            {"pets": parquet_paths["pets"]}, tmp_path / "o2", pets_breeds=[*BREEDS, "pug"]
        )


def test_external_records_go_only_to_test_external(tmp_path, parquet_paths):
    recs, facts = build_external(
        parquet_paths, tmp_path / "out", pets_breeds=BREEDS, pets_images=None
    )
    res = build_splits(recs, facts, seed=0)
    check_splits(res)
    assert len(res.splits["test-external"]) == len(recs)
    assert all(not res.splits[s] for s in res.splits if s != "test-external")


# ---- network layer (injected, so no download happens) -----------------------------------------
def test_download_external_lists_filters_and_skips_existing(tmp_path):
    listing = {
        "timm/oxford-iiit-pet": ["data/test-0.parquet", "data/train-0.parquet", "data/README.md"],
        "naver-clova-ix/cord-v2": [
            "data/test-0.parquet",
            "data/validation-0.parquet",
            "data/train-0.parquet",
        ],
        "rootsautomation/ScreenSpot": ["data/test-0.parquet", "data/test-1.parquet"],
    }
    fetched = []

    def fetch_json(url):
        repo = url.split("/api/datasets/")[1].split("/tree")[0]
        return [{"path": p} for p in listing[repo]]

    def fetch_file(url, dest):
        fetched.append(url)
        dest.write_bytes(b"x")

    paths = download_external(tmp_path, fetch_json, fetch_file)
    assert [p.name for p in paths["pets"]] == [
        "test-0.parquet"
    ]  # the train split is not downloaded
    assert [p.name for p in paths["cord"]] == ["test-0.parquet", "validation-0.parquet"]
    assert len(paths["screenspot"]) == 2 and len(fetched) == 5
    download_external(tmp_path, fetch_json, fetch_file)
    assert len(fetched) == 5  # everything already on disk: nothing re-downloaded
    with pytest.raises(RuntimeError, match="no parquet"):
        download_external(tmp_path / "x", lambda url: [], fetch_file)


def test_class_names_parses_dict_and_list_forms():
    as_dict = {
        "cardData": {
            "dataset_info": {
                "features": [
                    {"name": "label", "dtype": {"class_label": {"names": {"0": "a", "1": "b"}}}}
                ]
            }
        }
    }
    as_list = {
        "cardData": {
            "dataset_info": [
                {
                    "features": [
                        {"name": "label", "dtype": {"class_label": {"names": ["x", "y", "z"]}}}
                    ]
                }
            ]
        }
    }
    assert class_names("r", "label", lambda url: as_dict) == ["a", "b"]
    assert class_names("r", "label", lambda url: as_list) == ["x", "y", "z"]
    with pytest.raises(KeyError):
        class_names("r", "nope", lambda url: as_dict)
