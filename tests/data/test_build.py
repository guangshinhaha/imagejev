import json

from PIL import Image, ImageDraw

from imagejev.data.build import (
    assemble,
    build_coco,
    build_quality,
    build_rico,
    load_component,
)
from imagejev.data.coco import PERMISSIVE_LICENSE_IDS
from imagejev.data.records import ImageFacts, QuestionRecord, read_jsonl, write_jsonl

CATS = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "dog", "supercategory": "animal"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
]


def coco_instances(n=30):
    return {
        "categories": CATS,
        "images": [
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 100, "height": 100,
             "license": 4 if i % 2 else 1}
            for i in range(1, n + 1)
        ],
        "annotations": [
            {"id": i * 10 + k, "image_id": i, "category_id": 1 + (i + k) % 3, "area": 2000,
             "iscrowd": 0}
            for i in range(1, n + 1) for k in range(2)
        ],
    }  # fmt: skip


def test_build_coco_limit_license_and_absolute_paths(tmp_path):
    (tmp_path / "instances_val2017.json").write_text(json.dumps(coco_instances()))
    recs, facts = build_coco(tmp_path, tmp_path, limit=None, licenses=None, seed=0)
    assert len(facts) == 30 and all(f.image_path.startswith(str(tmp_path.resolve())) for f in facts)
    assert {r.image_id for r in recs} == {f.image_id for f in facts}
    few_r, few_f = build_coco(tmp_path, tmp_path, limit=5, licenses=None, seed=0)
    assert len(few_f) == 5 and {r.image_id for r in few_r} <= {f.image_id for f in few_f}
    assert any(r.image_id == few_f[-1].image_id for r in few_r)  # the last image kept its questions
    _, lic = build_coco(tmp_path, tmp_path, limit=None, licenses=PERMISSIVE_LICENSE_IDS, seed=0)
    assert len(lic) == 15 and {f.facts["license"] for f in lic} == {4}


def activity(kind):
    return {
        "root": {"klass": "DecorView"},
        "children": [
            {"klass": [kind], "ancestors": [["android.view.View"]], "bounds": [[0, 0, 50, 50]],
             "visible_to_user": [True], "visibility": ["visible"]}
        ],
    }  # fmt: skip


def test_build_rico_writes_screenshots_and_questions(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buf_rows = []
    for i in range(6):
        import io

        b = io.BytesIO()
        Image.new("RGB", (30, 50), (i * 10, 0, 0)).save(b, format="PNG")
        buf_rows.append(
            {"request_id": str(i), "activity": activity("android.widget.Spinner" if i % 2 else "x.View"),
             "is_keyboard_deployed": i == 3, "screenshot": {"bytes": b.getvalue(), "path": f"{i}.png"}}
        )  # fmt: skip
    pq.write_table(pa.Table.from_pylist(buf_rows), tmp_path / "rico.parquet")
    recs, facts = build_rico(tmp_path / "rico.parquet", tmp_path, limit=4, seed=0)
    assert len(facts) == 4 and all(Image.open(f.image_path).size == (30, 50) for f in facts)
    assert {r.task for r in recs} <= {"rico.element_present", "rico.keyboard_open"}
    assert any(r.task == "rico.keyboard_open" and r.answer is True for r in recs)


def textured(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.normal(128, 50, (256, 256, 3)).clip(0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    for k in range(30):
        d.line([k * 8, 0, 255 - k * 8, 255], fill=(0, 0, 0))
    return img


def test_build_quality_sources_and_absolute_paths(tmp_path):
    srcs = []
    for i in range(6):
        p = tmp_path / f"p{i}.png"
        textured(i).save(p)
        srcs.append((f"coco:{i:012d}", "photo", p, None))
    recs, facts = build_quality(srcs, tmp_path, seed=0)
    assert len(recs) == len(facts) == 12
    assert all(f.image_path.startswith("/") and Image.open(f.image_path) for f in facts)
    assert all(r.meta["source_id"].startswith("coco:") for r in recs)


def test_load_component_makes_paths_absolute(tmp_path):
    d = tmp_path / "web"
    (d / "images").mkdir(parents=True)
    write_jsonl(
        [ImageFacts("w1", "screenshot", "webgen", {"k": 1}, image_path="images/a.jpg")],
        d / "facts.jsonl",
    )
    write_jsonl([QuestionRecord("w1", "screenshot", "webgen", "t", {"type": "bool", "instructions": "x"}, True)],
                d / "questions.jsonl")  # fmt: skip
    recs, facts = load_component(d)
    assert facts[0].image_path == str((d / "images" / "a.jpg").resolve()) and len(recs) == 1


def test_assemble_writes_splits_and_combined_facts(tmp_path):
    recs, facts = build_coco_fixture(tmp_path)
    report = assemble(recs, facts, tmp_path / "out", seed=1)
    out = tmp_path / "out"
    assert (out / "splits" / "train.jsonl").exists() and (out / "facts_all.jsonl").exists()
    assert sum(report["questions"].values()) > 0 and "thin_families" in report
    combined = list(read_jsonl(out / "facts_all.jsonl", ImageFacts))
    assert len(combined) == len(facts)


def build_coco_fixture(tmp_path):
    (tmp_path / "instances_val2017.json").write_text(json.dumps(coco_instances(120)))
    return build_coco(tmp_path, tmp_path, limit=None, licenses=None, seed=0)
