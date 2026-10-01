import collections

import pytest

from imagejev.data.docgen import (
    ITEM_LEVELS,
    TOTAL_LEVELS,
    DocSpec,
    item_level,
    labels_from_measure,
    money,
    questions_for,
    render_html,
    sample_spec,
    total_level,
)
from imagejev.data.records import ImageFacts, QuestionRecord, read_jsonl


def test_money_and_levels():
    assert money(5) == "$0.05" and money(123456) == "$1,234.56" and money(100000) == "$1,000.00"
    assert [total_level(c) for c in (0, 999, 1000, 4999, 5000, 19999, 20000)] == [
        "under $10", "under $10", "$10 to $50", "$10 to $50", "$50 to $200", "$50 to $200",
        "over $200",
    ]  # fmt: skip
    assert [item_level(n) for n in (0, 1, 2, 3, 4, 5, 6, 7)] == [
        "one", "one", "two or three", "two or three", "four or five", "four or five",
        "six or more", "six or more",
    ]  # fmt: skip


def test_specs_deterministic_and_total_is_exact_integer_cents():
    assert sample_spec(5, 0) == sample_spec(5, 0) != sample_spec(5, 1)
    for i in range(300):
        s = sample_spec(i, 0)
        if s.doc_type == "form":
            assert s.total is None and not s.lines
        else:
            assert isinstance(s.total, int) and s.total == s.subtotal + s.tax
            assert 1 <= len(s.lines) <= 7
        assert (s.merchant_type is not None) == (s.doc_type == "receipt")
        assert s.table is False or s.doc_type in ("invoice", "form")


def test_every_total_bin_occurs_for_receipts_and_invoices():
    specs = [sample_spec(i, 0) for i in range(2500)]
    for t in ("receipt", "invoice"):
        bins = {total_level(s.total) for s in specs if s.doc_type == t}
        assert bins == set(TOTAL_LEVELS), t


def spec(**kw):
    base = dict(
        index=1, doc_type="invoice", variant="a", merchant_type=None, company="Acme Ltd",
        lines=(("Hosting", 2, 1050),), tax_pct=10, signature=False, stamp=None, table=True, seed=1,
    )  # fmt: skip
    return DocSpec(**{**base, **kw})


@pytest.mark.parametrize(
    ("kw", "marker"),
    [({"signature": True}, 'data-gt="signature"'), ({"stamp": "PAID"}, "data-gt='stamp'")],
)
def test_html_markers_follow_flags(kw, marker):
    assert marker in render_html(spec(**kw))
    assert marker not in render_html(spec())


def measure(s, **over):
    m = {"total": {"inside": True, "text": money(s.total or 0)}, "__size": [800, 1000]}
    if s.table:
        m["table"] = {"inside": True, "text": ""}
    if s.signature:
        m["signature"] = {"inside": True, "text": ""}
    if s.stamp:
        m["stamp"] = {"inside": True, "text": s.stamp}
    m.update(over)
    return m


def test_labels_match_spec_and_reject_clipping_or_wrong_text():
    s = spec(signature=True, stamp="VOID")
    labels = labels_from_measure(s, measure(s))
    assert (
        labels["has_signature"] and labels["stamp_text"] == "VOID" and labels["total_cents"] == 2310
    )
    assert labels_from_measure(s, measure(s, total={"inside": True, "text": "$1.00"})) is None
    assert labels_from_measure(s, measure(s, stamp={"inside": False, "text": "VOID"})) is None
    bad = measure(s)
    del bad["signature"]
    assert labels_from_measure(s, bad) is None
    extra = measure(
        spec(), signature={"inside": True, "text": ""}
    )  # signature drawn, spec says none
    assert labels_from_measure(spec(), extra) is None


def test_questions_are_valid_and_consistent_with_labels():
    s = spec(doc_type="receipt", merchant_type="cafe", table=False, stamp="PAID", signature=True)
    labels = labels_from_measure(s, measure(s))
    qs = {q.task: q for q in questions_for(s, labels, 0)}
    assert qs["doc.type"].answer == "receipt" and qs["doc.merchant_type"].answer == "cafe"
    assert qs["doc.has_stamp"].answer is True and qs["doc.stamp_text"].answer == "PAID"
    assert qs["doc.has_table"].answer is False and qs["doc.has_signature"].answer is True
    assert qs["doc.total_level"].answer == total_level(s.total)
    assert qs["doc.item_count"].answer == "one"
    assert qs["doc.total_level"].question["levels"] == TOTAL_LEVELS
    assert qs["doc.item_count"].question["levels"] == ITEM_LEVELS
    text = qs["doc.total_over"].question["instructions"]
    thresh = round(float(text.split("$")[1].rstrip("?").replace(",", "")) * 100)
    assert qs["doc.total_over"].answer == (s.total > thresh)
    assert all(q.style == "receipt:a" for q in qs.values())


def test_form_has_no_amount_or_merchant_questions():
    s = spec(doc_type="form", lines=(), table=False)
    labels = labels_from_measure(s, {"__size": [1, 1]})  # a form renders no total
    qs = {q.task for q in questions_for(s, labels, 0)}
    assert qs == {
        "doc.type",
        "doc.has_signature",
        "doc.has_table",
        "doc.has_stamp",
        "doc.stamp_text",
    }


@pytest.mark.slow
def test_generate_with_real_browser_matches_labels(tmp_path):
    pytest.importorskip("playwright")
    from PIL import Image

    from imagejev.data.docgen import generate

    try:
        counts = generate(18, tmp_path, seed=0)
    except Exception as e:  # no Chrome available
        pytest.skip(f"browser unavailable: {e}")
    assert counts["dropped"] == 0 and counts["pages"] == 18
    facts = list(read_jsonl(tmp_path / "facts.jsonl", ImageFacts))
    qs = list(read_jsonl(tmp_path / "questions.jsonl", QuestionRecord))
    assert len(qs) == counts["questions"]
    kinds = collections.Counter(f.facts["doc_type"] for f in facts)
    assert set(kinds) == {"receipt", "invoice", "form"}
    for f in facts:
        w, h = Image.open(tmp_path / f.image_path).size
        assert (w, h) != (0, 0) and w in (360, 794)
