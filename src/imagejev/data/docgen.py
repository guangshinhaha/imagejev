"""Synthetic documents (receipts, invoices, forms) with exact labels.

Every field is generated first and then drawn, so labels such as the total, the merchant type or
whether a signature exists are known by construction. Amounts are kept in integer cents. After
rendering, the DOM is checked: every tagged element must exist exactly when the spec says, and fit
inside the page, so a clipped total or a missing stamp never becomes a wrong label.

Styles: ``style = "<doc_type>:<variant>"`` (three layout variants per type). Hold out a variant per
type for ``test-styles``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import ImageFacts, QuestionRecord, write_jsonl

SOURCE = "docgen"
DOC_TYPES = ("receipt", "invoice", "form")
VARIANTS = ("a", "b", "c")
STAMPS = ("PAID", "VOID", "APPROVED", "DRAFT")
MERCHANT_TYPES = {
    "restaurant": (
        "a restaurant or diner",
        ["Trattoria", "Grill", "Noodle Bar", "Bistro", "Diner"],
    ),
    "grocery": ("a grocery store", ["Market", "Fresh Foods", "Grocer", "Supermart", "Greens"]),
    "pharmacy": ("a pharmacy", ["Pharmacy", "Drugs", "Apothecary", "Health Mart", "Rx"]),
    "hardware": ("a hardware store", ["Hardware", "Tools & More", "Lumber", "Fix-It", "Builders"]),
    "cafe": ("a coffee shop", ["Coffee", "Roasters", "Espresso Bar", "Bean House", "Cafe"]),
    "gas station": ("a gas station", ["Fuel", "Gas & Go", "Petro", "Fill-Up", "Service Station"]),
}
ITEMS = {
    "restaurant": ["Pasta", "Burger", "Salad", "Soup", "Steak", "Tea", "Dessert", "Fries"],
    "grocery": ["Milk", "Bread", "Apples", "Eggs", "Rice", "Cheese", "Coffee", "Pasta"],
    "pharmacy": ["Vitamin C", "Bandages", "Cough syrup", "Toothpaste", "Soap", "Aspirin"],
    "hardware": ["Hammer", "Screws", "Paint", "Tape", "Drill bit", "Gloves", "Ladder"],
    "cafe": ["Latte", "Espresso", "Muffin", "Bagel", "Tea", "Cold brew", "Cookie"],
    "gas station": ["Regular fuel", "Diesel", "Car wash", "Snack", "Water", "Oil"],
}
SERVICES = ["Consulting", "Design work", "Hosting", "Maintenance", "Training", "Support plan"]
TARGET_RANGES = [(200, 999), (1000, 4999), (5000, 19999), (20000, 90000)]  # cents, per bin
TOTAL_LEVELS = ["under $10", "$10 to $50", "$50 to $200", "over $200"]
ITEM_LEVELS = ["one", "two or three", "four or five", "six or more"]
COMPANY = ["Apex", "Birch", "Cobalt", "Dune", "Elm", "Fjord", "Gale", "Haven", "Iris", "Jade"]
FIELD_NAMES = ["Full name", "Date of birth", "Address", "Phone", "Email", "Occupation", "ID number"]
DOC_TYPE_DESCRIPTIONS = {
    "receipt": "a proof of purchase listing items bought and the amount paid",
    "invoice": "a bill from a company asking a customer for payment",
    "form": "a form with labelled fields to fill in",
}
FONTS = {
    "a": "'Courier New', monospace",
    "b": "Helvetica, Arial, sans-serif",
    "c": "Georgia, 'Times New Roman', serif",
}


def money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def total_level(cents: int) -> str:
    if cents < 1000:
        return TOTAL_LEVELS[0]
    if cents < 5000:
        return TOTAL_LEVELS[1]
    return TOTAL_LEVELS[2] if cents < 20000 else TOTAL_LEVELS[3]


def item_level(n: int) -> str:
    if n <= 1:
        return ITEM_LEVELS[0]
    if n <= 3:
        return ITEM_LEVELS[1]
    return ITEM_LEVELS[2] if n <= 5 else ITEM_LEVELS[3]


@dataclass(frozen=True)
class DocSpec:
    index: int
    doc_type: str
    variant: str
    merchant_type: str | None  # receipts only
    company: str
    lines: tuple[tuple[str, int, int], ...]  # (name, qty, unit cents)
    tax_pct: int
    signature: bool
    stamp: str | None
    table: bool
    seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def style(self) -> str:
        return f"{self.doc_type}:{self.variant}"

    @property
    def image_id(self) -> str:
        return f"doc:{self.index:07d}"

    @property
    def subtotal(self) -> int:
        return sum(q * c for _, q, c in self.lines)

    @property
    def tax(self) -> int:
        return round(self.subtotal * self.tax_pct / 100)

    @property
    def total(self) -> int | None:
        return None if self.doc_type == "form" else self.subtotal + self.tax


def sample_spec(index: int, seed: int) -> DocSpec:
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    doc_type = rng.choice(DOC_TYPES)
    merchant = rng.choice(sorted(MERCHANT_TYPES)) if doc_type == "receipt" else None
    suffixes = MERCHANT_TYPES[merchant][1] if merchant else ["Ltd", "Inc", "GmbH", "LLC"]
    company = f"{rng.choice(COMPANY)} {rng.choice(suffixes)}"
    lines: list[tuple[str, int, int]] = []
    if doc_type != "form":
        pool = ITEMS[merchant] if merchant else SERVICES
        names = rng.sample(pool, min(len(pool), rng.randint(1, 7)))
        qtys = [rng.randint(1, 3) for _ in names]
        # Choose the amount first, then scale unit prices to it, so every total bin gets coverage.
        weights = [0.05, 0.2, 0.35, 0.4] if doc_type == "invoice" else [0.15, 0.35, 0.3, 0.2]
        lo, hi = rng.choices(TARGET_RANGES, weights)[0]
        target = rng.randint(lo, hi)
        raw = [rng.uniform(0.5, 2.0) for _ in names]
        scale = target / sum(q * r for q, r in zip(qtys, raw, strict=True))
        lines = [
            (n, q, max(50, round(r * scale))) for n, q, r in zip(names, qtys, raw, strict=True)
        ]
    return DocSpec(
        index=index,
        doc_type=doc_type,
        variant=rng.choice(VARIANTS),
        merchant_type=merchant,
        company=company,
        lines=tuple(lines),
        tax_pct=rng.choice([0, 5, 8, 10]),
        signature=rng.random() < 0.5,
        stamp=rng.choice(STAMPS) if rng.random() < 0.4 else None,
        table=doc_type == "invoice"
        and rng.random() < 0.8
        or doc_type == "form"
        and rng.random() < 0.5,
        seed=int(h[16:24], 16),
    )


def _signature_svg(seed: int) -> str:
    rng = random.Random(seed)
    pts = [(10, 40)]
    x = 10
    for _ in range(rng.randint(5, 8)):
        x += rng.randint(14, 26)
        pts.append((x, rng.randint(8, 52)))
    d = f"M{pts[0][0]},{pts[0][1]} " + " ".join(
        f"Q{(a + c) // 2 + rng.randint(-8, 8)},{rng.randint(0, 60)} {c},{e}"
        for (a, _b), (c, e) in zip(pts, pts[1:], strict=False)
    )
    return (
        f'<svg width="{x + 20}" height="60" viewBox="0 0 {x + 20} 60" data-gt="signature">'
        f'<path d="{d}" fill="none" stroke="#1a237e" stroke-width="2.4" '
        f'stroke-linecap="round"/></svg>'
    )


def _items_html(spec: DocSpec) -> str:
    rows = "".join(
        f"<tr><td>{n}</td><td class='r'>{q}</td><td class='r'>{money(c)}</td>"
        f"<td class='r'>{money(q * c)}</td></tr>"
        for n, q, c in spec.lines
    )
    head = (
        "<tr><th>Item</th><th class='r'>Qty</th><th class='r'>Price</th>"
        "<th class='r'>Amount</th></tr>"
    )
    cls = "grid" if spec.table else "plain"
    return f'<table class="{cls}" {"data-gt=table" if spec.table else ""}>{head}{rows}</table>'


def _totals_html(spec: DocSpec) -> str:
    return (
        f"<div class='tot'><div>Subtotal <span>{money(spec.subtotal)}</span></div>"
        f"<div>Tax ({spec.tax_pct}%) <span>{money(spec.tax)}</span></div>"
        f"<div class='big'>Total <span data-gt='total'>{money(spec.total or 0)}</span></div></div>"
    )


def _sig_block(spec: DocSpec) -> str:
    sig = _signature_svg(spec.seed) if spec.signature else ""
    return (
        f"<div class='sig'><div class='sigbox'>{sig}</div><div class='line'>Signature</div></div>"
    )


def render_html(spec: DocSpec) -> str:
    v = spec.variant
    width = 360 if spec.doc_type == "receipt" else 794
    min_h = 0 if spec.doc_type == "receipt" else 1050
    head_align = "center" if (spec.doc_type == "receipt" or v == "b") else "left"
    border = {"a": "none", "b": "2px solid #333", "c": "1px double #666"}[v]
    if spec.doc_type == "receipt":
        body = (
            f"<h2>{spec.company}</h2><p class='c'>Thank you for your visit</p><hr>"
            f"{_items_html(spec)}<hr>{_totals_html(spec)}<hr>"
            f"<p class='c'>Card ending 4242</p>{_sig_block(spec)}"
        )
    elif spec.doc_type == "invoice":
        body = (
            f"<h1>INVOICE</h1><p><b>{spec.company}</b><br>Invoice #{1000 + spec.index}</p>"
            f"{_items_html(spec)}{_totals_html(spec)}<p class='muted'>Payment due in 30 days.</p>"
            f"{_sig_block(spec)}"
        )
    else:
        names = [FIELD_NAMES[(spec.index + k) % len(FIELD_NAMES)] for k in range(5)]
        fields = "".join(
            f"<div class='fld'><label>{n}</label><div class='ln'></div></div>" for n in names
        )
        grid = (
            "<table class='grid' data-gt=table><tr><th>Year</th><th>Employer</th><th>Role</th></tr>"
            "<tr><td>&nbsp;</td><td></td><td></td></tr><tr><td>&nbsp;</td><td></td><td></td></tr>"
            "</table>"
            if spec.table
            else ""
        )
        body = (
            f"<h1>Application Form</h1><p class='muted'>{spec.company}</p>"
            f"{fields}{grid}{_sig_block(spec)}"
        )
    stamp = f"<div class='stamp' data-gt='stamp'>{spec.stamp}</div>" if spec.stamp else ""
    small = spec.doc_type == "receipt"
    right, top, size = (20, 50, 24) if small else (30, 70, 34)
    css = f"""
body{{margin:0;background:#e8e8e8;font-family:{FONTS[v]};color:#111}}
#doc{{position:relative;width:{width}px;min-height:{min_h}px;margin:0;background:#fff;
padding:{22 if spec.doc_type == "receipt" else 56}px;box-sizing:border-box;border:{border}}}
h1,h2{{text-align:{head_align};margin:0 0 12px}} .c{{text-align:center}} .muted{{color:#666}}
table{{width:100%;border-collapse:collapse;margin:10px 0}} td,th{{padding:5px;text-align:left}}
.r{{text-align:right}} table.grid td,table.grid th{{border:1px solid #444}}
.tot{{margin:10px 0 10px auto;width:60%}} .tot div{{display:flex;justify-content:space-between}}
.big{{font-weight:700;font-size:1.15em;border-top:1px solid #333;margin-top:4px;padding-top:4px}}
hr{{border:0;border-top:1px dashed #999}} .sig{{margin-top:28px;width:55%}}
.sigbox{{height:62px}} .line{{border-top:1px solid #222;font-size:12px;color:#555}}
.fld{{margin:16px 0}} .fld label{{font-size:13px;color:#555}}
.ln{{border-bottom:1px solid #222;height:26px}}
.stamp{{position:absolute;right:{right}px;top:{top}px;transform:rotate(-14deg);
border:4px solid #c62828;color:#c62828;font-weight:800;font-size:{size}px;padding:2px 14px;
letter-spacing:3px;opacity:.82}}
"""
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head>"
        f"<body><div id='doc'>{stamp}{body}</div></body></html>"
    )


MEASURE_JS = """
() => {
  const doc = document.getElementById('doc'), D = doc.getBoundingClientRect(), out = {};
  for (const el of document.querySelectorAll('[data-gt]')) {
    const r = el.getBoundingClientRect(), k = el.getAttribute('data-gt');
    // rotated stamps have a larger bounding box than their content; allow a few pixels
    out[k] = {inside: r.left >= D.left - 8 && r.right <= D.right + 8 && r.top >= D.top - 8
              && r.bottom <= D.bottom + 8, text: (el.innerText || '').trim()};
  }
  out.__size = [Math.ceil(D.width), Math.ceil(D.height)];
  return out;
}
"""


def labels_from_measure(spec: DocSpec, m: dict[str, Any]) -> dict[str, Any] | None:
    """Labels, or ``None`` if the rendered page contradicts the spec or clips content."""
    expected = {
        "signature": spec.signature,
        "stamp": spec.stamp is not None,
        "table": spec.table,
        "total": spec.doc_type != "form",
    }
    for key, want in expected.items():
        if (key in m) != want or (want and not m[key]["inside"]):
            return None
    if spec.doc_type != "form" and m["total"]["text"] != money(spec.total or 0):
        return None
    if spec.stamp and m["stamp"]["text"] != spec.stamp:
        return None
    return {
        "doc_type": spec.doc_type,
        "merchant_type": spec.merchant_type,
        "total_cents": spec.total,
        "n_items": len(spec.lines),
        "has_signature": spec.signature,
        "has_stamp": spec.stamp is not None,
        "stamp_text": spec.stamp,
        "has_table": spec.table,
        "tax_pct": spec.tax_pct if spec.doc_type != "form" else None,
    }


def questions_for(spec: DocSpec, labels: dict[str, Any], seed: int) -> list[QuestionRecord]:
    h = hashlib.sha256(f"{seed}:q:{spec.index}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    iid = spec.image_id

    def rec(task: str, q: dict[str, Any], answer: Any) -> QuestionRecord:
        return QuestionRecord(iid, "document", SOURCE, task, q, answer, style=spec.style)

    def boolq(task: str, text: str, answer: bool) -> QuestionRecord:
        return rec(task, {"type": "bool", "instructions": text}, answer)

    out = [
        rec(
            "doc.type",
            {
                "type": "choice",
                "instructions": "What kind of document is this?",
                "criteria": dict(DOC_TYPE_DESCRIPTIONS),
            },
            spec.doc_type,
        ),
        boolq("doc.has_signature", "Is there a handwritten signature on the page?", spec.signature),
        boolq(
            "doc.has_table", "Does the page contain a table with ruled lines?", labels["has_table"]
        ),
        boolq("doc.has_stamp", "Is there a stamp on the document?", labels["has_stamp"]),
    ]
    stamp_opts = [*STAMPS, "no stamp"]
    out.append(
        rec(
            "doc.stamp_text",
            {
                "type": "choice",
                "instructions": "Which stamp is on the document?",
                "criteria": {
                    s: ("" if s != "no stamp" else "the document has no stamp") for s in stamp_opts
                },
            },
            labels["stamp_text"] or "no stamp",
        )
    )
    if spec.merchant_type:
        out.append(
            rec(
                "doc.merchant_type",
                {
                    "type": "choice",
                    "instructions": "What kind of business issued this receipt?",
                    "criteria": {k: v[0] for k, v in MERCHANT_TYPES.items()},
                },
                spec.merchant_type,
            )
        )
    if spec.total is not None:
        out.append(
            rec(
                "doc.total_level",
                {
                    "type": "score",
                    "instructions": "How large is the total amount due?",
                    "levels": TOTAL_LEVELS,
                },
                total_level(spec.total),
            )
        )
        threshold = rng.choice([1000, 2500, 5000, 10000, 20000])
        out.append(
            boolq(
                "doc.total_over",
                f"Is the total amount more than {money(threshold)}?",
                spec.total > threshold,
            )
        )
        out.append(
            rec(
                "doc.item_count",
                {
                    "type": "score",
                    "instructions": "How many line items are listed?",
                    "levels": ITEM_LEVELS,
                },
                item_level(len(spec.lines)),
            )
        )
    return out


def generate(
    n: int, out_dir: str | Path, *, seed: int = 0, start: int = 0, quality: int = 88
) -> dict[str, int]:
    """Render ``n`` documents into ``out_dir`` (see ``webgen.generate`` for the layout)."""
    from playwright.sync_api import sync_playwright

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    facts: list[ImageFacts] = []
    questions: list[QuestionRecord] = []
    dropped = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 900, "height": 1200})
        for i in range(start, start + n):
            spec = sample_spec(i, seed)
            page.set_content(render_html(spec))
            labels = labels_from_measure(spec, page.evaluate(MEASURE_JS))
            if labels is None:
                dropped += 1
                continue
            rel = f"images/{spec.image_id.split(':')[1]}.jpg"
            page.locator("#doc").screenshot(path=str(out / rel), type="jpeg", quality=quality)
            facts.append(
                ImageFacts(
                    spec.image_id, "document", SOURCE, labels, image_path=rel, style=spec.style
                )
            )
            questions.extend(questions_for(spec, labels, seed))
        browser.close()
    write_jsonl(facts, out / "facts.jsonl")
    write_jsonl(questions, out / "questions.jsonl")
    return {"pages": len(facts), "questions": len(questions), "dropped": dropped}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/docs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    print(generate(args.n, args.out, seed=args.seed, start=args.start))
