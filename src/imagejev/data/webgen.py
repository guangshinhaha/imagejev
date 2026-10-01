"""Synthetic web screenshots with exact labels.

Each page is rendered from a ``PageSpec`` (template x theme x overlay states), then the real DOM is
measured in the browser. Labels come from that measurement, and a page whose measurement disagrees
with its spec is dropped, so a label is never just "what we meant to draw".

Ground-truth elements are tagged with ``data-gt``: ``modal``, ``error``, ``spinner``, ``cookie``,
``captcha``, ``cta``. "Present" means at least half the element is inside the viewport. The CTA must
also be unoccluded at its centre (an open modal backdrop covers it), because the question is whether
it can be clicked right now without scrolling.

Styles: ``style = "<template>:<theme>"``. Hold out themes (or templates) for ``test-styles``.
"""

from __future__ import annotations

import colorsys
import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import ImageFacts, QuestionRecord, write_jsonl

SOURCE = "webgen"
TEMPLATES = ("login", "landing", "shop", "article", "dashboard", "checkout")
PAGE_TYPE_DESCRIPTIONS = {
    "login": "a sign-in page with email and password fields",
    "landing": "a marketing home page with a hero headline and call to action",
    "shop": "an online store with a product grid and a cart",
    "article": "a long-form article or blog post",
    "dashboard": "an analytics dashboard with stat cards and a table",
    "checkout": "a checkout page collecting shipping and payment details",
}
VIEWPORTS = {"desktop": (1280, 720), "mobile": (390, 844)}
INPUT_LEVELS = ["none", "one", "two or three", "four or more"]
FONTS = [
    "Helvetica, Arial, sans-serif",
    "Georgia, serif",
    "Verdana, sans-serif",
    "'Trebuchet MS', sans-serif",
    "'Courier New', monospace",
    "system-ui, sans-serif",
]
WORDS = (
    "northwind lumen harbor atlas juniper ember quartz meadow orbit fable cinder willow summit "
    "delta marble signal copper tidal vivid raven pixel anchor bloom crest drift echo flint"
).split()


@dataclass(frozen=True)
class Theme:
    id: str
    bg: str
    fg: str
    muted: str
    accent: str
    card: str
    border: str
    font: str
    radius: int
    pad: int
    dark: bool


def _hsl(h: float, s: float, l: float) -> str:  # noqa: E741
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def make_themes(n: int = 12, seed: int = 1234) -> list[Theme]:
    rng = random.Random(seed)
    themes = []
    for i in range(n):
        dark = i % 3 == 2
        hue = rng.random()
        themes.append(
            Theme(
                id=f"t{i:02d}",
                bg=_hsl(hue, 0.15, 0.1 if dark else 0.97),
                fg=_hsl(hue, 0.1, 0.92 if dark else 0.12),
                muted=_hsl(hue, 0.08, 0.65 if dark else 0.42),
                accent=_hsl((hue + rng.choice([0.0, 0.33, 0.5, 0.66])) % 1, 0.7, 0.5),
                card=_hsl(hue, 0.15, 0.16 if dark else 1.0),
                border=_hsl(hue, 0.1, 0.28 if dark else 0.86),
                font=FONTS[i % len(FONTS)],
                radius=rng.choice([0, 4, 8, 14]),
                pad=rng.choice([12, 16, 22]),
                dark=dark,
            )
        )
    return themes


@dataclass(frozen=True)
class PageSpec:
    index: int
    template: str
    theme: str
    viewport: str
    modal: bool
    error: bool
    spinner: bool
    cookie: bool
    captcha: bool
    cta_below_fold: bool
    cart_empty: bool
    brand: str = ""
    extras: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def style(self) -> str:
        return f"{self.template}:{self.theme}"

    @property
    def image_id(self) -> str:
        return f"web:{self.index:07d}"


def sample_spec(index: int, seed: int, themes: list[Theme]) -> PageSpec:
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    template = rng.choice(TEMPLATES)
    forms = template in ("login", "checkout")
    return PageSpec(
        index=index,
        template=template,
        theme=rng.choice(themes).id,
        viewport=rng.choices(["desktop", "mobile"], [0.6, 0.4])[0],
        modal=rng.random() < 0.4,
        error=rng.random() < 0.35,
        spinner=rng.random() < 0.25,
        cookie=rng.random() < 0.35,
        captcha=forms and rng.random() < 0.45,
        cta_below_fold=rng.random() < 0.5,
        cart_empty=template == "shop" and rng.random() < 0.5,
        brand=rng.choice(WORDS).capitalize() + rng.choice(["", "ly", "ify", " Co", "Hub"]),
    )


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------
def _css(t: Theme) -> str:
    return f"""
*{{box-sizing:border-box}} body{{margin:0;background:{t.bg};color:{t.fg};font-family:{t.font};
font-size:16px;line-height:1.45}} a{{color:{t.accent}}}
.nav{{display:flex;justify-content:space-between;align-items:center;padding:14px {t.pad}px;
border-bottom:1px solid {t.border};background:{t.card}}} .brand{{font-weight:700;font-size:20px}}
.wrap{{max-width:1000px;margin:0 auto;padding:{t.pad}px}}
.card{{background:{t.card};border:1px solid {t.border};border-radius:{t.radius}px;
padding:{t.pad}px}}
.btn{{display:inline-block;background:{t.accent};color:#fff;border:0;border-radius:{t.radius}px;
padding:12px 22px;font:inherit;font-weight:600;cursor:pointer}}
input,select{{width:100%;padding:11px;margin:6px 0 12px;border:1px solid {t.border};
border-radius:{t.radius}px;background:{t.bg};color:{t.fg};font:inherit}}
label{{font-size:14px;color:{t.muted}}} .grid{{display:grid;gap:{t.pad}px;
grid-template-columns:repeat(auto-fill,minmax(170px,1fr))}} .muted{{color:{t.muted}}}
.swatch{{height:110px;border-radius:{t.radius}px;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse}} td,th{{text-align:left;padding:8px;
border-bottom:1px solid {t.border}}}
.banner{{background:#b3261e;color:#fff;padding:12px {t.pad}px;font-weight:600}}
.backdrop{{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;
justify-content:center;z-index:50}} .modal{{background:{t.card};color:{t.fg};max-width:360px;
width:86%;padding:22px;border-radius:{t.radius}px;box-shadow:0 10px 40px rgba(0,0,0,.4)}}
.cookie{{position:fixed;left:0;right:0;bottom:0;background:{t.card};border-top:1px solid {t.border};
padding:14px {t.pad}px;z-index:40;display:flex;gap:12px;align-items:center;
justify-content:space-between}}
.spin{{position:fixed;inset:0;background:rgba(255,255,255,.7);display:flex;align-items:center;
justify-content:center;z-index:60}} .ring{{width:54px;height:54px;border:6px solid #ccc;
border-top-color:{t.accent};border-radius:50%;animation:none}}
.captcha{{display:flex;align-items:center;gap:10px;border:1px solid {t.border};padding:12px;
width:260px;background:{t.card};border-radius:{t.radius}px;margin-bottom:12px}}
.box{{width:24px;height:24px;border:2px solid {t.muted};border-radius:3px}}
"""


def _nav(spec: PageSpec, links: list[str]) -> str:
    items = "".join(f'<a href="#" style="margin-left:16px">{x}</a>' for x in links)
    return f'<div class="nav"><span class="brand">{spec.brand}</span><span>{items}</span></div>'


def _cta(label: str, spec: PageSpec, vh: int) -> str:
    spacer = f'<div style="height:{vh}px"></div>' if spec.cta_below_fold else ""
    return f'{spacer}<p><button class="btn" data-gt="cta">{label}</button></p>'


def _captcha(spec: PageSpec) -> str:
    if not spec.captcha:
        return ""
    return '<div class="captcha" data-gt="captcha"><span class="box"></span> I\'m not a robot</div>'


def _fields(names: list[str]) -> str:
    return "".join(f'<label>{n}</label><input type="text" placeholder="{n}">' for n in names)


def _body(spec: PageSpec, vh: int) -> str:
    cta = {
        "login": "Sign in",
        "landing": "Get started",
        "shop": "Checkout",
        "article": "Subscribe",
        "dashboard": "New report",
        "checkout": "Place order",
    }[spec.template]
    if spec.template == "login":
        inner = (
            f'<div class="card" style="max-width:420px;margin:30px auto"><h2>Welcome back</h2>'
            f"{_fields(['Email', 'Password'])}{_captcha(spec)}{_cta(cta, spec, vh)}</div>"
        )
    elif spec.template == "checkout":
        inner = (
            f'<div class="card"><h2>Checkout</h2>'
            f"{_fields(['Full name', 'Address', 'City', 'Card number'])}{_captcha(spec)}"
            f"{_cta(cta, spec, vh)}</div>"
        )
    elif spec.template == "landing":
        inner = (
            f'<div style="padding:40px 0"><h1 style="font-size:40px;margin:0 0 10px">'
            f"{spec.brand} makes work simpler</h1>"
            f'<p class="muted">Plan, track and ship in one place.'
            f"</p>{_cta(cta, spec, vh)}</div>"
        )
    elif spec.template == "shop":
        if spec.cart_empty:
            cart = '<div class="card" data-gt="cart-empty"><b>Your cart is empty</b></div>'
        else:
            cart = '<div class="card"><b>Cart</b><p>2 items · $48.00</p></div>'
        prods = "".join(
            f'<div class="card"><div class="swatch" style="background:hsl({i * 47 % 360},60%,60%)">'
            f'</div><b>{w.capitalize()}</b><div class="muted">${9 + i * 7}.00</div></div>'
            for i, w in enumerate(WORDS[spec.index % 5 : spec.index % 5 + 6])
        )
        inner = f'{cart}<h2>Products</h2><div class="grid">{prods}</div>{_cta(cta, spec, vh)}'
    elif spec.template == "article":
        paras = "".join(
            f"<p>{' '.join(WORDS[(spec.index + i + j) % len(WORDS)] for j in range(40))}.</p>"
            for i in range(5)
        )
        inner = (
            f'<h1>{spec.brand}: notes from the field</h1><p class="muted">5 min read</p>'
            f"{paras}{_cta(cta, spec, vh)}"
        )
    else:  # dashboard
        stats = "".join(
            f'<div class="card"><div class="muted">{n}</div><div style="font-size:28px">'
            f"{(spec.index * (k + 3)) % 900 + 100}</div></div>"
            for k, n in enumerate(["Visits", "Signups", "Revenue", "Churn"])
        )
        rows = "".join(
            f"<tr><td>{WORDS[(spec.index + r) % len(WORDS)]}</td><td>{r * 13 % 90 + 10}</td></tr>"
            for r in range(6)
        )
        inner = (
            f'<div class="grid">{stats}</div><div class="card" style="margin-top:16px"><table>'
            f"<tr><th>Name</th><th>Value</th></tr>{rows}</table></div>{_cta(cta, spec, vh)}"
        )
    return f'<div class="wrap">{inner}</div>'


def render_html(spec: PageSpec, theme: Theme) -> str:
    w, h = VIEWPORTS[spec.viewport]
    links = ["Home", "Pricing", "Help"] if w > 600 else ["Menu"]
    parts = [_nav(spec, links)]
    if spec.error:
        parts.append(
            '<div class="banner" data-gt="error">Something went wrong. Please try again.</div>'
        )
    parts.append(_body(spec, h))
    if spec.cookie:
        parts.append(
            '<div class="cookie" data-gt="cookie"><span>We use cookies to improve your experience.'
            '</span><button class="btn">Accept</button></div>'
        )
    if spec.modal:
        parts.append(
            '<div class="backdrop"><div class="modal" data-gt="modal"><h3 style="margin-top:0">'
            'Heads up</h3><p>Your session is about to expire.</p><button class="btn">OK</button>'
            "</div></div>"
        )
    if spec.spinner:
        parts.append('<div class="spin" data-gt="spinner"><div class="ring"></div></div>')
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><style>{_css(theme)}</style></head>'
        f"<body>{''.join(parts)}</body></html>"
    )


# ---------------------------------------------------------------------------------------------
# Measurement and labels
# ---------------------------------------------------------------------------------------------
MEASURE_JS = """
() => {
  const vw = innerWidth, vh = innerHeight, out = {};
  for (const el of document.querySelectorAll('[data-gt]')) {
    const key = el.getAttribute('data-gt'), r = el.getBoundingClientRect();
    const w = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0));
    const h = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    const area = r.width * r.height;
    const inView = area > 0 && (w * h) / area >= 0.5;
    let clear = false;
    if (inView) {
      const cx = Math.min(Math.max(r.left + r.width / 2, 0), vw - 1);
      const cy = Math.min(Math.max(r.top + r.height / 2, 0), vh - 1);
      const top = document.elementFromPoint(cx, cy);
      clear = !!top && (top === el || el.contains(top));
    }
    out[key] = {inView, clear};
  }
  out.__inputs = [...document.querySelectorAll('input, select, textarea')].filter(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  }).length;
  return out;
}
"""


def input_level(n: int) -> str:
    if n <= 0:
        return INPUT_LEVELS[0]
    if n == 1:
        return INPUT_LEVELS[1]
    return INPUT_LEVELS[2] if n <= 3 else INPUT_LEVELS[3]


def labels_from_measure(spec: PageSpec, m: dict[str, Any]) -> dict[str, Any] | None:
    """Final labels from DOM measurement, or ``None`` if it contradicts the spec.

    Overlays may be pushed out of view by nothing here, so they must be in view exactly when the
    spec says they exist. The CTA is checked separately: its measured clickability is the label.
    """
    expected = {
        "modal": spec.modal,
        "error": spec.error,
        "spinner": spec.spinner,
        "cookie": spec.cookie,
        "captcha": spec.captcha,
    }
    for key, want in expected.items():
        got = key in m and m[key]["inView"]
        if got != want:
            return None
    if "cta" not in m:
        return None
    cta_ok = bool(m["cta"]["inView"] and m["cta"]["clear"])
    if spec.cta_below_fold and m["cta"]["inView"]:
        return None  # the spacer should have pushed it off screen
    return {
        "page_type": spec.template,
        "modal_open": spec.modal,
        "error_banner": spec.error,
        "loading": spec.spinner,
        "cookie_banner": spec.cookie,
        "captcha": spec.captcha,
        "cart_empty": spec.cart_empty if spec.template == "shop" else None,
        "cta_clickable": cta_ok,
        "input_count": int(m["__inputs"]),
        "viewport": spec.viewport,
    }


BOOL_TASKS = {
    "modal_open": ("web.modal_open", "Is a dialog or popup open on top of the page?"),
    "error_banner": ("web.error_banner", "Is an error message banner shown?"),
    "loading": ("web.loading", "Is the page showing a loading spinner?"),
    "cookie_banner": ("web.cookie_banner", "Is a cookie consent banner shown?"),
    "captcha": ("web.captcha", "Is a CAPTCHA checkbox shown?"),
}


def questions_for(spec: PageSpec, labels: dict[str, Any], seed: int) -> list[QuestionRecord]:
    h = hashlib.sha256(f"{seed}:q:{spec.index}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    iid = spec.image_id

    def rec(task: str, q: dict[str, Any], answer: Any) -> QuestionRecord:
        return QuestionRecord(iid, "screenshot", SOURCE, task, q, answer, style=spec.style)

    out = [
        rec(
            "web.page_type",
            {
                "type": "choice",
                "instructions": "What kind of web page is this?",
                "criteria": dict(PAGE_TYPE_DESCRIPTIONS),
            },
            labels["page_type"],
        )
    ]
    keys = list(BOOL_TASKS)
    present = [k for k in keys if labels[k]]
    absent = [k for k in keys if not labels[k]]
    n_pos = min(len(present), rng.choice([1, 2]))
    chosen = rng.sample(present, n_pos) + rng.sample(absent, min(len(absent), 3 - n_pos))
    for k in chosen:
        task, text = BOOL_TASKS[k]
        out.append(rec(task, {"type": "bool", "instructions": text}, bool(labels[k])))
    if labels["cart_empty"] is not None:
        out.append(
            rec(
                "web.cart_empty",
                {"type": "bool", "instructions": "Does the page say the shopping cart is empty?"},
                bool(labels["cart_empty"]),
            )
        )
    out.append(
        rec(
            "web.cta_clickable",
            {
                "type": "bool",
                "instructions": "Can the main call-to-action button be clicked right now "
                "without scrolling?",
            },
            bool(labels["cta_clickable"]),
        )
    )
    out.append(
        rec(
            "web.input_count",
            {
                "type": "score",
                "instructions": "How many form input fields are visible?",
                "levels": INPUT_LEVELS,
            },
            input_level(labels["input_count"]),
        )
    )
    return out


def generate(
    n: int,
    out_dir: str | Path,
    *,
    seed: int = 0,
    start: int = 0,
    themes: list[Theme] | None = None,
    quality: int = 85,
) -> dict[str, int]:
    """Render ``n`` pages into ``out_dir/images`` and write ``facts.jsonl`` + ``questions.jsonl``.

    Needs Playwright with Chrome (``channel="chrome"``). Returns counts, including how many pages
    were dropped because the measured DOM contradicted the spec.
    """
    from playwright.sync_api import sync_playwright

    themes = themes or make_themes()
    by_id = {t.id: t for t in themes}
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    facts: list[ImageFacts] = []
    questions: list[QuestionRecord] = []
    dropped = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        pages = {
            vp: browser.new_page(viewport={"width": w, "height": h})
            for vp, (w, h) in VIEWPORTS.items()
        }
        for i in range(start, start + n):
            spec = sample_spec(i, seed, themes)
            page = pages[spec.viewport]
            page.set_content(render_html(spec, by_id[spec.theme]))
            labels = labels_from_measure(spec, page.evaluate(MEASURE_JS))
            if labels is None:
                dropped += 1
                continue
            rel = f"images/{spec.image_id.split(':')[1]}.jpg"
            page.screenshot(path=str(out / rel), type="jpeg", quality=quality)
            facts.append(
                ImageFacts(
                    spec.image_id, "screenshot", SOURCE, labels, image_path=rel, style=spec.style
                )
            )
            questions.extend(questions_for(spec, labels, seed))
        browser.close()
    write_jsonl(facts, out / "facts.jsonl")
    write_jsonl(questions, out / "questions.jsonl")
    return {"pages": len(facts), "questions": len(questions), "dropped": dropped}


def _iter_specs(n: int, seed: int) -> Iterator[PageSpec]:  # for tests and quick inspection
    themes = make_themes()
    for i in range(n):
        yield sample_spec(i, seed, themes)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/web")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    print(generate(args.n, args.out, seed=args.seed, start=args.start))
