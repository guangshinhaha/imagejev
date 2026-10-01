"""Image-quality score questions with exact answers.

A good source image is degraded by *one* operation at a strength drawn from a level's band, so the
answer is known by construction: blur, noise, JPEG compression, darkness/brightness, or (documents
only) tilt. Bands are separated by gaps so no sample sits on a level boundary.

Why no "crop": how much of the scene is missing is unanswerable without the original, so it can't
be a fair question. Tilt replaces it for documents, where text lines make it visible.

Derived images get new IDs (``<source>+<kind>-<n>``) and record ``source_id``, so splits must group
by source image (see the split builder).
"""

from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .records import ImageFacts, QuestionRecord, write_jsonl

SOURCE = "quality"


@dataclass(frozen=True)
class Kind:
    name: str
    instructions: str
    levels: tuple[str, ...]  # ordered low -> high
    bands: tuple[tuple[float, float] | None, ...]  # parameter range per level; None = untouched
    domains: tuple[str, ...] = ("photo", "document", "screenshot")


KINDS: dict[str, Kind] = {
    "blur": Kind(
        "blur",
        "How sharp is this image?",
        ("very blurry", "blurry", "slightly soft", "sharp"),
        ((0.022, 0.034), (0.009, 0.014), (0.0035, 0.0055), None),  # Gaussian radius / short side
    ),
    "noise": Kind(
        "noise",
        "How much grain or noise does this image have?",
        ("clean", "slight noise", "noisy", "very noisy"),
        (None, (9.0, 14.0), (24.0, 34.0), (50.0, 70.0)),  # Gaussian sigma on a 0-255 scale
    ),
    "jpeg": Kind(
        "jpeg",
        "How visible are compression artifacts?",
        ("none visible", "slight", "clear", "heavy"),
        (None, (28.0, 45.0), (9.0, 16.0), (2.0, 5.0)),  # JPEG quality
    ),
    "brightness": Kind(
        "brightness",
        "How bright is this image?",
        ("very dark", "dark", "normal", "very bright"),
        ((0.10, 0.20), (0.35, 0.50), None, (1.9, 2.6)),  # multiplier on pixel values
        # Photos only: a white page times 2 is still white, so "very bright" would be untrue there.
        domains=("photo",),
    ),
    "tilt": Kind(
        "tilt",
        "How tilted is this page?",
        ("level", "slightly tilted", "tilted", "strongly tilted"),
        (None, (1.5, 3.0), (5.0, 8.0), (12.0, 20.0)),  # degrees, random sign
        domains=("document",),
    ),
}


# ---------------------------------------------------------------------------------------------
# Measurements (used to vet sources and in tests to confirm levels are ordered)
# ---------------------------------------------------------------------------------------------
def _gray(img: Image.Image, short_side: int = 512) -> np.ndarray:
    g = img.convert("L")
    s = short_side / min(g.size)
    if s < 1:
        g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.LANCZOS)
    return np.asarray(g, dtype=np.float32)


def sharpness(img: Image.Image, short_side: int = 512) -> float:
    """Variance of the Laplacian with the short side capped at ``short_side``: higher is sharper.

    A small ``short_side`` (e.g. 96) tells heavy blurs apart; at 512 they all read near zero.
    """
    g = _gray(img, short_side)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def mean_luma(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L"), dtype=np.float32).mean())


def noise_level(img: Image.Image) -> float:
    """Robust high-frequency noise estimate (MAD of a Laplacian-like residual), in 0-255 units."""
    g = _gray(img)
    k = g[1:-1, 1:-1] - 0.25 * (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(np.median(np.abs(k - np.median(k))) / 0.6745 / 1.1)


def is_good_source(
    img: Image.Image,
    domain: str = "photo",
    *,
    min_sharpness: float = 60.0,
    luma: tuple[float, float] = (70.0, 190.0),
) -> bool:
    """A source can only be called 'sharp' / 'clean' / 'normal' if it already is.

    The brightness window applies to photos only: white documents and dark-theme screenshots are
    normal at the extremes.
    """
    if min(img.size) < 192 or sharpness(img) < min_sharpness:
        return False
    return domain != "photo" or luma[0] <= mean_luma(img) <= luma[1]


# ---------------------------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------------------------
def degrade(
    img: Image.Image, kind: str, level: int, rng: random.Random
) -> tuple[Image.Image, dict[str, float]]:
    """Apply ``kind`` at level index ``level``; returns the new image and the parameter used."""
    spec = KINDS[kind]
    img = img.convert("RGB")
    band = spec.bands[level]
    if band is None:
        return img.copy(), {}
    p = rng.uniform(*band)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(p * min(img.size))), {"radius_frac": p}
    if kind == "noise":
        a = np.asarray(img, dtype=np.float32)
        a = a + np.random.default_rng(rng.getrandbits(32)).normal(0, p, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), {"sigma": p}
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(1, round(p)))
        buf.seek(0)
        return Image.open(buf).convert("RGB"), {"quality": round(p)}
    if kind == "brightness":
        a = np.asarray(img, dtype=np.float32) * p
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), {"factor": p}
    if kind == "tilt":
        angle = p if rng.random() < 0.5 else -p
        out = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        return out, {"degrees": angle}
    raise ValueError(f"unknown kind {kind!r}")


def _rng(seed: int, key: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


@dataclass(frozen=True)
class Source:
    image_id: str
    domain: str
    image: Image.Image
    style: str | None = None


def iter_quality(
    sources: Iterable[Source],
    *,
    seed: int = 0,
    kinds_per_source: int = 2,
    check_source: bool = True,
) -> Iterator[tuple[Image.Image, ImageFacts, QuestionRecord]]:
    """For each good source, pick ``kinds_per_source`` applicable kinds and one level for each.

    Yields ``(degraded image, facts, question)``. A source that fails ``is_good_source`` is
    skipped, since its own defects would make 'sharp' / 'clean' / 'normal' labels untrue.
    """
    for src in sources:
        if check_source and not is_good_source(src.image, src.domain):
            continue
        rng = _rng(seed, src.image_id)
        applicable = [k for k, s in KINDS.items() if src.domain in s.domains]
        for kind in rng.sample(applicable, min(kinds_per_source, len(applicable))):
            spec = KINDS[kind]
            level = rng.randrange(len(spec.levels))
            out, params = degrade(src.image, kind, level, rng)
            iid = f"{src.image_id}+{kind}"
            facts = ImageFacts(
                iid,
                src.domain,
                SOURCE,
                {"source_id": src.image_id, "kind": kind, "level": spec.levels[level], **params},
                style=src.style,
            )
            q = QuestionRecord(
                iid,
                src.domain,
                SOURCE,
                f"quality.{kind}",
                {"type": "score", "instructions": spec.instructions, "levels": list(spec.levels)},
                spec.levels[level],
                style=src.style,
                meta={"source_id": src.image_id, **params},
            )
            yield out, facts, q


def generate(
    sources: Iterable[Source], out_dir: str | Path, *, seed: int = 0, **kw: Any
) -> dict[str, int]:
    """Write degraded images and JSONL. JPEG-artifact images are stored as PNG, since re-saving them
    as JPEG would add artifacts the label doesn't describe; the rest are JPEG at quality 95."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    facts, qs = [], []
    for img, f, q in iter_quality(sources, seed=seed, **kw):
        name = f.image_id.replace(":", "_").replace("+", "__")
        rel = f"images/{name}.png" if q.task == "quality.jpeg" else f"images/{name}.jpg"
        # JPEG-artifact images must be stored losslessly, or re-saving would add artifacts.
        img.save(out / rel, **({} if rel.endswith(".png") else {"quality": 95}))
        f.image_path = rel
        facts.append(f)
        qs.append(q)
    write_jsonl(facts, out / "facts.jsonl")
    write_jsonl(qs, out / "questions.jsonl")
    return {"images": len(facts)}
