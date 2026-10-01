"""The SigLIP zero-shot prior: aligned text-image knowledge the fusion model starts from.

SigLIP's text and vision towers were trained together, so its text-image match score already means
something. ModernBERT's features and SigLIP's image features were never aligned, so a fusion model
has to *learn* that alignment from scratch, which in a pilot (1,000 steps) it did not do for yes/no
questions (loss stayed at ln 2). The prior hands it the aligned signal directly:

* ``choice`` / ``score``: for each option, SigLIP's scaled similarity between the image and
  ``"<instructions> <option>"``;
* ``bool``: the similarity to the instructions minus the similarity to a generic caption.

These are exactly the logits of the zero-shot baseline. The fusion model adds a learned correction
(whose last layer starts at zero), so training begins at baseline performance and can only improve
on it, and compositional or negated questions, where the zero-shot score is wrong, are what the
correction learns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..schema import Question

GENERIC_CAPTION = "an image"


class TextEmbedder(Protocol):
    """Something that embeds text into the image-aligned space (rows L2-normalised)."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    @property
    def scale_bias(self) -> tuple[float, float]: ...


def prior_texts(q: Question) -> list[str]:
    """The strings embedded to score ``q``; ``prior_from_embeddings`` expects them in this order."""
    if q.type == "bool":
        return [q.instructions, GENERIC_CAPTION]
    prefix = f"{q.instructions} " if q.instructions else ""
    return [f"{prefix}{o.text()}" for o in q.options]


def prior_from_embeddings(
    text_emb: np.ndarray, image_global: np.ndarray, scale: float, bias: float, qtype: str
) -> np.ndarray:
    """Raw zero-shot logits: shape ``(n_options,)``, or ``(1,)`` for bool."""
    img = image_global.astype(np.float32)
    img = img / np.linalg.norm(img)
    sims = text_emb.astype(np.float32) @ img * scale + bias
    return np.array([sims[0] - sims[1]], dtype=np.float32) if qtype == "bool" else sims


class SigLIPTextEmbedder:
    """Adapter over a ``SigLIPBackend`` so the prior uses *identical* text handling (lower-casing,
    padding, caching) to the zero-shot baseline."""

    def __init__(self, backend):
        self.backend = backend

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.backend._text_embeds(list(texts))

    @property
    def scale_bias(self) -> tuple[float, float]:
        m = self.backend.model
        return float(m.logit_scale.exp().item()), float(m.logit_bias.item())


def warm(embedder: TextEmbedder, questions: Sequence[Question], chunk: int = 256) -> int:
    """Embed every string the given questions need, so training never waits on the text tower."""
    texts = sorted({t for q in questions for t in prior_texts(q)})
    for i in range(0, len(texts), chunk):
        embedder.embed(texts[i : i + chunk])
    return len(texts)
