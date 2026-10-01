"""Public API: ``Model.load(...)``, ``Model.encode(...)`` and ``Model.predict(...)``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .backends import Backend
from .images import CachedEncoder, ImageHandle, load_image
from .schema import Question, parse_questions, state_to_text, truncate_state

DEFAULT_TEMPERATURES = {"choice": 1.0, "score": 1.0, "bool": 1.0}


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(x: float, temperature: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-x / temperature)))


def format_answer(q: Question, logits: np.ndarray, temperature: float) -> dict[str, Any]:
    """Turn raw logits into the public per-question result dict."""
    if q.type == "bool":
        p_true = sigmoid(float(logits[0]), temperature)
        return {
            "answer": p_true >= 0.5,
            "p_true": p_true,
            "confidence": max(p_true, 1.0 - p_true),
        }
    probs = softmax(logits, temperature)
    best = int(np.argmax(probs))
    out: dict[str, Any] = {
        "answer": q.options[best].label,
        "probs": {o.label: float(p) for o, p in zip(q.options, probs, strict=True)},
        "confidence": float(probs[best]),
    }
    if q.type == "score":
        out["expected"] = float(np.dot(np.arange(len(probs)), probs))
    return out


class Model:
    def __init__(
        self,
        backend: Backend,
        temperatures: Mapping[str, float] | None = None,
        cache_size: int = 256,
    ):
        self.backend = backend
        self.temperatures = {**DEFAULT_TEMPERATURES, **(temperatures or {})}
        self._encoder = CachedEncoder(backend, max_items=cache_size)

    @classmethod
    def load(cls, name: str = "siglip2-zeroshot", device: str | None = None) -> Model:
        if name == "siglip2-zeroshot":
            from .backends.siglip import SigLIPBackend

            return cls(SigLIPBackend(device=device))
        raise ValueError(f"unknown model {name!r}; available: 'siglip2-zeroshot'")

    def encode(self, image: Any) -> ImageHandle:
        """Encode an image once; pass the handle to ``predict`` to skip the vision encoder."""
        return self._encoder.encode(image)

    def predict(
        self,
        image: Any,
        questions: Mapping[str, Mapping[str, Any]],
        state: str | Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        parsed = parse_questions(questions)  # validate before doing any expensive work
        state_text = truncate_state(state_to_text(state))
        handle = self.encode(image)
        return {
            q.id: format_answer(
                q, self.backend.logits(handle, state_text, q), self.temperatures[q.type]
            )
            for q in parsed
        }


__all__ = ["Model", "load_image"]
