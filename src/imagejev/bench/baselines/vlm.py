"""Small-VLM baseline: SmolVLM prompted with the question, answer read from next-token logits.

No text is generated. One forward pass per question; the logits of the answer tokens become the
option probabilities, so it is a fair "decide in one pass" comparison:

* ``choice`` / ``score``: options are lettered (A, B, ...); the logits of the letter tokens.
* ``bool``: the log-odds of "Yes" against "No".

Both bare and space-prefixed tokens are summed (``Yes`` and `` Yes``), because the chat template
ends in ``Assistant:`` and the model answers with a leading space.

The image can be encoded once and reused (``encode``): the connector output is cached and passed
back in, which reproduces the full pipeline's logits exactly (checked in the slow tests). Without
that, the VLM would pay for image encoding on every question and the latency comparison with the
encode-once model would be unfair to it.
"""

from __future__ import annotations

import string
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ...api import DEFAULT_TEMPERATURES, format_answer
from ...images import image_hash, load_image
from ...schema import Question, parse_questions, state_to_text, truncate_state

DEFAULT_CHECKPOINT = "HuggingFaceTB/SmolVLM-500M-Instruct"
LETTERS = string.ascii_uppercase + string.ascii_lowercase[:6]  # 32 options at most
YES = ("Yes", "yes")
NO = ("No", "no")


def option_letters(n: int) -> str:
    if not 2 <= n <= len(LETTERS):
        raise ValueError(f"need between 2 and {len(LETTERS)} options, got {n}")
    return LETTERS[:n]


def build_prompt(q: Question, state: str = "") -> str:
    """The text part of the user turn for one question."""
    parts = []
    if state:
        parts.append(f"Context: {state}")
    if q.instructions:
        parts.append(q.instructions)
    if q.type == "bool":
        parts.append("Answer Yes or No.")
        return "\n".join(parts)
    heading = "Options (ordered from lowest to highest):" if q.type == "score" else "Options:"
    parts.append(heading)
    for letter, option in zip(option_letters(len(q.options)), q.options, strict=True):
        parts.append(f"{letter}. {option.text()}")
    parts.append("Answer with the letter of the correct option.")
    return "\n".join(parts)


@dataclass(frozen=True)
class VLMHandle:
    key: str
    features: Any  # the connector output, a tensor on the model's device


class SmolVLM:
    """Prompted small VLM with the same ``predict`` / ``logits`` / ``encode`` surface as Model."""

    def __init__(
        self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None, cache_size: int = 64
    ):
        from ...backends.siglip import _use_system_trust_store, pick_device

        _use_system_trust_store()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = pick_device(device)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.name = f"smolvlm:{checkpoint.rsplit('/', 1)[-1]}"
        self.processor = AutoProcessor.from_pretrained(checkpoint, do_image_splitting=False)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.temperatures = dict(DEFAULT_TEMPERATURES)
        self._dtype = dtype
        self._cache: OrderedDict[str, VLMHandle] = OrderedDict()
        self._cache_size = cache_size
        tok = self.processor.tokenizer
        self._ids: dict[str, list[int]] = {}
        for word in (*YES, *NO, *LETTERS):
            self._ids[word] = [
                tok.encode(v, add_special_tokens=False)[0] for v in (word, " " + word)
            ]
        self._blank = Image.new(
            "RGB", (32, 32), "white"
        )  # tokenisation is independent of image size

    # -- image side ---------------------------------------------------------------------------
    def encode(self, image: Any) -> VLMHandle:
        if isinstance(image, VLMHandle):
            return image
        img = load_image(image)
        key = image_hash(img)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        torch = self._torch
        px = self.processor.image_processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            out = self.model.model.get_image_features(
                px["pixel_values"].to(self.device, self._dtype),
                px["pixel_attention_mask"].to(self.device),
            )
        handle = VLMHandle(key, out.pooler_output)
        self._cache[key] = handle
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return handle

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- question side ------------------------------------------------------------------------
    def _next_token_logits(self, handle: VLMHandle, text: str):
        torch = self._torch
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        enc = self.processor(text=prompt, images=[self._blank], return_tensors="pt")
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
                mm_encoder_outputs=handle.features,
            )
        return out.logits[0, -1].float()

    def _lse(self, logits, words: tuple[str, ...]) -> float:
        torch = self._torch
        ids = [i for w in words for i in self._ids[w]]
        return float(torch.logsumexp(logits[ids], dim=0))

    def _question_logits(self, handle: VLMHandle, state: str, q: Question) -> np.ndarray:
        logits = self._next_token_logits(handle, build_prompt(q, state))
        if q.type == "bool":
            return np.array([self._lse(logits, YES) - self._lse(logits, NO)])
        letters = option_letters(len(q.options))
        return np.array([self._lse(logits, (letter,)) for letter in letters])

    # -- public surface -----------------------------------------------------------------------
    def logits(
        self, image: Any, questions: Mapping[str, Any], state: Any = None
    ) -> dict[str, np.ndarray]:
        parsed = parse_questions(questions)
        text = truncate_state(state_to_text(state))
        handle = self.encode(image)
        return {q.id: self._question_logits(handle, text, q) for q in parsed}

    def predict(
        self, image: Any, questions: Mapping[str, Any], state: Any = None
    ) -> dict[str, dict[str, Any]]:
        parsed = parse_questions(questions)
        raw = self.logits(image, questions, state)
        return {
            q.id: format_answer(
                q,
                raw[q.id],
                self.temperatures[q.type],
                self.temperatures["bool_bias"] if q.type == "bool" else 0.0,
            )
            for q in parsed
        }
