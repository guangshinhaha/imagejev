"""Question schema: parsing and validation of ``choice`` / ``score`` / ``bool`` questions.

The shape deliberately matches Laya's so the same questions work on text or images.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .errors import QuestionSchemaError

MAX_OPTIONS = 32
MAX_STATE_TOKENS = 256
QUESTION_TYPES = ("choice", "score", "bool")

QuestionType = Literal["choice", "score", "bool"]


@dataclass(frozen=True)
class Option:
    """One answer option: a label and an optional description shown to the model."""

    label: str
    description: str = ""

    def text(self) -> str:
        return f"{self.label}: {self.description}" if self.description else self.label


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    instructions: str
    options: tuple[Option, ...] = ()  # empty for bool

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(o.label for o in self.options)


def _parse_options(qid: str, raw: Any, field: str, *, ordered: bool) -> tuple[Option, ...]:
    if isinstance(raw, Mapping):
        items = [(str(k), "" if v is None else str(v)) for k, v in raw.items()]
    elif isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        items = [(str(v), "") for v in raw]
    else:
        raise QuestionSchemaError(
            f"question {qid!r}: '{field}' must be a dict or a list, got {type(raw).__name__}"
        )
    min_n = 2 if ordered else 1
    if len(items) < min_n:
        raise QuestionSchemaError(
            f"question {qid!r}: '{field}' needs at least {min_n} entr{'ies' if min_n > 1 else 'y'}"
        )
    if len(items) > MAX_OPTIONS:
        raise QuestionSchemaError(
            f"question {qid!r}: '{field}' has {len(items)} entries; the maximum is {MAX_OPTIONS}"
        )
    labels = [label for label, _ in items]
    if any(not label.strip() for label in labels):
        raise QuestionSchemaError(f"question {qid!r}: '{field}' contains an empty label")
    if len(set(labels)) != len(labels):
        raise QuestionSchemaError(f"question {qid!r}: '{field}' contains duplicate labels")
    return tuple(Option(label, desc) for label, desc in items)


def parse_question(qid: str, spec: Mapping[str, Any]) -> Question:
    if not isinstance(spec, Mapping):
        raise QuestionSchemaError(f"question {qid!r} must be a dict, got {type(spec).__name__}")
    qtype = spec.get("type")
    if qtype not in QUESTION_TYPES:
        raise QuestionSchemaError(
            f"question {qid!r}: unknown type {qtype!r}; expected one of {QUESTION_TYPES}"
        )
    instructions = spec.get("instructions", "")
    if not isinstance(instructions, str):
        raise QuestionSchemaError(f"question {qid!r}: 'instructions' must be a string")
    if qtype == "bool":
        if not instructions.strip():
            raise QuestionSchemaError(f"question {qid!r}: bool questions need 'instructions'")
        return Question(qid, "bool", instructions.strip())
    if qtype == "choice":
        if "criteria" not in spec:
            raise QuestionSchemaError(f"question {qid!r}: choice questions need 'criteria'")
        options = _parse_options(qid, spec["criteria"], "criteria", ordered=False)
        if len(options) < 2:
            raise QuestionSchemaError(f"question {qid!r}: 'criteria' needs at least 2 entries")
        return Question(qid, "choice", instructions.strip(), options)
    if "levels" not in spec:
        raise QuestionSchemaError(f"question {qid!r}: score questions need 'levels'")
    options = _parse_options(qid, spec["levels"], "levels", ordered=True)
    return Question(qid, "score", instructions.strip(), options)


def parse_questions(questions: Mapping[str, Mapping[str, Any]]) -> list[Question]:
    if not isinstance(questions, Mapping) or not questions:
        raise QuestionSchemaError("'questions' must be a non-empty dict of id -> question")
    return [parse_question(str(qid), spec) for qid, spec in questions.items()]


class _Tokenizer(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...
    def decode(self, ids: list[int], **kwargs: Any) -> str: ...


_PIECE = re.compile(r"\w+|[^\w\s]")


def state_to_text(state: str | Mapping[str, Any] | None) -> str:
    """Serialise the optional ``state`` argument to text."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, sort_keys=True, ensure_ascii=False)
    except TypeError as e:
        raise QuestionSchemaError(f"'state' must be a string or JSON-serialisable: {e}") from e


def truncate_state(
    text: str, max_tokens: int = MAX_STATE_TOKENS, tokenizer: _Tokenizer | None = None
) -> str:
    """Truncate ``text`` to ``max_tokens`` tokens, warning when anything was cut.

    With no tokenizer, tokens are approximated as words and punctuation marks.
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        cut = tokenizer.decode(ids[:max_tokens])
    else:
        pieces = list(_PIECE.finditer(text))
        if len(pieces) <= max_tokens:
            return text
        cut = text[: pieces[max_tokens - 1].end()]
    warnings.warn(f"state longer than {max_tokens} tokens was truncated", UserWarning, stacklevel=3)
    return cut
