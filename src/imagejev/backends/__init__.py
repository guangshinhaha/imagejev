"""Model backends. A backend turns images and questions into option logits."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from ..images import ImageHandle
from ..schema import Question


class Backend(Protocol):
    encoder_id: str

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Features of shape ``(n_tokens, dim)``; row 0 is the global token."""
        ...

    def logits(self, handle: ImageHandle, state: str, question: Question) -> np.ndarray:
        """Logits over a question's options (shape ``(n_options,)``).

        For ``bool`` questions, return shape ``(1,)``: the log-odds that the answer is true.
        """
        ...
