"""Image loading and the ``ImageHandle`` feature cache."""

from __future__ import annotations

import hashlib
import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import ImageLoadError


def load_image(source: Any) -> Image.Image:
    """Decode a path, bytes or PIL image into an RGB PIL image.

    Raises ``ImageLoadError`` when the input cannot be decoded.
    """
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, bytes | bytearray):
            img = Image.open(io.BytesIO(bytes(source)))
        elif isinstance(source, str | os.PathLike):
            img = Image.open(source)
        else:
            raise ImageLoadError(f"unsupported image input type: {type(source).__name__}")
        img.load()
        return img.convert("RGB")
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise ImageLoadError(f"could not decode image: {e}") from e


def image_hash(img: Image.Image) -> str:
    """Stable content hash of the decoded pixels (size + RGB bytes)."""
    h = hashlib.sha256()
    h.update(f"{img.width}x{img.height}".encode())
    h.update(img.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ImageHandle:
    """Cached vision features for one image.

    ``features`` has shape ``(n_tokens, dim)``; for the default encoder that is 64 pooled patch
    tokens plus 1 global token. ``encoder_id`` guards against mixing handles across encoders.
    """

    key: str
    encoder_id: str
    features: np.ndarray


class ImageEncoder(Protocol):
    encoder_id: str

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return features of shape ``(n_tokens, dim)`` for an RGB image."""
        ...


class CachedEncoder:
    """Wraps an ``ImageEncoder`` with an LRU cache keyed by image content."""

    def __init__(self, encoder: ImageEncoder, max_items: int = 256):
        self.encoder = encoder
        self.max_items = max_items
        self._cache: OrderedDict[str, ImageHandle] = OrderedDict()

    def encode(self, source: Any) -> ImageHandle:
        if isinstance(source, ImageHandle):
            if source.encoder_id != self.encoder.encoder_id:
                raise ImageLoadError(
                    f"ImageHandle was made by encoder {source.encoder_id!r}, "
                    f"not {self.encoder.encoder_id!r}"
                )
            return source
        img = load_image(source)
        key = f"{self.encoder.encoder_id}:{image_hash(img)}"
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        feats = np.asarray(self.encoder.encode_image(img))
        if feats.ndim != 2:
            raise ValueError(f"encoder must return (n_tokens, dim); got shape {feats.shape}")
        handle = ImageHandle(key, self.encoder.encoder_id, feats)
        self._cache[key] = handle
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return handle
