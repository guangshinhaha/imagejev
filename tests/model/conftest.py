"""Shared fixtures: a tiny random ModernBERT and a fake tokenizer, so model tests run offline."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from imagejev.model.testing import FakeTokenizer, tiny_backbone  # noqa: E402


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def backbone():
    return tiny_backbone(hidden=64).eval()


@pytest.fixture
def make_backbone():
    return lambda: tiny_backbone(hidden=64).eval()
