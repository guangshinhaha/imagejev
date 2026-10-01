import io

import numpy as np
import pytest
from PIL import Image

from imagejev.errors import ImageLoadError
from imagejev.images import CachedEncoder, ImageHandle, image_hash, load_image


def _img(color=(255, 0, 0), size=(8, 6)):
    return Image.new("RGB", size, color)


class FakeEncoder:
    encoder_id = "fake"

    def __init__(self):
        self.calls = 0

    def encode_image(self, image):
        self.calls += 1
        return np.full((3, 4), image.getpixel((0, 0))[0], dtype=np.float16)


def test_load_from_all_input_kinds(tmp_path):
    img = _img()
    p = tmp_path / "a.png"
    img.save(p)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    for src in (p, str(p), buf.getvalue(), img):
        out = load_image(src)
        assert out.mode == "RGB" and out.size == (8, 6)


def test_load_converts_mode():
    assert load_image(Image.new("L", (4, 4))).mode == "RGB"
    assert load_image(Image.new("RGBA", (4, 4))).mode == "RGB"


@pytest.mark.parametrize("bad", [b"not an image", 123, None])
def test_undecodable_raises(bad):
    with pytest.raises(ImageLoadError):
        load_image(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "nope.png")


def test_hash_depends_on_content_and_size():
    assert image_hash(_img()) == image_hash(_img())
    assert image_hash(_img()) != image_hash(_img((0, 255, 0)))
    assert image_hash(_img(size=(8, 6))) != image_hash(_img(size=(6, 8)))


def test_encode_caches_by_content():
    enc = FakeEncoder()
    c = CachedEncoder(enc)
    h1 = c.encode(_img())
    h2 = c.encode(_img())  # distinct object, same pixels
    assert h1 is h2 and enc.calls == 1
    assert h1.features.shape == (3, 4)


def test_handle_passthrough_and_encoder_mismatch():
    c = CachedEncoder(FakeEncoder())
    h = c.encode(_img())
    assert c.encode(h) is h
    foreign = ImageHandle("k", "other", np.zeros((1, 1)))
    with pytest.raises(ImageLoadError):
        c.encode(foreign)


def test_lru_eviction():
    enc = FakeEncoder()
    c = CachedEncoder(enc, max_items=1)
    c.encode(_img((1, 0, 0)))
    c.encode(_img((2, 0, 0)))
    c.encode(_img((1, 0, 0)))
    assert enc.calls == 3


def test_bad_feature_shape_rejected():
    class Bad(FakeEncoder):
        def encode_image(self, image):
            return np.zeros(4)

    with pytest.raises(ValueError):
        CachedEncoder(Bad()).encode(_img())
