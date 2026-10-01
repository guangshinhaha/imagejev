import json

import numpy as np
import pytest
import torch
from PIL import Image

from imagejev.api import Model
from imagejev.backends import trained as trained_mod
from imagejev.backends.trained import TrainedBackend, export_model, load_trained
from imagejev.calibration import save_temperatures
from imagejev.model.testing import tiny_fusion, tiny_text_encoder

D_IMG = 24
CHOICE = {
    "type": "choice",
    "instructions": "Which colour?",
    "criteria": {"red": "warm", "green": "", "blue": "cool"},
}
SCORE = {"type": "score", "instructions": "How bright?", "levels": ["dark", "mid", "bright"]}
BOOL = {"type": "bool", "instructions": "Is it red?"}


class FakeVision:
    """Features are a deterministic function of the image, 65 tokens like the real encoder."""

    encoder_id = "fakevision:v1"

    def __init__(self):
        self.calls = 0

    def encode_image(self, image):
        self.calls += 1
        r, g, b = image.resize((1, 1)).getpixel((0, 0))
        rng = np.random.default_rng(r * 65536 + g * 256 + b)
        return rng.normal(size=(65, D_IMG)).astype(np.float16)


def make_checkpoint(path, seed=0):
    text, fusion = tiny_text_encoder(seed=seed), tiny_fusion(seed=seed)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():  # give the LoRA weights real values, so a lost weight would show
        for p in text.lora_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    cfg = {
        "backbone": "tiny",
        "lora_rank": 4,
        "lora_alpha": 32.0,
        "d_model": 32,
        "heads": 4,
        "n_blocks": 2,
    }
    torch.save(
        {
            "fusion": fusion.state_dict(),
            "lora": text.lora_state_dict(),
            "config": cfg,
            "step": 7,
            "best": 0.9,
        },
        path,
    )
    return text, fusion


@pytest.fixture
def exported(tmp_path):
    text, fusion = make_checkpoint(tmp_path / "ck.pt")
    export_model(tmp_path / "ck.pt", tmp_path / "model", vision_checkpoint="fake/vision")
    return tmp_path / "model", text, fusion


def img(color=(200, 30, 30)):
    return Image.new("RGB", (8, 8), color)


def test_export_writes_a_self_describing_directory(exported):
    d, _, _ = exported
    cfg = json.loads((d / "config.json").read_text())
    assert (d / "fusion.safetensors").exists() and (d / "lora.safetensors").exists()
    assert cfg["format_version"] == 1 and cfg["vision_checkpoint"] == "fake/vision"
    assert cfg["fusion"] == {
        "d_text": 32,
        "d_image": D_IMG,
        "d": 32,
        "heads": 4,
        "n_blocks": 2,
        "adapter_hidden": 48,
    }
    assert cfg["trained_steps"] == 7 and cfg["selection_score"] == 0.9 and cfg["lora"]["rank"] == 4


def test_loaded_model_reproduces_the_original_logits_exactly(exported):
    d, text, fusion = exported
    original = TrainedBackend(text, fusion, FakeVision(), "cpu")
    # The export holds only LoRA + fusion weights; the frozen base ModernBERT comes from the same
    # pretrained checkpoint in real use, so the stand-in must share the original's base weights.
    backend, temps = load_trained(
        d, device="cpu", text=tiny_text_encoder(seed=0), vision=FakeVision()
    )
    assert temps is None
    from imagejev.images import CachedEncoder
    from imagejev.schema import parse_question

    enc = CachedEncoder(original)
    h = enc.encode(img())
    for spec in (CHOICE, SCORE, BOOL):
        q = parse_question("q", spec)
        a = original.logits(h, "", q)
        b = backend.logits(h, "", q)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-6), spec["type"]


def test_model_load_serves_the_public_api_with_saved_calibration(exported, monkeypatch):
    d, _, _ = exported
    save_temperatures({"choice": 2.0, "score": 1.5, "bool": 3.0, "bool_bias": -0.5}, d)
    monkeypatch.setattr(
        trained_mod.TextEncoder,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_text_encoder(seed=5)),
    )
    monkeypatch.setattr(trained_mod, "SigLIPBackend", lambda *a, **k: FakeVision())
    m = Model.load(str(d), device="cpu")
    assert m.temperatures["choice"] == 2.0 and m.temperatures["bool_bias"] == -0.5
    out = m.predict(img(), {"c": CHOICE, "s": SCORE, "b": BOOL})
    assert set(out["c"]) == {"answer", "probs", "confidence"} and sum(
        out["c"]["probs"].values()
    ) == pytest.approx(1.0)
    assert set(out["s"]) == {"answer", "probs", "expected", "confidence"}
    assert set(out["b"]) == {"answer", "p_true", "confidence"} and 0 <= out["b"]["p_true"] <= 1
    # explicit temperatures override the saved ones
    m2 = Model.load(str(d), device="cpu", temperatures={"choice": 1.0})
    assert m2.temperatures["choice"] == 1.0


def test_encode_once_ask_many_and_state_matters(exported, monkeypatch):
    d, _, _ = exported
    vision = FakeVision()
    monkeypatch.setattr(
        trained_mod.TextEncoder,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_text_encoder()),
    )
    monkeypatch.setattr(trained_mod, "SigLIPBackend", lambda *a, **k: vision)
    m = Model.load(str(d), device="cpu")
    h = m.encode(img())
    base = m.predict(h, {"c": CHOICE})["c"]["probs"]
    m.predict(h, {"b": BOOL})
    assert vision.calls == 1  # the image was encoded once for both calls
    assert m.predict(h, {"c": CHOICE}, state="the customer returned the item")["c"]["probs"] != base
    assert m.predict(img(), {"c": CHOICE})["c"]["probs"] == base  # same pixels, same cached handle
    assert vision.calls == 1


def test_wrong_format_version_and_missing_dir(exported):
    d, _, _ = exported
    cfg = json.loads((d / "config.json").read_text())
    cfg["format_version"] = 99
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="format"):
        load_trained(d, device="cpu", text=tiny_text_encoder(), vision=FakeVision())
    with pytest.raises(ValueError, match="unknown model"):
        Model.load("/nonexistent/path")
