"""The trained model as a ``Backend``: frozen SigLIP 2 vision, ModernBERT + LoRA, fusion model.

``export_model`` turns a training checkpoint into a self-contained directory (safetensors weights,
``config.json``, optional ``temperatures.json``); ``load_trained`` rebuilds it, and ``Model.load``
accepts that directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..images import ImageHandle
from ..model.fusion import QTYPE_IDS, FusionBatch, FusionModel
from ..model.text import TextEncoder
from ..schema import Question
from ..train.data import option_texts
from .siglip import DEFAULT_CHECKPOINT, SigLIPBackend, pick_device

FORMAT_VERSION = 1
WEIGHT_FILES = ("fusion.safetensors", "lora.safetensors")


class TrainedBackend:
    def __init__(
        self,
        text: TextEncoder,
        fusion: FusionModel,
        vision: Any,
        device: str | torch.device,
        name: str = "imagejev",
    ):
        self.device = torch.device(device)
        self.text = text.to(self.device).eval()
        self.fusion = fusion.to(self.device).eval()
        self.vision = vision
        self.encoder_id = vision.encoder_id
        self.name = name

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.vision.encode_image(image)

    @torch.no_grad()
    def logits(self, handle: ImageHandle, state: str, question: Question) -> np.ndarray:
        segs = option_texts(question)
        enc = self.text.encode_pairs([question.instructions] * len(segs), segs)
        n = len(segs)
        st = self.text.encode_state(state)
        batch = FusionBatch(
            opt_tokens=enc.tokens,
            opt_mask=enc.mask,
            opt_focus=enc.focus,
            owner=torch.zeros(n, dtype=torch.long, device=self.device),
            slot=torch.arange(n, device=self.device),
            qtype=torch.tensor([QTYPE_IDS[question.type]], device=self.device),
            image=torch.from_numpy(np.asarray(handle.features)).float()[None].to(self.device),
            state_tokens=st.tokens if st else None,
            state_mask=st.mask if st else None,
        )
        return self.fusion(batch).float().cpu().numpy()


def export_model(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    vision_checkpoint: str = DEFAULT_CHECKPOINT,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a loadable model directory from a training checkpoint (``best.pt`` or ``last.pt``)."""
    from safetensors.torch import save_file

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck["fusion"].items()}, out / WEIGHT_FILES[0])
    save_file({k: v.contiguous() for k, v in ck["lora"].items()}, out / WEIGHT_FILES[1])
    d_text = int(ck["fusion"]["text_adapter.1.weight"].shape[1])
    d_image = int(ck["fusion"]["image_adapter.1.weight"].shape[1])
    config = {
        "format_version": FORMAT_VERSION,
        "vision_checkpoint": vision_checkpoint,
        "backbone": cfg["backbone"],
        "lora": {"rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"]},
        "fusion": {
            "d_text": d_text,
            "d_image": d_image,
            "d": cfg["d_model"],
            "heads": cfg["heads"],
            "n_blocks": cfg["n_blocks"],
            "adapter_hidden": int(ck["fusion"]["text_adapter.1.weight"].shape[0]),
        },
        "trained_steps": int(ck["step"]),
        "selection_score": float(ck["best"]),
        **(extra or {}),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out


def load_trained(
    model_dir: str | Path,
    *,
    device: str | None = None,
    text: TextEncoder | None = None,
    vision: Any = None,
) -> tuple[TrainedBackend, dict[str, float] | None]:
    """Rebuild a model from ``export_model``'s directory. ``text`` and ``vision`` can be injected
    (tests use tiny stand-ins); by default the pretrained backbones are downloaded."""
    from safetensors.torch import load_file

    from ..calibration import load_temperatures

    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported model format {cfg.get('format_version')!r}")
    dev = pick_device(device)
    if text is None:
        text = TextEncoder.from_pretrained(
            cfg["backbone"], rank=cfg["lora"]["rank"], alpha=cfg["lora"]["alpha"]
        )
    text.load_lora_state_dict(load_file(d / WEIGHT_FILES[1]))
    f = cfg["fusion"]
    fusion = FusionModel(
        d_text=f["d_text"],
        d_image=f["d_image"],
        d=f["d"],
        heads=f["heads"],
        n_blocks=f["n_blocks"],
        adapter_hidden=f["adapter_hidden"],
    )
    fusion.load_state_dict(load_file(d / WEIGHT_FILES[0]))
    if vision is None:
        vision = SigLIPBackend(cfg["vision_checkpoint"], device=dev)
    backend = TrainedBackend(text, fusion, vision, dev, name=f"imagejev:{d.name}")
    temps = load_temperatures(d) if (d / "temperatures.json").exists() else None
    return backend, temps
