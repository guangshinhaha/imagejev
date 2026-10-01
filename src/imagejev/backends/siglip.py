"""Stage 0 backend: SigLIP 2 zero-shot behind the typed API.

This is the baseline every trained model is compared against. It ignores ``state`` and has no
real notion of instructions beyond prepending them to each option's text.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..images import ImageHandle
from ..schema import Question

DEFAULT_CHECKPOINT = "google/siglip2-base-patch16-384"
POOL = 3  # 24x24 patch grid -> 8x8 = 64 tokens


def _use_system_trust_store() -> None:
    """Make HTTPS downloads honour the OS trust store (corporate proxies, custom CAs)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def pick_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SigLIPBackend:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        _use_system_trust_store()
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = pick_device(device)
        self.encoder_id = f"siglip2:{checkpoint}:pool{POOL}"
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self._text_cache: dict[str, np.ndarray] = {}

    # -- image side -------------------------------------------------------------------------
    def encode_image(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision = self.model.vision_model(pixel_values=inputs["pixel_values"])
            global_emb = vision.pooler_output  # (1, d): the vector used for zero-shot matching
            patches = vision.last_hidden_state  # (1, n, d)
        n = patches.shape[1]
        side = int(round(n**0.5))
        if side * side != n or side % POOL:
            raise ValueError(f"unexpected patch grid: {n} tokens")
        grid = patches.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.avg_pool2d(grid, POOL).flatten(2).transpose(1, 2)  # (1,64,d)
        feats = torch.cat([global_emb[:, None, :], pooled], dim=1)[0]
        return feats.float().cpu().numpy().astype(np.float16)

    # -- text side --------------------------------------------------------------------------
    def _text_embeds(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            torch = self._torch
            # SigLIP text towers were trained with lowercase text padded to max_length
            inputs = self.processor(
                text=[t.lower() for t in missing],
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
                emb = getattr(emb, "pooler_output", emb)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            for t, e in zip(missing, emb.float().cpu().numpy(), strict=True):
                self._text_cache[t] = e
        return np.stack([self._text_cache[t] for t in texts])

    def _scaled_similarity(self, handle: ImageHandle, texts: list[str]) -> np.ndarray:
        img = handle.features[0].astype(np.float32)
        img = img / np.linalg.norm(img)
        sims = self._text_embeds(texts) @ img
        scale = self.model.logit_scale.exp().item()
        bias = self.model.logit_bias.item()
        return sims * scale + bias

    # -- typed questions --------------------------------------------------------------------
    def logits(self, handle: ImageHandle, state: str, question: Question) -> np.ndarray:
        prefix = f"{question.instructions} " if question.instructions else ""
        if question.type == "bool":
            sims = self._scaled_similarity(handle, [f"{prefix}Yes.", f"{prefix}No."])
            return np.array([sims[0] - sims[1]])
        texts = [f"{prefix}{o.text()}" for o in question.options]
        return self._scaled_similarity(handle, texts)
