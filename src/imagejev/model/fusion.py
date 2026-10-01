"""Fusion: image + state memory, option encoder blocks, option mixing, and answer heads.

Data flow for a batch of ``Q`` questions with ``N`` options in total::

    option tokens (N, L, d_text) --adapter--> (N, L, d)
    memory per question = [image tokens (65) ; state tokens]   --adapters--> (Q, M, d)
    4 x FusionBlock: self-attention over an option's own tokens
                     -> cross-attention to its question's memory -> MLP
    pool each option -> vector (N, d)           (+ a rank embedding for ``score`` options only)
    OptionMixer: self-attention across the options of one question, no positional information
    head per question type -> one logit per option           (bool: the single logit is a log-odds)

Order handling. Options are encoded independently with shared weights and the mixer has no
positional encoding, so for ``choice`` and ``bool`` the output cannot depend on option order: a
permutation of the options permutes the logits and nothing else. ``score`` levels *are* ordered
(low to high), and a set-based model cannot know which end is "high" unless the words say so, so
score options also receive a rank embedding (position / (K - 1)). That is the one deliberate
exception to "no order information", and it is tested separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

QTYPE_IDS = {"choice": 0, "score": 1, "bool": 2}
N_IMAGE_TOKENS = 65  # 1 global + 8 x 8 pooled patches


def _adapter(d_in: int, d: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d)
    )


class FusionBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_ratio * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_pad: torch.Tensor,
        memory: torch.Tensor,
        mem_pad: torch.Tensor,
    ) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.drop(self.self_attn(h, h, h, key_padding_mask=x_pad, need_weights=False)[0])
        h = self.n2(x)
        x = x + self.drop(
            self.cross_attn(h, memory, memory, key_padding_mask=mem_pad, need_weights=False)[0]
        )
        return x + self.drop(self.mlp(self.n3(x)))


class OptionMixer(nn.Module):
    """Self-attention across the options of one question. No positional encoding."""

    def __init__(self, d: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d)
        )

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


@dataclass
class FusionBatch:
    """Everything the fusion model needs for ``Q`` questions with ``N`` options in total."""

    opt_tokens: torch.Tensor  # (N, L, d_text) text-encoder features of [instructions ; option]
    opt_mask: torch.Tensor  # (N, L) bool, True for real tokens
    owner: torch.Tensor  # (N,) long, index in [0, Q) of the question each option belongs to
    slot: torch.Tensor  # (N,) long, position of the option in its question (layout only)
    qtype: torch.Tensor  # (Q,) long, QTYPE_IDS
    image: torch.Tensor  # (Q, 65, d_image) cached SigLIP features
    state_tokens: torch.Tensor | None = None  # (Q, Ls, d_text)
    state_mask: torch.Tensor | None = None  # (Q, Ls) bool
    opt_focus: torch.Tensor | None = None  # (N, L) bool: tokens to pool; default = opt_mask

    def n_options(self) -> torch.Tensor:
        """Options per question, shape (Q,)."""
        return torch.bincount(self.owner, minlength=self.qtype.shape[0])


class FusionModel(nn.Module):
    def __init__(
        self,
        d_text: int = 768,
        d_image: int = 768,
        d: int = 512,
        heads: int = 8,
        n_blocks: int = 4,
        adapter_hidden: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.text_adapter = _adapter(d_text, d, adapter_hidden)
        self.image_adapter = _adapter(d_image, d, adapter_hidden)
        self.patch_pos = nn.Parameter(torch.zeros(N_IMAGE_TOKENS, d))  # global + 8x8 grid
        self.type_emb = nn.Embedding(2, d)  # 0 = image memory, 1 = state memory
        self.blocks = nn.ModuleList(
            [FusionBlock(d, heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(d)
        self.rank_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.mixer = OptionMixer(d, heads, dropout=dropout)
        self.mix_norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)) for _ in QTYPE_IDS]
        )
        nn.init.normal_(self.patch_pos, std=0.02)

    # -- pieces -----------------------------------------------------------------------------
    def build_memory(self, b: FusionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.image_adapter(b.image) + self.patch_pos + self.type_emb.weight[0]
        pad = torch.zeros(img.shape[:2], dtype=torch.bool, device=img.device)
        if b.state_tokens is None:
            return img, pad
        st = self.text_adapter(b.state_tokens) + self.type_emb.weight[1]
        st_mask = (
            b.state_mask
            if b.state_mask is not None
            else torch.ones(st.shape[:2], dtype=torch.bool, device=st.device)
        )
        return torch.cat([img, st], dim=1), torch.cat([pad, ~st_mask], dim=1)

    def option_vectors(self, b: FusionBatch) -> torch.Tensor:
        memory, mem_pad = self.build_memory(b)
        x = self.text_adapter(b.opt_tokens)
        x_pad = ~b.opt_mask
        memory, mem_pad = memory[b.owner], mem_pad[b.owner]  # each option sees its own question's
        for block in self.blocks:
            x = block(x, x_pad, memory, mem_pad)
        x = self.out_norm(x)
        focus = b.opt_focus if b.opt_focus is not None else b.opt_mask
        keep = focus.unsqueeze(-1).to(x.dtype)
        return (x * keep).sum(1) / keep.sum(1).clamp(min=1)  # mean over the option's own tokens

    def rank_features(self, b: FusionBatch) -> torch.Tensor:
        """(N, 2): normalised level position for score options, zeros (and no flag) otherwise."""
        k = b.n_options()[b.owner].to(torch.float32)
        is_score = (b.qtype[b.owner] == QTYPE_IDS["score"]).to(torch.float32)
        frac = b.slot.to(torch.float32) / (k - 1).clamp(min=1)
        return torch.stack([frac * is_score, is_score], dim=-1)

    def forward(self, b: FusionBatch) -> torch.Tensor:
        """One logit per option, shape (N,). For ``bool`` the logit is the log-odds of true."""
        vec = self.option_vectors(b)
        vec = vec + self.rank_mlp(self.rank_features(b)) * self.rank_features(b)[:, 1:2]
        q = b.qtype.shape[0]
        k_max = int(b.n_options().max())
        grid = vec.new_zeros(q, k_max, self.d)
        pad = torch.ones(q, k_max, dtype=torch.bool, device=vec.device)
        grid[b.owner, b.slot] = vec
        pad[b.owner, b.slot] = False
        mixed = self.mix_norm(self.mixer(grid, pad))[b.owner, b.slot]
        # Logits stay float32 whatever the autocast dtype: the heads emit half precision under
        # autocast, and writing that into a float32 buffer is an index_put dtype error.
        logits = vec.new_zeros(vec.shape[0], dtype=torch.float32)
        row_type = b.qtype[b.owner]
        for tid, head in enumerate(self.heads):
            sel = row_type == tid
            if sel.any():
                logits[sel] = head(mixed[sel]).squeeze(-1).float()
        return logits


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
