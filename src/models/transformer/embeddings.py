"""Scaled Token Embedding for Transformer models (Vaswani et al., 2017)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID


class ScaledEmbedding(nn.Module):
    """Token embedding initialized with N(0, d_model^-0.5) and scaled by sqrt(d_model)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int | None = PAD_ID,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size phai > 0, nhan duoc {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model phai > 0, nhan duoc {d_model}")
        if pad_id is not None and not (0 <= pad_id < vocab_size):
            raise ValueError(
                f"pad_id phai nam trong [0, {vocab_size - 1}], nhan duoc {pad_id}"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model**-0.5)
        if self.pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_id].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and scale by sqrt(d_model) so non-pad std stays near 1.0."""
        return self.embedding(token_ids) * self.scale
