"""Sinusoidal Positional Encoding for Transformer models (Vaswani et al., 2017)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings with dropout."""

    pe: torch.Tensor

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(
                f"d_model phai la so nguyen duong chan, nhan duoc {d_model}"
            )
        if max_len <= 0:
            raise ValueError(f"max_len phai > 0, nhan duoc {max_len}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout phai nam trong [0.0, 1.0), nhan duoc {dropout}")

        self.d_model = d_model
        self.dropout_rate = dropout
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # non-persistent buffer allows loading checkpoints cleanly with a different max_len
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding slice `[offset : offset + L]` to tensor `(B, L, d_model)`."""
        if x.dim() != 3:
            raise ValueError(
                f"x phai la tensor 3 chieu (B, L, d_model), nhan shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Chieu cuoi cua x phai bang d_model={self.d_model}, nhan {x.size(-1)}"
            )
        if offset < 0:
            raise ValueError(f"offset phai >= 0, nhan duoc {offset}")

        seq_len = x.size(1)
        if offset + seq_len > self.max_len:
            raise ValueError(
                f"offset + seq_len ({offset} + {seq_len}) vuot qua max_len={self.max_len}"
            )

        pos_emb = self.pe[:, offset : offset + seq_len, :].to(dtype=x.dtype)
        return self.dropout(x + pos_emb)


PositionalEncoding = SinusoidalPositionalEncoding
