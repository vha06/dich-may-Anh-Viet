"""Cơ chế Luong Attention trả về context vector và attention weights."""
from __future__ import annotations

import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    """Multiplicative General Attention (Luong et al., 2015)."""

    def __init__(self, enc_dim: int, dec_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(
        self,
        dec_hidden: torch.Tensor,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.w(dec_hidden).unsqueeze(1)
        scores = torch.bmm(query, enc_outputs.transpose(1, 2)).squeeze(1)
        if src_mask is not None:
            # Dùng min theo dtype thay vì -1e9 để tránh tràn số khi chạy AMP FP16
            scores = scores.masked_fill(src_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights
