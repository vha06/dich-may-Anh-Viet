"""Luong (Global General) Attention — tính trọng số chú ý cho Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LuongAttention(nn.Module):
    """Global General Multiplicative Attention (Luong et al., 2015).

    score(h_t, h_bar_s) = h_t^T · W_a · h_bar_s
    """

    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(
        self,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            decoder_hidden:  [batch, decoder_dim]
            encoder_outputs: [batch, src_len, encoder_dim]
            mask:            [batch, src_len]  — True ở vị trí PAD.

        Returns:
            context: [batch, encoder_dim]       — vector ngữ cảnh.
            weights: [batch, src_len]           — trọng số attention (dùng vẽ heatmap).
        """
        proj_h = self.W_a(encoder_outputs)
        attention_scores = torch.bmm(proj_h, decoder_hidden.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights
