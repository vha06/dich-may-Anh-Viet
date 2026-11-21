"""Bahdanau (Additive) Attention — tính trọng số chú ý cho Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BahdanauAttention(nn.Module):
    """Additive Attention (Bahdanau et al., 2015).

    score(s_t, h_j) = v^T · tanh(W_s · s_t + W_h · h_j)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_h = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

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
            weights: [batch, src_len]           — trọng số attention (dùng để vẽ heatmap).
        """
        score_s = self.W_s(decoder_hidden).unsqueeze(1)
        score_h = self.W_h(encoder_outputs)
        energy = torch.tanh(score_s + score_h)
        attention_scores = self.v(energy).squeeze(-1)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights
