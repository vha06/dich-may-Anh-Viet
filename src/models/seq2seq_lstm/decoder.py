"""Decoder: LSTM một chiều + Attention, sinh từng token đích."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.seq2seq_lstm.attention import BahdanauAttention


class Decoder(nn.Module):
    """LSTM Decoder kết hợp Bahdanau Attention.

    Mỗi bước:
      1. Nhúng token hiện tại.
      2. Nối embedding + context trước đó → đưa vào LSTM cell.
      3. Tính attention mới từ hidden state vừa ra + encoder_outputs.
      4. Nối hidden + context mới → chiếu ra vocab để dự đoán token tiếp.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        self.rnn = nn.LSTM(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token: Tensor,
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Chạy 1 bước giải mã.

        Args:
            token:           [batch]                       — token ID hiện tại.
            hidden:          [num_layers, batch, hidden]
            cell:            [num_layers, batch, hidden]
            encoder_outputs: [batch, src_len, encoder_dim]
            mask:            [batch, src_len]

        Returns:
            prediction:  [batch, vocab_size]
            hidden:      [num_layers, batch, hidden]
            cell:        [num_layers, batch, hidden]
            attn_weights: [batch, src_len]
        """
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))

        top_hidden = hidden[-1]
        context, attn_weights = self.attention(top_hidden, encoder_outputs, mask)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context], dim=-1)
        )

        return prediction, hidden, cell, attn_weights
