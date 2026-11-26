"""Decoder: LSTM một chiều + Luong Attention + Input Feeding (Luong et al., 2015)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.seq2seq_lstm.attention import LuongAttention


class Decoder(nn.Module):
    """LSTM Decoder kết hợp Luong Global Attention và Input Feeding.

    Mỗi bước:
      1. Nối embedding token hiện tại với attn_hidden bước trước (Input Feeding).
      2. Đưa qua LSTM để lấy hidden state h_t.
      3. Tính Luong Attention từ h_t và encoder_outputs → context vector c_t.
      4. Tính vector ẩn chú ý h_tilde = tanh(W_c · [h_t ; c_t]).
      5. Chiếu h_tilde ra vocab để dự đoán token tiếp theo.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = LuongAttention(encoder_dim, hidden_dim)
        self.fc_concat = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token: Tensor,
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
        mask: Tensor | None = None,
        prev_attn_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Chạy 1 bước giải mã.

        Returns:
            prediction:   [batch, vocab_size]
            hidden:       [num_layers, batch, hidden]
            cell:         [num_layers, batch, hidden]
            attn_weights: [batch, src_len]
            attn_hidden:  [batch, hidden] — dùng làm Input Feeding bước sau.
        """
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        if prev_attn_hidden is None:
            prev_attn_hidden = torch.zeros(
                token.size(0),
                self.hidden_dim,
                device=token.device,
                dtype=encoder_outputs.dtype,
            )

        rnn_input = torch.cat([embedded, prev_attn_hidden.unsqueeze(1)], dim=-1)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        dec_out = output.squeeze(1)
        context, attn_weights = self.attention(dec_out, encoder_outputs, mask)
        attn_hidden = self.dropout(
            torch.tanh(self.fc_concat(torch.cat([dec_out, context], dim=-1)))
        )
        prediction = self.fc_out(attn_hidden)

        return prediction, hidden, cell, attn_weights, attn_hidden
