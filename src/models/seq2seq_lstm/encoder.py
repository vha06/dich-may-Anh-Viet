"""Encoder: Bi-LSTM đọc câu nguồn (kèm pack_padded_sequence bỏ qua <pad>)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """Bi-directional LSTM Encoder.

    Đầu vào: chuỗi token ID nguồn (+ src_lengths thực tế để bỏ qua <pad>).
    Đầu ra: encoder_outputs [batch, seq_len, hidden*2], (hidden, cell).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_lengths: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            src:         [batch, src_len]
            src_lengths: [batch] — độ dài thực (không tính <pad>).
        """
        embedded = self.dropout(self.embedding(src))

        if src_lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, (hidden, cell) = self.rnn(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=src.size(1),
            )
        else:
            outputs, (hidden, cell) = self.rnn(embedded)

        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        hidden = torch.tanh(self.fc_hidden(hidden))

        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        cell = torch.tanh(self.fc_cell(cell))

        return outputs, (hidden, cell)
