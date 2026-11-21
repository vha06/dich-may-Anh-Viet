"""Encoder: Bi-LSTM đọc câu nguồn và tạo biểu diễn ẩn."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Bi-directional LSTM Encoder.

    Đầu vào: chuỗi token ID nguồn  →  Đầu ra: trạng thái ẩn của mỗi bước thời gian
    cùng (hidden, cell) cuối cùng đã biến đổi để khớp kích thước Decoder.
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

    def forward(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            src: [batch, src_len]  —  chuỗi token ID câu nguồn.

        Returns:
            encoder_outputs: [batch, src_len, hidden*2]
            (hidden, cell): mỗi tensor [num_layers, batch, hidden]
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # hidden: [num_layers*2, batch, hidden] → gộp 2 chiều lại
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        hidden = torch.tanh(self.fc_hidden(hidden))

        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        cell = torch.tanh(self.fc_cell(cell))

        return outputs, (hidden, cell)
