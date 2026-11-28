"""Decoder LSTM kết hợp Luong Attention, Input Feeding, Dropout đầu ra Attention và Residual."""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import LuongAttention


class DecoderLSTM(nn.Module):
    """Giải mã từng token đích dựa trên trạng thái ẩn và context vector từ Luong Attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
        input_feeding: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_feeding = input_feeding
        self.residual = residual

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        first_in_dim = embed_dim + hidden_dim if input_feeding else embed_dim
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    first_in_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )
                for i in range(num_layers)
            ]
        )
        self.attention = LuongAttention(enc_dim, hidden_dim)
        self.fc_combine = nn.Linear(hidden_dim + enc_dim, hidden_dim, bias=False)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        if embed_dim == hidden_dim:
            self.fc_out.weight = self.embedding.weight

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        attn_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chạy 1 bước giải mã; trả về (logits [B, V], hidden, cell, attn_weights, attn_hidden)."""
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        if self.input_feeding:
            if attn_hidden is None:
                attn_hidden = embedded.new_zeros((token.size(0), self.hidden_dim))
            x = torch.cat([embedded, attn_hidden.unsqueeze(1)], dim=-1)
        else:
            x = embedded

        next_h: list[torch.Tensor] = []
        next_c: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            layer_in = x if i == 0 else self.dropout(x)
            out, (h_i, c_i) = layer(layer_in, (hidden[i : i + 1], cell[i : i + 1]))
            x = (x + out) if (self.residual and i > 0) else out
            next_h.append(h_i)
            next_c.append(c_i)

        hidden = torch.cat(next_h, dim=0)
        cell = torch.cat(next_c, dim=0)
        dec_state = x.squeeze(1)

        context, attn_weights = self.attention(dec_state, enc_outputs, src_mask)
        attn_hidden = torch.tanh(self.fc_combine(torch.cat([dec_state, context], dim=-1)))
        attn_hidden = self.dropout(attn_hidden)
        logits = self.fc_out(attn_hidden)
        return logits, hidden, cell, attn_weights, attn_hidden
