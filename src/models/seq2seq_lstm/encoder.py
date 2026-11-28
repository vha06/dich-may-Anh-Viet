"""Encoder LSTM hai chiều cho mô hình Seq2Seq (hỗ trợ Residual Connections giữa các tầng)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderLSTM(nn.Module):
    """Mã hóa chuỗi nguồn bằng LSTM xếp tầng (hỗ trợ hai chiều, packing và residual)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_id: int = 0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.residual = residual
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    embed_dim if i == 0 else self.out_dim,
                    hidden_dim,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
                for i in range(num_layers)
            ]
        )
        if bidirectional:
            self.proj_h = nn.Linear(hidden_dim * 2, hidden_dim)
            self.proj_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def _merge_directions(self, state: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        # [num_layers * 2, B, H] -> [num_layers, B, 2H] -> [num_layers, B, H]
        state = state.view(self.num_layers, 2, state.size(1), self.hidden_dim)
        merged = torch.cat([state[:, 0], state[:, 1]], dim=-1)
        return torch.tanh(proj(merged))

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Trả về encoder_outputs [B, L_src, out_dim] và (h, c) [num_layers, B, hidden_dim]."""
        x = self.dropout(self.embedding(src))
        lengths = (~src_mask).sum(dim=1).clamp(min=1).cpu() if src_mask is not None else None

        h_list: list[torch.Tensor] = []
        c_list: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            layer_in = x if i == 0 else self.dropout(x)
            if lengths is not None:
                packed = pack_padded_sequence(layer_in, lengths, batch_first=True, enforce_sorted=False)
                packed_out, (h_i, c_i) = layer(packed)
                out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
            else:
                out, (h_i, c_i) = layer(layer_in)

            x = (x + out) if (self.residual and i > 0) else out
            h_list.append(h_i)
            c_list.append(c_i)

        h = torch.cat(h_list, dim=0)
        c = torch.cat(c_list, dim=0)
        if self.bidirectional:
            h = self._merge_directions(h, self.proj_h)
            c = self._merge_directions(c, self.proj_c)

        return x, (h, c)
