"""LSTM Encoder for Seq2Seq NMT with packed padding and bidirectional state projection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID

DEFAULT_EMBED_DIM = 256
DEFAULT_ENC_HIDDEN_DIM = 256
DEFAULT_DEC_HIDDEN_DIM = 512
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.3


class LSTMEncoder(nn.Module):
    """Multi-layer (bidirectional) LSTM Encoder with bridge projection for the Decoder.

    Follows Bahdanau et al. (2015) for bidirectional encoding + tanh state projection,
    while supporting configurable depth and unidirectionality for Luong (2015) and ablation studies.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = DEFAULT_ENC_HIDDEN_DIM,
        dec_hidden_dim: int | None = None,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        bidirectional: bool = True,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size phai > 0, nhan duoc {vocab_size}")
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim va hidden_dim phai > 0")
        if num_layers < 1:
            raise ValueError(f"num_layers phai >= 1, nhan duoc {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phai nam trong [0.0, 1.0), nhan duoc {dropout}")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_id = pad_id
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions
        self.dec_hidden_dim = self.output_dim if dec_hidden_dim is None else dec_hidden_dim
        if self.dec_hidden_dim <= 0:
            raise ValueError(f"dec_hidden_dim phai > 0, nhan duoc {self.dec_hidden_dim}")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )

        # Bridge forward+backward (or mismatched hidden dims) into Decoder's initial state size
        if bidirectional or self.output_dim != self.dec_hidden_dim:
            self.fc_hidden: nn.Linear | None = nn.Linear(
                self.output_dim, self.dec_hidden_dim
            )
            self.fc_cell: nn.Linear | None = nn.Linear(
                self.output_dim, self.dec_hidden_dim
            )
        else:
            self.fc_hidden = None
            self.fc_cell = None

    def _bridge_state(
        self, state: torch.Tensor, proj: nn.Linear | None
    ) -> torch.Tensor:
        """Project encoder final (h_n or c_n) to (num_layers, B, dec_hidden_dim)."""
        batch_size = state.size(1)
        if self.bidirectional:
            reshaped = state.view(self.num_layers, 2, batch_size, self.hidden_dim)
            merged = torch.cat((reshaped[:, 0], reshaped[:, 1]), dim=-1)
            assert proj is not None
            return torch.tanh(proj(merged))
        if proj is not None:
            return torch.tanh(proj(state))
        return state

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode padded source token IDs while ignoring padding steps via pack_padded_sequence."""
        if src_ids.dim() != 2:
            raise ValueError(f"src_ids phai la tensor 2D (B, S_max), nhan {tuple(src_ids.shape)}")
        if src_lengths.dim() != 1:
            raise ValueError(
                f"src_lengths phai la tensor 1D (B,), nhan {tuple(src_lengths.shape)}"
            )
        if src_ids.size(0) != src_lengths.size(0):
            raise ValueError(
                f"Batch size khong khop: src_ids={src_ids.size(0)}, src_lengths={src_lengths.size(0)}"
            )
        if not torch.all((src_lengths >= 1) & (src_lengths <= src_ids.size(1))):
            raise ValueError(
                f"src_lengths phai nam trong [1, {src_ids.size(1)}], nhan min={src_lengths.min().item()}, max={src_lengths.max().item()}"
            )

        embedded = self.dropout(self.embedding(src_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, (h_n, c_n) = self.rnn(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src_ids.size(1),
        )

        dec_h = self._bridge_state(h_n, self.fc_hidden)
        dec_c = self._bridge_state(c_n, self.fc_cell)
        return encoder_outputs, (dec_h, dec_c)

    def parameter_breakdown(self) -> dict[str, int]:
        """Return trainable parameter counts by submodule."""
        def _count(module: nn.Module | None) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        emb_params = _count(self.embedding)
        rnn_params = _count(self.rnn)
        bridge_params = _count(self.fc_hidden) + _count(self.fc_cell)
        return {
            "embedding": emb_params,
            "rnn": rnn_params,
            "bridge": bridge_params,
            "total": emb_params + rnn_params + bridge_params,
        }
