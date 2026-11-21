"""LSTM Decoder for Seq2Seq NMT with single-step decoding and optional Attention hook."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID
from src.models.seq2seq_lstm.encoder import (
    DEFAULT_DEC_HIDDEN_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_NUM_LAYERS,
)


class LSTMDecoder(nn.Module):
    """Multi-layer unidirectional LSTM Decoder with plug-and-play Attention support.

    Supports both pure Seq2Seq (`attention=None`, used for Phase 2 baseline validation and
    Tuần 4 no-attention ablation) and Luong-style post-RNN Attention (`attention != None`),
    consuming the exact `(h, c)` state tuple emitted by `LSTMEncoder`.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = DEFAULT_DEC_HIDDEN_DIM,
        enc_out_dim: int = DEFAULT_DEC_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        attention: nn.Module | None = None,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size phai > 0, nhan duoc {vocab_size}")
        if embed_dim <= 0 or hidden_dim <= 0 or enc_out_dim <= 0:
            raise ValueError("embed_dim, hidden_dim va enc_out_dim phai > 0")
        if num_layers < 1:
            raise ValueError(f"num_layers phai >= 1, nhan duoc {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phai nam trong [0.0, 1.0), nhan duoc {dropout}")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.enc_out_dim = enc_out_dim
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.attention = attention

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout,
        )

        # Luong attentional projection: tanh(W_c [h_t; c_t])
        self.concat_proj: nn.Linear | None = (
            nn.Linear(hidden_dim + enc_out_dim, hidden_dim)
            if attention is not None
            else None
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """Run one decoding step given current target token IDs and previous `(h, c)` state."""
        if input_token.dim() == 1:
            token_ids = input_token.unsqueeze(1)
        elif input_token.dim() == 2 and input_token.size(1) == 1:
            token_ids = input_token
        else:
            raise ValueError(
                f"input_token phai co shape (B,) hoac (B, 1), nhan {tuple(input_token.shape)}"
            )

        batch_size = token_ids.size(0)
        expected_state_shape = (self.num_layers, batch_size, self.hidden_dim)
        h_prev, c_prev = hidden
        if h_prev.shape != expected_state_shape or c_prev.shape != expected_state_shape:
            raise ValueError(
                f"hidden (h, c) phai co shape {expected_state_shape}, "
                f"nhan h={tuple(h_prev.shape)}, c={tuple(c_prev.shape)}"
            )

        embedded = self.dropout(self.embedding(token_ids))  # (B, 1, embed_dim)
        rnn_output, next_hidden = self.rnn(embedded, hidden)  # (B, 1, hidden_dim)
        dec_state = rnn_output.squeeze(1)  # (B, hidden_dim)

        if self.attention is not None:
            if encoder_outputs is None:
                raise ValueError(
                    "encoder_outputs la bat buoc khi Decoder co gan tang Attention."
                )
            context, attn_weights = self.attention(dec_state, encoder_outputs, src_mask)
            assert self.concat_proj is not None
            combined = torch.cat((dec_state, context), dim=-1)
            dec_rep = self.dropout(torch.tanh(self.concat_proj(combined)))
        else:
            attn_weights = None
            dec_rep = self.dropout(dec_state)

        logits = self.fc_out(dec_rep)  # (B, vocab_size)
        return logits, next_hidden, attn_weights

    def parameter_breakdown(self) -> dict[str, int]:
        """Return trainable parameter counts by submodule."""

        def _count(module: nn.Module | None) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        emb_params = _count(self.embedding)
        rnn_params = _count(self.rnn)
        attn_params = _count(self.attention)
        concat_params = _count(self.concat_proj)
        fc_out_params = _count(self.fc_out)
        total = emb_params + rnn_params + attn_params + concat_params + fc_out_params
        return {
            "embedding": emb_params,
            "rnn": rnn_params,
            "attention": attn_params,
            "concat_proj": concat_params,
            "fc_out": fc_out_params,
            "total": total,
        }
