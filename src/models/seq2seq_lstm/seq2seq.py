"""Mô hình Seq2Seq LSTM + Attention hoàn chỉnh."""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from .decoder import DecoderLSTM
from .encoder import EncoderLSTM


class Seq2SeqLSTM(nn.Module):
    """Ghép nối EncoderLSTM và DecoderLSTM có Attention."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_id: int = 0,
        input_feeding: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = EncoderLSTM(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_id=pad_id,
            residual=residual,
        )
        self.decoder = DecoderLSTM(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            enc_dim=self.encoder.out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
            input_feeding=input_feeding,
            residual=residual,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward teacher-forcing; trả về logits [B, L_tgt, V] (và attentions nếu cần)."""
        batch_size, tgt_len = tgt_input.shape
        enc_outputs, (h, c) = self.encoder(src, src_mask)

        logits = enc_outputs.new_empty((batch_size, tgt_len, self.tgt_vocab_size))
        attentions = enc_outputs.new_empty((batch_size, tgt_len, src.size(1))) if return_attention else None

        dec_token = tgt_input[:, 0]
        attn_hidden: torch.Tensor | None = None
        for t in range(tgt_len):
            step_logits, h, c, attn_w, attn_hidden = self.decoder(
                dec_token, h, c, enc_outputs, src_mask, attn_hidden
            )
            logits[:, t] = step_logits
            if attentions is not None:
                attentions[:, t] = attn_w
            if t + 1 < tgt_len:
                use_tf = teacher_forcing_ratio >= 1.0 or (
                    teacher_forcing_ratio > 0.0 and random.random() < teacher_forcing_ratio
                )
                dec_token = tgt_input[:, t + 1] if use_tf else step_logits.argmax(dim=-1)

        return (logits, attentions) if attentions is not None else logits
