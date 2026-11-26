"""Seq2Seq: Ghép Encoder + Decoder (Luong Attention + Input Feeding)."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch import Tensor

from src.models.seq2seq_lstm.decoder import Decoder
from src.models.seq2seq_lstm.encoder import Encoder


class Seq2Seq(nn.Module):
    """Mô hình Seq2Seq LSTM + Luong Global Attention + Input Feeding."""

    def __init__(self, encoder: Encoder, decoder: Decoder, pad_id: int = 0) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def create_src_mask(self, src: Tensor) -> Tensor:
        """Tạo mask cho vị trí PAD: True = PAD → bị che."""
        return src == self.pad_id

    def forward(
        self,
        src: Tensor,
        tgt_input: Tensor,
        src_lengths: Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            outputs:        [batch, tgt_len, vocab_size]
            attention_maps: [batch, tgt_len, src_len]
        """
        batch_size = src.size(0)
        tgt_len = tgt_input.size(1)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        attentions = []

        mask = self.create_src_mask(src)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths=src_lengths)

        current_token = tgt_input[:, 0]
        attn_hidden = None

        for t in range(tgt_len):
            prediction, hidden, cell, attn_w, attn_hidden = self.decoder(
                current_token,
                hidden,
                cell,
                encoder_outputs,
                mask,
                prev_attn_hidden=attn_hidden,
            )
            outputs[:, t] = prediction
            attentions.append(attn_w)

            if t + 1 < tgt_len:
                if random.random() < teacher_forcing_ratio:
                    current_token = tgt_input[:, t + 1]
                else:
                    current_token = prediction.argmax(dim=-1)

        attention_maps = torch.stack(attentions, dim=1)
        return outputs, attention_maps

    @torch.no_grad()
    def translate(
        self,
        src: Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
        src_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Greedy decoding cho suy luận."""
        self.eval()
        mask = self.create_src_mask(src)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths=src_lengths)

        batch_size = src.size(0)
        current_token = torch.full(
            (batch_size,), bos_id, dtype=torch.long, device=src.device
        )
        attn_hidden = None

        all_tokens: list[Tensor] = []
        all_attentions: list[Tensor] = []

        for _ in range(max_len):
            prediction, hidden, cell, attn_w, attn_hidden = self.decoder(
                current_token,
                hidden,
                cell,
                encoder_outputs,
                mask,
                prev_attn_hidden=attn_hidden,
            )
            current_token = prediction.argmax(dim=-1)
            all_tokens.append(current_token)
            all_attentions.append(attn_w)

            if (current_token == eos_id).all():
                break

        tokens = torch.stack(all_tokens, dim=1)
        attention_maps = torch.stack(all_attentions, dim=1)
        return tokens, attention_maps
