"""Seq2Seq: Ghép Encoder + Decoder thành mô hình dịch máy hoàn chỉnh."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch import Tensor

from src.models.seq2seq_lstm.encoder import Encoder
from src.models.seq2seq_lstm.decoder import Decoder


class Seq2Seq(nn.Module):
    """Mô hình Seq2Seq LSTM + Bahdanau Attention.

    Hỗ trợ teacher forcing khi huấn luyện
    và greedy decoding khi suy luận.
    """

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
        tgt: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            teacher_forcing_ratio: xác suất dùng token thật thay vì token dự đoán.

        Returns:
            outputs:         [batch, tgt_len - 1, vocab_size]
            attention_maps:  [batch, tgt_len - 1, src_len]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)
        attentions = []

        mask = self.create_src_mask(src)
        encoder_outputs, (hidden, cell) = self.encoder(src)

        current_token = tgt[:, 0]

        for t in range(1, tgt_len):
            prediction, hidden, cell, attn_w = self.decoder(
                current_token, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t - 1] = prediction
            attentions.append(attn_w)

            if random.random() < teacher_forcing_ratio:
                current_token = tgt[:, t]
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
        max_len: int = 100,
    ) -> tuple[Tensor, Tensor]:
        """Greedy decoding cho suy luận.

        Returns:
            tokens:          [batch, generated_len]
            attention_maps:  [batch, generated_len, src_len]
        """
        self.eval()
        mask = self.create_src_mask(src)
        encoder_outputs, (hidden, cell) = self.encoder(src)

        batch_size = src.size(0)
        current_token = torch.full(
            (batch_size,), bos_id, dtype=torch.long, device=src.device
        )

        all_tokens: list[Tensor] = []
        all_attentions: list[Tensor] = []

        for _ in range(max_len):
            prediction, hidden, cell, attn_w = self.decoder(
                current_token, hidden, cell, encoder_outputs, mask
            )
            current_token = prediction.argmax(dim=-1)
            all_tokens.append(current_token)
            all_attentions.append(attn_w)

            if (current_token == eos_id).all():
                break

        tokens = torch.stack(all_tokens, dim=1)
        attention_maps = torch.stack(all_attentions, dim=1)
        return tokens, attention_maps
