"""Complete Seq2Seq LSTM with Luong Global Attention model and factory helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.tokenizer import (
    BOS_ID,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_VOCAB_SIZE,
    EOS_ID,
    PAD_ID,
)
from src.models.seq2seq_lstm.attention import (
    DEFAULT_ATTENTION_METHOD,
    DEFAULT_ATTN_DIM,
    LuongAttention,
)
from src.models.seq2seq_lstm.decoder import LSTMDecoder
from src.models.seq2seq_lstm.encoder import (
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_ENC_HIDDEN_DIM,
    DEFAULT_NUM_LAYERS,
    LSTMEncoder,
)


class Seq2SeqLSTM(nn.Module):
    """End-to-end Seq2Seq NMT wrapping LSTMEncoder, LuongAttention, and LSTMDecoder."""

    def __init__(
        self,
        encoder: LSTMEncoder,
        decoder: LSTMDecoder,
        pad_id: int = PAD_ID,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
    ) -> None:
        super().__init__()
        if encoder.num_layers != decoder.num_layers:
            raise ValueError(
                f"num_layers khong khop: encoder={encoder.num_layers}, decoder={decoder.num_layers}"
            )
        if encoder.dec_hidden_dim != decoder.hidden_dim:
            raise ValueError(
                f"hidden_dim khong khop: encoder.dec_hidden_dim={encoder.dec_hidden_dim}, "
                f"decoder.hidden_dim={decoder.hidden_dim}"
            )
        if decoder.attention is not None and encoder.output_dim != decoder.enc_out_dim:
            raise ValueError(
                f"enc_out_dim khong khop voi Attention: encoder.output_dim={encoder.output_dim}, "
                f"decoder.enc_out_dim={decoder.enc_out_dim}"
            )
        if encoder.pad_id != pad_id or decoder.pad_id != pad_id:
            raise ValueError(
                f"pad_id khong dong nhat: model={pad_id}, encoder={encoder.pad_id}, decoder={decoder.pad_id}"
            )

        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run full-batch training/eval forward pass with configurable Teacher Forcing."""
        if tgt_input_ids.dim() != 2:
            raise ValueError(
                f"tgt_input_ids phai la tensor 2D (B, T_max), nhan {tuple(tgt_input_ids.shape)}"
            )
        if tgt_input_ids.size(0) != src_ids.size(0):
            raise ValueError(
                f"Batch size khong khop: src={src_ids.size(0)}, tgt={tgt_input_ids.size(0)}"
            )
        if tgt_input_ids.size(1) < 1:
            raise ValueError("tgt_input_ids phai co it nhat 1 buoc thoi gian.")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError(
                f"teacher_forcing_ratio phai nam trong [0.0, 1.0], nhan {teacher_forcing_ratio}"
            )

        src_mask = src_ids != self.pad_id
        encoder_outputs, hidden = self.encoder(src_ids, src_lengths)

        t_max = tgt_input_ids.size(1)
        input_token = tgt_input_ids[:, 0]
        logits_steps: list[torch.Tensor] = []
        attn_steps: list[torch.Tensor] = []

        for t in range(t_max):
            logits_t, hidden, attn_t = self.decoder(
                input_token,
                hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            logits_steps.append(logits_t)
            if attn_t is not None:
                attn_steps.append(attn_t)

            if t + 1 < t_max:
                use_tf = (
                    (not self.training)
                    or teacher_forcing_ratio >= 1.0
                    or (
                        teacher_forcing_ratio > 0.0
                        and torch.rand(1).item() < teacher_forcing_ratio
                    )
                )
                input_token = (
                    tgt_input_ids[:, t + 1]
                    if use_tf
                    else logits_t.argmax(dim=-1).detach()
                )

        logits = torch.stack(logits_steps, dim=1)
        attn_weights = torch.stack(attn_steps, dim=1) if attn_steps else None
        return logits, attn_weights

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Autoregressively decode a batch using argmax with per-sentence EOS early stop."""
        if max_len < 1:
            raise ValueError(f"max_len phai >= 1, nhan {max_len}")

        batch_size = src_ids.size(0)
        device = src_ids.device
        src_mask = src_ids != self.pad_id
        encoder_outputs, hidden = self.encoder(src_ids, src_lengths)

        input_token = torch.full(
            (batch_size,), self.bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        pred_steps: list[torch.Tensor] = []
        attn_steps: list[torch.Tensor] = []

        for _ in range(max_len):
            logits_t, hidden, attn_t = self.decoder(
                input_token,
                hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            next_token = logits_t.argmax(dim=-1).masked_fill(finished, self.pad_id)
            pred_steps.append(next_token)
            if attn_t is not None:
                attn_steps.append(attn_t)

            finished = finished | (next_token == self.eos_id)
            if finished.all():
                break
            input_token = next_token

        pred_ids = torch.stack(pred_steps, dim=1)
        attn_weights = torch.stack(attn_steps, dim=1) if attn_steps else None
        return pred_ids, attn_weights

    def parameter_breakdown(self) -> dict[str, int]:
        """Return trainable parameter counts grouped by encoder, decoder core, and attention."""
        enc_params = self.encoder.parameter_breakdown()["total"]
        dec_stats = self.decoder.parameter_breakdown()
        attn_params = dec_stats["attention"]
        dec_core_params = dec_stats["total"] - attn_params
        return {
            "encoder": enc_params,
            "decoder_core": dec_core_params,
            "attention": attn_params,
            "total": enc_params + dec_core_params + attn_params,
        }


def build_seq2seq_lstm(
    src_vocab_size: int = DEFAULT_VOCAB_SIZE,
    tgt_vocab_size: int = DEFAULT_VOCAB_SIZE,
    embed_dim: int = DEFAULT_EMBED_DIM,
    enc_hidden_dim: int = DEFAULT_ENC_HIDDEN_DIM,
    dec_hidden_dim: int | None = None,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    bidirectional: bool = True,
    attention_method: str | None = DEFAULT_ATTENTION_METHOD,
    attn_dim: int = DEFAULT_ATTN_DIM,
    pad_id: int = PAD_ID,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> Seq2SeqLSTM:
    """Factory assembling LSTMEncoder, optional LuongAttention, and LSTMDecoder."""
    enc_out_dim = enc_hidden_dim * (2 if bidirectional else 1)
    resolved_dec_hidden = enc_out_dim if dec_hidden_dim is None else dec_hidden_dim

    encoder = LSTMEncoder(
        vocab_size=src_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=enc_hidden_dim,
        dec_hidden_dim=resolved_dec_hidden,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        pad_id=pad_id,
    )

    attn_module: LuongAttention | None
    if attention_method is None or attention_method.strip().lower() == "none":
        attn_module = None
    else:
        attn_module = LuongAttention(
            dec_hidden_dim=resolved_dec_hidden,
            enc_out_dim=enc_out_dim,
            method=attention_method,
            attn_dim=attn_dim,
        )

    decoder = LSTMDecoder(
        vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=resolved_dec_hidden,
        enc_out_dim=enc_out_dim,
        num_layers=num_layers,
        dropout=dropout,
        attention=attn_module,
        pad_id=pad_id,
    )

    return Seq2SeqLSTM(
        encoder=encoder,
        decoder=decoder,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )
