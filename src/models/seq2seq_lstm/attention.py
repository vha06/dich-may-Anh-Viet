"""Luong Global Attention mechanism supporting dot, general, and concat score functions."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.seq2seq_lstm.encoder import DEFAULT_DEC_HIDDEN_DIM

DEFAULT_ATTENTION_METHOD = "general"
DEFAULT_ATTN_DIM = 256
VALID_ATTENTION_METHODS = ("dot", "general", "concat")


class LuongAttention(nn.Module):
    """Global multiplicative/additive Attention from Luong, Pham & Manning (EMNLP 2015).

    Supports all three global alignment score variants (Eq. 8 of the paper):
      - `dot`:     h_t^T * h_bar_s              (requires dec_hidden_dim == enc_out_dim)
      - `general`: h_t^T * W_a * h_bar_s        (default; learns source-target alignment)
      - `concat`:  v_a^T * tanh(W_q h_t + W_k h_bar_s)
    """

    def __init__(
        self,
        dec_hidden_dim: int = DEFAULT_DEC_HIDDEN_DIM,
        enc_out_dim: int = DEFAULT_DEC_HIDDEN_DIM,
        method: str = DEFAULT_ATTENTION_METHOD,
        attn_dim: int = DEFAULT_ATTN_DIM,
    ) -> None:
        super().__init__()
        norm_method = method.strip().lower()
        if norm_method not in VALID_ATTENTION_METHODS:
            raise ValueError(
                f"Khong ho tro method={method!r}; chi ho tro {VALID_ATTENTION_METHODS}"
            )
        if dec_hidden_dim <= 0 or enc_out_dim <= 0:
            raise ValueError("dec_hidden_dim va enc_out_dim phai > 0")

        self.dec_hidden_dim = dec_hidden_dim
        self.enc_out_dim = enc_out_dim
        self.method = norm_method
        self.attn_dim = attn_dim

        if norm_method == "dot":
            if dec_hidden_dim != enc_out_dim:
                raise ValueError(
                    f"Phuong phap 'dot' yeu cau dec_hidden_dim == enc_out_dim, "
                    f"nhan {dec_hidden_dim} != {enc_out_dim}"
                )
        elif norm_method == "general":
            self.w_a = nn.Linear(enc_out_dim, dec_hidden_dim, bias=False)
        elif norm_method == "concat":
            if attn_dim <= 0:
                raise ValueError(f"attn_dim phai > 0 voi method='concat', nhan {attn_dim}")
            self.w_q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
            self.w_k = nn.Linear(enc_out_dim, attn_dim, bias=False)
            self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def _compute_scores(
        self,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unnormalized alignment scores of shape (B, S_max)."""
        if self.method == "dot":
            return torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)
        if self.method == "general":
            projected_keys = self.w_a(encoder_outputs)  # (B, S_max, dec_hidden_dim)
            return torch.bmm(projected_keys, query.unsqueeze(2)).squeeze(2)

        # method == "concat": v_a^T * tanh(W_q * h_t + W_k * h_bar_s)
        q_proj = self.w_q(query).unsqueeze(1)  # (B, 1, attn_dim)
        k_proj = self.w_k(encoder_outputs)  # (B, S_max, attn_dim)
        energy = torch.tanh(q_proj + k_proj)  # (B, S_max, attn_dim)
        return self.v_a(energy).squeeze(-1)

    def forward(
        self,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector (B, enc_out_dim) and attention weights (B, S_max)."""
        if query.dim() != 2 or query.size(1) != self.dec_hidden_dim:
            raise ValueError(
                f"query phai co shape (B, {self.dec_hidden_dim}), nhan {tuple(query.shape)}"
            )
        if (
            encoder_outputs.dim() != 3
            or encoder_outputs.size(0) != query.size(0)
            or encoder_outputs.size(2) != self.enc_out_dim
            or encoder_outputs.size(1) < 1
        ):
            raise ValueError(
                f"encoder_outputs phai co shape ({query.size(0)}, S_max>=1, {self.enc_out_dim}), "
                f"nhan {tuple(encoder_outputs.shape)}"
            )

        scores = self._compute_scores(query, encoder_outputs)  # (B, S_max)

        if src_mask is not None:
            if src_mask.shape != scores.shape:
                raise ValueError(
                    f"src_mask shape {tuple(src_mask.shape)} != scores shape {tuple(scores.shape)}"
                )
            mask = src_mask.to(dtype=torch.bool)
            if not torch.all(mask.any(dim=-1)):
                raise ValueError(
                    "Moi cau trong batch phai co it nhat 1 vi tri src_mask=True de tranh NaN khi softmax."
                )
            scores = scores.masked_fill(~mask, float("-inf"))

        # Cast to float32 before softmax for fp16/bf16 AMP stability
        attn_weights = torch.softmax(scores.float(), dim=-1).to(encoder_outputs.dtype)  # (B, S_max)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
