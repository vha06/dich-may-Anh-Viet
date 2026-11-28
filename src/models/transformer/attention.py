"""Scaled Dot-Product Attention and Multi-Head Attention implemented from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_4d_mask(
    mask: torch.Tensor, target_shape: tuple[int, int, int, int]
) -> None:
    """Ensure attention mask is a 4D bool tensor broadcastable to `(B, H, L_q, L_k)`."""
    if mask.dtype != torch.bool:
        raise ValueError(
            f"mask phai co kieu torch.bool (True=giu, False=che), nhan {mask.dtype}"
        )
    if mask.ndim != 4:
        raise ValueError(
            f"mask phai la tensor 4 chieu (B/1, H/1, L_q/1, L_k/1), nhan shape {tuple(mask.shape)}"
        )
    for axis, (m_dim, t_dim) in enumerate(zip(mask.shape, target_shape)):
        if m_dim not in (1, t_dim):
            raise ValueError(
                f"Khong the broadcast mask shape {tuple(mask.shape)} toi {target_shape} o truc {axis}"
            )


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Scaled Dot-Product Attention using `torch.matmul` and `F.softmax`.

    Mask convention matches Seq2Seq Attention: `True` keeps a position, `False` masks it.
    Returns `(context, attn_weights)` where `context` preserves `v.dtype` and
    `attn_weights` stays in `torch.float32` (pre-dropout) for exact row sums.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"q, k, v phai la tensor 4D (B, H, L, d), nhan "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"q, k, v phai khop (batch, heads), nhan "
            f"q={q.shape[:2]}, k={k.shape[:2]}, v={v.shape[:2]}"
        )
    if q.size(-1) != k.size(-1):
        raise ValueError(
            f"d_k cua q va k phai bang nhau, nhan q={q.size(-1)}, k={k.size(-1)}"
        )
    if k.size(2) != v.size(2):
        raise ValueError(
            f"Do dai L_k cua k va v phai bang nhau, nhan k={k.size(2)}, v={v.size(2)}"
        )

    bsz, num_heads, l_q, d_k = q.shape
    l_k = k.size(2)

    # Evaluate scores and softmax in float32 for numerical stability under AMP (bfloat16/fp16)
    scores = torch.matmul(q, k.transpose(-2, -1)).float() / math.sqrt(d_k)

    if mask is not None:
        _validate_4d_mask(mask, (bsz, num_heads, l_q, l_k))
        all_masked = ~mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, float("-inf"))
        # Replace all-False rows with 0 before softmax so neither forward nor backward emits NaN
        scores = scores.masked_fill(all_masked, 0.0)
        attn_weights = F.softmax(scores, dim=-1).masked_fill(all_masked, 0.0)
    else:
        attn_weights = F.softmax(scores, dim=-1)

    attn_probs = dropout(attn_weights) if dropout is not None else attn_weights
    context = torch.matmul(attn_probs.to(dtype=v.dtype), v)
    return context, attn_weights


class MultiHeadAttention(nn.Module):
    """Unified Multi-Head Attention shared across Encoder Self, Masked Decoder Self, and Cross Attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        store_attn: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model phai > 0, nhan duoc {d_model}")
        if num_heads <= 0:
            raise ValueError(f"num_heads phai > 0, nhan duoc {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) phai chia het cho num_heads ({num_heads})"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout phai nam trong [0.0, 1.0), nhan duoc {dropout}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout
        self.store_attn = store_attn
        self.last_attn_weights: torch.Tensor | None = None

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for proj in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        store_attn: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project Q/K/V, run multi-head scaled dot-product attention, and return `(output, attn_weights)`."""
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.ndim != 3:
                raise ValueError(
                    f"{name} phai la tensor 3D (B, L, d_model), nhan shape {tuple(tensor.shape)}"
                )
            if tensor.size(-1) != self.d_model:
                raise ValueError(
                    f"Chieu cuoi cua {name} phai bang d_model={self.d_model}, nhan {tensor.size(-1)}"
                )

        bsz = query.size(0)
        if key.size(0) != bsz or value.size(0) != bsz:
            raise ValueError(
                f"Batch size cua query ({bsz}), key ({key.size(0)}), value ({value.size(0)}) khong khop"
            )
        if key.size(1) != value.size(1):
            raise ValueError(
                f"Do dai L_k cua key ({key.size(1)}) va value ({value.size(1)}) phai bang nhau"
            )

        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        context, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )
        output = self.w_o(self._combine_heads(context))

        should_store = self.store_attn if store_attn is None else store_attn
        if should_store:
            self.last_attn_weights = attn_weights.detach()
        else:
            self.last_attn_weights = None

        return output, attn_weights
