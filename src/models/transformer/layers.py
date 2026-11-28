"""Residual Connection and Layer Normalization (`SublayerConnection`) for Transformer layers."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class SublayerConnection(nn.Module):
    """Apply Residual Connection, Dropout, and LayerNorm around a sublayer.

    Supports both Post-LN (`norm_first=False`, default per Vaswani et al., 2017):
        `LayerNorm(x + Dropout(Sublayer(x)))`
    and Pre-LN (`norm_first=True`, for ablation):
        `x + Dropout(Sublayer(LayerNorm(x)))`
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model phai > 0, nhan duoc {d_model}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout phai nam trong [0.0, 1.0), nhan duoc {dropout}")
        if eps <= 0.0:
            raise ValueError(f"eps phai > 0, nhan duoc {eps}")
        if not isinstance(norm_first, bool):
            raise ValueError(
                f"norm_first phai la bool (True=Pre-LN, False=Post-LN), nhan duoc {norm_first!r}"
            )

        self.d_model = d_model
        self.dropout_rate = dropout
        self.norm_first = norm_first
        self.eps = eps

        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply Pre-LN or Post-LN residual connection around `sublayer` on `(B, L, d_model)` tensor."""
        if x.ndim != 3:
            raise ValueError(
                f"x phai la tensor 3D (B, L, d_model), nhan shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Chieu cuoi cua x phai bang d_model={self.d_model}, nhan {x.size(-1)}"
            )
        if not callable(sublayer):
            raise TypeError("sublayer phai la mot callable nhan tensor (B, L, d_model)")

        sublayer_input = self.norm(x) if self.norm_first else x
        sublayer_out = sublayer(sublayer_input)

        if not isinstance(sublayer_out, torch.Tensor):
            raise TypeError(
                f"sublayer phai tra ve dung 1 torch.Tensor, nhan {type(sublayer_out).__name__}"
            )
        if sublayer_out.shape != x.shape:
            raise ValueError(
                f"Output cua sublayer phai co shape {tuple(x.shape)}, nhan {tuple(sublayer_out.shape)}"
            )

        residual = x + self.dropout(sublayer_out)
        if self.norm_first:
            return residual
        return self.norm(residual)
