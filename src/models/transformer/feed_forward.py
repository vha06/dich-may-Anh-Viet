"""Position-wise Feed-Forward Network for Transformer Encoder and Decoder layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """Two-layer token-wise MLP: `fc2(dropout(act(fc1(x))))` (Vaswani et al., 2017)."""

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model phai > 0, nhan duoc {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff phai > 0, nhan duoc {d_ff}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout phai nam trong [0.0, 1.0), nhan duoc {dropout}")

        act_name = activation.strip().lower()
        if act_name not in self._ACTIVATIONS:
            raise ValueError(
                f"Khong ho tro activation={activation!r}; chon mot trong {tuple(self._ACTIVATIONS)}"
            )

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.activation_name = act_name

        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.act = self._ACTIVATIONS[act_name]()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise expansion-projection FFN to `(B, L, d_model)` tensor."""
        if x.ndim != 3:
            raise ValueError(
                f"x phai la tensor 3D (B, L, d_model), nhan shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Chieu cuoi cua x phai bang d_model={self.d_model}, nhan {x.size(-1)}"
            )
        return self.fc2(self.dropout(self.act(self.fc1(x))))
