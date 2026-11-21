"""Loss function cho bài toán dịch máy — bỏ qua vị trí PAD."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class LabelSmoothedCrossEntropy(nn.Module):
    """CrossEntropy có label smoothing, bỏ qua token PAD.

    Khi smoothing = 0.0, hoạt động giống CrossEntropyLoss(ignore_index=pad_id).
    """

    def __init__(self, pad_id: int = 0, smoothing: float = 0.0) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=smoothing,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        return self.criterion(logits, targets)
