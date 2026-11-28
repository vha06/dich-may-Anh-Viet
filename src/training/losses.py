"""Hàm mất mát CrossEntropy cho dịch máy (bỏ qua token <pad>, hỗ trợ Label Smoothing)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TranslationLoss(nn.Module):
    """CrossEntropyLoss tự động gộp chiều [B, L, V] -> [B*L, V] và bỏ qua pad_id."""

    def __init__(self, pad_id: int = 0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính loss từ logits [B, L, V] (hoặc [N, V]) và targets [B, L] (hoặc [N])."""
        return self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


if __name__ == "__main__":
    loss_fn = TranslationLoss(pad_id=0, label_smoothing=0.1)
    sample_logits = torch.randn(2, 4, 8000, requires_grad=True)
    sample_targets = torch.tensor([[5, 6, 3, 0], [7, 3, 0, 0]])
    loss = loss_fn(sample_logits, sample_targets)
    loss.backward()
    # Đổi logits tại vị trí target == pad_id (0) không được làm đổi giá trị loss
    logits_pad_changed = sample_logits.detach().clone()
    logits_pad_changed[0, 3] += 100.0
    assert torch.allclose(loss.detach(), loss_fn(logits_pad_changed, sample_targets))
    print("TranslationLoss OK:", round(loss.item(), 4))
