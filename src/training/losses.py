"""Loss functions and numerically safe perplexity helpers for NMT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID

DEFAULT_MAX_PPL_LOSS = 20.0


def compute_perplexity(
    loss_value: float, max_exp_input: float = DEFAULT_MAX_PPL_LOSS
) -> float:
    """Compute Perplexity exp(loss) clamped at max_exp_input to avoid OverflowError."""
    value = float(loss_value)
    if math.isnan(value):
        raise ValueError("Khong the tinh Perplexity tu gia tri loss NaN.")
    return math.exp(min(value, max_exp_input))


class TranslationLoss(nn.Module):
    """Cross-entropy loss over sequence logits ignoring PAD tokens with optional smoothing."""

    def __init__(
        self,
        pad_id: int = PAD_ID,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing phai nam trong [0.0, 1.0), nhan duoc {label_smoothing}"
            )
        self.pad_id = pad_id
        self.label_smoothing = float(label_smoothing)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compute mean token cross-entropy loss and return non-pad token count."""
        if logits.dim() != 3:
            raise ValueError(
                f"logits phai co shape (B, T_max, V), nhan {tuple(logits.shape)}"
            )
        if targets.dim() != 2:
            raise ValueError(
                f"targets phai co shape (B, T_max), nhan {tuple(targets.shape)}"
            )
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"Kich thuoc logits {tuple(logits.shape[:2])} != targets {tuple(targets.shape)}"
            )

        num_tokens = int((targets != self.pad_id).sum().item())
        if num_tokens == 0:
            raise ValueError("Batch khong co token thuc nao khac <pad> de tinh Loss.")

        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return loss, num_tokens


def build_loss_fn(
    pad_id: int = PAD_ID,
    label_smoothing: float = 0.0,
) -> TranslationLoss:
    """Instantiate a single TranslationLoss instance."""
    return TranslationLoss(pad_id=pad_id, label_smoothing=label_smoothing)


def build_loss_pair(
    pad_id: int = PAD_ID,
    train_label_smoothing: float = 0.0,
) -> tuple[TranslationLoss, TranslationLoss]:
    """Return (train_loss_fn, eval_loss_fn) enforcing label_smoothing=0.0 on evaluation."""
    train_loss_fn = TranslationLoss(
        pad_id=pad_id, label_smoothing=train_label_smoothing
    )
    eval_loss_fn = TranslationLoss(pad_id=pad_id, label_smoothing=0.0)
    return train_loss_fn, eval_loss_fn
