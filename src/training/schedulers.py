"""Unified learning rate schedulers (Plateau and Noam) for Seq2Seq and Transformer."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class NoamScheduler(LRScheduler):
    """Transformer learning rate schedule from Vaswani et al. (NeurIPS 2017).

    Sets the exact target learning rate directly (ignoring optimizer's initial lr to
    prevent double-multiplication) and avoids zero-division at step 0 via 1-based step:
        lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    step_mode: Literal["batch", "epoch"] = "batch"
    requires_metric: bool = False

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model phai > 0, nhan duoc {d_model}")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps phai >= 1, nhan duoc {warmup_steps}")
        if factor <= 0.0:
            raise ValueError(f"factor phai > 0.0, nhan duoc {factor}")

        self.d_model = int(d_model)
        self.warmup_steps = int(warmup_steps)
        self.factor = float(factor)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _compute_lr(self, step: int) -> float:
        effective_step = max(step, 1)
        scale = min(
            effective_step ** (-0.5),
            effective_step * (self.warmup_steps ** (-1.5)),
        )
        return self.factor * (self.d_model ** (-0.5)) * scale

    def get_lr(self) -> list[float]:
        # LRScheduler.__init__ leaves last_epoch=0 before batch 1 -> step = 1
        step = self.last_epoch + 1
        target_lr = self._compute_lr(step)
        return [target_lr for _ in self.base_lrs]

    def step(self, metric: float | None = None) -> None:
        """Advance scheduler by one training batch; ignores optional metric."""
        _ = metric
        super().step()


class PlateauScheduler:
    """Epoch-level ReduceLROnPlateau adapter matching the unified scheduler contract."""

    step_mode: Literal["batch", "epoch"] = "epoch"
    requires_metric: bool = True

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 2,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
    ) -> None:
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor phai nam trong (0.0, 1.0), nhan duoc {factor}")
        if patience < 0:
            raise ValueError(f"patience phai >= 0, nhan duoc {patience}")
        if min_lr < 0.0:
            raise ValueError(f"min_lr phai >= 0.0, nhan duoc {min_lr}")

        self.optimizer = optimizer
        self._scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=threshold,
        )

    def step(self, metric: float | None = None) -> None:
        """Advance scheduler using the monitored epoch metric (e.g. val_loss)."""
        if metric is None:
            raise ValueError("PlateauScheduler yeu cau truyen gia tri metric (val_loss).")
        self._scheduler.step(float(metric))

    def get_last_lr(self) -> list[float]:
        """Return current learning rates from all optimizer parameter groups."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scheduler.load_state_dict(state_dict)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "plateau",
    **kwargs: Any,
) -> NoamScheduler | PlateauScheduler:
    """Factory returning a unified scheduler ('plateau' or 'noam')."""
    kind = scheduler_type.strip().lower()
    if kind == "plateau":
        return PlateauScheduler(optimizer, **kwargs)
    if kind == "noam":
        return NoamScheduler(optimizer, **kwargs)
    raise ValueError(
        f"Khong ho tro scheduler_type={scheduler_type!r}; chi ho tro 'plateau' hoac 'noam'."
    )
