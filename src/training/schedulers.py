"""Learning rate scheduler cho quá trình huấn luyện."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "plateau",
    **kwargs,
) -> _LRScheduler | ReduceLROnPlateau:
    """Khởi tạo LR scheduler theo tên.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_type: "plateau" hoặc "step".
    """
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 3),
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    if scheduler_type == "step":
        return __import__("torch").optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 5),
            gamma=kwargs.get("gamma", 0.5),
        )
    raise ValueError(f"Scheduler '{scheduler_type}' chưa được hỗ trợ.")
