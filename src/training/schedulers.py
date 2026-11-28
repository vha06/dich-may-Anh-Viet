"""Learning rate schedulers cho Seq2Seq LSTM (ReduceLROnPlateau, MultiStepLR) và Transformer (NoamScheduler)."""
from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR, ReduceLROnPlateau


class NoamScheduler(LRScheduler):
    """Lịch học warmup + inverse square root decay theo Vaswani et al. (2017)."""

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        scale: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = max(1, self.last_epoch + 1)
        lr = (
            self.scale
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )
        return [lr for _ in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer, cfg: dict[str, Any] | None
) -> ReduceLROnPlateau | LRScheduler | None:
    """Khởi tạo scheduler từ mục `training.scheduler` trong file cấu hình YAML."""
    if not cfg or cfg.get("type", "none") in ("none", None):
        return None

    sched_type = cfg["type"]
    if sched_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "min"),
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 2)),
            min_lr=float(cfg.get("min_lr", 1e-5)),
        )
    if sched_type == "multistep":
        milestones = [int(m) for m in cfg["milestones"]]
        return MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(cfg.get("gamma", 0.5)),
        )
    if sched_type == "noam":
        return NoamScheduler(
            optimizer,
            d_model=int(cfg["d_model"]),
            warmup_steps=int(cfg.get("warmup_steps", 4000)),
            scale=float(cfg.get("scale", 1.0)),
        )
    raise ValueError(f"Scheduler không hỗ trợ: '{sched_type}'")


if __name__ == "__main__":
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-3)

    # 1. Kiểm tra ReduceLROnPlateau (LSTM)
    plateau = build_scheduler(opt, {"type": "reduce_on_plateau", "factor": 0.5, "patience": 1})
    assert isinstance(plateau, ReduceLROnPlateau)
    plateau.step(2.0)
    plateau.step(2.0)
    plateau.step(2.0)
    assert opt.param_groups[0]["lr"] == 5e-4

    # 2. Kiểm tra MultiStepLR (LSTM): E01-E12 = 1e-3, E13-E16 = 5e-4, E17-E20 = 2.5e-4
    opt_ms = torch.optim.Adam([param], lr=1e-3)
    ms = build_scheduler(opt_ms, {"type": "multistep", "milestones": [12, 16], "gamma": 0.5})
    assert isinstance(ms, MultiStepLR)
    ms_lrs = []
    for _ in range(20):
        ms_lrs.append(opt_ms.param_groups[0]["lr"])
        opt_ms.step()
        ms.step()
    assert ms_lrs[:12] == [1e-3] * 12
    assert ms_lrs[12:16] == [5e-4] * 4
    assert ms_lrs[16:20] == [2.5e-4] * 4

    # 3. Kiểm tra NoamScheduler (Transformer): tăng tới đỉnh ở warmup_steps rồi giảm
    noam = NoamScheduler(opt, d_model=512, warmup_steps=10)
    lrs = []
    for _ in range(15):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        noam.step()
    assert lrs[0] < lrs[9] and lrs[9] > lrs[14]
    print("Schedulers OK | peak_noam_lr:", round(lrs[9], 6))
