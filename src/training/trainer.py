"""Vòng lặp huấn luyện, đánh giá, lưu checkpoint và ghi log cho Seq2Seq LSTM & Transformer."""
from __future__ import annotations

import inspect
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.losses import TranslationLoss
from src.training.schedulers import NoamScheduler, build_scheduler


class Trainer:
    """Quản lý vòng lặp train/val, AMP FP16, grad clipping, early stopping và checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: dict[str, Any],
        pad_id: int = 0,
        device: torch.device | str | None = None,
        eval_bleu_fn: Callable[[nn.Module], float | dict[str, Any]] | None = None,
    ) -> None:
        train_cfg = config.get("training", {})
        log_cfg = config.get("logging", {})

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.pad_id = pad_id
        self.eval_bleu_fn = eval_bleu_fn

        self.epochs = int(train_cfg.get("epochs", 20))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        self.teacher_forcing_ratio = float(train_cfg.get("teacher_forcing_ratio", 0.5))
        self.min_tf_ratio = float(
            train_cfg.get("min_teacher_forcing_ratio", self.teacher_forcing_ratio)
        )
        self.patience = int(train_cfg.get("early_stopping_patience", 5))
        self.use_amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"

        self.criterion = TranslationLoss(
            pad_id=pad_id,
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        ).to(self.device)
        self.optimizer: Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self.scheduler: ReduceLROnPlateau | NoamScheduler | LRScheduler | None = build_scheduler(
            self.optimizer, train_cfg.get("scheduler")
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ckpt_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_path = self.ckpt_dir / log_cfg.get("checkpoint_name", "best_model.pt")
        self.log_dir = Path(log_cfg.get("log_dir", "runs/experiment"))
        self.use_wandb = bool(log_cfg.get("use_wandb", False))
        self.wandb_entity = log_cfg.get("wandb_entity")
        self.wandb_project = str(log_cfg.get("wandb_project", "dich-may-Anh-Viet"))
        self.wandb_run_name = log_cfg.get("wandb_run_name")
        self.global_step = 0

        forward_params = inspect.signature(self.model.forward).parameters
        self._accepts_tf = "teacher_forcing_ratio" in forward_params
        self._accepts_tgt_mask = "tgt_mask" in forward_params

    def _unpack_batch(
        self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(batch, dict):
            src = batch["src_ids"].to(self.device, non_blocking=True)
            tgt_in = batch["tgt_input_ids"].to(self.device, non_blocking=True)
            tgt_out = batch["tgt_output_ids"].to(self.device, non_blocking=True)
            return src, tgt_in, tgt_out, src == self.pad_id, tgt_in == self.pad_id
        tensors = [t.to(self.device, non_blocking=True) for t in batch]
        if len(tensors) == 5:
            return tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
        return tensors[0], tensors[1], tensors[2], None, None

    def _forward_model(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
        tf_ratio: float,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"src_mask": src_mask}
        if self._accepts_tf:
            kwargs["teacher_forcing_ratio"] = tf_ratio
        if self._accepts_tgt_mask:
            kwargs["tgt_mask"] = tgt_mask
        out = self.model(src, tgt_in, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def train_epoch(self, tf_ratio: float | None = None) -> float:
        """Chạy 1 epoch huấn luyện (hỗ trợ gradient accumulation) và trả về train loss trung bình."""
        self.model.train()
        total_loss = 0.0
        cur_tf = self.teacher_forcing_ratio if tf_ratio is None else tf_ratio
        num_batches = len(self.train_loader)
        accum = self.grad_accum_steps
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(self.train_loader, start=1):
            src, tgt_in, tgt_out, src_mask, tgt_mask = self._unpack_batch(batch)
            window_start = ((step_idx - 1) // accum) * accum
            window_size = min(accum, num_batches - window_start)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self._forward_model(src, tgt_in, src_mask, tgt_mask, cur_tf)
                loss = self.criterion(logits, tgt_out)

            total_loss += loss.item()
            self.scaler.scale(loss / window_size).backward()
            del logits, loss

            if step_idx % accum == 0 or step_idx == num_batches:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if (
                    isinstance(self.scheduler, NoamScheduler)
                    and self.scaler.get_scale() >= scale_before
                ):
                    self.scheduler.step()

                self.global_step += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self) -> float:
        """Đánh giá loss chuẩn trên tập validation (bật Teacher Forcing = 1.0, tắt Dropout)."""
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            src, tgt_in, tgt_out, src_mask, tgt_mask = self._unpack_batch(batch)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self._forward_model(src, tgt_in, src_mask, tgt_mask, 1.0)
                loss = self.criterion(logits, tgt_out)
            total_loss += loss.item()

        return total_loss / max(1, len(self.val_loader))

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )

    def fit(self) -> list[dict[str, Any]]:
        """Chạy toàn bộ các epoch, lưu best checkpoint và log JSONL (+ W&B nếu bật)."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "metrics.jsonl"
        wb: Any = None
        if self.use_wandb:
            import importlib

            wb = importlib.import_module("wandb")
            wb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.config,
            )

        best_score = -math.inf if self.eval_bleu_fn is not None else math.inf
        bad_epochs = 0
        history: list[dict[str, Any]] = []

        sep = "-" * 94
        header = (
            f"{'Epoch':<7} | {'Train Loss (PPL)':<17} | {'Val Loss (PPL)':<17} | "
            f"{'Val BLEU':>8} | {'LR':<9} | {'Time':>8} | {'Best'}"
        )

        try:
            print(f"\n{sep}\n{header}\n{sep}", flush=True)
            for epoch in range(1, self.epochs + 1):
                progress = (epoch - 1) / max(1, self.epochs - 1)
                cur_tf = round(
                    self.teacher_forcing_ratio
                    - progress * (self.teacher_forcing_ratio - self.min_tf_ratio),
                    4,
                )
                t_train0 = time.time()
                train_loss = self.train_epoch(cur_tf)
                train_time = round(time.time() - t_train0, 2)

                t_val0 = time.time()
                val_loss = self.validate()
                raw_bleu = self.eval_bleu_fn(self.model) if self.eval_bleu_fn is not None else None
                val_time = round(time.time() - t_val0, 2)

                if isinstance(raw_bleu, dict):
                    val_bleu: float | None = float(raw_bleu["bleu"])
                    bleu_detail: dict[str, Any] = raw_bleu
                elif raw_bleu is not None:
                    val_bleu = float(raw_bleu)
                    bleu_detail = {}
                else:
                    val_bleu = None
                    bleu_detail = {}

                lr = float(self.optimizer.param_groups[0]["lr"])
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    sched_metric = (
                        val_bleu
                        if (self.scheduler.mode == "max" and val_bleu is not None)
                        else val_loss
                    )
                    self.scheduler.step(sched_metric)
                elif self.scheduler is not None and not isinstance(self.scheduler, NoamScheduler):
                    self.scheduler.step()

                improved = (
                    (val_bleu > best_score) if val_bleu is not None else (val_loss < best_score)
                )
                if improved:
                    best_score = val_bleu if val_bleu is not None else val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                elapsed = round(train_time + val_time, 2)
                train_ppl = round(math.exp(min(train_loss, 20.0)), 2)
                val_ppl = round(math.exp(min(val_loss, 20.0)), 2)
                gpu_mb = (
                    round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2), 1)
                    if self.device.type == "cuda"
                    else 0.0
                )

                record: dict[str, Any] = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "train_loss": round(train_loss, 4),
                    "train_ppl": train_ppl,
                    "val_loss": round(val_loss, 4),
                    "val_ppl": val_ppl,
                    "val_bleu": round(val_bleu, 2) if val_bleu is not None else None,
                    "val_precisions": bleu_detail.get("precisions"),
                    "val_bp": bleu_detail.get("bp"),
                    "best_metric": round(best_score, 4),
                    "early_stopping_counter": bad_epochs,
                    "lr": lr,
                    "gpu_mem_mb": gpu_mb,
                    "train_time_sec": train_time,
                    "val_time_sec": val_time,
                    "time_sec": elapsed,
                }
                if improved:
                    self.save_checkpoint(self.ckpt_path, epoch, record)
                history.append(record)
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if wb is not None:
                    wb_payload: dict[str, Any] = {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "early_stopping_counter": bad_epochs,
                        "best_metric": round(best_score, 4),
                        "train/loss": record["train_loss"],
                        "train/ppl": train_ppl,
                        "train/lr": lr,
                        "train/time_sec": train_time,
                        "val/loss": record["val_loss"],
                        "val/ppl": val_ppl,
                        "val/time_sec": val_time,
                    }
                    if val_bleu is not None:
                        wb_payload["val/bleu"] = record["val_bleu"]
                    if "bp" in bleu_detail:
                        wb_payload["val/bp"] = bleu_detail["bp"]
                    wb.log(wb_payload, step=self.global_step)

                train_str = f"{record['train_loss']:.4f} ({train_ppl:.2f})"
                val_str = f"{record['val_loss']:.4f} ({val_ppl:.2f})"
                bleu_col = f"{record['val_bleu']:.2f}" if val_bleu is not None else "-"
                time_col = f"{elapsed:.1f}s"
                best_col = "✓ BEST" if improved else "-"
                print(
                    f"{epoch:02d}/{self.epochs:02d}   | {train_str:<17} | {val_str:<17} | "
                    f"{bleu_col:>8} | {lr:<9.2e} | {time_col:>8} | {best_col}",
                    flush=True,
                )

                if self.patience > 0 and bad_epochs >= self.patience:
                    print(
                        f"Early stopping tại epoch {epoch} (không cải thiện sau {self.patience} epochs)."
                    )
                    break
            print(sep, flush=True)
        finally:
            if wb is not None:
                wb.finish()

        return history


if __name__ == "__main__":
    import tempfile
    from torch.utils.data import TensorDataset
    from src.models.seq2seq_lstm import Seq2SeqLSTM

    src = torch.randint(4, 50, (8, 6))
    tgt_in = torch.randint(4, 50, (8, 5))
    tgt_out = torch.randint(4, 50, (8, 5))
    src_mask = torch.zeros_like(src, dtype=torch.bool)
    tgt_mask = torch.zeros_like(tgt_in, dtype=torch.bool)
    loader: DataLoader[Any] = DataLoader(
        TensorDataset(src, tgt_in, tgt_out, src_mask, tgt_mask), batch_size=4
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "training": {
                "epochs": 2,
                "lr": 1e-3,
                "amp": True,
                "label_smoothing": 0.1,
                "grad_accum_steps": 2,
            },
            "logging": {"checkpoint_dir": tmp, "checkpoint_name": "best.pt", "log_dir": tmp},
        }
        tiny_model = Seq2SeqLSTM(50, 50, embed_dim=16, hidden_dim=32, num_layers=1)
        trainer = Trainer(tiny_model, loader, loader, cfg, eval_bleu_fn=lambda _: {"bleu": 12.34})
        hist = trainer.fit()
        assert len(hist) == 2 and (Path(tmp) / "best.pt").exists()
        print("Trainer self-check OK")
