"""Model-agnostic NMT Trainer supporting AMP, checkpoint resume, Early Stopping, and W&B."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import DEFAULT_MAX_SEQ_LEN, BaseTokenizer
from src.evaluation.decode import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_LENGTH_PENALTY,
    DecodeResult,
    translate_dataloader,
)
from src.training.losses import TranslationLoss, compute_perplexity


def build_forward_adapter(
    model: nn.Module,
) -> Callable[[nn.Module, dict[str, torch.Tensor], float], torch.Tensor]:
    """Build a model-agnostic forward callable working across Seq2SeqLSTM and Transformers."""
    params = set(inspect.signature(model.forward).parameters.keys())
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(model.forward).parameters.values()
    )

    def _adapter(
        mdl: nn.Module,
        batch: dict[str, torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        call_kwargs: dict[str, Any] = {}
        if has_kwargs or "src_lengths" in params:
            call_kwargs["src_lengths"] = batch["src_lengths"]
        if has_kwargs or "teacher_forcing_ratio" in params:
            call_kwargs["teacher_forcing_ratio"] = teacher_forcing_ratio

        output = mdl(batch["src_ids"], tgt_input_ids=batch["tgt_input_ids"], **call_kwargs)
        return output[0] if isinstance(output, tuple) else output

    return _adapter


def _resolve_amp_setup(
    use_amp: bool,
    amp_dtype: str,
    device: torch.device,
) -> tuple[bool, torch.dtype | None, bool]:
    """Select safe AMP configuration (bfloat16 preferred on Ada/RTX 4060, float16 fallback)."""
    if not use_amp or device.type != "cuda":
        return False, None, False

    kind = amp_dtype.strip().lower()
    if kind == "auto":
        resolved = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif kind in ("bf16", "bfloat16"):
        resolved = torch.bfloat16
    elif kind in ("fp16", "float16"):
        resolved = torch.float16
    else:
        raise ValueError(
            f"Khong ho tro amp_dtype={amp_dtype!r}; chon 'auto', 'bfloat16', hoac 'float16'."
        )

    use_scaler = resolved == torch.float16
    return True, resolved, use_scaler


class Trainer:
    """Unified training, validation, checkpointing, and W&B orchestrator for NMT models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        train_loss_fn: TranslationLoss,
        eval_loss_fn: TranslationLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tgt_tokenizer: BaseTokenizer,
        config: dict[str, Any],
        device: torch.device | str = "cpu",
        checkpoint_dir: str | Path = Path("checkpoints/baseline_lstm"),
        output_dir: str | Path = Path("outputs/baseline_lstm"),
        forward_fn: Callable[[nn.Module, dict[str, torch.Tensor], float], torch.Tensor]
        | None = None,
        decode_fn: Callable[..., DecodeResult] | None = None,
        use_amp: bool = True,
        amp_dtype: str = "auto",
        grad_clip: float = 1.0,
        teacher_forcing_ratio: float = 1.0,
        save_best_by: str = "val_bleu",
        early_stopping_patience: int = 5,
        eval_decode_strategy: str = "greedy",
        eval_beam_size: int = DEFAULT_BEAM_SIZE,
        eval_length_penalty: float = DEFAULT_LENGTH_PENALTY,
        max_decode_len: int = DEFAULT_MAX_SEQ_LEN,
        wandb_run: Any = None,
        show_progress: bool = True,
    ) -> None:
        if save_best_by not in ("val_bleu", "val_loss"):
            raise ValueError(
                f"save_best_by phai la 'val_bleu' hoac 'val_loss', nhan {save_best_by!r}"
            )
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience phai >= 0")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss_fn = train_loss_fn
        self.eval_loss_fn = eval_loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tgt_tokenizer = tgt_tokenizer
        self.config = config

        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.forward_fn = (
            forward_fn if forward_fn is not None else build_forward_adapter(self.model)
        )
        self.decode_fn = decode_fn
        self.grad_clip = float(grad_clip)
        self.teacher_forcing_ratio = float(teacher_forcing_ratio)
        self.save_best_by = save_best_by
        self.early_stopping_patience = int(early_stopping_patience)
        self.eval_decode_strategy = eval_decode_strategy
        self.eval_beam_size = int(eval_beam_size)
        self.eval_length_penalty = float(eval_length_penalty)
        self.max_decode_len = int(max_decode_len)
        self.wandb_run = wandb_run
        self.show_progress = show_progress

        self.amp_enabled, self.amp_torch_dtype, use_scaler = _resolve_amp_setup(
            use_amp=use_amp, amp_dtype=amp_dtype, device=self.device
        )
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda") if use_scaler else None
        )

        self.best_metric: float = (
            float("-inf") if self.save_best_by == "val_bleu" else float("inf")
        )
        self.best_epoch: int = 0
        self.early_stopping_counter: int = 0
        self.global_step: int = 0
        self.start_epoch: int = 1
        self.history: list[dict[str, Any]] = []

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _current_lr(self) -> float:
        if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
            return float(self.scheduler.get_last_lr()[0])
        return float(self.optimizer.param_groups[0]["lr"])

    def train_one_epoch(
        self, epoch: int, max_batches: int | None = None
    ) -> dict[str, float]:
        """Run one training epoch with token-weighted loss, grad clipping, and batch LR step."""
        self.model.train()
        start_time = time.perf_counter()

        total_loss_sum = 0.0
        total_tokens = 0
        grad_norm_sum = 0.0
        num_steps = 0

        total_iter = (
            min(len(self.train_loader), max_batches)
            if max_batches is not None
            else len(self.train_loader)
        )
        iterator = tqdm(
            self.train_loader,
            total=total_iter,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            disable=not self.show_progress,
        )

        for batch_idx, raw_batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = self._move_batch(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_torch_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.forward_fn(self.model, batch, self.teacher_forcing_ratio)
                loss, batch_tokens = self.train_loss_fn(logits, batch["tgt_output_ids"])

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0.0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    )
                else:
                    grad_norm = 0.0
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                step_executed = self.scaler.get_scale() >= scale_before
            else:
                loss.backward()
                if self.grad_clip > 0.0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    )
                else:
                    grad_norm = 0.0
                self.optimizer.step()
                step_executed = True

            if (
                self.scheduler is not None
                and self.scheduler.step_mode == "batch"
                and step_executed
            ):
                self.scheduler.step()

            loss_val = float(loss.item())
            total_loss_sum += loss_val * batch_tokens
            total_tokens += batch_tokens
            grad_norm_sum += grad_norm
            num_steps += 1
            self.global_step += 1

            if self.show_progress:
                iterator.set_postfix(
                    loss=f"{loss_val:.4f}",
                    lr=f"{self._current_lr():.2e}",
                )

        if total_tokens == 0:
            raise RuntimeError("Khong co batch nao duoc xu ly trong train_one_epoch.")

        avg_loss = total_loss_sum / total_tokens
        return {
            "train_loss": round(avg_loss, 4),
            "train_ppl": round(compute_perplexity(avg_loss), 4),
            "grad_norm": round(grad_norm_sum / max(num_steps, 1), 4),
            "lr": self._current_lr(),
            "train_tokens": total_tokens,
            "train_time_sec": round(time.perf_counter() - start_time, 2),
        }

    def validate(self, max_batches: int | None = None) -> dict[str, Any]:
        """Compute unsmoothed val_loss, val_ppl, and SacreBLEU while preserving train/eval mode."""
        was_training = self.model.training
        self.model.eval()
        start_time = time.perf_counter()

        total_loss_sum = 0.0
        total_tokens = 0
        try:
            with torch.no_grad():
                for batch_idx, raw_batch in enumerate(self.val_loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    batch = self._move_batch(raw_batch)
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_torch_dtype,
                        enabled=self.amp_enabled,
                    ):
                        logits = self.forward_fn(self.model, batch, 1.0)
                        loss, batch_tokens = self.eval_loss_fn(
                            logits, batch["tgt_output_ids"]
                        )
                    total_loss_sum += float(loss.item()) * batch_tokens
                    total_tokens += batch_tokens

            val_loss = total_loss_sum / max(total_tokens, 1)
            val_ppl = compute_perplexity(val_loss)

            decode_res = translate_dataloader(
                self.model,
                self.val_loader,
                self.tgt_tokenizer,
                strategy=self.eval_decode_strategy,
                beam_size=self.eval_beam_size,
                length_penalty=self.eval_length_penalty,
                max_len=self.max_decode_len,
                device=self.device,
                decode_fn=self.decode_fn,
                max_batches=max_batches,
            )

            sample_pairs = [
                {
                    "prediction": decode_res["predictions"][i],
                    "reference": decode_res["references"][i],
                }
                for i in range(min(3, decode_res["num_sentences"]))
            ]

            return {
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 4),
                "val_bleu": decode_res["bleu"],
                "val_precisions": decode_res["metrics"]["precisions"],
                "val_bp": decode_res["metrics"]["brevity_penalty"],
                "bleu_signature": decode_res["metrics"]["signature"],
                "val_num_sentences": decode_res["num_sentences"],
                "sample_pairs": sample_pairs,
                "val_time_sec": round(time.perf_counter() - start_time, 2),
            }
        finally:
            self.model.train(was_training)

    def _is_better_metric(self, current: float) -> bool:
        if self.save_best_by == "val_bleu":
            return current > self.best_metric
        return current < self.best_metric

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Persist full training state (including best_metric & early_stopping_counter)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "early_stopping_counter": self.early_stopping_counter,
            "save_best_by": self.save_best_by,
            "history": self.history,
            "config": self.config,
        }
        torch.save(payload, out_path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore complete training state cleanly for seamless --resume."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Khong tim thay checkpoint: {in_path}")

        payload = torch.load(in_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if self.scaler is not None and payload.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(payload["scaler_state_dict"])

        self.global_step = int(payload.get("global_step", 0))
        self.best_metric = float(
            payload.get(
                "best_metric",
                float("-inf") if self.save_best_by == "val_bleu" else float("inf"),
            )
        )
        self.best_epoch = int(payload.get("best_epoch", 0))
        self.early_stopping_counter = int(payload.get("early_stopping_counter", 0))
        self.history = list(payload.get("history", []))
        self.start_epoch = int(payload["epoch"]) + 1
        return self.start_epoch

    def _log_to_wandb(self, record: dict[str, Any], sample_pairs: Sequence[dict[str, str]]) -> None:
        if self.wandb_run is None:
            return
        log_payload = {
            "epoch": record["epoch"],
            "global_step": record["global_step"],
            "train/loss": record["train_loss"],
            "train/ppl": record["train_ppl"],
            "train/grad_norm": record["grad_norm"],
            "train/lr": record["lr"],
            "val/loss": record["val_loss"],
            "val/ppl": record["val_ppl"],
            "val/bleu": record["val_bleu"],
            "best_metric": self.best_metric,
            "early_stopping_counter": self.early_stopping_counter,
        }
        try:
            import wandb

            if sample_pairs:
                table = wandb.Table(columns=["epoch", "prediction", "reference"])
                for pair in sample_pairs:
                    table.add_data(record["epoch"], pair["prediction"], pair["reference"])
                log_payload["val/samples"] = table
        except Exception:
            pass
        self.wandb_run.log(log_payload, step=record["global_step"])

    def fit(
        self,
        num_epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute multi-epoch training loop with validation, checkpointing, and Early Stopping."""
        if self.start_epoch > num_epochs:
            return self.history

        best_path = self.checkpoint_dir / "best_model.pt"
        last_path = self.checkpoint_dir / "last_model.pt"
        history_path = self.output_dir / "history.json"

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_stats = self.train_one_epoch(epoch, max_batches=max_train_batches)
            val_stats = self.validate(max_batches=max_val_batches)

            if self.scheduler is not None and self.scheduler.step_mode == "epoch":
                metric_val = val_stats["val_loss"] if self.scheduler.requires_metric else None
                self.scheduler.step(metric_val)

            current_metric = float(val_stats[self.save_best_by])
            improved = self._is_better_metric(current_metric)
            if improved:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                self.save_checkpoint(best_path, epoch)
            else:
                self.early_stopping_counter += 1

            record: dict[str, Any] = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": train_stats["train_loss"],
                "train_ppl": train_stats["train_ppl"],
                "grad_norm": train_stats["grad_norm"],
                "lr": train_stats["lr"],
                "train_time_sec": train_stats["train_time_sec"],
                "val_loss": val_stats["val_loss"],
                "val_ppl": val_stats["val_ppl"],
                "val_bleu": val_stats["val_bleu"],
                "val_precisions": val_stats["val_precisions"],
                "val_bp": val_stats["val_bp"],
                "val_time_sec": val_stats["val_time_sec"],
                "is_best": improved,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "early_stopping_counter": self.early_stopping_counter,
            }
            self.history.append(record)
            self.save_checkpoint(last_path, epoch)
            history_path.write_text(
                json.dumps(self.history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._log_to_wandb(record, val_stats["sample_pairs"])

            tag = "* BEST *" if improved else f"patience {self.early_stopping_counter}/{self.early_stopping_patience}"
            print(
                f"[Epoch {epoch:02d}/{num_epochs:02d}] "
                f"train_loss={record['train_loss']:.4f} (ppl={record['train_ppl']:.2f}) | "
                f"val_loss={record['val_loss']:.4f} (ppl={record['val_ppl']:.2f}) | "
                f"val_bleu={record['val_bleu']:.2f} | lr={record['lr']:.2e} | "
                f"{record['train_time_sec'] + record['val_time_sec']:.1f}s | {tag}"
            )

            if (
                self.early_stopping_patience > 0
                and self.early_stopping_counter >= self.early_stopping_patience
            ):
                print(
                    f"Early Stopping kich hoat tai Epoch {epoch} "
                    f"(khong cai thien {self.save_best_by} trong {self.early_stopping_patience} epoch lien tiep)."
                )
                break

        self.start_epoch = (self.history[-1]["epoch"] + 1) if self.history else 1
        return self.history
