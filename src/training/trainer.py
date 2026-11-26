"""Trainer: vòng lặp huấn luyện và đánh giá cho mô hình dịch máy."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import BaseTokenizer
from src.evaluation.decode import greedy_decode
from src.evaluation.metrics import compute_corpus_bleu


class Trainer:
    """Quản lý toàn bộ quá trình train / validate / lưu checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        tokenizer: BaseTokenizer,
        device: torch.device,
        scheduler=None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        teacher_forcing_ratio: float = 0.5,
        max_decode_len: int = 128,
        early_stopping_patience: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_decode_len = max_decode_len
        self.early_stopping_patience = early_stopping_patience
        self.best_bleu = 0.0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Chạy 1 epoch huấn luyện, trả về loss trung bình."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch in pbar:
            src = batch["src_ids"].to(self.device)
            src_lengths = batch["src_lengths"].to(self.device)
            tgt_in = batch["tgt_input_ids"].to(self.device)
            tgt_out = batch["tgt_output_ids"].to(self.device)

            self.optimizer.zero_grad()

            outputs, _ = self.model(
                src,
                tgt_in,
                src_lengths=src_lengths,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
            )
            loss = self.criterion(outputs, tgt_out)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, dict]:
        """Đánh giá trên tập val/test, trả về (loss, dict metrics SacreBLEU)."""
        self.model.eval()
        total_loss = 0.0
        all_predictions: list[str] = []
        fallback_references: list[str] = []

        for batch in dataloader:
            src = batch["src_ids"].to(self.device)
            src_lengths = batch["src_lengths"].to(self.device)
            tgt_in = batch["tgt_input_ids"].to(self.device)
            tgt_out = batch["tgt_output_ids"].to(self.device)

            # 1. Tính Val Loss với Teacher Forcing chuẩn
            outputs, _ = self.model(
                src,
                tgt_in,
                src_lengths=src_lengths,
                teacher_forcing_ratio=1.0,
            )
            loss = self.criterion(outputs, tgt_out)
            total_loss += loss.item()

            # 2. Suy luận tự do hoàn toàn (không nhìn tgt) để chấm Val BLEU
            preds, _ = greedy_decode(
                self.model,
                src,
                self.tokenizer,
                max_len=self.max_decode_len,
                device=self.device,
                src_lengths=src_lengths,
            )
            all_predictions.extend(preds)

            if not hasattr(dataloader.dataset, "tgt_lines"):
                for i in range(src.size(0)):
                    fallback_references.append(self.tokenizer.decode(tgt_out[i]))

        # 3. Đối chiếu với văn bản gốc nguyên vẹn trên đĩa (Gold Reference)
        if hasattr(dataloader.dataset, "tgt_lines"):
            raw_references = list(dataloader.dataset.tgt_lines)
        else:
            raw_references = fallback_references

        if len(all_predictions) != len(raw_references):
            raise ValueError(
                f"Số lượng predictions ({len(all_predictions)}) ≠ references ({len(raw_references)})"
            )

        avg_loss = total_loss / len(dataloader)
        bleu_metrics = compute_corpus_bleu(all_predictions, raw_references)
        return avg_loss, bleu_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
    ) -> dict:
        """Huấn luyện đầy đủ num_epochs, hỗ trợ Early Stopping và lưu checkpoint tốt nhất."""
        history = {"train_loss": [], "val_loss": [], "val_bleu": []}
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, bleu_metrics = self.evaluate(val_loader)
            val_bleu = bleu_metrics["bleu"]
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_bleu"].append(val_bleu)

            p_str = "/".join(f"{p:.1f}" for p in bleu_metrics["precisions"])
            bp = bleu_metrics["brevity_penalty"]
            print(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val BLEU: {val_bleu:.2f} (P={p_str}, BP={bp:.3f}) | "
                f"Time: {elapsed:.1f}s"
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_bleu)
                else:
                    self.scheduler.step()

            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                epochs_no_improve = 0
                self._save_checkpoint(epoch, val_bleu)
                print(f"  → Lưu checkpoint mới (BLEU = {val_bleu:.2f})")
            else:
                epochs_no_improve += 1
                if (
                    self.early_stopping_patience is not None
                    and epochs_no_improve >= self.early_stopping_patience
                ):
                    print(
                        f"\nEarly stopping: Val BLEU không cải thiện sau "
                        f"{self.early_stopping_patience} epochs liên tiếp."
                    )
                    break

        print(f"\nHuấn luyện xong! Best BLEU: {self.best_bleu:.2f}")
        return history

    def _save_checkpoint(self, epoch: int, bleu: float) -> None:
        path = self.checkpoint_dir / "best_seq2seq.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "bleu": bleu,
            },
            path,
        )
