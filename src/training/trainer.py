"""Trainer: vòng lặp huấn luyện và đánh giá cho mô hình dịch máy."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.tokenizer import BPETokenizer
from src.evaluation.metrics import compute_bleu


class Trainer:
    """Quản lý toàn bộ quá trình train / validate / lưu checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        tokenizer: BPETokenizer,
        device: torch.device,
        scheduler=None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        teacher_forcing_ratio: float = 0.5,
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
        self.best_bleu = 0.0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Chạy 1 epoch huấn luyện, trả về loss trung bình."""
        self.model.train()
        total_loss = 0.0

        for src, tgt in dataloader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            self.optimizer.zero_grad()

            outputs, _ = self.model(src, tgt, self.teacher_forcing_ratio)
            loss = self.criterion(outputs, tgt[:, 1:])

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Đánh giá trên tập val/test, trả về (loss, BLEU)."""
        self.model.eval()
        total_loss = 0.0
        all_predictions: list[str] = []
        all_references: list[str] = []

        for src, tgt in dataloader:
            src, tgt = src.to(self.device), tgt.to(self.device)

            outputs, _ = self.model(src, tgt, teacher_forcing_ratio=0.0)
            loss = self.criterion(outputs, tgt[:, 1:])
            total_loss += loss.item()

            pred_ids = outputs.argmax(dim=-1)
            for i in range(src.size(0)):
                pred_text = self.tokenizer.decode(pred_ids[i].cpu().tolist())
                ref_text = self.tokenizer.decode(tgt[i, 1:].cpu().tolist())
                all_predictions.append(pred_text)
                all_references.append(ref_text)

        avg_loss = total_loss / len(dataloader)
        bleu = compute_bleu(all_predictions, all_references)
        return avg_loss, bleu

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
    ) -> dict:
        """Huấn luyện đầy đủ num_epochs, lưu checkpoint tốt nhất."""
        history = {"train_loss": [], "val_loss": [], "val_bleu": []}

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_bleu = self.evaluate(val_loader)
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_bleu"].append(val_bleu)

            print(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val BLEU: {val_bleu:.2f} | "
                f"Time: {elapsed:.1f}s"
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_bleu)
                else:
                    self.scheduler.step()

            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                self._save_checkpoint(epoch, val_bleu)
                print(f"  → Lưu checkpoint mới (BLEU = {val_bleu:.2f})")

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
