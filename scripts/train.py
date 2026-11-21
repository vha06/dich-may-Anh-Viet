"""Điểm chạy huấn luyện: python scripts/train.py --config configs/baseline_lstm.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path để luôn tìm thấy module 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.data.tokenizer import BPETokenizer
from src.data.dataset import build_dataloader
from src.models.seq2seq_lstm import Encoder, Decoder, Seq2Seq
from src.training.losses import LabelSmoothedCrossEntropy
from src.training.schedulers import build_scheduler
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, pad_id: int, device: torch.device) -> Seq2Seq:
    m = cfg["model"]
    encoder = Encoder(
        vocab_size=m["vocab_size"],
        embed_dim=m["embed_dim"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        pad_id=pad_id,
    )
    decoder = Decoder(
        vocab_size=m["vocab_size"],
        embed_dim=m["embed_dim"],
        encoder_dim=m["hidden_dim"] * 2,
        hidden_dim=m["hidden_dim"],
        attention_dim=m["attention_dim"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        pad_id=pad_id,
    )
    model = Seq2Seq(encoder, decoder, pad_id=pad_id).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable:,} trainable)")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dịch máy")
    parser.add_argument(
        "--config", type=str, default="configs/baseline_lstm.yaml",
        help="Đường dẫn file cấu hình YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BPETokenizer(cfg["data"]["tokenizer_model"])

    train_loader = build_dataloader(
        en_path=cfg["data"]["train_en"],
        vi_path=cfg["data"]["train_vi"],
        tokenizer=tokenizer,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        max_len=cfg["data"]["max_len"],
    )
    val_loader = build_dataloader(
        en_path=cfg["data"]["val_en"],
        vi_path=cfg["data"]["val_vi"],
        tokenizer=tokenizer,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        max_len=cfg["data"]["max_len"],
    )

    model = build_model(cfg, pad_id=tokenizer.pad_id, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    scheduler = build_scheduler(
        optimizer,
        scheduler_type=cfg["training"]["scheduler"],
        patience=cfg["training"]["scheduler_patience"],
        factor=cfg["training"]["scheduler_factor"],
        min_lr=cfg["training"]["min_lr"],
    )

    criterion = LabelSmoothedCrossEntropy(
        pad_id=tokenizer.pad_id,
        smoothing=cfg["training"]["label_smoothing"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        scheduler=scheduler,
        grad_clip=cfg["training"]["grad_clip"],
        checkpoint_dir=cfg["checkpoint"]["dir"],
        teacher_forcing_ratio=cfg["training"]["teacher_forcing_ratio"],
    )

    trainer.fit(train_loader, val_loader, num_epochs=cfg["training"]["num_epochs"])


if __name__ == "__main__":
    main()
