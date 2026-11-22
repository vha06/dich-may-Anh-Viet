"""CLI training entry point for NMT models configured via YAML."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.dataset import build_dataloaders
from src.data.tokenizer import PAD_ID, load_tokenizer
from src.models.seq2seq_lstm import build_seq2seq_lstm
from src.training import Trainer, build_loss_pair, build_scheduler


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load experiment YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"File YAML khong hop le: {path}")
    return cfg


def set_seed(seed: int) -> None:
    """Set reproducible random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve target compute device ('auto', 'cpu', 'cuda')."""
    kind = device_arg.strip().lower()
    if kind == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind in ("cpu", "cuda"):
        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Yeu cau CUDA nhung PyTorch hien tai khong nhan GPU.")
        return torch.device(kind)
    return torch.device(device_arg)


def build_model_from_config(cfg: dict[str, Any]) -> nn.Module:
    """Instantiate an NMT architecture from the `model` block of a YAML config."""
    model_cfg = dict(cfg["model"])
    arch = str(model_cfg.pop("arch", "seq2seq_lstm")).strip().lower()
    if arch == "seq2seq_lstm":
        return build_seq2seq_lstm(**model_cfg)
    if arch == "transformer":
        raise NotImplementedError(
            "Kien truc 'transformer' se duoc bo sung o Giai doan 3 (Tuan 3)."
        )
    raise ValueError(f"Khong ho tro kien truc model.arch={arch!r}")


def _init_wandb_run(
    cfg: dict[str, Any],
    disable_wandb: bool,
    resume_path: str | None,
) -> Any:
    wandb_cfg = cfg.get("wandb", {})
    if disable_wandb or not wandb_cfg.get("enabled", False):
        return None
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Da bat wandb trong config nhung moi truong chua cai `wandb`. "
            "Chay `pip install wandb` hoac them co `--no-wandb` khi test nhanh."
        ) from exc

    exp_cfg = cfg.get("experiment", {})
    return wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project", "dich-may-Anh-Viet"),
        group=wandb_cfg.get("group"),
        name=exp_cfg.get("name"),
        tags=wandb_cfg.get("tags"),
        config=cfg,
        resume="allow" if resume_path is not None else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huan luyen mo hinh Dich may Anh-Viet tu file cau hinh YAML"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_lstm.yaml"),
        help="Duong dan file YAML cau hinh (mac dinh: configs/baseline_lstm.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Duong dan checkpoint (last_model.pt) de tiep tuc huan luyen",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Ghi de tong so epoch trong config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Ghi de batch_size trong config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiet bi tinh toan: auto | cpu | cuda (mac dinh: auto)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Tat ghi log Weights & Biases cho lan chay nay",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Ghi de ten Run hien thi tren W&B (vi du: ha_seq2seq_lstm_v1)",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Ghi de nhom thu nghiem W&B group (vi du: phase2-baseline-lstm)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Gioi han so batch train moi epoch (dung de chay smoke test nhanh)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Gioi han so batch val moi epoch (dung de chay smoke test nhanh)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.run_name is not None:
        exp_cfg["name"] = args.run_name
    if args.wandb_group is not None:
        cfg.setdefault("wandb", {})["group"] = args.wandb_group

    set_seed(int(exp_cfg.get("seed", 42)))
    device = resolve_device(args.device)

    tokenizer_dir = Path(data_cfg["tokenizer_dir"])
    tok_type = str(data_cfg.get("tokenizer_type", "bpe"))
    src_tok = load_tokenizer(tokenizer_dir / f"{tok_type}_en.json", tokenizer_type=tok_type)
    tgt_tok = load_tokenizer(tokenizer_dir / f"{tok_type}_vi.json", tokenizer_type=tok_type)

    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    loaders = build_dataloaders(
        processed_dir=data_cfg["processed_dir"],
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=int(data_cfg.get("batch_size", 64)),
        max_len=int(data_cfg.get("max_len", 128)),
        pre_tokenize=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    model = build_model_from_config(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    sched_params = dict(train_cfg.get("scheduler_params", {}))
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=str(train_cfg.get("scheduler_type", "plateau")),
        **sched_params,
    )
    train_loss_fn, eval_loss_fn = build_loss_pair(
        pad_id=PAD_ID,
        train_label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )

    wandb_run = _init_wandb_run(
        cfg=cfg,
        disable_wandb=args.no_wandb,
        resume_path=str(args.resume) if args.resume is not None else None,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        tgt_tokenizer=tgt_tok,
        config=cfg,
        device=device,
        checkpoint_dir=exp_cfg.get("checkpoint_dir", f"checkpoints/{exp_cfg['name']}"),
        output_dir=exp_cfg.get("output_dir", f"outputs/{exp_cfg['name']}"),
        use_amp=bool(train_cfg.get("use_amp", True)),
        amp_dtype=str(train_cfg.get("amp_dtype", "auto")),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        teacher_forcing_ratio=float(train_cfg.get("teacher_forcing_ratio", 1.0)),
        save_best_by=str(train_cfg.get("save_best_by", "val_bleu")),
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
        eval_decode_strategy=str(train_cfg.get("eval_decode_strategy", "greedy")),
        eval_beam_size=int(train_cfg.get("eval_beam_size", 4)),
        eval_length_penalty=float(train_cfg.get("eval_length_penalty", 0.6)),
        max_decode_len=int(data_cfg.get("max_len", 128)),
        wandb_run=wandb_run,
    )

    if args.resume is not None:
        resumed_epoch = trainer.load_checkpoint(args.resume)
        print(
            f"Da khoi phuc checkpoint {args.resume} -> tiep tuc tu Epoch {resumed_epoch} "
            f"(best_{trainer.save_best_by}={trainer.best_metric:.4f} tai Epoch {trainer.best_epoch})"
        )

    print(
        f"=== Huan luyen [{exp_cfg['name']}] | device={device} | "
        f"amp={trainer.amp_enabled} ({trainer.amp_torch_dtype}) | "
        f"params={total_params:,} | wandb={'ON' if wandb_run else 'OFF'} ==="
    )

    try:
        history = trainer.fit(
            num_epochs=int(train_cfg["epochs"]),
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(
        f"\nHoan tat! Best Epoch={trainer.best_epoch} | "
        f"Best {trainer.save_best_by}={trainer.best_metric:.4f} | "
        f"Tong so epoch da chay={len(history)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
