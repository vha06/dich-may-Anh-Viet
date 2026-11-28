"""Điểm chạy huấn luyện mô hình dịch máy: python scripts/train.py --config configs/baseline_lstm.yaml"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import build_dataloaders
from src.data.tokenizer import BaseTokenizer, load_tokenizer
from src.evaluation.metrics import compute_corpus_bleu
from src.models.seq2seq_lstm import Seq2SeqLSTM
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _beam_decode_batch(
    model: Seq2SeqLSTM,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    unk_id: int,
    max_len: int,
    beam_size: int = 5,
    length_penalty: float = 0.9,
) -> list[list[int]]:
    """Giải mã Beam Search song song theo batch [B * K] có phạt độ dài (GNMT length penalty)."""
    device = src.device
    batch_size = src.size(0)
    vocab_size = model.tgt_vocab_size

    enc_outputs, (h, c) = model.encoder(src, src_mask)
    if beam_size <= 1:
        token = torch.full((batch_size,), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        attn_hidden: torch.Tensor | None = None
        steps: list[torch.Tensor] = []
        for _ in range(max_len):
            logits, h, c, _, attn_hidden = model.decoder(
                token, h, c, enc_outputs, src_mask, attn_hidden
            )
            logits[:, pad_id] = -1e9
            logits[:, unk_id] = -1e9
            token = logits.argmax(dim=-1)
            steps.append(token)
            finished |= token == eos_id
            if bool(finished.all()):
                break
        return torch.stack(steps, dim=1).tolist() if steps else [[] for _ in range(batch_size)]

    enc_bk = enc_outputs.repeat_interleave(beam_size, dim=0)
    mask_bk = src_mask.repeat_interleave(beam_size, dim=0)
    h = h.repeat_interleave(beam_size, dim=1)
    c = c.repeat_interleave(beam_size, dim=1)
    attn_hidden = None

    seqs = torch.full((batch_size, beam_size, 1), sos_id, dtype=torch.long, device=device)
    scores = torch.full((batch_size, beam_size), -1e9, device=device)
    scores[:, 0] = 0.0
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
    lengths = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
    batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size

    for _ in range(max_len):
        last_tokens = seqs[:, :, -1].reshape(batch_size * beam_size)
        logits, h, c, _, attn_hidden = model.decoder(
            last_tokens, h, c, enc_bk, mask_bk, attn_hidden
        )
        logits[:, pad_id] = -1e9
        logits[:, unk_id] = -1e9
        log_probs = torch.log_softmax(logits, dim=-1).view(batch_size, beam_size, vocab_size)

        if bool(finished.any()):
            log_probs = log_probs.masked_fill(finished.unsqueeze(-1), -1e9)
            log_probs[:, :, pad_id] = torch.where(finished, 0.0, log_probs[:, :, pad_id])

        cand_scores = scores.unsqueeze(-1) + log_probs
        cand_lengths = lengths + (~finished).long()
        lp = (((5.0 + cand_lengths.float()) / 6.0) ** length_penalty).unsqueeze(-1)
        penalized = (cand_scores / lp).view(batch_size, beam_size * vocab_size)

        _, top_idx = penalized.topk(beam_size, dim=-1)
        beam_idx = top_idx // vocab_size
        token_idx = top_idx % vocab_size

        scores = cand_scores.view(batch_size, beam_size * vocab_size).gather(1, top_idx)
        seqs = torch.cat(
            [
                seqs.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, seqs.size(2))),
                token_idx.unsqueeze(-1),
            ],
            dim=-1,
        )
        finished = finished.gather(1, beam_idx) | (token_idx == eos_id)
        lengths = cand_lengths.gather(1, beam_idx)

        flat_idx = (batch_offsets + beam_idx).reshape(-1)
        h = h.index_select(1, flat_idx)
        c = c.index_select(1, flat_idx)
        attn_hidden = attn_hidden.index_select(0, flat_idx)

        if bool(finished.all()):
            break

    return seqs[:, 0, 1:].tolist()


@torch.no_grad()
def evaluate_val_bleu(
    model: nn.Module,
    val_loader: DataLoader[Any],
    val_refs: list[str],
    tgt_tokenizer: BaseTokenizer,
    max_len: int,
    device: torch.device,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> dict[str, Any]:
    """Giải mã Beam Search (hoặc Greedy nếu beam_size=1) theo batch trên tập val và tính SacreBLEU."""
    model.eval()
    hyps: list[str] = []
    sos_id = tgt_tokenizer.sos_id
    eos_id = tgt_tokenizer.eos_id
    pad_id = tgt_tokenizer.pad_id
    unk_id = tgt_tokenizer.unk_id

    for batch in val_loader:
        if isinstance(batch, dict):
            src = batch["src_ids"].to(device, non_blocking=True)
            src_mask = src == pad_id
        else:
            src = batch[0].to(device, non_blocking=True)
            src_mask = batch[3].to(device, non_blocking=True) if len(batch) >= 4 else (src == pad_id)

        if isinstance(model, Seq2SeqLSTM):
            preds = _beam_decode_batch(
                model,
                src,
                src_mask,
                sos_id=sos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                unk_id=unk_id,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
        else:
            raise ValueError(f"Chưa hỗ trợ giải mã cho kiểu mô hình: {type(model).__name__}")

        for seq in preds:
            if eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            hyps.append(tgt_tokenizer.decode(seq, skip_special_tokens=True))

    return compute_corpus_bleu(hyps, val_refs)


def build_model(model_cfg: dict[str, Any], pad_id: int) -> nn.Module:
    cfg = dict(model_cfg)
    model_type = cfg.pop("type", "seq2seq_lstm")
    if model_type == "seq2seq_lstm":
        return Seq2SeqLSTM(**cfg, pad_id=pad_id)
    raise ValueError(f"Kiểu mô hình không hợp lệ: '{model_type}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dịch máy Anh-Việt")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_lstm.yaml"),
        help="Đường dẫn tới file cấu hình YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Tắt ghi log lên Weights & Biases")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False

    set_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    src_tok = load_tokenizer(data_cfg["src_tokenizer"])
    tgt_tok = load_tokenizer(data_cfg["tgt_tokenizer"])

    processed_dir = data_cfg.get("processed_dir", Path(data_cfg["train_src"]).parent)
    loaders = build_dataloaders(
        processed_dir=processed_dir,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=int(data_cfg.get("batch_size", 64)),
        max_len=int(data_cfg.get("max_len", 128)),
        use_bucketing=bool(data_cfg.get("use_bucketing", True)),
        pre_tokenize=bool(data_cfg.get("cache_tokenized", True)),
        num_workers=int(data_cfg.get("num_workers", 2)),
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    with Path(data_cfg["val_tgt"]).open("r", encoding="utf-8") as f:
        val_refs = [line.rstrip("\n") for line in f]

    model = build_model(config["model"], pad_id=src_tok.pad_id)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(data_cfg.get("batch_size", 64))
    grad_accum = max(1, int(config.get("training", {}).get("grad_accum_steps", 1)))
    dec_cfg = config.get("decoding", {})
    max_decode_len = int(dec_cfg.get("max_len", 160))
    val_method = str(dec_cfg.get("val_method", "beam"))
    val_beam_size = int(dec_cfg.get("beam_size", 4)) if val_method == "beam" else 1
    length_penalty = float(dec_cfg.get("length_penalty", 0.6))

    print(
        f"Mô hình: {config['model'].get('type')} | Tham số: {num_params:,} | "
        f"Thiết bị: {device} | Batch: {batch_size}x{grad_accum}={batch_size * grad_accum} | "
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
        f"Val decode: {val_method} (beam={val_beam_size}, lp={length_penalty})"
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        pad_id=tgt_tok.pad_id,
        device=device,
        eval_bleu_fn=lambda m: evaluate_val_bleu(
            m,
            val_loader,
            val_refs,
            tgt_tok,
            max_decode_len,
            device,
            beam_size=val_beam_size,
            length_penalty=length_penalty,
        ),
    )
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
