"""CLI checkpoint evaluation script for Greedy and Beam Search SacreBLEU scoring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from scripts.train import build_model_from_config, resolve_device
from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import DEFAULT_MAX_SEQ_LEN, load_tokenizer
from src.evaluation.decode import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_LENGTH_PENALTY,
    translate_dataloader,
)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    strategy: str | None = None,
    beam_size: int | None = None,
    length_penalty: float | None = None,
    max_len: int | None = None,
    batch_size: int | None = None,
    device: str = "auto",
    output_path: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Load a saved checkpoint, translate val/test split, compute SacreBLEU, and export JSON."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt_path}")
    if split not in ("val", "test"):
        raise ValueError(f"split phai la 'val' hoac 'test', nhan {split!r}")

    torch_device = resolve_device(device)
    payload = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    cfg = payload["config"]

    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})

    resolved_strategy = strategy or "beam"
    resolved_beam = (
        beam_size
        if beam_size is not None
        else int(train_cfg.get("test_beam_size", DEFAULT_BEAM_SIZE))
    )
    resolved_lp = (
        length_penalty
        if length_penalty is not None
        else float(train_cfg.get("test_length_penalty", DEFAULT_LENGTH_PENALTY))
    )
    resolved_max_len = (
        max_len if max_len is not None else int(data_cfg.get("max_len", DEFAULT_MAX_SEQ_LEN))
    )
    resolved_batch_size = (
        batch_size if batch_size is not None else int(data_cfg.get("batch_size", 64))
    )

    tokenizer_dir = Path(data_cfg["tokenizer_dir"])
    tok_type = str(data_cfg.get("tokenizer_type", "bpe"))
    src_tok = load_tokenizer(tokenizer_dir / f"{tok_type}_en.json", tokenizer_type=tok_type)
    tgt_tok = load_tokenizer(tokenizer_dir / f"{tok_type}_vi.json", tokenizer_type=tok_type)

    proc_dir = Path(data_cfg["processed_dir"])
    dataset = TranslationDataset(
        src_path=proc_dir / f"{split}.en",
        tgt_path=proc_dir / f"{split}.vi",
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        max_len=resolved_max_len,
        pre_tokenize=True,
    )
    loader = create_dataloader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(torch_device.type == "cuda"),
    )

    model = build_model_from_config(cfg).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    start_time = time.perf_counter()
    res = translate_dataloader(
        model=model,
        dataloader=loader,
        tgt_tokenizer=tgt_tok,
        strategy=resolved_strategy,
        beam_size=resolved_beam,
        length_penalty=resolved_lp,
        max_len=resolved_max_len,
        device=torch_device,
        max_batches=max_batches,
    )
    elapsed_sec = round(time.perf_counter() - start_time, 2)
    speed = round(res["num_sentences"] / max(elapsed_sec, 1e-6), 2)

    examples = [
        {
            "index": idx,
            "src": dataset.src_lines[idx],
            "prediction": res["predictions"][idx],
            "reference": res["references"][idx],
        }
        for idx in range(res["num_sentences"])
    ]

    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "experiment_name": cfg.get("experiment", {}).get("name", "unknown"),
        "split": split,
        "strategy": resolved_strategy,
        "beam_size": resolved_beam if resolved_strategy == "beam" else 1,
        "length_penalty": resolved_lp if resolved_strategy == "beam" else 0.0,
        "device": str(torch_device),
        "num_sentences": res["num_sentences"],
        "elapsed_sec": elapsed_sec,
        "sentences_per_sec": speed,
        "bleu": res["bleu"],
        "metrics": res["metrics"],
        "translations": examples,
    }

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Danh gia checkpoint NMT tren tap val/test bang Greedy hoac Beam Search"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Duong dan file checkpoint (.pt), vi du checkpoints/baseline_lstm/best_model.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Tap danh gia (mac dinh: test)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "beam"],
        default="beam",
        help="Chien luoc giai ma (mac dinh: beam)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="So nhanh Beam (mac dinh doc tu config trong checkpoint, hoac 4)",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="He so phat do dai alpha (mac dinh doc tu config, hoac 0.6)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size suy luan",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiet bi: auto | cpu | cuda",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Duong dan file JSON xuat ket qua dich (phuc vu Phan tich loi Tuan 5)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Gioi han so batch danh gia (dung khi test nhanh)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    out_file = args.output
    if out_file is None:
        exp_dir = Path("outputs") / args.checkpoint.parent.name
        out_file = exp_dir / f"eval_{args.split}_{args.strategy}.json"

    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        strategy=args.strategy,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        batch_size=args.batch_size,
        device=args.device,
        output_path=out_file,
        max_batches=args.max_batches,
    )

    m = report["metrics"]
    print(f"=== Ket qua Danh gia [{report['experiment_name']}] tren tap {report['split']} ===")
    print(
        f"  - Checkpoint : {report['checkpoint']} (epoch {report['checkpoint_epoch']})\n"
        f"  - Giai ma    : strategy={report['strategy']} | beam_size={report['beam_size']} | "
        f"alpha={report['length_penalty']} | device={report['device']}\n"
        f"  - SacreBLEU  : {report['bleu']:.2f} | precisions={m['precisions']} | "
        f"BP={m['brevity_penalty']:.4f}\n"
        f"  - Signature  : {m['signature']}\n"
        f"  - Toc do     : {report['num_sentences']} cau trong {report['elapsed_sec']}s "
        f"({report['sentences_per_sec']} cau/giay)\n"
        f"  - Da luu JSON: {out_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
