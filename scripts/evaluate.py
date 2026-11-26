"""Chấm điểm SacreBLEU chính thức trên tập val/test bằng Beam Search hoặc Greedy Search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from scripts.train import build_model, load_config
from src.data.dataset import build_dataloader
from src.data.tokenizer import create_tokenizer
from src.evaluation.decode import beam_search_decode, greedy_decode
from src.evaluation.metrics import compute_corpus_bleu


def evaluate_split(
    model,
    src_path: str | Path,
    tgt_path: str | Path,
    src_tok,
    tgt_tok,
    max_len: int,
    beam_width: int,
    length_penalty: float,
    device: torch.device,
) -> dict:
    """Giải mã toàn bộ tập nguồn độc lập và chấm SacreBLEU so với Gold Reference gốc."""
    batch_size = 1 if beam_width > 1 else 64
    loader = build_dataloader(
        src_path=src_path,
        tgt_path=tgt_path,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=batch_size,
        shuffle=False,
        use_bucketing=False,
        max_len=max_len,
        pin_memory=(device.type == "cuda"),
    )

    predictions: list[str] = []
    desc = f"Decoding ({Path(src_path).name}, beam={beam_width})"

    for batch in tqdm(loader, desc=desc):
        src_ids = batch["src_ids"]
        src_lengths = batch["src_lengths"]
        if beam_width <= 1:
            preds, _ = greedy_decode(
                model,
                src_ids,
                tgt_tok,
                max_len=max_len,
                device=device,
                src_lengths=src_lengths,
            )
            predictions.extend(preds)
        else:
            preds = beam_search_decode(
                model,
                src_ids,
                tgt_tok,
                beam_width=beam_width,
                max_len=max_len,
                device=device,
                src_lengths=src_lengths,
                length_penalty=length_penalty,
            )
            predictions.extend(preds)

    raw_references = list(loader.dataset.tgt_lines)
    if len(predictions) != len(raw_references):
        raise ValueError(
            f"Số lượng predictions ({len(predictions)}) ≠ references ({len(raw_references)})"
        )

    return compute_corpus_bleu(predictions, raw_references)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chấm điểm SacreBLEU cho mô hình dịch máy")
    parser.add_argument("--config", type=str, default="configs/baseline_lstm.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_seq2seq.pt")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Beam width: {args.beam_width} | Length penalty: {args.length_penalty}")

    tok_dir = Path(cfg["data"]["tokenizer_dir"])
    tok_type = cfg["data"].get("tokenizer_type", "bpe")
    ext = "json" if tok_type == "bpe" else "model"
    prefix = "bpe" if tok_type == "bpe" else "spm"

    src_tok = create_tokenizer(tok_dir / f"{prefix}_en.{ext}", tokenizer_type=tok_type)
    tgt_tok = create_tokenizer(tok_dir / f"{prefix}_vi.{ext}", tokenizer_type=tok_type)

    model = build_model(
        cfg,
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        pad_id=src_tok.pad_id,
        device=device,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint '{args.checkpoint}' (epoch={ckpt.get('epoch')}, val_greedy_bleu={ckpt.get('bleu', 0.0):.2f})")

    splits = ["val", "test"] if args.split == "both" else [args.split]
    results: dict = {"checkpoint": args.checkpoint, "beam_width": args.beam_width, "splits": {}}

    for sp in splits:
        metrics = evaluate_split(
            model=model,
            src_path=cfg["data"][f"{sp}_en"],
            tgt_path=cfg["data"][f"{sp}_vi"],
            src_tok=src_tok,
            tgt_tok=tgt_tok,
            max_len=cfg["data"]["max_len"],
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            device=device,
        )
        results["splits"][sp] = metrics
        print(
            f"\n[{sp.upper()}] SacreBLEU = {metrics['bleu']:.2f} | "
            f"Precisions = {metrics['precisions']} | "
            f"BP = {metrics['brevity_penalty']:.4f} | "
            f"Length ratio = {metrics['length_ratio']:.4f}\n"
            f"  Signature: {metrics['signature']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Đã lưu kết quả tại: {args.output}")


if __name__ == "__main__":
    main()
