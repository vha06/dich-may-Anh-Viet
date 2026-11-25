"""Đánh giá chất lượng bản dịch chuẩn hóa với SacreBLEU (WMT/ACL)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from sacrebleu.metrics import BLEU

from src.data.tokenizer import BaseTokenizer


def _normalize_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng đầu/cuối và liên tiếp trước khi chấm BLEU."""
    return " ".join(text.strip().split())


def _normalize_references(
    references: Sequence[str] | Sequence[Sequence[str]],
    num_hypotheses: int,
) -> list[list[str]]:
    """Chuẩn hóa references về dạng list[list[str]] cho SacreBLEU."""
    if not references:
        raise ValueError("references không được rỗng!")

    first = references[0]
    if isinstance(first, str):
        ref_sets = [[_normalize_text(r) for r in references]]  # type: ignore[arg-type]
    else:
        ref_sets = [[_normalize_text(r) for r in ref_set] for ref_set in references]  # type: ignore[union-attr]

    for idx, ref_set in enumerate(ref_sets):
        if len(ref_set) != num_hypotheses:
            raise ValueError(
                f"Số lượng predictions ({num_hypotheses}) ≠ references[{idx}] ({len(ref_set)})"
            )
    return ref_sets


def _format_bleu_result(result, metric: BLEU) -> dict:
    """Đóng gói kết quả SacreBLEU đầy đủ 8 khóa phục vụ log và báo cáo."""
    return {
        "bleu": round(float(result.score), 2),
        "bleu_exact": float(result.score),
        "precisions": [round(float(p), 2) for p in result.precisions],
        "brevity_penalty": round(float(result.bp), 4),
        "length_ratio": round(float(result.sys_len) / max(int(result.ref_len), 1), 4),
        "sys_len": int(result.sys_len),
        "ref_len": int(result.ref_len),
        "signature": str(metric.get_signature()),
    }


def compute_corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    lowercase: bool = False,
    tokenize: str = "13a",
    smooth_method: str = "exp",
    force: bool = True,
) -> dict:
    """Tính điểm SacreBLEU cấp tập dữ liệu (0.0 – 100.0), trả về dict 8 khóa."""
    if not hypotheses:
        raise ValueError("hypotheses/predictions không được rỗng!")

    hyps = [_normalize_text(h) for h in hypotheses]
    refs = _normalize_references(references, num_hypotheses=len(hyps))

    metric = BLEU(
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        force=force,
    )
    result = metric.corpus_score(hyps, refs)
    return _format_bleu_result(result, metric)


compute_bleu = compute_corpus_bleu


def compute_sentence_bleu(
    hypothesis: str,
    reference: str | Sequence[str],
    lowercase: bool = False,
    tokenize: str = "13a",
    smooth_method: str = "exp",
    force: bool = True,
) -> dict:
    """Tính điểm SacreBLEU cấp câu đơn lẻ (effective_order=True), trả về dict 8 khóa."""
    hyp = _normalize_text(hypothesis)
    if isinstance(reference, str):
        refs = [_normalize_text(reference)]
    else:
        if not reference:
            raise ValueError("reference không được rỗng!")
        refs = [_normalize_text(r) for r in reference]

    metric = BLEU(
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        effective_order=True,
        force=force,
    )
    result = metric.sentence_score(hyp, refs)
    return _format_bleu_result(result, metric)


def decode_and_compute_bleu(
    pred_ids,
    references: Sequence[str] | Sequence[Sequence[str]],
    tokenizer: BaseTokenizer,
    lowercase: bool = False,
    tokenize: str = "13a",
    smooth_method: str = "exp",
    force: bool = True,
) -> dict:
    """Giải mã Tensor/danh sách ID dự đoán bằng tokenizer và chấm SacreBLEU."""
    predictions = [
        tokenizer.decode(row, skip_special_tokens=True) for row in pred_ids
    ]
    return compute_corpus_bleu(
        predictions,
        references,
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        force=force,
    )


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def evaluate_split_sanity(
    processed_dir: Path = Path("data/processed"),
) -> dict:
    """Chạy 5 kịch bản sanity check SacreBLEU trên tập val.vi và test.vi."""
    rng = random.Random(42)
    report: dict = {"splits": {}}

    for split in ("val", "test"):
        ref_path = processed_dir / f"{split}.vi"
        if not ref_path.exists():
            continue

        refs = _read_lines(ref_path)
        empty_hyps = ["" for _ in refs]
        half_hyps = [
            " ".join(line.split()[: max(1, len(line.split()) // 2)]) for line in refs
        ]
        shuffled_hyps = []
        for line in refs:
            words = line.split()
            rng.shuffle(words)
            shuffled_hyps.append(" ".join(words))
        lower_hyps = [line.lower() for line in refs]

        report["splits"][split] = {
            "num_sentences": len(refs),
            "perfect_match": compute_corpus_bleu(refs, refs),
            "empty_predictions": compute_corpus_bleu(empty_hyps, refs),
            "truncated_half": compute_corpus_bleu(half_hyps, refs),
            "word_order_shuffled": compute_corpus_bleu(shuffled_hyps, refs),
            "lowercased": compute_corpus_bleu(lower_hyps, refs),
        }

    out_path = processed_dir / "metrics_sanity_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Đã lưu kết quả sanity check tại: {out_path}")
    for split, data in report["splits"].items():
        print(f"\n=== {split.upper()} ({data['num_sentences']} câu) ===")
        for scenario in (
            "perfect_match",
            "empty_predictions",
            "truncated_half",
            "word_order_shuffled",
            "lowercased",
        ):
            m = data[scenario]
            print(
                f"  {scenario:20s}: BLEU={m['bleu']:6.2f} | "
                f"BP={m['brevity_penalty']:.4f} | P={m['precisions']}"
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm định độ đo SacreBLEU.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_split_sanity(processed_dir=args.processed_dir)
