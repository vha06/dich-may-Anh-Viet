"""Đánh giá chất lượng bản dịch chuẩn hóa với SacreBLEU (WMT/ACL)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from sacrebleu.metrics import BLEU

from src.data.tokenizer import BaseTokenizer, load_tokenizer


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
    tokenizer_path: Path = Path("data/tokenizer/bpe_vi.json"),
    max_len: int = 128,
    seed: int = 42,
) -> dict:
    """Chạy bộ kiểm định SacreBLEU chuẩn nhóm trên tập val.vi, test.vi và mức câu đơn."""
    tgt_tok = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else None
    report: dict = {
        "config": {"max_len": max_len, "seed": seed, "tokenizer": str(tokenizer_path)},
        "splits": {},
        "sentence_sanity": {},
    }

    for split in ("val", "test"):
        ref_path = processed_dir / f"{split}.vi"
        if not ref_path.exists():
            continue

        refs = _read_lines(ref_path)
        rng = random.Random(seed)

        half_hyps = [
            " ".join(line.split()[: max(1, len(line.split()) // 2)]) for line in refs
        ]
        shuffled_hyps = []
        for line in refs:
            words = line.split()
            rng.shuffle(words)
            shuffled_hyps.append(" ".join(words))

        split_res: dict = {
            "num_sentences": len(refs),
            "identity": compute_corpus_bleu(refs, refs),
        }
        if tgt_tok is not None:
            oracle_hyps = [
                tgt_tok.decode(
                    tgt_tok.encode(line, add_special_tokens=True, max_len=max_len),
                    skip_special_tokens=True,
                )
                for line in refs
            ]
            split_res["oracle_bpe_ceiling"] = compute_corpus_bleu(oracle_hyps, refs)

        split_res["truncated_half"] = compute_corpus_bleu(half_hyps, refs)
        split_res["word_order_shuffled"] = compute_corpus_bleu(shuffled_hyps, refs)
        report["splits"][split] = split_res

    # 4 test mức câu đơn chuẩn
    sent_ref = "Hôm nay bầu trời ở Hà Nội rất trong xanh và mát mẻ ."
    sent_words = sent_ref.split()
    sent_half = " ".join(sent_words[: max(1, len(sent_words) // 2)])
    sent_scrambled_words = list(sent_words)
    random.Random(seed).shuffle(sent_scrambled_words)
    sent_scrambled = " ".join(sent_scrambled_words)

    report["sentence_sanity"] = {
        "reference": sent_ref,
        "exact_match": compute_sentence_bleu(sent_ref, sent_ref),
        "one_word_diff": compute_sentence_bleu(sent_ref.replace("ở", "tại"), sent_ref),
        "half_truncated": compute_sentence_bleu(sent_half, sent_ref),
        "word_scrambled": compute_sentence_bleu(sent_scrambled, sent_ref),
    }

    out_path = processed_dir / "metrics_sanity_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Đã lưu kết quả sanity check tại: {out_path}")
    for split, data in report["splits"].items():
        print(f"\n=== {split.upper()} ({data['num_sentences']} câu) ===")
        for scenario in ("identity", "oracle_bpe_ceiling", "truncated_half", "word_order_shuffled"):
            if scenario not in data:
                continue
            m = data[scenario]
            print(
                f"  {scenario:20s}: BLEU={m['bleu']:6.2f} | BP={m['brevity_penalty']:.4f} | "
                f"len={m['sys_len']}/{m['ref_len']} | P={m['precisions']}"
            )

    print("\n=== SENTENCE SANITY ===")
    for k in ("exact_match", "one_word_diff", "half_truncated", "word_scrambled"):
        m = report["sentence_sanity"][k]
        print(f"  {k:16s}: BLEU={m['bleu']:6.2f} | BP={m['brevity_penalty']:.4f}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm định độ đo SacreBLEU.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/tokenizer/bpe_vi.json"))
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_split_sanity(
        processed_dir=args.processed_dir,
        tokenizer_path=args.tokenizer_path,
        max_len=args.max_len,
        seed=args.seed,
    )
