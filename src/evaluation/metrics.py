"""Tính chỉ số SacreBLEU chuẩn (corpus, sentence, decode+BLEU) và kiểm định sanity."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sacrebleu

from src.data.tokenizer import BaseTokenizer, load_tokenizer


def _normalize_corpus_refs(
    num_hyps: int, references: Sequence[str] | Sequence[Sequence[str]]
) -> list[list[str]]:
    if references and isinstance(references[0], str):
        ref_streams: list[list[str]] = [[str(x) for x in references]]
    else:
        ref_streams = [[str(x) for x in r] for r in references]

    if not ref_streams or any(len(r) != num_hyps for r in ref_streams):
        raise ValueError(
            f"Số câu hypotheses ({num_hyps}) không khớp với references "
            f"({[len(r) for r in ref_streams]})."
        )
    return ref_streams


def compute_corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    tokenize: str = "13a",
    lowercase: bool = False,
    smooth_method: str = "exp",
) -> dict[str, Any]:
    """Tính corpus SacreBLEU chuẩn cho danh sách câu dịch và câu tham chiếu."""
    hyps = [str(h) for h in hypotheses]
    if not hyps:
        raise ValueError("Danh sách hypotheses không được rỗng.")

    ref_streams = _normalize_corpus_refs(len(hyps), references)
    metric = sacrebleu.BLEU(
        tokenize=tokenize,
        lowercase=lowercase,
        smooth_method=smooth_method,
        force=True,
    )
    bleu = metric.corpus_score(hyps, ref_streams)
    bp = round(float(bleu.bp), 4)
    sys_len = int(bleu.sys_len)
    ref_len = int(bleu.ref_len)
    return {
        "bleu": round(float(bleu.score), 2),
        "bleu_exact": float(bleu.score),
        "precisions": [round(float(p), 2) for p in bleu.precisions],
        "brevity_penalty": bp,
        "bp": bp,
        "length_ratio": round(sys_len / max(1, ref_len), 4),
        "sys_len": sys_len,
        "ref_len": ref_len,
        "signature": str(metric.get_signature()),
    }


compute_bleu = compute_corpus_bleu


def compute_sentence_bleu(
    hypothesis: str,
    references: str | Sequence[str],
    tokenize: str = "13a",
    lowercase: bool = False,
    smooth_method: str = "exp",
) -> dict[str, Any]:
    """Tính sentence-level SacreBLEU (effective_order=True) phục vụ phân tích lỗi từng câu."""
    ref_list = [references] if isinstance(references, str) else [str(r) for r in references]
    if not ref_list:
        raise ValueError("Danh sách references không được rỗng.")

    metric = sacrebleu.BLEU(
        tokenize=tokenize,
        lowercase=lowercase,
        smooth_method=smooth_method,
        effective_order=True,
        force=True,
    )
    bleu = metric.sentence_score(str(hypothesis), ref_list)
    bp = round(float(bleu.bp), 4)
    sys_len = int(bleu.sys_len)
    ref_len = int(bleu.ref_len)
    return {
        "bleu": round(float(bleu.score), 2),
        "bleu_exact": float(bleu.score),
        "precisions": [round(float(p), 2) for p in bleu.precisions],
        "brevity_penalty": bp,
        "bp": bp,
        "length_ratio": round(sys_len / max(1, ref_len), 4),
        "sys_len": sys_len,
        "ref_len": ref_len,
        "signature": str(metric.get_signature()),
    }


def decode_and_compute_bleu(
    pred_ids: Sequence[Sequence[int]] | Any,
    references: Sequence[str] | Sequence[Sequence[str]],
    tokenizer: BaseTokenizer,
    tokenize: str = "13a",
    lowercase: bool = False,
    smooth_method: str = "exp",
) -> dict[str, Any]:
    """Giải mã danh sách/tensor IDs dự đoán bằng tokenizer rồi chấm corpus SacreBLEU."""
    seq_list = pred_ids.tolist() if hasattr(pred_ids, "tolist") else list(pred_ids)
    hyps = [tokenizer.decode(seq, skip_special_tokens=True) for seq in seq_list]
    return compute_corpus_bleu(
        hyps,
        references,
        tokenize=tokenize,
        lowercase=lowercase,
        smooth_method=smooth_method,
    )


def evaluate_split_sanity(
    ref_path: Path,
    tokenizer: BaseTokenizer,
    max_len: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Chạy 4 bài kiểm định sanity trên 1 tập dữ liệu (val/test)."""
    refs = ref_path.read_text(encoding="utf-8").splitlines()

    # 1. Identity (Gốc vs Gốc)
    identity = compute_corpus_bleu(refs, refs)

    # 2. Oracle BPE Ceiling (encode + truncate max_len + decode)
    oracle_ids = [
        tokenizer.encode(line, add_special_tokens=True, max_len=max_len) for line in refs
    ]
    oracle_bpe = decode_and_compute_bleu(oracle_ids, refs, tokenizer)

    # 3. Cắt ngắn 50% độ dài (kiểm tra phạt Brevity Penalty)
    half_hyps = [" ".join(line.split()[: max(1, len(line.split()) // 2)]) for line in refs]
    half_truncated = compute_corpus_bleu(half_hyps, refs)

    # 4. Xáo trộn trật tự từ
    rng = random.Random(seed)
    scrambled_hyps: list[str] = []
    for line in refs:
        words = line.split()
        rng.shuffle(words)
        scrambled_hyps.append(" ".join(words))
    word_scrambled = compute_corpus_bleu(scrambled_hyps, refs)

    return {
        "num_sentences": len(refs),
        "identity": identity,
        "oracle_bpe": oracle_bpe,
        "half_truncated": half_truncated,
        "word_scrambled": word_scrambled,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm định sanity SacreBLEU chuẩn nhóm")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/metrics_sanity_stats.json"),
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    tgt_tok = load_tokenizer(args.tokenizer_dir / "bpe_vi.json")

    splits_report: dict[str, Any] = {}
    print("=== Kiểm định Sanity SacreBLEU (IWSLT'15 En-Vi) ===")
    for split in ("val", "test"):
        res = evaluate_split_sanity(
            args.processed_dir / f"{split}.vi",
            tgt_tok,
            max_len=args.max_len,
            seed=args.seed,
        )
        splits_report[split] = res
        print(f"\n[{split.upper()}] ({res['num_sentences']:,} câu)")
        for key in ("identity", "oracle_bpe", "half_truncated", "word_scrambled"):
            m = res[key]
            print(
                f"  - {key:<14}: bleu={m['bleu']:6.2f} | bp={m['brevity_penalty']:.4f} | "
                f"len={m['sys_len']:,}/{m['ref_len']:,} | prec={m['precisions']}"
            )

    sent_ref = "Hôm nay bầu trời ở Hà Nội rất trong xanh và mát mẻ ."
    sent_words = sent_ref.split()
    sent_rng = random.Random(args.seed)
    sent_scrambled = list(sent_words)
    sent_rng.shuffle(sent_scrambled)

    sentence_tests = {
        "exact_match": compute_sentence_bleu(sent_ref, sent_ref),
        "one_word_diff": compute_sentence_bleu(sent_ref.replace(" ở ", " tại "), sent_ref),
        "half_truncated": compute_sentence_bleu(
            " ".join(sent_words[: len(sent_words) // 2]), sent_ref
        ),
        "word_scrambled": compute_sentence_bleu(" ".join(sent_scrambled), sent_ref),
    }

    print("\n[SENTENCE LEVEL]")
    for name, m in sentence_tests.items():
        print(f"  - {name:<14}: bleu={m['bleu']:6.2f} | bp={m['brevity_penalty']:.4f}")
    print(f"\nSignature: {splits_report['val']['identity']['signature']}")

    full_report = {
        "config": {"max_len": args.max_len, "seed": args.seed, "tokenize": "13a", "lowercase": False},
        "splits": splits_report,
        "sentence_tests": sentence_tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu báo cáo tại: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
