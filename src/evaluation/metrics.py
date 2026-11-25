"""SacreBLEU evaluation utilities for English-Vietnamese NMT."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from sacrebleu.metrics import BLEU

from src.data.tokenizer import DEFAULT_MAX_SEQ_LEN, BaseTokenizer, load_tokenizer

DEFAULT_TOKENIZE = "13a"
DEFAULT_SMOOTH_METHOD = "exp"


def _normalize_line(text: str) -> str:
    """Strip outer whitespace and collapse internal whitespace runs to a single space."""
    return " ".join(text.strip().split())


def compute_corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    lowercase: bool = False,
    tokenize: str = DEFAULT_TOKENIZE,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    force: bool = True,
) -> dict[str, Any]:
    """Compute corpus-level SacreBLEU-4, n-gram precisions, brevity penalty, and signature."""
    if not predictions or not references:
        raise ValueError("Danh sach predictions va references khong duoc rong.")

    hypotheses = [_normalize_line(str(p)) for p in predictions]
    if isinstance(references[0], str):
        ref_streams = [[_normalize_line(str(r)) for r in references]]  # type: ignore[arg-type]
    else:
        ref_streams = [
            [_normalize_line(str(r)) for r in stream] for stream in references
        ]

    if not ref_streams or any(len(stream) != len(hypotheses) for stream in ref_streams):
        raise ValueError(
            f"So luong cau khong khop: predictions={len(hypotheses)}, "
            f"references={[len(stream) for stream in ref_streams]}"
        )

    bleu_metric = BLEU(
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        force=force,
    )
    score = bleu_metric.corpus_score(hypotheses, ref_streams)
    sig = str(bleu_metric.get_signature())

    return {
        "bleu": round(float(score.score), 2),
        "bleu_exact": round(float(score.score), 4),
        "precisions": [round(float(p), 2) for p in score.precisions],
        "brevity_penalty": round(float(score.bp), 4),
        "length_ratio": (
            round(float(score.sys_len / score.ref_len), 4)
            if score.ref_len > 0
            else 0.0
        ),
        "sys_len": int(score.sys_len),
        "ref_len": int(score.ref_len),
        "num_sentences": len(hypotheses),
        "tokenize": tokenize,
        "lowercase": lowercase,
        "smooth_method": smooth_method,
        "signature": sig,
    }


compute_bleu = compute_corpus_bleu


def compute_sentence_bleu(
    prediction: str,
    reference: str,
    lowercase: bool = False,
    tokenize: str = DEFAULT_TOKENIZE,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    force: bool = True,
) -> dict[str, Any]:
    """Compute smoothed sentence-level SacreBLEU for individual error analysis & demos."""
    hyp = _normalize_line(prediction)
    ref = _normalize_line(reference)

    bleu_metric = BLEU(
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        effective_order=True,
        force=force,
    )
    score = bleu_metric.sentence_score(hyp, [ref])

    return {
        "bleu": round(float(score.score), 2),
        "bleu_exact": round(float(score.score), 4),
        "precisions": [round(float(p), 2) for p in score.precisions],
        "brevity_penalty": round(float(score.bp), 4),
        "length_ratio": (
            round(float(score.sys_len / score.ref_len), 4)
            if score.ref_len > 0
            else 0.0
        ),
        "sys_len": int(score.sys_len),
        "ref_len": int(score.ref_len),
    }


def decode_and_compute_bleu(
    predicted_ids: Sequence[Sequence[int]] | Any,
    references: Sequence[str] | Sequence[Sequence[int]] | Any,
    tgt_tokenizer: BaseTokenizer,
    lowercase: bool = False,
    tokenize: str = DEFAULT_TOKENIZE,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    force: bool = True,
) -> dict[str, Any]:
    """Decode predicted token IDs via tgt_tokenizer and score against target references."""
    raw_preds = (
        predicted_ids.tolist()
        if hasattr(predicted_ids, "tolist")
        else list(predicted_ids)
    )
    if not raw_preds:
        raise ValueError("predicted_ids khong duoc rong.")

    pred_texts = [
        tgt_tokenizer.decode(seq, skip_special_tokens=True) for seq in raw_preds
    ]

    raw_refs = (
        references.tolist() if hasattr(references, "tolist") else list(references)
    )
    if not raw_refs:
        raise ValueError("references khong duoc rong.")

    if isinstance(raw_refs[0], str):
        ref_texts = [str(r) for r in raw_refs]
    else:
        ref_texts = [
            tgt_tokenizer.decode(seq, skip_special_tokens=True) for seq in raw_refs
        ]

    metrics = compute_corpus_bleu(
        pred_texts,
        ref_texts,
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        force=force,
    )
    metrics["sample_pairs"] = [
        {"prediction": pred_texts[i], "reference": ref_texts[i]}
        for i in range(min(3, len(pred_texts)))
    ]
    return metrics


def evaluate_split_sanity(
    ref_path: Path,
    tgt_tokenizer: BaseTokenizer,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
    seed: int = 42,
) -> dict[str, Any]:
    """Run identity, BPE upper-bound ceiling, brevity-penalty, and word-order sanity checks."""
    if not ref_path.exists():
        raise FileNotFoundError(f"Khong tim thay file tham chieu: {ref_path}")

    refs = ref_path.read_text(encoding="utf-8").splitlines()
    if not refs:
        raise ValueError(f"File tham chieu rong: {ref_path}")

    # 1. Identity test (reference vs reference)
    identity = compute_corpus_bleu(refs, refs, tokenize="13a")
    if identity["bleu"] != 100.0 or identity["brevity_penalty"] != 1.0:
        raise AssertionError(
            f"Identity sanity check that bai: bleu={identity['bleu']}, bp={identity['brevity_penalty']}"
        )

    # 2. Oracle BPE upper-bound (encode with max_len truncation, decode, score against raw refs)
    encoded_ids = [
        tgt_tokenizer.encode(line, add_special_tokens=True, max_len=max_len)
        for line in refs
    ]
    oracle_13a = decode_and_compute_bleu(
        encoded_ids, refs, tgt_tokenizer, tokenize="13a"
    )
    oracle_none = decode_and_compute_bleu(
        encoded_ids, refs, tgt_tokenizer, tokenize="none"
    )

    # 3. Brevity penalty test (truncate every hypothesis to half its words)
    half_hyps = []
    for line in refs:
        words = line.split()
        half_hyps.append(" ".join(words[: max(1, len(words) // 2)]))
    brevity_half = compute_corpus_bleu(half_hyps, refs, tokenize="13a")

    # 4. Word-order permutation test (keep identical bag-of-words, shuffle word order)
    rng = random.Random(seed)
    shuffled_hyps = []
    for line in refs:
        words = line.split()
        rng.shuffle(words)
        shuffled_hyps.append(" ".join(words))
    word_order_shuffled = compute_corpus_bleu(shuffled_hyps, refs, tokenize="13a")

    return {
        "ref_path": str(ref_path),
        "num_sentences": len(refs),
        "identity_13a": identity,
        "oracle_bpe_ceiling_13a": oracle_13a,
        "oracle_bpe_ceiling_none": oracle_none,
        "brevity_penalty_half_len": brevity_half,
        "word_order_shuffled": word_order_shuffled,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiem dinh module danh gia SacreBLEU cho IWSLT'15 En-Vi"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thu muc chua val.vi va test.vi (mac dinh: data/processed)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Thu muc chua bpe_vi.json (mac dinh: data/tokenizer)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Do dai toi da cua chuoi token (mac dinh: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho kiem thu xao tron trat tu tu (mac dinh: 42)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    tgt_tok = load_tokenizer(args.tokenizer_dir / "bpe_vi.json", tokenizer_type="bpe")

    print("=== Kiem dinh Module Danh gia SacreBLEU (IWSLT'15 En-Vi) ===")

    splits_report: dict[str, Any] = {}
    for split_name in ("val", "test"):
        ref_path = args.processed_dir / f"{split_name}.vi"
        res = evaluate_split_sanity(
            ref_path=ref_path,
            tgt_tokenizer=tgt_tok,
            max_len=args.max_len,
            seed=args.seed,
        )
        splits_report[split_name] = res

        print(f"\n[Tap {split_name} ({res['num_sentences']:,} cau) - {ref_path}]")
        for label, key in (
            ("1. Identity (Goc vs Goc)           ", "identity_13a"),
            ("2a. Oracle BPE Ceiling (tok=13a)   ", "oracle_bpe_ceiling_13a"),
            ("2b. Oracle BPE Ceiling (tok=none)  ", "oracle_bpe_ceiling_none"),
            ("3. Cat ngan 50% (Brevity Penalty)  ", "brevity_penalty_half_len"),
            ("4. Xao tron trat tu tu (Permuted)  ", "word_order_shuffled"),
        ):
            m = res[key]
            print(
                f"  - {label}: BLEU={m['bleu']:6.2f} | BP={m['brevity_penalty']:.4f} | "
                f"p={m['precisions']} | len_ratio={m['length_ratio']:.4f}"
            )
        print(f"  - Signature (13a): {res['identity_13a']['signature']}")

    # Illustrative sentence-level BLEU cases
    base_ref = "Hôm nay bầu trời ở Hà Nội rất trong xanh và mát mẻ ."
    sentence_cases = [
        ("exact_match", base_ref, base_ref),
        (
            "one_word_diff",
            "Hôm nay bầu trời tại Hà Nội rất trong xanh và mát mẻ .",
            base_ref,
        ),
        ("half_truncated", "Hôm nay bầu trời ở Hà Nội", base_ref),
        ("word_scrambled", "mẻ mát và xanh trong rất Nội Hà ở trời bầu nay Hôm .", base_ref),
    ]
    sentence_examples: list[dict[str, Any]] = []
    print("\n[Vi du Sentence-level SacreBLEU (effective_order=True)]")
    for case_name, hyp, ref in sentence_cases:
        s_score = compute_sentence_bleu(hyp, ref)
        sentence_examples.append(
            {
                "case": case_name,
                "prediction": hyp,
                "reference": ref,
                "score": s_score,
            }
        )
        print(
            f"  - {case_name:<15}: BLEU={s_score['bleu']:6.2f} | "
            f"BP={s_score['brevity_penalty']:.4f} | p={s_score['precisions']}"
        )

    report = {
        "config": {
            "tokenize": DEFAULT_TOKENIZE,
            "smooth_method": DEFAULT_SMOOTH_METHOD,
            "max_len": args.max_len,
            "seed": args.seed,
        },
        "splits": splits_report,
        "sentence_examples": sentence_examples,
    }

    out_path = args.processed_dir / "metrics_sanity_stats.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDa luu ket qua kiem dinh SacreBLEU tai: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
