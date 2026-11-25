"""Preprocess and analyze the IWSLT'15 En-Vi dataset."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Decode HTML entities, normalize Vietnamese Unicode to NFC, strip control chars, and collapse spaces."""
    # Unescape twice in case of nested entities such as &amp;apos;
    cleaned = html.unescape(html.unescape(text))
    cleaned = unicodedata.normalize("NFC", cleaned)
    # Remove invisible Unicode control/format chars while keeping whitespace to avoid word merging
    cleaned = "".join(
        ch for ch in cleaned if unicodedata.category(ch)[0] != "C" or ch.isspace()
    )
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _percentile(sorted_values: list[int], q: float) -> float:
    """Compute linear-interpolated percentile (q in [0, 100]) on sorted non-empty integers."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0, "max": 0}
    sorted_lens = sorted(lengths)
    return {
        "mean": round(sum(sorted_lens) / len(sorted_lens), 2),
        "median": round(_percentile(sorted_lens, 50), 2),
        "p95": round(_percentile(sorted_lens, 95), 2),
        "p99": round(_percentile(sorted_lens, 99), 2),
        "min": sorted_lens[0],
        "max": sorted_lens[-1],
    }


def compute_split_stats(en_lines: list[str], vi_lines: list[str]) -> dict[str, Any]:
    """Compute word-count and length-ratio statistics for parallel En-Vi sentences."""
    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Line count mismatch: en={len(en_lines)}, vi={len(vi_lines)}")
    en_lengths = [len(line.split()) for line in en_lines]
    vi_lengths = [len(line.split()) for line in vi_lines]

    ratios: list[float] = []
    abnormal_count = 0
    for l_en, l_vi in zip(en_lengths, vi_lengths):
        if l_en > 0 and l_vi > 0:
            ratio = l_vi / l_en
            ratios.append(ratio)
            if (ratio > 3.0 or ratio < 0.33) and (l_en >= 4 or l_vi >= 4):
                abnormal_count += 1

    ratio_stats = {
        "mean_ratio": round(sum(ratios) / len(ratios), 2) if ratios else 0.0,
        "abnormal_count": abnormal_count,
        "abnormal_pct": round(abnormal_count / len(ratios) * 100, 2) if ratios else 0.0,
    }

    return {
        "num_sentences": len(en_lines),
        "en_word_len": _length_summary(en_lengths),
        "vi_word_len": _length_summary(vi_lengths),
        "length_ratio": ratio_stats,
    }


def process_split(
    src_path: Path,
    tgt_path: Path,
    out_src_path: Path,
    out_tgt_path: Path,
    is_train: bool = False,
    max_words: int | None = 128,
    remove_duplicates: bool = True,
) -> dict[str, Any]:
    """Clean a parallel split and write processed outputs.

    For validation and test splits (is_train=False), every sentence pair is preserved
    to ensure fair and standardized BLEU evaluation. Pass `max_words <= 0` or `None`
    to disable length filtering on the training split.
    """
    with src_path.open("r", encoding="utf-8") as f_src:
        raw_src_lines = [line.rstrip("\r\n") for line in f_src]
    with tgt_path.open("r", encoding="utf-8") as f_tgt:
        raw_tgt_lines = [line.rstrip("\r\n") for line in f_tgt]

    if len(raw_src_lines) != len(raw_tgt_lines):
        raise ValueError(
            f"So dong nguon va dich khong khop ({src_path.name}={len(raw_src_lines)}, "
            f"{tgt_path.name}={len(raw_tgt_lines)})"
        )

    effective_max_words = max_words if (max_words is not None and max_words > 0) else None

    cleaned_src: list[str] = []
    cleaned_tgt: list[str] = []
    samples: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    dropped_empty = 0
    dropped_duplicate = 0
    dropped_too_long = 0

    for raw_s, raw_t in zip(raw_src_lines, raw_tgt_lines):
        clean_s = clean_text(raw_s)
        clean_t = clean_text(raw_t)

        if is_train:
            if not clean_s or not clean_t:
                dropped_empty += 1
                continue
            if effective_max_words is not None and (
                len(clean_s.split()) > effective_max_words
                or len(clean_t.split()) > effective_max_words
            ):
                dropped_too_long += 1
                continue
            pair = (clean_s, clean_t)
            if remove_duplicates:
                if pair in seen_pairs:
                    dropped_duplicate += 1
                    continue
                seen_pairs.add(pair)

        if len(samples) < 3 and (raw_s != clean_s or raw_t != clean_t):
            samples.append(
                {
                    "raw_en": raw_s,
                    "clean_en": clean_s,
                    "raw_vi": raw_t,
                    "clean_vi": clean_t,
                }
            )

        cleaned_src.append(clean_s)
        cleaned_tgt.append(clean_t)

    out_src_path.parent.mkdir(parents=True, exist_ok=True)
    out_tgt_path.parent.mkdir(parents=True, exist_ok=True)
    out_src_path.write_text("\n".join(cleaned_src) + "\n", encoding="utf-8")
    out_tgt_path.write_text("\n".join(cleaned_tgt) + "\n", encoding="utf-8")

    stats = compute_split_stats(cleaned_src, cleaned_tgt)
    stats["raw_num_sentences"] = len(raw_src_lines)
    stats["dropped"] = {
        "empty": dropped_empty,
        "duplicate": dropped_duplicate,
        "too_long": dropped_too_long,
    }
    stats["samples"] = samples
    stats["pairs"] = set(zip(cleaned_src, cleaned_tgt))
    stats["src_set"] = set(cleaned_src)
    stats["tgt_set"] = set(cleaned_tgt)
    return stats


def verify_split_separation(
    split_pairs: dict[str, set[tuple[str, str]]],
    split_srcs: dict[str, set[str]],
    split_tgts: dict[str, set[str]],
) -> dict[str, Any]:
    """Check 3-level overlap (pairs, source_only, target_only) across train, val, and test splits."""
    val_pairs = split_pairs["val"]
    test_pairs = split_pairs["test"]
    train_pairs = split_pairs["train"]

    overlap = {
        "pairs": {
            "val_test": len(val_pairs & test_pairs),
            "train_val": len(train_pairs & val_pairs),
            "train_test": len(train_pairs & test_pairs),
        },
        "source_only": {
            "val_test": len(split_srcs["val"] & split_srcs["test"]),
            "train_val": len(split_srcs["train"] & split_srcs["val"]),
            "train_test": len(split_srcs["train"] & split_srcs["test"]),
        },
        "target_only": {
            "val_test": len(split_tgts["val"] & split_tgts["test"]),
            "train_val": len(split_tgts["train"] & split_tgts["val"]),
            "train_test": len(split_tgts["train"] & split_tgts["test"]),
        },
    }
    if val_pairs == test_pairs:
        raise ValueError("Canh bao: tap val va test bi trung nhau hoan toan!")
    return overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tien xu ly va thong ke bo du lieu IWSLT'15 En-Vi"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Thu muc chua du lieu tho (mac dinh: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thu muc luu du lieu da lam sach (mac dinh: data/processed)",
    )
    parser.add_argument(
        "--max-words",
        dest="max_words",
        type=int,
        default=128,
        help="Loc bo cac cau o tap train co do dai > max-words tu (mac dinh: 128; truyen <= 0 de tat loc)",
    )
    parser.add_argument(
        "--max-len",
        dest="max_words",
        type=int,
        help="Alias cho --max-words",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Giu lai cac cap cau trung lap trong tap train",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    effective_max_words = (
        args.max_words if (args.max_words is not None and args.max_words > 0) else None
    )

    split_specs = [
        ("train", "train.en", "train.vi", "train.en", "train.vi", True),
        ("val", "tst2012.en", "tst2012.vi", "val.en", "val.vi", False),
        ("test", "tst2013.en", "tst2013.vi", "test.en", "test.vi", False),
    ]

    all_stats: dict[str, Any] = {}
    split_pairs: dict[str, set[tuple[str, str]]] = {}
    split_srcs: dict[str, set[str]] = {}
    split_tgts: dict[str, set[str]] = {}
    sample_preview: list[dict[str, str]] = []

    for split_name, raw_en, raw_vi, out_en, out_vi, is_train in split_specs:
        res = process_split(
            src_path=raw_dir / raw_en,
            tgt_path=raw_dir / raw_vi,
            out_src_path=processed_dir / out_en,
            out_tgt_path=processed_dir / out_vi,
            is_train=is_train,
            max_words=effective_max_words,
            remove_duplicates=not args.keep_duplicates,
        )
        split_pairs[split_name] = res.pop("pairs")
        split_srcs[split_name] = res.pop("src_set")
        split_tgts[split_name] = res.pop("tgt_set")
        samples = res.pop("samples")
        if split_name == "val" or not sample_preview:
            sample_preview = samples or sample_preview
        all_stats[split_name] = res

    overlap = verify_split_separation(split_pairs, split_srcs, split_tgts)
    report = {
        "config": {
            "max_words": effective_max_words,
            "remove_duplicates": not args.keep_duplicates,
        },
        "splits": all_stats,
        "overlap_check": overlap,
    }

    stats_path = processed_dir / "stats.json"
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Bao cao Tien xu ly du lieu IWSLT'15 En-Vi ===")
    for name, s in all_stats.items():
        en_s = s["en_word_len"]
        vi_s = s["vi_word_len"]
        drop = s["dropped"]
        ratio_s = s["length_ratio"]
        print(
            f"\n[{name.upper()}] Goc: {s['raw_num_sentences']:,} -> Sau lam sach: {s['num_sentences']:,} cap cau"
        )
        if name == "train":
            len_label = f" (>{effective_max_words} tu)" if effective_max_words is not None else ""
            print(
                f"  Da loai bo -> rong: {drop['empty']}, trung lap: {drop['duplicate']}, "
                f"qua dai{len_label}: {drop['too_long']}"
            )
        print(
            f"  Do dai EN (tu): mean={en_s['mean']}, median={en_s['median']}, "
            f"p95={en_s['p95']}, p99={en_s['p99']}, min={en_s['min']}, max={en_s['max']}"
        )
        print(
            f"  Do dai VI (tu): mean={vi_s['mean']}, median={vi_s['median']}, "
            f"p95={vi_s['p95']}, p99={vi_s['p99']}, min={vi_s['min']}, max={vi_s['max']}"
        )
        print(
            f"  Ty le do dai (VI/EN): trung binh={ratio_s['mean_ratio']}, "
            f"lech bat thuong (>3x hoac <0.33x): {ratio_s['abnormal_count']} cau ({ratio_s['abnormal_pct']}%)"
        )

    print("\n=== Kiem tra phan chia tap du lieu (Overlap Check 3 tang) ===")
    print("  Trung ca cap (pairs):")
    print(f"    val vs test  : {overlap['pairs']['val_test']} cap cau")
    print(f"    train vs val : {overlap['pairs']['train_val']} cap cau")
    print(f"    train vs test: {overlap['pairs']['train_test']} cap cau")
    print("  Trung cau nguon EN (source_only):")
    print(f"    val vs test  : {overlap['source_only']['val_test']} cau")
    print(f"    train vs val : {overlap['source_only']['train_val']} cau")
    print(f"    train vs test: {overlap['source_only']['train_test']} cau")
    print("  Trung cau dich VI (target_only):")
    print(f"    val vs test  : {overlap['target_only']['val_test']} cau")
    print(f"    train vs val : {overlap['target_only']['train_val']} cau")
    print(f"    train vs test: {overlap['target_only']['train_test']} cau")

    print("\n=== 3 cap cau mau truoc va sau khi lam sach ===")
    for idx, sample in enumerate(sample_preview[:3], start=1):
        print(f"Mau {idx}:")
        print(f"  [Raw EN]  : {sample['raw_en']}")
        print(f"  [Clean EN]: {sample['clean_en']}")
        print(f"  [Raw VI]  : {sample['raw_vi']}")
        print(f"  [Clean VI]: {sample['clean_vi']}")

    print(f"\nDa luu thong ke chi tiet tai: {stats_path}")

    assert all_stats["val"]["num_sentences"] == 1553, "Loi: Tap val khong du 1,553 dong!"
    assert all_stats["test"]["num_sentences"] == 1268, "Loi: Tap test khong du 1,268 dong!"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
