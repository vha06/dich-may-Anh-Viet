"""Tiền xử lý dữ liệu thô IWSLT'15 En–Vi: giải mã HTML, NFC, lọc nhiễu và thống kê."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import unicodedata
from pathlib import Path

DEFAULT_MAX_WORDS = 128
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Giải mã HTML entities 2 lần, chuẩn hóa NFC, bỏ ký tự điều khiển ẩn và gom khoảng trắng."""
    cleaned = html.unescape(html.unescape(text))
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = "".join(
        ch for ch in cleaned if unicodedata.category(ch)[0] != "C" or ch.isspace()
    )
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def read_lines(path: Path) -> list[str]:
    """Đọc dòng theo đúng \\n/\\r\\n, tránh tách nhầm \\x0c hay \\u2028 như splitlines()."""
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def write_lines(path: Path, lines: list[str]) -> None:
    """Ghi danh sách dòng ra file theo chuẩn POSIX (có \\n ở cuối file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(content, encoding="utf-8")


def _percentile(sorted_vals: list[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compute_length_stats(lines: list[str]) -> dict[str, float | int]:
    """Tính mean, median, p95, p99, min, max theo số từ."""
    if not lines:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0, "max": 0}
    lengths = sorted(len(line.split()) for line in lines)
    return {
        "mean": round(statistics.mean(lengths), 2),
        "median": round(statistics.median(lengths), 2),
        "p95": round(_percentile(lengths, 95), 2),
        "p99": round(_percentile(lengths, 99), 2),
        "min": lengths[0],
        "max": lengths[-1],
    }


def preprocess_split(
    en_path: Path,
    vi_path: Path,
    out_en_path: Path,
    out_vi_path: Path,
    is_train: bool = False,
    max_words: int | None = DEFAULT_MAX_WORDS,
    keep_duplicates: bool = False,
) -> dict:
    """Làm sạch 1 cặp file .en/.vi. Chỉ lọc rỗng, độ dài và trùng lặp khi is_train=True."""
    en_lines = read_lines(en_path)
    vi_lines = read_lines(vi_path)

    if len(en_lines) != len(vi_lines):
        raise ValueError(
            f"Số dòng không khớp: {en_path.name}={len(en_lines)}, "
            f"{vi_path.name}={len(vi_lines)}"
        )

    total = len(en_lines)
    kept_en: list[str] = []
    kept_vi: list[str] = []
    samples: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    dropped_empty = 0
    dropped_too_long = 0
    dropped_duplicate = 0

    for raw_s, raw_t in zip(en_lines, vi_lines):
        clean_s = clean_text(raw_s)
        clean_t = clean_text(raw_t)

        if is_train:
            if not clean_s or not clean_t:
                dropped_empty += 1
                continue

            len_s = len(clean_s.split())
            len_t = len(clean_t.split())
            if max_words is not None and max_words > 0 and (len_s > max_words or len_t > max_words):
                dropped_too_long += 1
                continue

            if not keep_duplicates:
                pair = (clean_s, clean_t)
                if pair in seen_pairs:
                    dropped_duplicate += 1
                    continue
                seen_pairs.add(pair)

        kept_en.append(clean_s)
        kept_vi.append(clean_t)

        if len(samples) < 3 and (raw_s != clean_s or raw_t != clean_t):
            samples.append({
                "raw_en": raw_s,
                "clean_en": clean_s,
                "raw_vi": raw_t,
                "clean_vi": clean_t,
            })

    if not is_train and len(kept_en) != total:
        raise AssertionError(f"Tập non-train bị rơi dòng: {len(kept_en)} != {total}")

    write_lines(out_en_path, kept_en)
    write_lines(out_vi_path, kept_vi)

    en_stats = compute_length_stats(kept_en)
    vi_stats = compute_length_stats(kept_vi)
    ratio = round(vi_stats["mean"] / en_stats["mean"], 4) if en_stats["mean"] > 0 else 0.0

    return {
        "total": total,
        "kept": len(kept_en),
        "dropped_empty": dropped_empty,
        "dropped_too_long": dropped_too_long,
        "dropped_duplicate": dropped_duplicate,
        "en_stats": en_stats,
        "vi_stats": vi_stats,
        "length_ratio": ratio,
        "samples": samples,
        "_pairs": list(zip(kept_en, kept_vi)),
    }


def compute_overlap(
    split_pairs: dict[str, list[tuple[str, str]]],
) -> dict[str, dict[str, int]]:
    """Kiểm tra rò rỉ dữ liệu giữa các tập ở 3 mức: pairs, source_only, target_only."""
    if "val" in split_pairs and "test" in split_pairs:
        if set(split_pairs["val"]) == set(split_pairs["test"]):
            raise ValueError("Phát hiện tập val và test trùng hệt nhau!")

    overlap: dict[str, dict[str, int]] = {}
    names = list(split_pairs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa = set(split_pairs[a])
            pb = set(split_pairs[b])
            sa = {s for s, _ in split_pairs[a]}
            sb = {s for s, _ in split_pairs[b]}
            ta = {t for _, t in split_pairs[a]}
            tb = {t for _, t in split_pairs[b]}
            overlap[f"{a}_vs_{b}"] = {
                "pairs": len(pa & pb),
                "source_only": len(sa & sb),
                "target_only": len(ta & tb),
            }
    return overlap


def preprocess_all(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    max_words: int | None = DEFAULT_MAX_WORDS,
    keep_duplicates: bool = False,
) -> dict:
    """Tiền xử lý toàn bộ 3 tập train/val/test, kiểm tra toàn vẹn và xuất stats.json."""
    splits = {
        "train": ("train.en", "train.vi", True),
        "val": ("tst2012.en", "tst2012.vi", False),
        "test": ("tst2013.en", "tst2013.vi", False),
    }

    report: dict = {
        "config": {
            "max_words": max_words,
            "keep_duplicates": keep_duplicates,
            "lowercase": False,
        },
        "splits": {},
    }
    split_pairs: dict[str, list[tuple[str, str]]] = {}

    for split_name, (en_file, vi_file, is_train) in splits.items():
        en_path = raw_dir / en_file
        vi_path = raw_dir / vi_file

        if not en_path.exists() or not vi_path.exists():
            print(f"[BỎ QUA] Không tìm thấy file cho split '{split_name}'.")
            continue

        res = preprocess_split(
            en_path=en_path,
            vi_path=vi_path,
            out_en_path=processed_dir / f"{split_name}.en",
            out_vi_path=processed_dir / f"{split_name}.vi",
            is_train=is_train,
            max_words=max_words,
            keep_duplicates=keep_duplicates,
        )

        pairs = res.pop("_pairs")
        samples = res.pop("samples")
        split_pairs[split_name] = pairs
        report["splits"][split_name] = res

        print(
            f"[{split_name:5s}] Tổng: {res['total']:>7,} | Giữ: {res['kept']:>7,} | "
            f"Rỗng: {res['dropped_empty']:>3} | Dài: {res['dropped_too_long']:>3} | "
            f"Trùng: {res['dropped_duplicate']:>4} | Ratio(VI/EN): {res['length_ratio']:.2f}"
        )

        if split_name == "train" and samples:
            print("\n=== 3 cặp câu mẫu trước / sau khi làm sạch (train) ===")
            for idx, sm in enumerate(samples, 1):
                print(f"  [{idx}] RAW EN  : {sm['raw_en'][:100]}")
                print(f"      CLEAN EN: {sm['clean_en'][:100]}")
                print(f"      RAW VI  : {sm['raw_vi'][:100]}")
                print(f"      CLEAN VI: {sm['clean_vi'][:100]}")
            print()

    report["overlap_check"] = compute_overlap(split_pairs)

    if "val" in report["splits"]:
        assert report["splits"]["val"]["kept"] == report["splits"]["val"]["total"]
    if "test" in report["splits"]:
        assert report["splits"]["test"]["kept"] == report["splits"]["test"]["total"]

    stats_path = processed_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã lưu thống kê tại: {stats_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu IWSLT'15 En–Vi.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help="Số từ tối đa trên tập train (nhập 0 để tắt lọc độ dài).",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Giữ lại các cặp câu trùng lặp trên tập train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_words = None if args.max_words <= 0 else args.max_words
    preprocess_all(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_words=max_words,
        keep_duplicates=args.keep_duplicates,
    )


if __name__ == "__main__":
    main()
