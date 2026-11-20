"""Tiền xử lý dữ liệu thô IWSLT'15 En–Vi: làm sạch câu, lọc độ dài, lưu vào data/processed/."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

MIN_LEN = 1
MAX_LEN = 100


def clean_sentence(text: str) -> str:
    """Chuẩn hóa Unicode NFC, lowercase, bỏ ký tự thừa, thu gọn khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s.,!?'\"\-()]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_length(en_sentence: str, vi_sentence: str) -> bool:
    """Kiểm tra cả 2 câu có nằm trong [MIN_LEN, MAX_LEN] từ không."""
    en_len = len(en_sentence.split())
    vi_len = len(vi_sentence.split())
    return (MIN_LEN <= en_len <= MAX_LEN) and (MIN_LEN <= vi_len <= MAX_LEN)


def preprocess_split(
    en_path: Path, vi_path: Path, out_en_path: Path, out_vi_path: Path
) -> dict:
    """Đọc 1 cặp file .en/.vi, làm sạch, lọc, lưu kết quả."""
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    vi_lines = vi_path.read_text(encoding="utf-8").splitlines()

    if len(en_lines) != len(vi_lines):
        raise ValueError(
            f"Số dòng không khớp: {en_path.name}={len(en_lines)}, "
            f"{vi_path.name}={len(vi_lines)}"
        )

    total = len(en_lines)
    kept_en, kept_vi = [], []

    for en_raw, vi_raw in zip(en_lines, vi_lines):
        en_clean = clean_sentence(en_raw)
        vi_clean = clean_sentence(vi_raw)
        if not en_clean or not vi_clean:
            continue
        if not is_valid_length(en_clean, vi_clean):
            continue
        kept_en.append(en_clean)
        kept_vi.append(vi_clean)

    out_en_path.parent.mkdir(parents=True, exist_ok=True)
    out_en_path.write_text("\n".join(kept_en), encoding="utf-8")
    out_vi_path.write_text("\n".join(kept_vi), encoding="utf-8")

    return {"total": total, "kept": len(kept_en), "filtered": total - len(kept_en)}


def preprocess_all(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> None:
    """Tiền xử lý toàn bộ 3 tập train/val/test."""
    splits = {
        "train": ("train.en", "train.vi"),
        "val": ("tst2012.en", "tst2012.vi"),
        "test": ("tst2013.en", "tst2013.vi"),
    }

    for split_name, (en_file, vi_file) in splits.items():
        en_path = raw_dir / en_file
        vi_path = raw_dir / vi_file

        if not en_path.exists() or not vi_path.exists():
            print(f"[BỎ QUA] Không tìm thấy file cho split '{split_name}'.")
            continue

        stats = preprocess_split(
            en_path=en_path,
            vi_path=vi_path,
            out_en_path=processed_dir / f"{split_name}.en",
            out_vi_path=processed_dir / f"{split_name}.vi",
        )
        print(
            f"[{split_name:5s}] Tổng: {stats['total']:>7,} | "
            f"Giữ: {stats['kept']:>7,} | Lọc: {stats['filtered']:>5,}"
        )


if __name__ == "__main__":
    preprocess_all()
