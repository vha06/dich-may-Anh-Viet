"""Download the Stanford NMT IWSLT'15 En-Vi dataset files and verify integrity."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATASET_PAGE_URL = "https://huggingface.co/datasets/dtruong46me/standford-nmt-iwslt15"
BASE_URL = f"{DATASET_PAGE_URL}/resolve/main"
SPLITS = {
    "train": ("train.en", "train.vi"),
    "val (tst2012)": ("tst2012.en", "tst2012.vi"),
    "test (tst2013)": ("tst2013.en", "tst2013.vi"),
}
FILES = [fname for pair in SPLITS.values() for fname in pair]


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a single file from URL to dest. Return True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, tmp_dest.open("wb") as out_f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
        tmp_dest.replace(dest)
    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise
    return True


def count_lines(path: Path) -> int:
    """Count lines in a UTF-8 text file using splitlines() to match text parsing."""
    with path.open("r", encoding="utf-8") as f:
        return len(f.read().splitlines())


def verify_raw_files(raw_dir: Path) -> None:
    """Verify all 6 files exist, are non-empty, and have matching En-Vi line counts."""
    print("\n=== Kiem tra tinh toan ven du lieu goc (data/raw) ===")
    print(f"{'Split':<16} {'File':<14} {'Size (KB)':>12} {'Lines':>10}")
    print("-" * 56)

    for split_name, (en_name, vi_name) in SPLITS.items():
        en_path = raw_dir / en_name
        vi_path = raw_dir / vi_name
        if not en_path.exists() or not vi_path.exists():
            raise FileNotFoundError(f"Thieu file cho tap {split_name}: {en_name} / {vi_name}")

        en_lines = count_lines(en_path)
        vi_lines = count_lines(vi_path)
        en_kb = en_path.stat().st_size / 1024
        vi_kb = vi_path.stat().st_size / 1024

        print(f"{split_name:<16} {en_name:<14} {en_kb:>12.1f} {en_lines:>10,}")
        print(f"{'':<16} {vi_name:<14} {vi_kb:>12.1f} {vi_lines:>10,}")

        if en_lines != vi_lines:
            raise ValueError(
                f"So dong khong khop o tap {split_name}: {en_name}={en_lines}, {vi_name}={vi_lines}"
            )
        if en_lines == 0:
            raise ValueError(f"Tap {split_name} bi rong (0 dong).")

    print("-" * 56)
    print("OK: Tat ca 6 file hop le va so dong Anh-Viet khop nhau 100%.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tai bo du lieu IWSLT'15 En-Vi (Stanford NMT) ve data/raw/"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Thu muc luu du lieu tho (mac dinh: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tai lai ngay ca khi file da ton tai",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    raw_dir: Path = args.raw_dir

    print(f"Thu muc luu du lieu tho: {raw_dir.resolve()}")
    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        dest = raw_dir / filename
        try:
            downloaded = download_file(url, dest, force=args.force)
            status = "Da tai xong" if downloaded else "Da ton tai (bo qua)"
            print(f"[{status}] {filename} ({dest.stat().st_size:,} bytes)")
        except urllib.error.URLError as exc:
            print(f"[LOI] Khong the tai {url}: {exc}", file=sys.stderr)
            return 1

    verify_raw_files(raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
