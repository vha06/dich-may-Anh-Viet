"""Unit tests cho pipeline tiền xử lý dữ liệu (src/data/preprocess.py)."""

import tempfile
import unittest
from pathlib import Path

from src.data.preprocess import clean_text, preprocess_split, read_lines


class TestPreprocess(unittest.TestCase):
    def test_unescape_nested_html_and_preserve_case_punctuation(self):
        raw = "I &amp;apos;d like &quot;AI: 100% / $50;&quot; &#91;OK&#93;!"
        expected = 'I \'d like "AI: 100% / $50;" [OK]!'
        self.assertEqual(clean_text(raw), expected)

    def test_nfd_to_nfc_and_strip_control_chars(self):
        # "ế" dạng tổ hợp NFD: e + \u0302 + \u0301
        nfd = "Tie\u0302\u0301ng Vie\u0323\u0302t\x00\x07   kho\u0302ng  lo\u0302\u0303i"
        cleaned = clean_text(nfd)
        self.assertEqual(cleaned, "Tiếng Việt không lỗi")

    def test_read_lines_does_not_split_on_form_feed_or_u2028(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.txt"
            p.write_text("Hello\x0cworld\nSecond\u2028line\n", encoding="utf-8")
            lines = read_lines(p)
            self.assertEqual(len(lines), 2)
            self.assertEqual(clean_text(lines[0]), "Hello world")
            self.assertEqual(clean_text(lines[1]), "Second line")

    def test_train_filters_but_eval_keeps_all_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_en = tmp_dir / "raw.en"
            raw_vi = tmp_dir / "raw.vi"
            raw_en.write_text(
                "Hello world .\n\nToo long sentence here\nHello world .\n",
                encoding="utf-8",
            )
            raw_vi.write_text(
                "Xin chào .\nTrống bên Anh\nCâu dài\nXin chào .\n",
                encoding="utf-8",
            )

            # is_train=True: lọc rỗng, lọc > 3 từ, lọc trùng lặp
            train_res = preprocess_split(
                raw_en,
                raw_vi,
                tmp_dir / "train.en",
                tmp_dir / "train.vi",
                is_train=True,
                max_words=3,
            )
            self.assertEqual(train_res["total"], 4)
            self.assertEqual(train_res["kept"], 1)
            self.assertEqual(train_res["dropped_empty"], 1)
            self.assertEqual(train_res["dropped_too_long"], 1)
            self.assertEqual(train_res["dropped_duplicate"], 1)

            # is_train=False: giữ nguyên 100% số dòng
            eval_res = preprocess_split(
                raw_en,
                raw_vi,
                tmp_dir / "val.en",
                tmp_dir / "val.vi",
                is_train=False,
                max_words=3,
            )
            self.assertEqual(eval_res["kept"], 4)


if __name__ == "__main__":
    unittest.main()
