"""Unit test cho pipeline dữ liệu: python3 -m unittest tests/test_data_pipeline.py"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from src.data.preprocess import clean_text, process_split

HAS_TORCH = torch is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
TOK_DIR = Path("data/tokenizer")


class StubTokenizer:
    """Tokenizer giả: mỗi từ -> 1 id (=10), để test không phụ thuộc file BPE."""

    pad_id, unk_id, sos_id, eos_id = 0, 1, 2, 3

    def encode(self, text, add_special_tokens=True):
        ids = [10] * len(text.split())
        return [2] + ids + [3] if add_special_tokens else ids


class TestPreprocess(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("life &amp;apos;s  work ."), "life 's work .")
        self.assertEqual(clean_text("ho\u0323c"), "học")  # NFD -> NFC
        self.assertEqual(clean_text("a\u200bb\x00c\t d"), "abc d")

    def test_process_split_keeps_alignment_with_odd_line_breaks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # \x0c và \u2028 nằm giữa câu không được tách thành dòng mới
            (d / "a.en").write_text("one\x0ctwo\nthree\n", encoding="utf-8")
            (d / "a.vi").write_text("một\u2028hai\nba\n", encoding="utf-8")
            stats = process_split(d / "a.en", d / "a.vi", d / "o.en", d / "o.vi")
            self.assertEqual(stats["num_sentences"], 2)
            self.assertEqual((d / "o.en").read_text(encoding="utf-8"), "one two\nthree\n")
            self.assertEqual((d / "o.vi").read_text(encoding="utf-8"), "một hai\nba\n")

    def test_train_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "a.en").write_text("hi\nhi\n\nx x x\n", encoding="utf-8")
            (d / "a.vi").write_text("chào\nchào\nrỗng\ny y y\n", encoding="utf-8")
            stats = process_split(
                d / "a.en", d / "a.vi", d / "o.en", d / "o.vi", is_train=True, max_words=2
            )
            self.assertEqual(stats["dropped"], {"empty": 1, "duplicate": 1, "too_long": 1})
            self.assertEqual(stats["num_sentences"], 1)


@unittest.skipUnless(HAS_TORCH, "cần torch")
class TestDataset(unittest.TestCase):
    def _dataset(self, src, tgt, max_len=10):
        from src.data.dataset import TranslationDataset

        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "s").write_text(src, encoding="utf-8")
        (d / "t").write_text(tgt, encoding="utf-8")
        tok = cast(Any, StubTokenizer())
        return TranslationDataset(d / "s", d / "t", tok, tok, max_len=max_len)

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_truncation_always_preserves_eos(self):
        src, tgt = self._dataset(" ".join(["w"] * 20) + "\n", "a b\n")[0]
        self.assertEqual(len(src), 10)
        self.assertEqual((src[0].item(), src[-1].item()), (2, 3))
        self.assertEqual(tgt.tolist(), [2, 10, 10, 3])

    def test_line_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._dataset("a\nb\n", "a\n")

    def test_target_shifting_logic(self):
        from src.data.dataset import collate_fn

        assert torch is not None
        batch = [
            (torch.tensor([2, 5, 3]), torch.tensor([2, 7, 8, 3])),
            (torch.tensor([2, 3]), torch.tensor([2, 9, 3])),
        ]
        src, tgt_in, tgt_out, src_mask, tgt_mask = collate_fn(batch, return_masks=True)
        self.assertEqual(src.tolist(), [[2, 5, 3], [2, 3, 0]])
        self.assertEqual(tgt_in.tolist(), [[2, 7, 8], [2, 9, 3]])
        self.assertEqual(tgt_out.tolist(), [[7, 8, 3], [9, 3, 0]])
        self.assertEqual(src_mask.tolist(), [[False, False, False], [False, False, True]])
        self.assertEqual(tgt_mask.shape, tgt_in.shape)

    def test_bucketing_batch_sampler(self):
        from src.data.dataset import BucketBatchSampler

        lengths = [50, 3, 40, 2, 60, 1]
        batches = list(BucketBatchSampler(lengths, batch_size=2, shuffle=True))
        self.assertEqual(sorted(sorted(lengths[i] for i in b) for b in batches), [[1, 2], [3, 40], [50, 60]])


@unittest.skipUnless(HAS_TOKENIZERS and (TOK_DIR / "tokenizer_vi.json").exists(), "cần tokenizers + file tokenizer")
class TestTokenizer(unittest.TestCase):
    def test_roundtrip_and_special_ids(self):
        from src.data.tokenizer import BPETokenizer

        for lang, text in [("en", "Climate change costs $1,000,000 ."), ("vi", "Các từ có dấu: ồ, ế, ắ, ự, ỡ .")]:
            tok = BPETokenizer.from_file(TOK_DIR / f"tokenizer_{lang}.json")
            self.assertEqual((tok.pad_id, tok.unk_id, tok.sos_id, tok.eos_id), (0, 1, 2, 3))
            ids = tok.encode(text)
            self.assertEqual((ids[0], ids[-1]), (2, 3))
            self.assertNotIn(tok.unk_id, ids)
            self.assertEqual(tok.decode(ids), text)


if __name__ == "__main__":
    unittest.main()
