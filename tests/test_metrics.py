"""Unit test kiểm thử thước đo SacreBLEU (src/evaluation/metrics.py)."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, BPETokenizer
from src.evaluation.metrics import (
    compute_bleu,
    compute_corpus_bleu,
    compute_sentence_bleu,
    decode_and_compute_bleu,
)

REQUIRED_KEYS = {
    "bleu",
    "bleu_exact",
    "precisions",
    "brevity_penalty",
    "length_ratio",
    "sys_len",
    "ref_len",
    "signature",
}


class TestMetrics(unittest.TestCase):
    def test_perfect_match_and_schema(self):
        ref = "tôi đang học bài dịch máy với mô hình transformer"
        hyp = "  tôi đang học   bài dịch máy với mô hình transformer  "
        res = compute_sentence_bleu(hyp, ref)
        self.assertEqual(set(res.keys()), REQUIRED_KEYS)
        self.assertAlmostEqual(res["bleu"], 100.0, places=1)
        self.assertEqual(res["brevity_penalty"], 1.0)
        self.assertIn("tok:13a", res["signature"])
        self.assertIn("case:mixed", res["signature"])

    def test_zero_match(self):
        ref = "hôm nay thời tiết rất đẹp"
        hyp = "chiếc xe đạp màu đỏ bị hỏng"
        res = compute_sentence_bleu(hyp, ref)
        self.assertEqual(res["bleu"], 0.0)

    def test_brevity_penalty(self):
        ref = "chúng tôi đang cố gắng hoàn thành bài tập lớn môn học này"
        hyp = "chúng tôi"
        res = compute_sentence_bleu(hyp, ref)
        self.assertLess(res["brevity_penalty"], 1.0)
        self.assertLess(res["bleu"], 5.0)

    def test_word_order_shuffled(self):
        ref = "hôm nay chúng tôi học dịch máy Anh Việt trên lớp"
        shuffled = "lớp trên Việt Anh máy dịch học tôi chúng nay hôm"
        res = compute_sentence_bleu(shuffled, ref)
        self.assertGreater(res["precisions"][0], 99.0)
        self.assertLess(res["bleu"], 30.0)

    def test_corpus_bleu_and_multi_reference_and_case_sensitive(self):
        refs = ["Tôi yêu Việt Nam", "Chào buổi sáng"]
        hyps = ["Tôi yêu Việt Nam", "Chào buổi sáng"]
        res = compute_corpus_bleu(hyps, refs)
        res_alias = compute_bleu(hyps, refs)
        self.assertEqual(res, res_alias)
        self.assertAlmostEqual(res["bleu"], 100.0, places=1)

        # Kiểm tra phân biệt hoa/thường (lowercase=False)
        lower_hyps = ["tôi yêu việt nam", "chào buổi sáng"]
        res_lower = compute_corpus_bleu(lower_hyps, refs)
        self.assertLess(res_lower["bleu"], 50.0)

        # Kiểm tra nhiều bộ tham chiếu
        multi_refs = [
            ["Tôi yêu Việt Nam", "Chào buổi sáng"],
            ["Tôi thích Việt Nam", "Xin chào buổi sáng"],
        ]
        res_multi = compute_corpus_bleu(["Tôi thích Việt Nam", "Chào buổi sáng"], multi_refs)
        self.assertAlmostEqual(res_multi["bleu"], 100.0, places=1)

    def test_decode_and_compute_bleu(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_vi = tmp_dir / "train.vi"
            train_vi.write_text("Tôi yêu Việt Nam .\nChào buổi sáng .\n" * 3, encoding="utf-8")
            tok = BPETokenizer.train([train_vi], tmp_dir / "bpe_vi.json", vocab_size=60, min_frequency=1)

            ids1 = tok.encode("Tôi yêu Việt Nam .", add_special_tokens=True)
            ids2 = tok.encode("Chào buổi sáng .", add_special_tokens=True)
            max_l = max(len(ids1), len(ids2))
            t1 = ids1 + [PAD_ID] * (max_l - len(ids1))
            t2 = ids2 + [PAD_ID] * (max_l - len(ids2))
            pred_tensor = torch.tensor([t1, t2], dtype=torch.long)

            res = decode_and_compute_bleu(
                pred_tensor, ["Tôi yêu Việt Nam .", "Chào buổi sáng ."], tok
            )
            self.assertEqual(set(res.keys()), REQUIRED_KEYS)
            self.assertAlmostEqual(res["bleu"], 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
