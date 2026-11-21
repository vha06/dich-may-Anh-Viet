"""Unit test kiểm thử thước đo BLEU (src/evaluation/metrics.py)."""

import unittest

from src.evaluation.metrics import compute_bleu, compute_sentence_bleu


class TestMetrics(unittest.TestCase):
    def test_perfect_match(self):
        ref = "tôi đang học bài dịch máy với mô hình transformer"
        hyp = "tôi đang học bài dịch máy với mô hình transformer"
        score = compute_sentence_bleu(hyp, ref)
        self.assertAlmostEqual(score, 100.0, places=1)

    def test_zero_match(self):
        ref = "hôm nay thời tiết rất đẹp"
        hyp = "chiếc xe đạp màu đỏ bị hỏng"
        score = compute_sentence_bleu(hyp, ref)
        self.assertEqual(score, 0.0)

    def test_brevity_penalty(self):
        ref = "chúng tôi đang cố gắng hoàn thành bài tập lớn môn học này"
        hyp = "chúng tôi"
        score = compute_sentence_bleu(hyp, ref)
        self.assertLess(score, 5.0)

    def test_corpus_bleu(self):
        refs = ["tôi yêu việt nam", "chào buổi sáng"]
        hyps = ["tôi yêu việt nam", "chào buổi sáng"]
        score = compute_bleu(hyps, refs)
        self.assertAlmostEqual(score, 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
