"""Unit tests cho Tokenizer (HF BPE & SentencePiece) và Dataset 2 Tokenizer."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.data.dataset import build_dataloader
from src.data.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    BPETokenizer,
    SentencePieceTokenizer,
    create_tokenizer,
)


class TestTokenizerAndDataset(unittest.TestCase):
    def test_bpe_contract_and_decode_stops_at_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_en = tmp_dir / "train.en"
            train_en.write_text(
                "Hello world .\nI love machine translation .\nHello AI 2025 !\n" * 5,
                encoding="utf-8",
            )
            tok = BPETokenizer.train([train_en], tmp_dir / "bpe_en.json", vocab_size=80, min_frequency=1)

            self.assertEqual(tok.pad_id, PAD_ID)
            self.assertEqual(tok.unk_id, UNK_ID)
            self.assertEqual(tok.bos_id, BOS_ID)
            self.assertEqual(tok.eos_id, EOS_ID)
            self.assertEqual(tok.sos_id, BOS_ID)

            ids = tok.encode("Hello world .", add_special_tokens=True, max_len=5)
            self.assertEqual(ids[0], BOS_ID)
            self.assertEqual(ids[-1], EOS_ID)
            self.assertLessEqual(len(ids), 5)

            tokens = tok.encode_tokens("Hello world .", add_special_tokens=False)
            self.assertTrue(any(t.endswith("</w>") for t in tokens))

            # Kiểm tra decode nhận Tensor 1D và dừng ngay tại EOS đầu tiên
            hello_ids = tok.encode("Hello", add_special_tokens=False)
            world_ids = tok.encode("world", add_special_tokens=False)
            tensor_ids = torch.tensor([BOS_ID, *hello_ids, EOS_ID, *world_ids, PAD_ID])
            self.assertEqual(tok.decode(tensor_ids), "Hello")

            loaded = create_tokenizer(tmp_dir / "bpe_en.json", tokenizer_type="bpe")
            self.assertIsInstance(loaded, BPETokenizer)

    def test_sentencepiece_decode_stops_at_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_vi = tmp_dir / "train.vi"
            train_vi.write_text(
                "Xin chào thế giới .\nTôi yêu học máy .\nTiếng Việt có dấu .\n" * 5,
                encoding="utf-8",
            )
            tok = SentencePieceTokenizer.train(train_vi, tmp_dir / "spm_vi", vocab_size=60)
            a_ids = tok.encode("Xin chào", add_special_tokens=False)
            b_ids = tok.encode("thế giới", add_special_tokens=False)
            seq = [BOS_ID, *a_ids, EOS_ID, *b_ids]
            self.assertEqual(tok.decode(seq), "Xin chào")

    def test_dataset_with_separate_tokenizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            en_file = tmp_dir / "train.en"
            vi_file = tmp_dir / "train.vi"
            en_file.write_text("Hello world .\nGood morning .\n" * 3, encoding="utf-8")
            vi_file.write_text("Xin chào .\nChào buổi sáng .\n" * 3, encoding="utf-8")

            src_tok = BPETokenizer.train([en_file], tmp_dir / "bpe_en.json", vocab_size=60, min_frequency=1)
            tgt_tok = BPETokenizer.train([vi_file], tmp_dir / "bpe_vi.json", vocab_size=60, min_frequency=1)

            loader = build_dataloader(
                en_file, vi_file, src_tokenizer=src_tok, tgt_tokenizer=tgt_tok, batch_size=2, shuffle=False
            )
            src_batch, tgt_batch = next(iter(loader))
            self.assertEqual(src_batch.shape[0], 2)
            self.assertEqual(tgt_batch.shape[0], 2)
            self.assertEqual(src_batch[0, 0].item(), BOS_ID)
            self.assertEqual(tgt_batch[0, 0].item(), BOS_ID)


if __name__ == "__main__":
    unittest.main()
