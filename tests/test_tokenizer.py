"""Unit tests cho Tokenizer (HF BPE & SentencePiece) và Dataset/DataLoader chuẩn nhóm."""

import functools
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    TranslationDataset,
    _verify_batch_invariants,
    build_dataloaders,
    collate_translation_batch,
    create_dataloader,
)
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

    def test_dataset_batch_contract_bucketing_and_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for split in ("train", "val", "test"):
                (tmp_dir / f"{split}.en").write_text(
                    "Hello world .\nGood morning everyone today .\nYes .\n" * 2,
                    encoding="utf-8",
                )
                (tmp_dir / f"{split}.vi").write_text(
                    "Xin chào .\nChào buổi sáng mọi người hôm nay .\nVâng .\n" * 2,
                    encoding="utf-8",
                )

            src_tok = BPETokenizer.train(
                [tmp_dir / "train.en"], tmp_dir / "bpe_en.json", vocab_size=80, min_frequency=1
            )
            tgt_tok = BPETokenizer.train(
                [tmp_dir / "train.vi"], tmp_dir / "bpe_vi.json", vocab_size=80, min_frequency=1
            )

            # Kiểm tra get_raw_pair
            ds = TranslationDataset(
                tmp_dir / "train.en", tmp_dir / "train.vi", src_tok, tgt_tok, max_len=128
            )
            self.assertEqual(ds.get_raw_pair(0), ("Hello world .", "Xin chào ."))

            # Kiểm tra collate_fn cấp module có thể pickle trên Windows
            fn = functools.partial(collate_translation_batch, pad_id=PAD_ID)
            self.assertIsNotNone(pickle.dumps(fn))

            # Kiểm tra build_dataloaders (gồm cả bucketing trên train, khóa cứng val/test)
            loaders = build_dataloaders(
                tmp_dir, src_tok, tgt_tok, batch_size=2, max_len=128, use_bucketing=True
            )
            self.assertEqual(set(loaders.keys()), {"train", "val", "test"})

            for split, loader in loaders.items():
                batch = next(iter(loader))
                _verify_batch_invariants(batch, BOS_ID, EOS_ID, PAD_ID)
                self.assertEqual(batch["src_ids"].shape[0], 2)

            # Kiểm tra val giữ nguyên thứ tự gốc 1-1
            val_loader = create_dataloader(
                tmp_dir / "val.en", tmp_dir / "val.vi", src_tok, tgt_tok, batch_size=3, shuffle=False
            )
            val_batch = next(iter(val_loader))
            self.assertEqual(src_tok.decode(val_batch["src_ids"][0]), "Hello world .")
            self.assertEqual(tgt_tok.decode(val_batch["tgt_output_ids"][0]), "Xin chào .")


if __name__ == "__main__":
    unittest.main()
