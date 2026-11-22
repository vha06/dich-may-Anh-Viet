"""Unit, overfit, and real-batch integration tests for complete Seq2SeqLSTM."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset import (
    TranslationDataset,
    collate_translation_batch,
    create_dataloader,
)
from src.data.tokenizer import BOS_ID, DEFAULT_VOCAB_SIZE, EOS_ID, PAD_ID, load_tokenizer
from src.evaluation.metrics import decode_and_compute_bleu
from src.models.seq2seq_lstm import (
    LSTMDecoder,
    LSTMEncoder,
    Seq2SeqLSTM,
    build_seq2seq_lstm,
)


class TestSeq2SeqLSTM(unittest.TestCase):
    """Test suite verifying assembly, Teacher Forcing, greedy decode, overfit, and full 15.8M step."""

    def test_factory_modes_and_mismatch_validation(self) -> None:
        expected_totals = {
            "general": 15_820_096,
            "dot": 15_557_952,
            "concat": 15_820_352,
            None: 15_033_152,
        }
        for method, exp_total in expected_totals.items():
            model = build_seq2seq_lstm(attention_method=method)
            self.assertEqual(model.parameter_breakdown()["total"], exp_total)
            if method is None:
                self.assertIsNone(model.decoder.attention)
            else:
                self.assertIsNotNone(model.decoder.attention)

        model_str_none = build_seq2seq_lstm(attention_method="none")
        self.assertIsNone(model_str_none.decoder.attention)
        self.assertEqual(model_str_none.parameter_breakdown()["total"], 15_033_152)

        enc = LSTMEncoder(vocab_size=200, embed_dim=32, hidden_dim=32, num_layers=2)
        dec_bad_layers = LSTMDecoder(
            vocab_size=200, embed_dim=32, hidden_dim=64, enc_out_dim=64, num_layers=1
        )
        with self.assertRaises(ValueError):
            Seq2SeqLSTM(enc, dec_bad_layers)

        dec_bad_hidden = LSTMDecoder(
            vocab_size=200, embed_dim=32, hidden_dim=128, enc_out_dim=64, num_layers=2
        )
        with self.assertRaises(ValueError):
            Seq2SeqLSTM(enc, dec_bad_hidden)

    def test_forward_teacher_forcing_ratios(self) -> None:
        torch.manual_seed(42)
        bsz, s_max, t_max, vocab = 3, 9, 7, 120
        model = build_seq2seq_lstm(
            src_vocab_size=vocab,
            tgt_vocab_size=vocab,
            embed_dim=32,
            enc_hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            attention_method="general",
        )
        src_ids = torch.tensor(
            [
                [2, 10, 11, 12, 13, 3, PAD_ID, PAD_ID, PAD_ID],
                [2, 20, 21, 22, 23, 24, 25, 26, 3],
                [2, 30, 31, 3, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
            ],
            dtype=torch.long,
        )
        src_lengths = torch.tensor([6, 9, 4], dtype=torch.long)
        tgt_in = torch.randint(4, vocab, (bsz, t_max))
        tgt_in[:, 0] = BOS_ID

        pad_mask = (src_ids == PAD_ID).unsqueeze(1).expand(-1, t_max, -1)

        model.train()
        for tf_ratio in (1.0, 0.5, 0.0):
            logits, attn = model(src_ids, src_lengths, tgt_in, teacher_forcing_ratio=tf_ratio)
            self.assertEqual(logits.shape, (bsz, t_max, vocab))
            self.assertIsNotNone(attn)
            assert attn is not None
            self.assertEqual(attn.shape, (bsz, t_max, s_max))
            self.assertTrue(
                torch.allclose(attn.sum(dim=-1), torch.ones(bsz, t_max), atol=1e-5)
            )
            self.assertTrue(torch.equal(attn[pad_mask], torch.zeros_like(attn[pad_mask])))

        model.eval()
        logits_eval, attn_eval = model(src_ids, src_lengths, tgt_in, teacher_forcing_ratio=0.0)
        self.assertEqual(logits_eval.shape, (bsz, t_max, vocab))
        self.assertIsNotNone(attn_eval)

    def test_greedy_decode_early_stop_and_bleu_integration(self) -> None:
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not bpe_vi.exists():
            self.skipTest("Thieu data/tokenizer/bpe_vi.json")

        torch.manual_seed(42)
        model = build_seq2seq_lstm(
            src_vocab_size=DEFAULT_VOCAB_SIZE,
            tgt_vocab_size=DEFAULT_VOCAB_SIZE,
            embed_dim=32,
            enc_hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            attention_method="dot",
        )
        model.eval()

        # Bias fc_out so the model emits EOS quickly, verifying early stopping + padding
        with torch.no_grad():
            model.decoder.fc_out.weight.zero_()
            model.decoder.fc_out.bias.fill_(-10.0)
            model.decoder.fc_out.bias[EOS_ID] = 10.0

        src_ids = torch.tensor(
            [
                [2, 50, 60, 70, 3],
                [2, 80, 90, 3, PAD_ID],
            ],
            dtype=torch.long,
        )
        src_lengths = torch.tensor([5, 4], dtype=torch.long)

        pred_ids, attn_weights = model.greedy_decode(src_ids, src_lengths, max_len=20)
        self.assertEqual(pred_ids.shape, (2, 1))
        self.assertTrue(torch.all(pred_ids[:, 0] == EOS_ID))
        self.assertIsNotNone(attn_weights)
        assert attn_weights is not None
        self.assertEqual(attn_weights.shape, (2, 1, 5))

        # Test no-attention greedy decode capped at max_len and connected to SacreBLEU
        model_no_attn = build_seq2seq_lstm(
            src_vocab_size=DEFAULT_VOCAB_SIZE,
            tgt_vocab_size=DEFAULT_VOCAB_SIZE,
            embed_dim=32,
            enc_hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            attention_method=None,
        )
        model_no_attn.eval()
        pred_no_attn, attn_none = model_no_attn.greedy_decode(
            src_ids, src_lengths, max_len=6
        )
        self.assertIsNone(attn_none)
        self.assertEqual(pred_no_attn.size(0), 2)
        self.assertLessEqual(pred_no_attn.size(1), 6)

        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        bleu_res = decode_and_compute_bleu(
            pred_no_attn, ["câu mẫu một .", "câu mẫu hai ."], tgt_tok
        )
        self.assertGreaterEqual(bleu_res["bleu"], 0.0)
        self.assertLessEqual(bleu_res["bleu"], 100.0)

    def test_overfit_mini_batch_real_data(self) -> None:
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu hoac model BPE de chay overfit test.")

        torch.manual_seed(42)
        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, pre_tokenize=True)

        # Indices 1..4 are concise, varied sentences in val
        indices = [1, 2, 3, 4]
        batch = collate_translation_batch([ds[i] for i in indices], pad_id=PAD_ID)
        raw_refs = [ds.get_raw_pair(i)[1] for i in indices]

        model = build_seq2seq_lstm(
            src_vocab_size=src_tok.vocab_size,
            tgt_vocab_size=tgt_tok.vocab_size,
            embed_dim=64,
            enc_hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            attention_method="general",
        )
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

        src_ids = batch["src_ids"]
        src_lengths = batch["src_lengths"]
        tgt_in = batch["tgt_input_ids"]
        tgt_out = batch["tgt_output_ids"]

        model.eval()
        pred_before, _ = model.greedy_decode(src_ids, src_lengths, max_len=tgt_out.size(1) + 5)
        bleu_before = decode_and_compute_bleu(pred_before, raw_refs, tgt_tok)["bleu"]

        model.train()
        losses: list[float] = []
        for _ in range(30):
            logits, _ = model(src_ids, src_lengths, tgt_in, teacher_forcing_ratio=1.0)
            loss = criterion(logits.reshape(-1, tgt_tok.vocab_size), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        loss_start, loss_end = losses[0], losses[-1]
        self.assertLess(loss_end, 1.5, f"Overfit loss cuoi van cao: {loss_end:.4f}")
        self.assertLess(loss_end, loss_start * 0.25)

        model.eval()
        pred_after, _ = model.greedy_decode(src_ids, src_lengths, max_len=tgt_out.size(1) + 5)
        bleu_after = decode_and_compute_bleu(pred_after, raw_refs, tgt_tok)["bleu"]
        self.assertGreater(
            bleu_after,
            bleu_before,
            f"BLEU khong tang sau overfit: before={bleu_before}, after={bleu_after}",
        )

    def test_full_baseline_15m_real_val_step(self) -> None:
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu hoac model BPE de chay integration test.")

        torch.manual_seed(42)
        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, pre_tokenize=True)
        loader = create_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))

        model = build_seq2seq_lstm()
        self.assertEqual(model.parameter_breakdown()["total"], 15_820_096)

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        src_ids = batch["src_ids"]
        src_lengths = batch["src_lengths"]
        tgt_in = batch["tgt_input_ids"]
        tgt_out = batch["tgt_output_ids"]

        model.train()
        logits_1, attn_1 = model(src_ids, src_lengths, tgt_in, teacher_forcing_ratio=1.0)
        self.assertEqual(logits_1.shape, (8, tgt_in.size(1), DEFAULT_VOCAB_SIZE))
        self.assertIsNotNone(attn_1)
        assert attn_1 is not None
        self.assertEqual(attn_1.shape, (8, tgt_in.size(1), src_ids.size(1)))

        loss_1 = criterion(logits_1.reshape(-1, DEFAULT_VOCAB_SIZE), tgt_out.reshape(-1))
        self.assertTrue(torch.isfinite(loss_1))
        expected_ln_v = math.log(DEFAULT_VOCAB_SIZE)
        self.assertAlmostEqual(float(loss_1.item()), expected_ln_v, delta=0.5)

        optimizer.zero_grad()
        loss_1.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Thieu gradient cho {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient {name} co NaN/Inf")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_2, _ = model(src_ids, src_lengths, tgt_in)
            loss_2 = criterion(
                logits_2.reshape(-1, DEFAULT_VOCAB_SIZE), tgt_out.reshape(-1)
            )
        self.assertLess(float(loss_2.item()), float(loss_1.item()))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Mo hinh Hoan chinh Seq2SeqLSTM ===")
    for method in ("general", "dot", "concat", None):
        m = build_seq2seq_lstm(attention_method=method)
        stats = m.parameter_breakdown()
        label = "None (Khong Attention)" if method is None else method
        print(
            f"  - mode={label:<22}: total={stats['total']:>10,} "
            f"(enc={stats['encoder']:,}, dec={stats['decoder_core']:,}, attn={stats['attention']:,})"
        )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSeq2SeqLSTM)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
