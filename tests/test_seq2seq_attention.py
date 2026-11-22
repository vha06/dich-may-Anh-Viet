"""Unit and integration tests for LuongAttention."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import DEFAULT_VOCAB_SIZE, PAD_ID, load_tokenizer
from src.models.seq2seq_lstm import LSTMDecoder, LSTMEncoder, LuongAttention


class TestLuongAttention(unittest.TestCase):
    """Test suite verifying dot/general/concat variants, masking invariance, gradients, and integration."""

    def test_all_three_score_methods_shapes_and_params(self) -> None:
        torch.manual_seed(42)
        bsz, s_max, dec_dim, enc_dim = 4, 11, 512, 512
        query = torch.randn(bsz, dec_dim)
        encoder_outputs = torch.randn(bsz, s_max, enc_dim)

        expected_params = {
            "dot": 0,
            "general": 512 * 512,  # 262,144
            "concat": (512 * 256) + (512 * 256) + 256,  # 262,400
        }

        for method, exp_params in expected_params.items():
            attn = LuongAttention(
                dec_hidden_dim=dec_dim,
                enc_out_dim=enc_dim,
                method=method,
                attn_dim=256,
            )
            self.assertEqual(attn.count_parameters(), exp_params)

            context, attn_weights = attn(query, encoder_outputs)
            self.assertEqual(context.shape, (bsz, enc_dim))
            self.assertEqual(attn_weights.shape, (bsz, s_max))
            self.assertTrue(torch.all(attn_weights >= 0.0))
            self.assertTrue(
                torch.allclose(attn_weights.sum(dim=-1), torch.ones(bsz), atol=1e-5)
            )

        with self.assertRaises(ValueError):
            LuongAttention(dec_hidden_dim=256, enc_out_dim=512, method="dot")
        with self.assertRaises(ValueError):
            LuongAttention(dec_hidden_dim=512, enc_out_dim=512, method="bahdanau")

    def test_padding_mask_invariance_extreme_values(self) -> None:
        torch.manual_seed(42)
        bsz, s_max, dec_dim, enc_dim = 3, 8, 128, 128
        src_mask = torch.tensor(
            [
                [True, True, True, True, True, False, False, False],
                [True, True, True, True, True, True, True, True],
                [True, True, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        query = torch.randn(bsz, dec_dim)
        enc_zero_pad = torch.randn(bsz, s_max, enc_dim)
        enc_zero_pad[~src_mask] = 0.0

        enc_extreme_pad = enc_zero_pad.clone()
        enc_extreme_pad[~src_mask] = 1e6

        for method in ("dot", "general", "concat"):
            attn = LuongAttention(dec_dim, enc_dim, method=method, attn_dim=64)
            attn.eval()
            with torch.no_grad():
                ctx_a, w_a = attn(query, enc_zero_pad, src_mask=src_mask)
                ctx_b, w_b = attn(query, enc_extreme_pad, src_mask=src_mask)

            self.assertTrue(
                torch.allclose(ctx_a, ctx_b, atol=1e-6),
                f"{method}: context thay doi khi padding thay doi gia tri.",
            )
            self.assertTrue(
                torch.allclose(w_a, w_b, atol=1e-6),
                f"{method}: attn_weights thay doi khi padding thay doi gia tri.",
            )
            self.assertTrue(
                torch.equal(w_b[~src_mask], torch.zeros_like(w_b[~src_mask])),
                f"{method}: vi tri padding phai co trong so bang 0.0.",
            )

    def test_gradient_flow_all_methods(self) -> None:
        torch.manual_seed(42)
        bsz, s_max, dec_dim, enc_dim = 2, 6, 64, 96
        src_mask = torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, True, True, True],
            ],
            dtype=torch.bool,
        )

        for method in ("general", "concat"):
            attn = LuongAttention(dec_dim, enc_dim, method=method, attn_dim=32)
            query = torch.randn(bsz, dec_dim, requires_grad=True)
            enc_out = torch.randn(bsz, s_max, enc_dim, requires_grad=True)

            context, weights = attn(query, enc_out, src_mask=src_mask)
            loss = context.sum() + weights.sum()
            attn.zero_grad()
            loss.backward()

            self.assertIsNotNone(query.grad)
            self.assertTrue(torch.isfinite(query.grad).all())
            self.assertIsNotNone(enc_out.grad)
            self.assertTrue(torch.isfinite(enc_out.grad).all())
            for name, param in attn.named_parameters():
                self.assertIsNotNone(param.grad, f"{method}: thieu gradient cho {name}")
                self.assertTrue(torch.isfinite(param.grad).all())

        # Dot requires dec_dim == enc_dim
        attn_dot = LuongAttention(64, 64, method="dot")
        q_dot = torch.randn(bsz, 64, requires_grad=True)
        e_dot = torch.randn(bsz, s_max, 64, requires_grad=True)
        ctx_dot, w_dot = attn_dot(q_dot, e_dot, src_mask=src_mask)
        (ctx_dot.sum() + w_dot.sum()).backward()
        self.assertIsNotNone(q_dot.grad)
        self.assertTrue(torch.isfinite(q_dot.grad).all())
        self.assertIsNotNone(e_dot.grad)
        self.assertTrue(torch.isfinite(e_dot.grad).all())

    def test_integration_encoder_attention_decoder_real_val_batch(self) -> None:
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
        loader = create_dataloader(ds, batch_size=64, shuffle=False)
        batch = next(iter(loader))

        encoder = LSTMEncoder(vocab_size=src_tok.vocab_size)
        attention = LuongAttention(dec_hidden_dim=512, enc_out_dim=512)
        decoder = LSTMDecoder(vocab_size=tgt_tok.vocab_size, attention=attention)
        encoder.eval()
        decoder.eval()

        src_ids = batch["src_ids"]
        src_lengths = batch["src_lengths"]
        tgt_input_ids = batch["tgt_input_ids"]
        tgt_output_ids = batch["tgt_output_ids"]
        src_mask = src_ids != PAD_ID

        bsz, s_max = src_ids.shape
        _, t_max = tgt_input_ids.shape

        with torch.no_grad():
            enc_outputs, hidden = encoder(src_ids, src_lengths)
            logits_steps = []
            attn_steps = []
            for t in range(t_max):
                logits_t, hidden, attn_w = decoder(
                    tgt_input_ids[:, t],
                    hidden,
                    encoder_outputs=enc_outputs,
                    src_mask=src_mask,
                )
                self.assertIsNotNone(attn_w)
                logits_steps.append(logits_t)
                attn_steps.append(attn_w)

            logits = torch.stack(logits_steps, dim=1)  # (64, 127, 8000)
            attn_map = torch.stack(attn_steps, dim=1)  # (64, 127, 103)

        self.assertEqual(logits.shape, (bsz, t_max, DEFAULT_VOCAB_SIZE))
        self.assertEqual(attn_map.shape, (bsz, t_max, s_max))
        self.assertTrue(
            torch.allclose(attn_map.sum(dim=-1), torch.ones(bsz, t_max), atol=1e-5)
        )
        expanded_pad_mask = (~src_mask).unsqueeze(1).expand(-1, t_max, -1)
        self.assertTrue(
            torch.equal(
                attn_map[expanded_pad_mask],
                torch.zeros_like(attn_map[expanded_pad_mask]),
            )
        )

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        loss = criterion(logits.reshape(-1, DEFAULT_VOCAB_SIZE), tgt_output_ids.reshape(-1))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 8.5)
        self.assertLess(loss.item(), 9.5)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu LuongAttention (Global Attention) ===")
    for method in ("dot", "general", "concat"):
        attn = LuongAttention(dec_hidden_dim=512, enc_out_dim=512, method=method)
        print(f"  - method='{method:<7}': {attn.count_parameters():,} tham so")

    enc = LSTMEncoder(vocab_size=DEFAULT_VOCAB_SIZE)
    dec = LSTMDecoder(
        vocab_size=DEFAULT_VOCAB_SIZE,
        attention=LuongAttention(dec_hidden_dim=512, enc_out_dim=512, method="general"),
    )
    dec_stats = dec.parameter_breakdown()
    total_seq2seq = enc.parameter_breakdown()["total"] + dec_stats["total"]
    print(
        f"Tong tham so Seq2Seq (Encoder + Decoder + Luong general): {total_seq2seq:,} "
        f"(Encoder={enc.parameter_breakdown()['total']:,}, "
        f"Decoder_core={dec_stats['total'] - dec_stats['attention']:,}, "
        f"Attention={dec_stats['attention']:,})"
    )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLuongAttention)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
