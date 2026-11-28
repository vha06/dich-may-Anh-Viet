"""Unit and integration tests for Transformer ScaledEmbedding & SinusoidalPositionalEncoding."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import PAD_ID, load_tokenizer
from src.models.transformer import (
    PositionalEncoding,
    ScaledEmbedding,
    SinusoidalPositionalEncoding,
)


class TestTransformerPositionalEncoding(unittest.TestCase):
    """Test suite verifying shapes, formulas, rotation property, AMP, and embedding integration."""

    def test_shape_persistent_false_buffer_and_zero_trainable_params(self) -> None:
        self.assertIs(PositionalEncoding, SinusoidalPositionalEncoding)
        pe_layer = SinusoidalPositionalEncoding(d_model=256, dropout=0.2, max_len=512)
        pe_layer.eval()

        self.assertEqual(sum(p.numel() for p in pe_layer.parameters()), 0)
        buffers = dict(pe_layer.named_buffers())
        self.assertIn("pe", buffers)
        self.assertEqual(tuple(buffers["pe"].shape), (1, 512, 256))
        self.assertNotIn("pe", pe_layer.state_dict())

        for seq_len in (1, 16, 128):
            x = torch.zeros(4, seq_len, 256)
            out = pe_layer(x)
            self.assertEqual(tuple(out.shape), (4, seq_len, 256))

        # Flexible checkpoint reload with a larger max_len
        pe_larger = SinusoidalPositionalEncoding(d_model=256, dropout=0.2, max_len=1024)
        pe_larger.load_state_dict(pe_layer.state_dict(), strict=True)
        self.assertEqual(tuple(pe_larger.pe.shape), (1, 1024, 256))

    def test_exact_sinusoidal_formula_and_value_bounds(self) -> None:
        d_model = 64
        max_len = 100
        pe_layer = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.0, max_len=max_len)
        table = pe_layer.pe.squeeze(0)

        self.assertTrue(torch.isfinite(table).all())
        self.assertGreaterEqual(float(table.min().item()), -1.0)
        self.assertLessEqual(float(table.max().item()), 1.0)

        # At pos = 0: sin(0) == 0.0 on even dims, cos(0) == 1.0 on odd dims
        self.assertTrue(torch.allclose(table[0, 0::2], torch.zeros(d_model // 2), atol=1e-6))
        self.assertTrue(torch.allclose(table[0, 1::2], torch.ones(d_model // 2), atol=1e-6))

        for pos in (1, 7, 42, 99):
            for i in (0, 5, 16, 31):
                angle = pos / (10000.0 ** ((2 * i) / d_model))
                self.assertAlmostEqual(float(table[pos, 2 * i].item()), math.sin(angle), places=5)
                self.assertAlmostEqual(
                    float(table[pos, 2 * i + 1].item()), math.cos(angle), places=5
                )

    def test_relative_position_linear_rotation_property(self) -> None:
        d_model = 64
        pe_layer = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.0, max_len=128)
        table = pe_layer.pe.squeeze(0)  # (128, 64)

        i_vals = torch.arange(0, d_model, 2, dtype=torch.float32)
        omegas = torch.exp(i_vals * (-math.log(10000.0) / d_model))

        for k in (3, 17):
            cos_k = torch.cos(k * omegas)
            sin_k = torch.sin(k * omegas)
            for pos in (0, 5, 42):
                sin_pos = table[pos, 0::2]
                cos_pos = table[pos, 1::2]
                # Exact 2x2 rotation transform M_k * [sin(w*pos), cos(w*pos)]^T
                pred_sin = cos_k * sin_pos + sin_k * cos_pos
                pred_cos = -sin_k * sin_pos + cos_k * cos_pos
                self.assertTrue(torch.allclose(pred_sin, table[pos + k, 0::2], atol=1e-5))
                self.assertTrue(torch.allclose(pred_cos, table[pos + k, 1::2], atol=1e-5))

    def test_offset_dropout_amp_bfloat16_and_validation_errors(self) -> None:
        torch.manual_seed(42)
        pe_layer = SinusoidalPositionalEncoding(d_model=32, dropout=0.2, max_len=64)
        x = torch.randn(2, 10, 32)

        # In eval mode, dropout is off and slicing via offset matches full sequence
        pe_layer.eval()
        full_out = pe_layer(x)
        for t in (0, 4, 9):
            step_out = pe_layer(x[:, t : t + 1, :], offset=t)
            self.assertTrue(torch.allclose(step_out, full_out[:, t : t + 1, :], atol=1e-6))

        # In train mode, dropout=0.2 randomly zeroes entries and differs from eval output
        pe_layer.train()
        train_out = pe_layer(x)
        self.assertFalse(torch.allclose(train_out, full_out))

        # AMP bfloat16 check (CPU and CUDA if available)
        pe_layer.eval()
        x_bf16 = x.to(torch.bfloat16)
        out_bf16 = pe_layer(x_bf16)
        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out_bf16).all())

        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            pe_cuda = SinusoidalPositionalEncoding(d_model=32, dropout=0.2, max_len=64).to(cuda_dev)
            pe_cuda.eval()
            out_cuda = pe_cuda(x_bf16.to(cuda_dev))
            self.assertEqual(out_cuda.device.type, "cuda")
            self.assertEqual(out_cuda.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(out_cuda).all())

        # Validation error checks
        for bad_kwargs in (
            {"d_model": 0},
            {"d_model": -8},
            {"d_model": 15},
            {"d_model": 32, "max_len": 0},
            {"d_model": 32, "max_len": -5},
            {"d_model": 32, "dropout": -0.1},
            {"d_model": 32, "dropout": 1.0},
        ):
            with self.assertRaises(ValueError):
                SinusoidalPositionalEncoding(**bad_kwargs)

        with self.assertRaises(ValueError):
            pe_layer(torch.zeros(2, 32))  # 2D input
        with self.assertRaises(ValueError):
            pe_layer(torch.zeros(2, 5, 16))  # wrong d_model
        with self.assertRaises(ValueError):
            pe_layer(torch.zeros(2, 5, 32), offset=-1)
        with self.assertRaises(ValueError):
            pe_layer(torch.zeros(2, 5, 32), offset=60)  # 60 + 5 > 64

        for bad_emb in (
            {"vocab_size": 0, "d_model": 32},
            {"vocab_size": 100, "d_model": 0},
            {"vocab_size": 100, "d_model": 32, "pad_id": 100},
            {"vocab_size": 100, "d_model": 32, "pad_id": -1},
        ):
            with self.assertRaises(ValueError):
                ScaledEmbedding(**bad_emb)

    def test_scaled_embedding_and_pe_integration_with_padding_and_gradients(self) -> None:
        torch.manual_seed(42)
        d_model = 256
        emb = ScaledEmbedding(vocab_size=8000, d_model=d_model, pad_id=PAD_ID)
        pe_layer = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.2, max_len=512)
        pe_layer.eval()

        # Batch of real tokens + trailing padding tokens (pad_id=0)
        tokens = torch.randint(4, 8000, (8, 32), dtype=torch.long)
        tokens[0, 20:] = PAD_ID
        tokens[3, 12:] = PAD_ID

        scaled = emb(tokens)
        self.assertEqual(tuple(scaled.shape), (8, 32, d_model))

        # (a) pad_id=0 must output exact zeros before adding PE
        pad_mask = tokens == PAD_ID
        self.assertTrue(torch.all(scaled[pad_mask] == 0.0))

        # (b) real token embeddings scaled by sqrt(d_model) must have std ~ 1.0
        real_std = float(scaled[~pad_mask].std(unbiased=False).item())
        self.assertGreater(real_std, 0.85)
        self.assertLess(real_std, 1.15)

        out = pe_layer(scaled)
        self.assertEqual(tuple(out.shape), (8, 32, d_model))

        # (c) gradient at pad_id=0 row must stay exact 0.0 after backward()
        loss = (out[~pad_mask] ** 2).mean()
        loss.backward()
        grad = emb.embedding.weight.grad
        assert grad is not None
        self.assertTrue(torch.all(grad[PAD_ID] == 0.0))
        active_id = int(tokens[1, 0].item())
        self.assertGreater(float(grad[active_id].abs().sum().item()), 0.0)

    def test_real_val_batch_integration_optional(self) -> None:
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu val.en hoac BPE tokenizer de chay test batch that.")

        torch.manual_seed(42)
        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, max_len=128, pre_tokenize=True)
        loader = create_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))

        src_ids = batch["src_ids"]
        emb = ScaledEmbedding(vocab_size=src_tok.vocab_size, d_model=256, pad_id=PAD_ID)
        pe_layer = SinusoidalPositionalEncoding(d_model=256, dropout=0.2, max_len=512)
        pe_layer.train()

        scaled = emb(src_ids)
        pad_mask = src_ids == PAD_ID
        if pad_mask.any():
            self.assertTrue(torch.all(scaled[pad_mask] == 0.0))

        encoded = pe_layer(scaled)
        self.assertEqual(tuple(encoded.shape), (8, src_ids.size(1), 256))
        encoded.mean().backward()
        assert emb.embedding.weight.grad is not None
        self.assertTrue(torch.all(emb.embedding.weight.grad[PAD_ID] == 0.0))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Transformer ScaledEmbedding & SinusoidalPositionalEncoding ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformerPositionalEncoding)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
