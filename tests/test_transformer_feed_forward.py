"""Unit and integration tests for Transformer PositionWiseFeedForward."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import PAD_ID, load_tokenizer
from src.models.transformer import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    ScaledEmbedding,
    SinusoidalPositionalEncoding,
)


class TestTransformerFeedForward(unittest.TestCase):
    """Test suite verifying shapes, params, activations, position-wise independence, AMP, and grads."""

    def test_shape_and_param_count_with_and_without_bias(self) -> None:
        ffn_bias = PositionWiseFeedForward(d_model=256, d_ff=1024, dropout=0.2, bias=True)
        # (256*1024 + 1024) + (1024*256 + 256) = 525,568
        self.assertEqual(sum(p.numel() for p in ffn_bias.parameters()), 525_568)
        x = torch.randn(4, 16, 256)
        out_bias = ffn_bias(x)
        self.assertEqual(tuple(out_bias.shape), (4, 16, 256))

        ffn_nobias = PositionWiseFeedForward(d_model=256, d_ff=1024, dropout=0.2, bias=False)
        self.assertIsNone(ffn_nobias.fc1.bias)
        self.assertIsNone(ffn_nobias.fc2.bias)
        # 2 * (256 * 1024) = 524,288
        self.assertEqual(sum(p.numel() for p in ffn_nobias.parameters()), 524_288)
        out_nobias = ffn_nobias(x)
        self.assertEqual(tuple(out_nobias.shape), (4, 16, 256))

    def test_exact_formula_relu_and_gelu(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 5, 32)

        for act_input, act_fn in (("ReLU", F.relu), ("GELU", F.gelu)):
            ffn = PositionWiseFeedForward(
                d_model=32, d_ff=128, dropout=0.0, activation=act_input, bias=True
            )
            ffn.eval()
            out = ffn(x)
            manual_hidden = act_fn(F.linear(x, ffn.fc1.weight, ffn.fc1.bias))
            manual_out = F.linear(manual_hidden, ffn.fc2.weight, ffn.fc2.bias)
            self.assertTrue(torch.allclose(out, manual_out, atol=1e-6))

    def test_position_wise_independence_on_cpu_float32(self) -> None:
        torch.manual_seed(42)
        # Keep strictly on CPU float32 so GPU TF32 rounding cannot affect exact torch.equal
        ffn = PositionWiseFeedForward(d_model=64, d_ff=256, dropout=0.2, activation="relu").cpu()
        ffn.eval()

        x = torch.randn(3, 10, 64, dtype=torch.float32, device="cpu")
        x_perturbed = x.clone()
        x_perturbed[:, 3, :] += 1e5

        out = ffn(x)
        out_perturbed = ffn(x_perturbed)

        for pos in range(10):
            if pos != 3:
                self.assertTrue(torch.equal(out[:, pos, :], out_perturbed[:, pos, :]))
        self.assertFalse(torch.equal(out[:, 3, :], out_perturbed[:, 3, :]))

    def test_dropout_amp_bfloat16_and_validation_errors(self) -> None:
        torch.manual_seed(42)
        ffn = PositionWiseFeedForward(d_model=256, d_ff=1024, dropout=0.2, activation="relu")
        x = torch.randn(2, 8, 256)

        ffn.eval()
        out_eval_1 = ffn(x)
        out_eval_2 = ffn(x)
        self.assertTrue(torch.equal(out_eval_1, out_eval_2))

        ffn.train()
        out_train = ffn(x)
        self.assertFalse(torch.allclose(out_eval_1, out_train))

        # AMP bfloat16 check (CPU + CUDA if available)
        ffn.eval()
        ffn_bf16 = ffn.to(torch.bfloat16)
        x_bf16 = x.to(torch.bfloat16)
        out_bf16 = ffn_bf16(x_bf16)
        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out_bf16).all())

        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            ffn_cuda = ffn_bf16.to(cuda_dev)
            out_cuda = ffn_cuda(x_bf16.to(cuda_dev))
            self.assertEqual(out_cuda.device.type, "cuda")
            self.assertEqual(out_cuda.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(out_cuda).all())

        # Validation error checks
        for bad_cfg in (
            {"d_model": 0, "d_ff": 128},
            {"d_model": -16, "d_ff": 128},
            {"d_model": 32, "d_ff": 0},
            {"d_model": 32, "d_ff": -64},
            {"d_model": 32, "d_ff": 128, "dropout": -0.1},
            {"d_model": 32, "d_ff": 128, "dropout": 1.0},
            {"d_model": 32, "d_ff": 128, "activation": "swish"},
        ):
            with self.assertRaises(ValueError):
                PositionWiseFeedForward(**bad_cfg)

        ffn_chk = PositionWiseFeedForward(d_model=32, d_ff=128)
        with self.assertRaises(ValueError):
            ffn_chk(torch.randn(2, 32))  # 2D
        with self.assertRaises(ValueError):
            ffn_chk(torch.randn(2, 4, 8, 32))  # 4D
        with self.assertRaises(ValueError):
            ffn_chk(torch.randn(2, 8, 64))  # wrong last dim

    def test_end_to_end_pipeline_gradients_and_optional_val_batch(self) -> None:
        torch.manual_seed(42)
        d_model, num_heads, d_ff = 256, 4, 1024
        emb = ScaledEmbedding(vocab_size=8000, d_model=d_model, pad_id=PAD_ID)
        pe = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.2, max_len=512)
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.2)
        ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.2)

        src_ids = torch.randint(4, 8000, (4, 14))
        src_ids[0, 10:] = PAD_ID
        src_mask = (src_ids != PAD_ID).unsqueeze(1).unsqueeze(2)

        x = pe(emb(src_ids))
        attn_out, _ = mha(x, x, x, mask=src_mask)
        ffn_out = ffn(x + attn_out)
        self.assertEqual(tuple(ffn_out.shape), (4, 14, d_model))

        ffn_out.mean().backward()
        for name, layer in (("fc1", ffn.fc1), ("fc2", ffn.fc2)):
            assert isinstance(layer, nn.Linear)
            self.assertIsNotNone(layer.weight.grad, name)
            assert layer.weight.grad is not None
            self.assertTrue(torch.isfinite(layer.weight.grad).all(), name)
            self.assertGreater(float(layer.weight.grad.abs().sum().item()), 0.0, name)
            if layer.bias is not None:
                self.assertIsNotNone(layer.bias.grad, name)
                assert layer.bias.grad is not None
                self.assertTrue(torch.isfinite(layer.bias.grad).all(), name)

        assert emb.embedding.weight.grad is not None
        self.assertTrue(torch.all(emb.embedding.weight.grad[PAD_ID] == 0.0))

        # Optional real val batch check
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu val hoac BPE tokenizer de chay test batch that.")

        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, max_len=128, pre_tokenize=True)
        batch = next(iter(create_dataloader(ds, batch_size=8, shuffle=False)))

        b_src = batch["src_ids"]
        b_mask = (b_src != PAD_ID).unsqueeze(1).unsqueeze(2)
        b_x = pe(emb(b_src))
        b_attn, _ = mha(b_x, b_x, b_x, mask=b_mask)
        b_ffn = ffn(b_x + b_attn)
        self.assertEqual(tuple(b_ffn.shape), (8, b_src.size(1), d_model))
        self.assertTrue(torch.isfinite(b_ffn).all())
        b_ffn.mean().backward()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Transformer PositionWiseFeedForward ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformerFeedForward)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
