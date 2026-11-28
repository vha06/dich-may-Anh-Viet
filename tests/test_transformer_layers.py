"""Unit and integration tests for Transformer SublayerConnection (Post-LN / Pre-LN)."""

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
    SublayerConnection,
)


class TestTransformerSublayerConnection(unittest.TestCase):
    """Test suite verifying Post-LN default, Pre-LN behavior, position independence, AMP, and grads."""

    def test_default_post_ln_params_init_and_zero_sublayer_behavior(self) -> None:
        torch.manual_seed(42)
        default_sub = SublayerConnection(d_model=256)
        self.assertFalse(default_sub.norm_first)
        self.assertEqual(sum(p.numel() for p in default_sub.parameters()), 512)
        self.assertTrue(torch.equal(default_sub.norm.weight, torch.ones(256)))
        self.assertTrue(torch.equal(default_sub.norm.bias, torch.zeros(256)))

        x = torch.randn(3, 10, 256) * 2.5 + 1.5

        # Pre-LN with sublayer=0 must keep x bit-exact in BOTH eval() and train(dropout=0.2)
        pre_ln = SublayerConnection(d_model=256, dropout=0.2, norm_first=True)
        pre_ln.eval()
        self.assertTrue(torch.equal(pre_ln(x, torch.zeros_like), x))
        pre_ln.train()
        self.assertTrue(torch.equal(pre_ln(x, torch.zeros_like), x))

        # Post-LN with sublayer=0 in eval() normalizes each token vector to mean~0, var~1
        post_ln = SublayerConnection(d_model=256, dropout=0.2, norm_first=False)
        post_ln.eval()
        out_post = post_ln(x, torch.zeros_like)
        self.assertTrue(torch.allclose(out_post.mean(dim=-1), torch.zeros(3, 10), atol=1e-5))
        self.assertTrue(
            torch.allclose(out_post.var(dim=-1, unbiased=False), torch.ones(3, 10), atol=1e-4)
        )
        self.assertTrue(torch.allclose(out_post, F.layer_norm(x, (256,), eps=1e-5), atol=1e-6))

    def test_exact_formula_pre_ln_vs_post_ln_and_position_independence_cpu_float32(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 6, 32, dtype=torch.float32, device="cpu") * 3.0 + 5.0
        ffn = PositionWiseFeedForward(d_model=32, d_ff=128, dropout=0.0, activation="relu").cpu()
        ffn.eval()

        pre_ln = SublayerConnection(d_model=32, dropout=0.0, norm_first=True).cpu()
        post_ln = SublayerConnection(d_model=32, dropout=0.0, norm_first=False).cpu()
        pre_ln.eval()
        post_ln.eval()

        out_pre = pre_ln(x, ffn)
        expected_pre = x + ffn(pre_ln.norm(x))
        self.assertTrue(torch.allclose(out_pre, expected_pre, atol=1e-6))

        out_post = post_ln(x, ffn)
        expected_post = post_ln.norm(x + ffn(x))
        self.assertTrue(torch.allclose(out_post, expected_post, atol=1e-6))

        self.assertFalse(torch.allclose(out_pre, out_post))

        # Strictly on CPU float32 so GPU TF32 rounding cannot affect exact torch.equal
        x_pert = x.clone()
        x_pert[:, 2, :] += 1e5
        for sub in (pre_ln, post_ln):
            y_base = sub(x, ffn)
            y_pert = sub(x_pert, ffn)
            for pos in range(6):
                if pos != 2:
                    self.assertTrue(torch.equal(y_base[:, pos, :], y_pert[:, pos, :]))
            self.assertFalse(torch.equal(y_base[:, 2, :], y_pert[:, 2, :]))

    def test_dropout_amp_bfloat16_and_validation_errors(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 8, 256)
        ffn = PositionWiseFeedForward(d_model=256, d_ff=1024, dropout=0.0)
        ffn.eval()

        for norm_first in (False, True):
            sub = SublayerConnection(d_model=256, dropout=0.2, norm_first=norm_first)
            sub.eval()
            out_eval_1 = sub(x, ffn)
            out_eval_2 = sub(x, ffn)
            self.assertTrue(torch.equal(out_eval_1, out_eval_2))

            sub.train()
            out_train = sub(x, ffn)
            self.assertFalse(torch.allclose(out_eval_1, out_train))

            # AMP bfloat16 check (CPU + CUDA if available) even on large scale input
            sub.eval()
            sub_bf16 = sub.to(torch.bfloat16)
            ffn_bf16 = ffn.to(torch.bfloat16)
            x_bf16 = (x * 100.0).to(torch.bfloat16)
            out_bf16 = sub_bf16(x_bf16, ffn_bf16)
            self.assertEqual(out_bf16.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(out_bf16).all())

            if torch.cuda.is_available():
                cuda_dev = torch.device("cuda")
                out_cuda = sub_bf16.to(cuda_dev)(x_bf16.to(cuda_dev), ffn_bf16.to(cuda_dev))
                self.assertEqual(out_cuda.device.type, "cuda")
                self.assertEqual(out_cuda.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(out_cuda).all())
                sub_bf16.cpu()
                ffn_bf16.cpu()

            ffn.float()

        # Validation error checks for __init__
        for bad_cfg in (
            {"d_model": 0},
            {"d_model": -8},
            {"d_model": 32, "dropout": -0.1},
            {"d_model": 32, "dropout": 1.0},
            {"d_model": 32, "eps": 0.0},
            {"d_model": 32, "eps": -1e-5},
            {"d_model": 32, "norm_first": "post"},
            {"d_model": 32, "norm_first": 1},
        ):
            with self.assertRaises(ValueError):
                SublayerConnection(**bad_cfg)

        # Validation error checks for forward
        sub_chk = SublayerConnection(d_model=32)
        with self.assertRaises(ValueError):
            sub_chk(torch.randn(2, 32), lambda h: h)  # 2D
        with self.assertRaises(ValueError):
            sub_chk(torch.randn(2, 4, 8, 32), lambda h: h)  # 4D
        with self.assertRaises(ValueError):
            sub_chk(torch.randn(2, 8, 64), lambda h: h)  # wrong d_model
        with self.assertRaises(TypeError):
            sub_chk(torch.randn(2, 8, 32), "not_callable")  # type: ignore[arg-type]

        mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
        with self.assertRaises(TypeError):
            # Passing MHA tuple output directly instead of tensor
            sub_chk(torch.randn(2, 8, 32), lambda h: mha(h, h, h))  # type: ignore[arg-type, return-value]
        with self.assertRaises(ValueError):
            # Mismatched shape output
            sub_chk(torch.randn(2, 8, 32), lambda h: h[:, :1, :])

    def test_gradient_flow_pre_ln_and_post_ln(self) -> None:
        torch.manual_seed(42)
        d_model, num_heads, d_ff = 64, 4, 256

        for norm_first in (False, True):
            x = torch.randn(3, 10, d_model, requires_grad=True)
            mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
            ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
            sub_attn = SublayerConnection(d_model=d_model, dropout=0.1, norm_first=norm_first)
            sub_ffn = SublayerConnection(d_model=d_model, dropout=0.1, norm_first=norm_first)

            h = sub_attn(x, lambda t: mha(t, t, t)[0])
            out = sub_ffn(h, ffn)
            out.mean().backward()

            assert x.grad is not None
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum().item()), 0.0)

            for mod in (sub_attn.norm, sub_ffn.norm, mha.w_q, mha.w_o, ffn.fc1, ffn.fc2):
                for param in mod.parameters():
                    assert param.grad is not None
                    self.assertTrue(torch.isfinite(param.grad).all())
                    self.assertGreater(float(param.grad.abs().sum().item()), 0.0)

        # Clean identity gradient highway on Pre-LN when sublayer output is zero
        x_id = torch.randn(2, 5, d_model, requires_grad=True)
        pre_ln = SublayerConnection(d_model=d_model, dropout=0.2, norm_first=True)
        grad_out = torch.randn(2, 5, d_model)
        out_id = pre_ln(x_id, torch.zeros_like)
        out_id.backward(grad_out)
        assert x_id.grad is not None
        self.assertTrue(torch.allclose(x_id.grad, grad_out, atol=1e-6))

    def test_end_to_end_encoder_block_pipeline_and_optional_val_batch(self) -> None:
        torch.manual_seed(42)
        d_model, num_heads, d_ff = 256, 4, 1024

        def run_block_check(src_ids: torch.Tensor, norm_first: bool) -> None:
            emb = ScaledEmbedding(vocab_size=8000, d_model=d_model, pad_id=PAD_ID)
            pe = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.2, max_len=512)
            mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.2)
            ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.2)
            sub1 = SublayerConnection(d_model=d_model, dropout=0.2, norm_first=norm_first)
            sub2 = SublayerConnection(d_model=d_model, dropout=0.2, norm_first=norm_first)

            src_mask = (src_ids != PAD_ID).unsqueeze(1).unsqueeze(2)
            x = pe(emb(src_ids))
            h = sub1(x, lambda t: mha(t, t, t, mask=src_mask)[0])
            out = sub2(h, ffn)

            self.assertEqual(tuple(out.shape), (src_ids.size(0), src_ids.size(1), d_model))
            self.assertTrue(torch.isfinite(out).all())

            out.mean().backward()
            for layer in (sub1.norm, sub2.norm, mha.w_q, mha.w_k, mha.w_v, mha.w_o, ffn.fc1, ffn.fc2):
                assert isinstance(layer, (nn.LayerNorm, nn.Linear))
                assert layer.weight.grad is not None
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(float(layer.weight.grad.abs().sum().item()), 0.0)

            assert emb.embedding.weight.grad is not None
            self.assertTrue(torch.all(emb.embedding.weight.grad[PAD_ID] == 0.0))

        synth_ids = torch.randint(4, 8000, (4, 14))
        synth_ids[0, 10:] = PAD_ID
        run_block_check(synth_ids, norm_first=False)
        run_block_check(synth_ids, norm_first=True)

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

        run_block_check(batch["src_ids"], norm_first=False)
        run_block_check(batch["src_ids"], norm_first=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Transformer SublayerConnection ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformerSublayerConnection)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
