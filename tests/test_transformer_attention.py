"""Unit and integration tests for Transformer Scaled Dot-Product & Multi-Head Attention."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import PAD_ID, load_tokenizer
from src.models.transformer import (
    MultiHeadAttention,
    ScaledEmbedding,
    SinusoidalPositionalEncoding,
    scaled_dot_product_attention,
)


class TestTransformerAttention(unittest.TestCase):
    """Test suite verifying SDPA math, 3 MHA roles, pad/causal invariance, NaN guard, and AMP."""

    def test_scaled_dot_product_attention_exact_formula_and_scaling(self) -> None:
        q = torch.tensor([[[[1.0, 0.0, 2.0, -1.0], [0.5, 1.5, -0.5, 0.5]]]])
        k = torch.tensor([[[[1.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, -1.0], [2.0, -1.0, 0.5, 1.0]]]])
        v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])

        ctx, attn = scaled_dot_product_attention(q, k, v)
        self.assertEqual(tuple(ctx.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(attn.shape), (1, 1, 2, 3))

        # Manual closed-form reference
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(4)
        expected_attn = torch.softmax(raw_scores, dim=-1)
        expected_ctx = torch.matmul(expected_attn, v)

        self.assertTrue(torch.allclose(attn, expected_attn, atol=1e-6))
        self.assertTrue(torch.allclose(ctx, expected_ctx, atol=1e-6))
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(1, 1, 2), atol=1e-6))

        # Verify scaling by 1/sqrt(d_k) keeps dot-product variance ~ 1.0 at large d_k=256
        torch.manual_seed(42)
        q_big = torch.randn(2, 4, 10, 256)
        k_big = torch.randn(2, 4, 10, 256)
        scaled_scores = torch.matmul(q_big, k_big.transpose(-2, -1)) / math.sqrt(256)
        scaled_std = float(scaled_scores.std().item())
        self.assertGreater(scaled_std, 0.8)
        self.assertLess(scaled_std, 1.2)

    def test_three_attention_roles_and_pad_invariance(self) -> None:
        torch.manual_seed(42)
        d_model, num_heads = 256, 4
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        mha.eval()

        # 1. Encoder Self-Attention (padding invariance up to 1e5 perturbation)
        bsz, s_len = 3, 12
        src = torch.randn(bsz, s_len, d_model)
        src_valid = torch.ones(bsz, s_len, dtype=torch.bool)
        src_valid[0, 8:] = False
        src_valid[1, 5:] = False
        src_mask = src_valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        out_enc, attn_enc = mha(src, src, src, mask=src_mask)
        self.assertEqual(tuple(out_enc.shape), (bsz, s_len, d_model))
        self.assertEqual(tuple(attn_enc.shape), (bsz, num_heads, s_len, s_len))
        self.assertTrue(torch.all(attn_enc[0, :, :, 8:] == 0.0))
        self.assertTrue(torch.all(attn_enc[1, :, :, 5:] == 0.0))

        # Mutate <pad> positions to 1e5 and confirm real query outputs remain identical
        src_perturbed = src.clone()
        src_perturbed[0, 8:] = 1e5
        src_perturbed[1, 5:] = -1e5
        out_enc_p, _ = mha(src_perturbed, src_perturbed, src_perturbed, mask=src_mask)
        self.assertTrue(torch.allclose(out_enc[0, :8], out_enc_p[0, :8], atol=1e-5))
        self.assertTrue(torch.allclose(out_enc[1, :5], out_enc_p[1, :5], atol=1e-5))

        # 2. Decoder Masked Self-Attention (zero future leakage)
        t_len = 10
        tgt = torch.randn(bsz, t_len, d_model)
        causal_mask = torch.tril(torch.ones(t_len, t_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out_dec, attn_dec = mha(tgt, tgt, tgt, mask=causal_mask)
        self.assertEqual(tuple(out_dec.shape), (bsz, t_len, d_model))
        upper = torch.triu(torch.ones(t_len, t_len, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.all(attn_dec[:, :, upper] == 0.0))

        # Corrupt future tokens after t0=4 -> outputs at 0..4 must remain invariant
        tgt_future = tgt.clone()
        tgt_future[:, 5:, :] += 1e4
        out_dec_f, _ = mha(tgt_future, tgt_future, tgt_future, mask=causal_mask)
        self.assertTrue(torch.allclose(out_dec[:, :5, :], out_dec_f[:, :5, :], atol=1e-5))

        # 3. Cross-Attention (L_q=9 != L_k=12) + source <pad> invariance
        tgt_cross = torch.randn(bsz, 9, d_model)
        out_cross, attn_cross = mha(tgt_cross, src, src, mask=src_mask)
        self.assertEqual(tuple(out_cross.shape), (bsz, 9, d_model))
        self.assertEqual(tuple(attn_cross.shape), (bsz, num_heads, 9, s_len))
        self.assertTrue(torch.all(attn_cross[0, :, :, 8:] == 0.0))

        out_cross_p, _ = mha(tgt_cross, src_perturbed, src_perturbed, mask=src_mask)
        self.assertTrue(torch.allclose(out_cross, out_cross_p, atol=1e-5))

    def test_all_masked_row_no_nan_in_forward_and_backward(self) -> None:
        q = torch.randn(2, 4, 5, 16, requires_grad=True)
        k = torch.randn(2, 4, 6, 16, requires_grad=True)
        v = torch.randn(2, 4, 6, 16, requires_grad=True)

        mask = torch.ones(2, 1, 5, 6, dtype=torch.bool)
        mask[0, :, 2, :] = False  # Entire query row masked out for sentence 0, pos 2
        mask[1, :, :, :] = False  # Entire sentence 1 masked out

        ctx, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        self.assertTrue(torch.isfinite(ctx).all())
        self.assertTrue(torch.isfinite(attn).all())
        self.assertTrue(torch.all(attn[0, :, 2, :] == 0.0))
        self.assertTrue(torch.all(ctx[0, :, 2, :] == 0.0))
        self.assertTrue(torch.all(attn[1] == 0.0))
        self.assertTrue(torch.all(ctx[1] == 0.0))

        ctx.sum().backward()
        for grad in (q.grad, k.grad, v.grad):
            assert grad is not None
            self.assertTrue(torch.isfinite(grad).all())

    def test_param_count_store_attn_dropout_and_amp_bfloat16(self) -> None:
        torch.manual_seed(42)
        mha = MultiHeadAttention(d_model=256, num_heads=4, dropout=0.2, bias=True)
        total_params = sum(p.numel() for p in mha.parameters())
        # 4 * (256 * 256 + 256) = 263,168
        self.assertEqual(total_params, 263_168)
        self.assertIsNone(mha.last_attn_weights)

        x = torch.randn(2, 6, 256)
        mha.eval()
        out_eval, attn_eval = mha(x, x, x, store_attn=True)
        self.assertIsNotNone(mha.last_attn_weights)
        assert mha.last_attn_weights is not None
        self.assertFalse(mha.last_attn_weights.requires_grad)

        # Calling with store_attn=False must clear last_attn_weights to free VRAM
        _, _ = mha(x, x, x, store_attn=False)
        self.assertIsNone(mha.last_attn_weights)

        # In train mode with dropout=0.2, output differs while returned attn_weights still sum to 1.0
        mha.train()
        out_train, attn_train = mha(x, x, x)
        self.assertFalse(torch.allclose(out_eval, out_train))
        self.assertEqual(attn_train.dtype, torch.float32)
        self.assertTrue(torch.allclose(attn_train.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-5))
        self.assertTrue(torch.allclose(attn_eval.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-5))

        # AMP bfloat16 check (CPU + CUDA if available)
        mha.eval()
        x_bf16 = x.to(torch.bfloat16)
        mha_bf16 = mha.to(torch.bfloat16)
        out_bf16, attn_bf16 = mha_bf16(x_bf16, x_bf16, x_bf16)
        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertEqual(attn_bf16.dtype, torch.float32)
        self.assertTrue(torch.isfinite(out_bf16).all())
        self.assertTrue(torch.allclose(attn_bf16.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-5))

        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            mha_cuda = mha_bf16.to(cuda_dev)
            out_cuda, attn_cuda = mha_cuda(
                x_bf16.to(cuda_dev), x_bf16.to(cuda_dev), x_bf16.to(cuda_dev), store_attn=True
            )
            self.assertEqual(out_cuda.device.type, "cuda")
            self.assertEqual(out_cuda.dtype, torch.bfloat16)
            self.assertEqual(attn_cuda.dtype, torch.float32)
            self.assertTrue(torch.isfinite(out_cuda).all())
            _, _ = mha_cuda(x_bf16.to(cuda_dev), x_bf16.to(cuda_dev), x_bf16.to(cuda_dev), store_attn=False)
            self.assertIsNone(mha_cuda.last_attn_weights)

    def test_validation_errors(self) -> None:
        for bad_cfg in (
            {"d_model": 0, "num_heads": 4},
            {"d_model": 256, "num_heads": 0},
            {"d_model": 250, "num_heads": 4},
            {"d_model": 256, "num_heads": 4, "dropout": -0.1},
            {"d_model": 256, "num_heads": 4, "dropout": 1.0},
        ):
            with self.assertRaises(ValueError):
                MultiHeadAttention(**bad_cfg)

        mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
        q = torch.randn(4, 4, 64)  # Note B == L_q == 4 intentionally!
        # Must reject 2D/3D masks so B == L_q cannot silently mis-broadcast
        with self.assertRaises(ValueError):
            mha(q, q, q, mask=torch.ones(4, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            mha(q, q, q, mask=torch.ones(4, 1, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            mha(q, q, q, mask=torch.ones(4, 1, 1, 4, dtype=torch.float32))
        with self.assertRaises(ValueError):
            mha(q, q, q, mask=torch.ones(4, 2, 1, 4, dtype=torch.bool))  # 2 != 4 heads

        with self.assertRaises(ValueError):
            mha(torch.randn(4, 64), q, q)  # 2D query
        with self.assertRaises(ValueError):
            mha(q, torch.randn(3, 4, 64), torch.randn(3, 4, 64))  # batch mismatch
        with self.assertRaises(ValueError):
            mha(q, torch.randn(4, 5, 64), torch.randn(4, 6, 64))  # L_k != L_v

    def test_end_to_end_embedding_pe_mha_gradients_and_optional_val_batch(self) -> None:
        torch.manual_seed(42)
        d_model, num_heads = 256, 4
        src_emb = ScaledEmbedding(vocab_size=8000, d_model=d_model, pad_id=PAD_ID)
        tgt_emb = ScaledEmbedding(vocab_size=8000, d_model=d_model, pad_id=PAD_ID)
        pe = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.2, max_len=512)
        self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.2)
        cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.2)

        src_ids = torch.randint(4, 8000, (4, 12))
        src_ids[0, 9:] = PAD_ID
        tgt_ids = torch.randint(4, 8000, (4, 9))
        tgt_ids[1, 6:] = PAD_ID

        src_mask = (src_ids != PAD_ID).unsqueeze(1).unsqueeze(2)  # (4, 1, 1, 12)
        tgt_pad = (tgt_ids != PAD_ID).unsqueeze(1).unsqueeze(2)  # (4, 1, 1, 9)
        causal = torch.tril(torch.ones(9, 9, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad & causal

        enc_in = pe(src_emb(src_ids))
        enc_out, _ = self_attn(enc_in, enc_in, enc_in, mask=src_mask)
        dec_in = pe(tgt_emb(tgt_ids))
        dec_self, _ = self_attn(dec_in, dec_in, dec_in, mask=tgt_mask)
        dec_cross, _ = cross_attn(dec_self, enc_out, enc_out, mask=src_mask)

        dec_cross.mean().backward()

        for proj_name, proj in (
            ("w_q", cross_attn.w_q),
            ("w_k", cross_attn.w_k),
            ("w_v", cross_attn.w_v),
            ("w_o", cross_attn.w_o),
            ("self.w_q", self_attn.w_q),
            ("self.w_k", self_attn.w_k),
            ("self.w_v", self_attn.w_v),
            ("self.w_o", self_attn.w_o),
        ):
            assert isinstance(proj, nn.Linear)
            self.assertIsNotNone(proj.weight.grad, proj_name)
            assert proj.weight.grad is not None
            self.assertTrue(torch.isfinite(proj.weight.grad).all(), proj_name)
            self.assertGreater(float(proj.weight.grad.abs().sum().item()), 0.0, proj_name)
            if proj.bias is not None:
                self.assertIsNotNone(proj.bias.grad, proj_name)
                assert proj.bias.grad is not None
                self.assertTrue(torch.isfinite(proj.bias.grad).all(), proj_name)

        assert src_emb.embedding.weight.grad is not None
        assert tgt_emb.embedding.weight.grad is not None
        self.assertTrue(torch.all(src_emb.embedding.weight.grad[PAD_ID] == 0.0))
        self.assertTrue(torch.all(tgt_emb.embedding.weight.grad[PAD_ID] == 0.0))

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
        b_tgt = batch["tgt_input_ids"]
        b_src_mask = (b_src != PAD_ID).unsqueeze(1).unsqueeze(2)
        b_t_len = b_tgt.size(1)
        b_tgt_mask = (b_tgt != PAD_ID).unsqueeze(1).unsqueeze(2) & (
            torch.tril(torch.ones(b_t_len, b_t_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        )

        mha_real = MultiHeadAttention(d_model=256, num_heads=4, dropout=0.2)
        out_r_enc, attn_r_enc = mha_real(pe(src_emb(b_src)), pe(src_emb(b_src)), pe(src_emb(b_src)), mask=b_src_mask)
        out_r_dec, attn_r_cross = mha_real(pe(tgt_emb(b_tgt)), out_r_enc, out_r_enc, mask=b_src_mask)
        self.assertEqual(tuple(out_r_enc.shape), (8, b_src.size(1), 256))
        self.assertEqual(tuple(attn_r_enc.shape), (8, 4, b_src.size(1), b_src.size(1)))
        self.assertEqual(tuple(out_r_dec.shape), (8, b_t_len, 256))
        self.assertEqual(tuple(attn_r_cross.shape), (8, 4, b_t_len, b_src.size(1)))
        # Ensure masked self-attention on real batch preserves exact causal shape
        out_r_self, attn_r_self = mha_real(pe(tgt_emb(b_tgt)), pe(tgt_emb(b_tgt)), pe(tgt_emb(b_tgt)), mask=b_tgt_mask)
        self.assertEqual(tuple(out_r_self.shape), (8, b_t_len, 256))
        self.assertEqual(tuple(attn_r_self.shape), (8, 4, b_t_len, b_t_len))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Transformer Scaled Dot-Product & Multi-Head Attention ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformerAttention)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
