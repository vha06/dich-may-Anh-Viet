"""Unit and integration tests for Transformer Padding and Causal Look-Ahead Masks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import PAD_ID, load_tokenizer
from src.models.transformer import (
    MultiHeadAttention,
    ScaledEmbedding,
    SinusoidalPositionalEncoding,
    SublayerConnection,
    create_causal_mask,
    create_masks,
    create_padding_mask,
    create_target_mask,
)


class TestTransformerMasks(unittest.TestCase):
    """Test suite verifying mask shapes, boolean semantics, edge validation, and MHA integration."""

    def test_create_padding_mask_shapes_values_and_custom_pad_id(self) -> None:
        token_ids = torch.tensor(
            [
                [5, 6, 7, 8],
                [5, 6, PAD_ID, PAD_ID],
                [PAD_ID, PAD_ID, PAD_ID, PAD_ID],
            ],
            dtype=torch.long,
        )
        mask = create_padding_mask(token_ids, pad_id=PAD_ID)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(tuple(mask.shape), (3, 1, 1, 4))
        self.assertTrue(torch.all(mask[0, 0, 0]).item())
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, False, False])
        self.assertFalse(torch.any(mask[2, 0, 0]).item())

        custom_ids = torch.tensor([[10, 99, 20], [99, 99, 30]], dtype=torch.int32)
        custom_mask = create_padding_mask(custom_ids, pad_id=99)
        self.assertEqual(custom_mask.dtype, torch.bool)
        self.assertEqual(custom_mask[0, 0, 0].tolist(), [True, False, True])
        self.assertEqual(custom_mask[1, 0, 0].tolist(), [False, False, True])

    def test_create_causal_mask_lower_triangular_and_device(self) -> None:
        m1 = create_causal_mask(1)
        self.assertEqual(m1.dtype, torch.bool)
        self.assertEqual(tuple(m1.shape), (1, 1, 1, 1))
        self.assertTrue(bool(m1.item()))

        t_len = 6
        m6 = create_causal_mask(t_len)
        self.assertEqual(m6.dtype, torch.bool)
        self.assertEqual(tuple(m6.shape), (1, 1, t_len, t_len))
        for i in range(t_len):
            for j in range(t_len):
                self.assertEqual(bool(m6[0, 0, i, j].item()), j <= i)

        if torch.cuda.is_available():
            m_cuda = create_causal_mask(4, device="cuda")
            self.assertEqual(m_cuda.device.type, "cuda")
            self.assertEqual(m_cuda.dtype, torch.bool)

    def test_create_target_mask_create_masks_and_validation_errors(self) -> None:
        src_ids = torch.tensor([[4, 5, 6, 0, 0], [7, 8, 9, 10, 11]], dtype=torch.long)
        tgt_ids = torch.tensor([[2, 12, 13, 0], [2, 14, 0, 0]], dtype=torch.long)

        tgt_mask = create_target_mask(tgt_ids, pad_id=PAD_ID)
        self.assertEqual(tgt_mask.dtype, torch.bool)
        self.assertEqual(tuple(tgt_mask.shape), (2, 1, 4, 4))

        for b in range(2):
            for i in range(4):
                for j in range(4):
                    expected = (j <= i) and (int(tgt_ids[b, j].item()) != PAD_ID)
                    self.assertEqual(bool(tgt_mask[b, 0, i, j].item()), expected)

        s_mask, t_mask = create_masks(src_ids, tgt_ids, pad_id=PAD_ID)
        self.assertEqual(tuple(s_mask.shape), (2, 1, 1, 5))
        self.assertEqual(tuple(t_mask.shape), (2, 1, 4, 4))
        self.assertEqual(s_mask.device, src_ids.device)
        self.assertEqual(t_mask.device, tgt_ids.device)

        # Validation error checks
        with self.assertRaises(ValueError):
            create_padding_mask(torch.empty((0, 5), dtype=torch.long))  # B == 0
        with self.assertRaises(ValueError):
            create_target_mask(torch.empty((0, 4), dtype=torch.long))  # B == 0
        with self.assertRaises(ValueError):
            create_padding_mask(torch.empty((2, 0), dtype=torch.long))  # L == 0
        with self.assertRaises(ValueError):
            create_padding_mask(torch.tensor([1, 2, 3], dtype=torch.long))  # 1D
        with self.assertRaises(ValueError):
            create_padding_mask(torch.ones((1, 2, 3), dtype=torch.long))  # 3D

        for bad_dtype in (torch.float32, torch.bool, torch.complex64, torch.complex128):
            with self.assertRaises(ValueError):
                create_padding_mask(torch.ones((2, 4), dtype=bad_dtype))

        for bad_pad in (True, 0.0, "0"):
            with self.assertRaises(ValueError):
                create_padding_mask(src_ids, pad_id=bad_pad)  # type: ignore[arg-type]

        for bad_len in (0, -3, True, 4.0):
            with self.assertRaises(ValueError):
                create_causal_mask(bad_len)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            create_masks(src_ids[:1], tgt_ids)  # mismatched batch size

        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                create_masks(src_ids.cuda(), tgt_ids.cpu())

    def test_mha_three_roles_invariance_on_cpu_float32_and_all_pad_safety(self) -> None:
        torch.manual_seed(42)
        # Keep strictly on CPU float32 and set mha.eval() before torch.equal assertions
        mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.2).cpu()
        mha.eval()

        src_ids = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 0, 0, 0]], dtype=torch.long)
        tgt_ids = torch.tensor([[2, 10, 11, 12, 0, 0], [2, 13, 14, 0, 0, 0]], dtype=torch.long)
        src_mask, tgt_mask = create_masks(src_ids, tgt_ids, pad_id=PAD_ID)

        enc_x = torch.randn(2, 5, 64, dtype=torch.float32)
        enc_x_pad_pert = enc_x.clone()
        enc_x_pad_pert[0, 3:, :] += 1e5
        enc_x_pad_pert[1, 2:, :] += 1e5

        # 1. Encoder Self-Attention
        enc_out, enc_attn = mha(enc_x, enc_x, enc_x, mask=src_mask)
        enc_out_pert, _ = mha(enc_x_pad_pert, enc_x_pad_pert, enc_x_pad_pert, mask=src_mask)
        self.assertTrue(torch.all(enc_attn[0, :, :3, 3:] == 0.0))
        self.assertTrue(torch.all(enc_attn[1, :, :2, 2:] == 0.0))
        self.assertTrue(
            torch.allclose(enc_attn.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)
        )
        self.assertTrue(torch.equal(enc_out[0, :3], enc_out_pert[0, :3]))
        self.assertTrue(torch.equal(enc_out[1, :2], enc_out_pert[1, :2]))

        # 2. Decoder Masked Self-Attention
        dec_x = torch.randn(2, 6, 64, dtype=torch.float32)
        dec_out, dec_self_attn = mha(dec_x, dec_x, dec_x, mask=tgt_mask)
        self.assertTrue(torch.all(dec_self_attn[0, :, :, 4:] == 0.0))
        self.assertTrue(torch.all(dec_self_attn[1, :, :, 3:] == 0.0))
        upper = ~create_causal_mask(6).squeeze(0).squeeze(0)
        self.assertTrue(torch.all(dec_self_attn[:, :, upper] == 0.0))

        k_cutoff = 3
        dec_x_future_pert = dec_x.clone()
        dec_x_future_pert[:, k_cutoff:, :] += 1e5
        dec_out_pert, _ = mha(dec_x_future_pert, dec_x_future_pert, dec_x_future_pert, mask=tgt_mask)
        self.assertTrue(torch.equal(dec_out[:, :k_cutoff], dec_out_pert[:, :k_cutoff]))

        # 3. Decoder Cross-Attention (T=6 != S=5)
        cross_out, cross_attn = mha(dec_x, enc_x, enc_x, mask=src_mask)
        cross_out_pert, _ = mha(dec_x, enc_x_pad_pert, enc_x_pad_pert, mask=src_mask)
        self.assertEqual(tuple(cross_attn.shape), (2, 4, 6, 5))
        self.assertTrue(torch.all(cross_attn[0, :, :, 3:] == 0.0))
        self.assertTrue(torch.all(cross_attn[1, :, :, 2:] == 0.0))
        self.assertTrue(torch.equal(cross_out, cross_out_pert))

        # Edge case: all-<pad> batch in train() mode must remain finite in forward and backward
        mha.train()
        all_pad_ids = torch.full((2, 5), PAD_ID, dtype=torch.long)
        ap_src_mask, ap_tgt_mask = create_masks(all_pad_ids, all_pad_ids, pad_id=PAD_ID)
        ap_x = torch.randn(2, 5, 64, requires_grad=True)
        ap_out_enc, _ = mha(ap_x, ap_x, ap_x, mask=ap_src_mask)
        ap_out_dec, _ = mha(ap_x, ap_x, ap_x, mask=ap_tgt_mask)
        loss = (ap_out_enc + ap_out_dec).sum()
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        assert ap_x.grad is not None
        self.assertTrue(torch.isfinite(ap_x.grad).all())

    def test_real_val_batch_integration_and_cuda_bfloat16_optional(self) -> None:
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu val hoac BPE tokenizer de chay test batch that.")

        torch.manual_seed(42)
        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, max_len=128, pre_tokenize=True)
        batch = next(iter(create_dataloader(ds, batch_size=8, shuffle=False)))

        d_model, num_heads = 256, 4
        src_emb = ScaledEmbedding(src_tok.vocab_size, d_model, pad_id=PAD_ID)
        tgt_emb = ScaledEmbedding(tgt_tok.vocab_size, d_model, pad_id=PAD_ID)
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.1, max_len=256)
        mha_enc = MultiHeadAttention(d_model, num_heads, dropout=0.1)
        mha_dec = MultiHeadAttention(d_model, num_heads, dropout=0.1)
        mha_cross = MultiHeadAttention(d_model, num_heads, dropout=0.1)
        sub_enc = SublayerConnection(d_model, dropout=0.1, norm_first=False)
        sub_dec = SublayerConnection(d_model, dropout=0.1, norm_first=False)
        sub_cross = SublayerConnection(d_model, dropout=0.1, norm_first=False)

        for mod in (src_emb, tgt_emb, pe, mha_enc, mha_dec, mha_cross, sub_enc, sub_dec, sub_cross):
            mod.eval()

        def verify_device_dtype(
            src_ids: torch.Tensor,
            tgt_in_ids: torch.Tensor,
            device: torch.device,
            use_bf16: bool,
        ) -> None:
            for mod in (src_emb, tgt_emb, pe, mha_enc, mha_dec, mha_cross, sub_enc, sub_dec, sub_cross):
                mod.to(device=device, dtype=torch.bfloat16 if use_bf16 else torch.float32)

            src_ids_d = src_ids.to(device)
            tgt_in_d = tgt_in_ids.to(device)
            src_mask, tgt_mask = create_masks(src_ids_d, tgt_in_d, pad_id=PAD_ID)

            enc_in = pe(src_emb(src_ids_d))
            enc_weights_box: list[torch.Tensor] = []

            def run_enc_attn(t: torch.Tensor) -> torch.Tensor:
                out, w = mha_enc(t, t, t, mask=src_mask)
                enc_weights_box.append(w)
                return out

            enc_out = sub_enc(enc_in, run_enc_attn)

            dec_in = pe(tgt_emb(tgt_in_d))
            dec_weights_box: list[torch.Tensor] = []

            def run_dec_attn(t: torch.Tensor) -> torch.Tensor:
                out, w = mha_dec(t, t, t, mask=tgt_mask)
                dec_weights_box.append(w)
                return out

            dec_h = sub_dec(dec_in, run_dec_attn)

            cross_weights_box: list[torch.Tensor] = []

            def run_cross_attn(t: torch.Tensor) -> torch.Tensor:
                out, w = mha_cross(t, enc_out, enc_out, mask=src_mask)
                cross_weights_box.append(w)
                return out

            dec_out = sub_cross(dec_h, run_cross_attn)

            expected_dtype = torch.bfloat16 if use_bf16 else torch.float32
            self.assertEqual(enc_out.dtype, expected_dtype)
            self.assertEqual(dec_out.dtype, expected_dtype)
            self.assertTrue(torch.isfinite(enc_out).all())
            self.assertTrue(torch.isfinite(dec_out).all())

            for w, m in (
                (enc_weights_box[0], src_mask),
                (dec_weights_box[0], tgt_mask),
                (cross_weights_box[0], src_mask),
            ):
                self.assertEqual(w.dtype, torch.float32)
                self.assertTrue(torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5))
                masked_pos = (~m).expand_as(w)
                self.assertTrue(torch.all(w[masked_pos] == 0.0))

        verify_device_dtype(batch["src_ids"], batch["tgt_input_ids"], torch.device("cpu"), False)
        if torch.cuda.is_available():
            verify_device_dtype(batch["src_ids"], batch["tgt_input_ids"], torch.device("cuda"), True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Transformer Masks ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformerMasks)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
