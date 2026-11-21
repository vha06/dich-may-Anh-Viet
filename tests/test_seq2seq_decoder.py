"""Unit and integration tests for LSTMDecoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import DEFAULT_VOCAB_SIZE, PAD_ID, load_tokenizer
from src.models.seq2seq_lstm import LSTMDecoder, LSTMEncoder


class _DummyAttention(nn.Module):
    """Minimal Attention mock matching the `(query, encoder_outputs, src_mask)` hook."""

    def __init__(self, dec_dim: int, enc_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.proj(query).unsqueeze(1)  # (B, 1, enc_dim)
        scores = torch.bmm(projected, encoder_outputs.transpose(1, 2)).squeeze(1)
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class TestLSTMDecoder(unittest.TestCase):
    """Test suite verifying single-step shapes, Attention hook, gradients, and Encoder->Decoder flow."""

    def test_single_step_shapes_no_attention(self) -> None:
        bsz, vocab_size, embed_dim, hidden_dim = 4, 200, 64, 512
        for num_layers in (1, 2):
            decoder = LSTMDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.2,
                attention=None,
            )
            self.assertIsNone(decoder.concat_proj)

            hidden = (
                torch.zeros(num_layers, bsz, hidden_dim),
                torch.zeros(num_layers, bsz, hidden_dim),
            )
            for token_input in (
                torch.randint(2, vocab_size, (bsz,)),
                torch.randint(2, vocab_size, (bsz, 1)),
            ):
                logits, (h_next, c_next), attn_weights = decoder(token_input, hidden)
                self.assertEqual(logits.shape, (bsz, vocab_size))
                self.assertEqual(h_next.shape, (num_layers, bsz, hidden_dim))
                self.assertEqual(c_next.shape, (num_layers, bsz, hidden_dim))
                self.assertIsNone(attn_weights)

    def test_plug_and_play_attention_hook(self) -> None:
        bsz, s_max, vocab_size, hidden_dim, enc_dim = 3, 9, 250, 128, 128
        attn = _DummyAttention(dec_dim=hidden_dim, enc_dim=enc_dim)
        decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=64,
            hidden_dim=hidden_dim,
            enc_out_dim=enc_dim,
            num_layers=2,
            dropout=0.1,
            attention=attn,
        )
        self.assertIsNotNone(decoder.concat_proj)

        token_ids = torch.randint(2, vocab_size, (bsz,))
        hidden = (
            torch.zeros(2, bsz, hidden_dim),
            torch.zeros(2, bsz, hidden_dim),
        )
        encoder_outputs = torch.randn(bsz, s_max, enc_dim)
        src_mask = torch.tensor(
            [
                [True, True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )

        with self.assertRaises(ValueError):
            decoder(token_ids, hidden, encoder_outputs=None, src_mask=src_mask)

        logits, (h_next, c_next), attn_weights = decoder(
            token_ids, hidden, encoder_outputs=encoder_outputs, src_mask=src_mask
        )
        self.assertEqual(logits.shape, (bsz, vocab_size))
        self.assertEqual(h_next.shape, (2, bsz, hidden_dim))
        self.assertEqual(c_next.shape, (2, bsz, hidden_dim))
        self.assertIsNotNone(attn_weights)
        assert attn_weights is not None
        self.assertEqual(attn_weights.shape, (bsz, s_max))
        self.assertTrue(
            torch.allclose(attn_weights.sum(dim=-1), torch.ones(bsz), atol=1e-5)
        )
        self.assertTrue(
            torch.equal(attn_weights[~src_mask], torch.zeros_like(attn_weights[~src_mask]))
        )

    def test_multi_step_gradient_flow_and_pad_zero_grad(self) -> None:
        torch.manual_seed(42)
        bsz, t_steps, s_max, vocab_size, hidden_dim = 2, 4, 5, 150, 64

        tgt_in = torch.tensor(
            [
                [2, 11, 22, 33],
                [2, 44, PAD_ID, PAD_ID],
            ],
            dtype=torch.long,
        )
        tgt_out = torch.tensor(
            [
                [11, 22, 33, 3],
                [44, 3, PAD_ID, PAD_ID],
            ],
            dtype=torch.long,
        )
        encoder_outputs = torch.randn(bsz, s_max, hidden_dim, requires_grad=True)
        src_mask = torch.ones(bsz, s_max, dtype=torch.bool)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        for use_attention in (False, True):
            attn = _DummyAttention(hidden_dim, hidden_dim) if use_attention else None
            decoder = LSTMDecoder(
                vocab_size=vocab_size,
                embed_dim=32,
                hidden_dim=hidden_dim,
                enc_out_dim=hidden_dim,
                num_layers=2,
                dropout=0.1,
                attention=attn,
            )
            decoder.train()
            hidden = (
                torch.zeros(2, bsz, hidden_dim),
                torch.zeros(2, bsz, hidden_dim),
            )
            step_logits = []
            for t in range(t_steps):
                logits_t, hidden, _ = decoder(
                    tgt_in[:, t],
                    hidden,
                    encoder_outputs=encoder_outputs if use_attention else None,
                    src_mask=src_mask if use_attention else None,
                )
                step_logits.append(logits_t)

            stacked = torch.stack(step_logits, dim=1)
            loss = criterion(stacked.reshape(-1, vocab_size), tgt_out.reshape(-1))
            decoder.zero_grad()
            loss.backward()

            for name, param in decoder.named_parameters():
                self.assertIsNotNone(param.grad, f"Thieu gradient cho {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Gradient cua {name} chua NaN/Inf",
                )

            pad_grad = decoder.embedding.weight.grad[PAD_ID]
            self.assertTrue(
                torch.equal(pad_grad, torch.zeros_like(pad_grad)),
                "Gradient tai padding_idx=0 cua Decoder Embedding phai bang 0.",
            )

    def test_integration_encoder_to_decoder_real_val_batch(self) -> None:
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
        decoder = LSTMDecoder(vocab_size=tgt_tok.vocab_size, attention=None)
        encoder.eval()
        decoder.eval()

        src_ids = batch["src_ids"]
        src_lengths = batch["src_lengths"]
        tgt_input_ids = batch["tgt_input_ids"]
        tgt_output_ids = batch["tgt_output_ids"]

        bsz, t_max = tgt_input_ids.shape
        with torch.no_grad():
            _, hidden = encoder(src_ids, src_lengths)
            logits_steps = []
            for t in range(t_max):
                logits_t, hidden, attn_w = decoder(tgt_input_ids[:, t], hidden)
                self.assertIsNone(attn_w)
                logits_steps.append(logits_t)
            logits = torch.stack(logits_steps, dim=1)

        self.assertEqual(logits.shape, (bsz, t_max, DEFAULT_VOCAB_SIZE))
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        loss = criterion(logits.reshape(-1, DEFAULT_VOCAB_SIZE), tgt_output_ids.reshape(-1))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 8.5)
        self.assertLess(loss.item(), 9.5)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu LSTMDecoder (Seq2Seq Baseline) ===")
    dec_no_attn = LSTMDecoder(vocab_size=DEFAULT_VOCAB_SIZE, attention=None)
    b_no = dec_no_attn.parameter_breakdown()
    print(
        f"[Khong Attention] embed={b_no['embedding']:,} | rnn={b_no['rnn']:,} | "
        f"fc_out={b_no['fc_out']:,} | tong={b_no['total']:,}"
    )

    dummy_attn = _DummyAttention(dec_dim=512, enc_dim=512)
    dec_with_attn = LSTMDecoder(vocab_size=DEFAULT_VOCAB_SIZE, attention=dummy_attn)
    b_att = dec_with_attn.parameter_breakdown()
    print(
        f"[Co hook Attention (cong them concat_proj)] concat_proj={b_att['concat_proj']:,} | "
        f"decoder_khong_tinh_attn={b_att['total'] - b_att['attention']:,}"
    )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLSTMDecoder)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
