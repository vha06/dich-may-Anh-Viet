"""Unit and integration tests for Greedy Search, Beam Search, and translate_dataloader."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, load_tokenizer
from src.evaluation import (
    DecodeResult,
    beam_search_seq2seq,
    greedy_decode_seq2seq,
    translate_dataloader,
)
from src.models.seq2seq_lstm import build_seq2seq_lstm


class _StubEncoder(nn.Module):
    """Deterministic Encoder stub emitting fixed source encodings."""

    def __init__(self, enc_dim: int = 4) -> None:
        super().__init__()
        self.enc_dim = enc_dim

    def forward(
        self, src_ids: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, s_max = src_ids.shape
        enc_out = torch.ones(bsz, s_max, self.enc_dim)
        h = torch.zeros(1, bsz, self.enc_dim)
        c = torch.zeros(1, bsz, self.enc_dim)
        return enc_out, (h, c)


class _StubDecoder(nn.Module):
    """Deterministic Decoder stub transitioning over a controlled token score table."""

    def __init__(
        self,
        vocab_size: int,
        transitions: dict[tuple[int, int], tuple[dict[int, float], list[float]]],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # Key: (prev_token_id, step_state_id) -> ({tok_id: target_prob}, attention_row)
        self.transitions = transitions

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        bsz = input_token.size(0)
        s_len = encoder_outputs.size(1)
        h_prev, c_prev = hidden

        logits = torch.full((bsz, self.vocab_size), -50.0)
        attn = torch.full((bsz, s_len), 1.0 / s_len)
        h_next = h_prev.clone()

        for b in range(bsz):
            prev_tok = int(input_token[b].item())
            step_state = int(round(float(h_prev[0, b, 0].item())))
            rule = self.transitions.get((prev_tok, step_state))
            if rule is not None:
                probs_map, attn_weights = rule
                for tok_id, prob in probs_map.items():
                    logits[b, tok_id] = math.log(prob)
                attn[b] = torch.tensor(attn_weights[:s_len], dtype=torch.float)
            else:
                logits[b, EOS_ID] = 0.0
            h_next[0, b, 0] = float(step_state + 1)

        return logits, (h_next, c_prev), attn


class _StubSeq2Seq(nn.Module):
    """Minimal duck-typed model exposing encoder/decoder/special IDs for decode tests."""

    def __init__(
        self,
        transitions: dict[tuple[int, int], tuple[dict[int, float], list[float]]],
        vocab_size: int = 10,
    ) -> None:
        super().__init__()
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.encoder = _StubEncoder(enc_dim=4)
        self.decoder = _StubDecoder(vocab_size=vocab_size, transitions=transitions)


class TestDecoding(unittest.TestCase):
    """Test suite verifying masking, mode preservation, beam=1 equivalence, trap graph, and BLEU."""

    def test_invalid_pad_bos_masked_and_train_mode_restored(self) -> None:
        torch.manual_seed(42)
        model = build_seq2seq_lstm(
            src_vocab_size=100,
            tgt_vocab_size=100,
            embed_dim=32,
            enc_hidden_dim=32,
            num_layers=1,
            dropout=0.2,
            attention_method="general",
        )
        with torch.no_grad():
            model.decoder.fc_out.bias[PAD_ID] = 100.0
            model.decoder.fc_out.bias[BOS_ID] = 100.0

        src_ids = torch.tensor([[2, 10, 20, 3], [2, 30, 3, PAD_ID]], dtype=torch.long)
        src_lengths = torch.tensor([4, 3], dtype=torch.long)

        model.train()
        res_greedy = greedy_decode_seq2seq(model, src_ids, src_lengths, max_len=8)
        self.assertTrue(model.training, "greedy_decode phai khoi phuc lai mode train().")

        res_beam = beam_search_seq2seq(
            model, src_ids, src_lengths, beam_size=4, length_penalty=0.6, max_len=8
        )
        self.assertTrue(model.training, "beam_search phai khoi phuc lai mode train().")

        for seq in res_greedy.token_ids + res_beam.token_ids:
            self.assertNotIn(PAD_ID, seq)
            self.assertNotIn(BOS_ID, seq)

    def test_beam_size_1_exact_equivalence_with_greedy(self) -> None:
        torch.manual_seed(42)
        model = build_seq2seq_lstm(
            src_vocab_size=120,
            tgt_vocab_size=120,
            embed_dim=32,
            enc_hidden_dim=32,
            num_layers=2,
            dropout=0.2,
            attention_method="general",
        )
        model.eval()
        src_ids = torch.tensor(
            [
                [2, 15, 25, 35, 45, 55, 3],
                [2, 12, 22, 3, PAD_ID, PAD_ID, PAD_ID],
                [2, 41, 51, 61, 71, 3, PAD_ID],
            ],
            dtype=torch.long,
        )
        src_lengths = torch.tensor([7, 4, 6], dtype=torch.long)

        res_greedy = greedy_decode_seq2seq(
            model, src_ids, src_lengths, max_len=12, length_penalty=0.0
        )
        res_beam1 = beam_search_seq2seq(
            model, src_ids, src_lengths, beam_size=1, length_penalty=0.0, max_len=12
        )

        self.assertEqual(res_greedy.token_ids, res_beam1.token_ids)
        for s_g, s_b in zip(res_greedy.scores, res_beam1.scores):
            self.assertAlmostEqual(s_g, s_b, places=5)

        assert res_greedy.attn_weights is not None and res_beam1.attn_weights is not None
        for i, (a_g, a_b) in enumerate(zip(res_greedy.attn_weights, res_beam1.attn_weights)):
            self.assertEqual(a_g.shape, (len(res_greedy.token_ids[i]), int(src_lengths[i].item())))
            self.assertEqual(a_b.shape, (len(res_beam1.token_ids[i]), int(src_lengths[i].item())))
            self.assertTrue(torch.allclose(a_g, a_b, atol=1e-5))

    def test_beam_search_beats_greedy_on_trap_graph_and_tracks_attention(self) -> None:
        # Normalized probability graph over vocab IDs 4..9 (EOS=3):
        # Step 0 from BOS(2): A(4) p=0.55, B(5) p=0.45 -> Greedy picks A(4)!
        # Step 1 from A(4): spread across 6(0.35), 8(0.33), 9(0.32) -> max branch total = 0.1925
        # Step 1 from B(5): 7(0.95), 8(0.05); then from 7: EOS(0.95), 8(0.05) -> total = 0.4061 >> 0.1925!
        transitions = {
            (BOS_ID, 0): ({4: 0.55, 5: 0.45}, [0.8, 0.1, 0.1]),
            (4, 1): ({6: 0.35, 8: 0.33, 9: 0.32}, [0.1, 0.8, 0.1]),
            (6, 2): ({EOS_ID: 1.0}, [0.1, 0.1, 0.8]),
            (8, 2): ({EOS_ID: 1.0}, [0.3, 0.3, 0.4]),
            (9, 2): ({EOS_ID: 1.0}, [0.4, 0.3, 0.3]),
            (5, 1): ({7: 0.95, 8: 0.05}, [0.2, 0.7, 0.1]),
            (7, 2): ({EOS_ID: 0.95, 8: 0.05}, [0.1, 0.2, 0.7]),
        }
        stub = _StubSeq2Seq(transitions, vocab_size=10)
        src_ids = torch.tensor([[2, 8, 3]], dtype=torch.long)
        src_lengths = torch.tensor([3], dtype=torch.long)

        res_greedy = greedy_decode_seq2seq(stub, src_ids, src_lengths, max_len=6)
        res_beam = beam_search_seq2seq(
            stub, src_ids, src_lengths, beam_size=4, length_penalty=0.0, max_len=6
        )

        self.assertEqual(res_greedy.token_ids[0], [4, 6, EOS_ID])
        self.assertEqual(res_beam.token_ids[0], [5, 7, EOS_ID])
        self.assertGreater(res_beam.scores[0], res_greedy.scores[0])

        # Verify winning beam's exact per-step attention history was preserved
        assert res_beam.attn_weights is not None
        expected_attn = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=torch.float,
        )
        self.assertTrue(torch.allclose(res_beam.attn_weights[0], expected_attn, atol=1e-6))

    def test_length_penalty_and_no_attention_mode(self) -> None:
        # Compare short path [4, EOS] (logp = ln(0.6) + ln(0.8) = -0.734)
        # vs longer path [5, 7, 8, EOS] (logp = ln(0.4) + 3*ln(0.95) = -1.070)
        # With alpha=0.0: short (-0.734) > long (-1.070)
        # With alpha=2.0: LP(2)=(7/6)^2=1.361 (-0.539) vs LP(4)=(9/6)^2=2.250 (-0.476) -> long wins!
        transitions = {
            (BOS_ID, 0): ({4: 0.60, 5: 0.40}, [0.5, 0.5]),
            (4, 1): ({EOS_ID: 0.80, 9: 0.20}, [0.5, 0.5]),
            (5, 1): ({7: 0.95, 9: 0.05}, [0.5, 0.5]),
            (7, 2): ({8: 0.95, 9: 0.05}, [0.5, 0.5]),
            (8, 3): ({EOS_ID: 0.95, 9: 0.05}, [0.5, 0.5]),
        }
        stub = _StubSeq2Seq(transitions, vocab_size=12)
        src_ids = torch.tensor([[2, 3]], dtype=torch.long)
        src_lengths = torch.tensor([2], dtype=torch.long)

        res_no_lp = beam_search_seq2seq(
            stub, src_ids, src_lengths, beam_size=4, length_penalty=0.0, max_len=6
        )
        res_high_lp = beam_search_seq2seq(
            stub, src_ids, src_lengths, beam_size=4, length_penalty=2.0, max_len=6
        )
        self.assertEqual(res_no_lp.token_ids[0], [4, EOS_ID])
        self.assertEqual(res_high_lp.token_ids[0], [5, 7, 8, EOS_ID])

        # Also check no-attention Seq2Seq mode + max_len fallback without EOS
        torch.manual_seed(42)
        model_no_attn = build_seq2seq_lstm(
            src_vocab_size=50,
            tgt_vocab_size=50,
            embed_dim=16,
            enc_hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            attention_method=None,
        )
        with torch.no_grad():
            model_no_attn.decoder.fc_out.weight.zero_()
            model_no_attn.decoder.fc_out.bias.fill_(-5.0)
            model_no_attn.decoder.fc_out.bias[10] = 5.0  # Never emit EOS

        res_g = greedy_decode_seq2seq(model_no_attn, src_ids, src_lengths, max_len=4)
        res_b = beam_search_seq2seq(
            model_no_attn, src_ids, src_lengths, beam_size=3, max_len=4
        )
        self.assertIsNone(res_g.attn_weights)
        self.assertIsNone(res_b.attn_weights)
        self.assertEqual(res_g.token_ids[0], [10, 10, 10, 10])
        self.assertEqual(res_b.token_ids[0], [10, 10, 10, 10])

    def test_translate_dataloader_gold_refs_and_custom_decode_fn(self) -> None:
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

        model = build_seq2seq_lstm(
            src_vocab_size=src_tok.vocab_size,
            tgt_vocab_size=tgt_tok.vocab_size,
            embed_dim=64,
            enc_hidden_dim=64,
            num_layers=1,
            dropout=0.1,
            attention_method="general",
        )

        # Put model in train() intentionally to verify mode restoration
        model.train()
        greedy_eval = translate_dataloader(
            model,
            loader,
            tgt_tok,
            strategy="greedy",
            max_len=16,
            max_batches=1,
        )
        self.assertTrue(model.training)
        self.assertEqual(greedy_eval["num_sentences"], 8)
        self.assertEqual(greedy_eval["references"], ds.tgt_lines[:8])
        # Line 0 in val.vi ends with '?' even though DataLoader tgt_output truncates it
        self.assertTrue(greedy_eval["references"][0].endswith("?"))

        beam_eval = translate_dataloader(
            model,
            loader,
            tgt_tok,
            strategy="beam",
            beam_size=4,
            length_penalty=0.6,
            max_len=16,
            max_batches=1,
        )
        self.assertTrue(model.training)
        self.assertEqual(beam_eval["num_sentences"], 8)
        self.assertEqual(beam_eval["strategy"], "beam")
        self.assertIn("signature", beam_eval["metrics"])

        # Custom decode_fn hook (simulating Week 3 Transformer plug-in)
        def _oracle_stub_decoder(
            _model: nn.Module,
            batch_src: torch.Tensor,
            _batch_lens: torch.Tensor,
            max_len: int = 128,
        ) -> DecodeResult:
            ids = [
                tgt_tok.encode(ds.tgt_lines[idx], add_special_tokens=False, max_len=max_len)
                + [EOS_ID]
                for idx in range(batch_src.size(0))
            ]
            return DecodeResult(
                token_ids=ids,
                scores=[0.0] * len(ids),
                attn_weights=None,
            )

        custom_eval = translate_dataloader(
            model,
            loader,
            tgt_tok,
            decode_fn=_oracle_stub_decoder,
            max_len=128,
            max_batches=1,
        )
        self.assertEqual(custom_eval["strategy"], "custom")
        self.assertGreater(custom_eval["bleu"], 99.0)

    def test_beam_search_contiguous_hidden_states_and_cuda_compat(self) -> None:
        torch.manual_seed(42)
        model = build_seq2seq_lstm(
            src_vocab_size=80,
            tgt_vocab_size=80,
            embed_dim=32,
            enc_hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            attention_method="general",
        )
        model.eval()
        src_ids = torch.tensor(
            [
                [2, 11, 21, 31, 41, 3],
                [2, 12, 22, 3, PAD_ID, PAD_ID],
                [2, 13, 23, 33, 3, PAD_ID],
            ],
            dtype=torch.long,
        )
        src_lengths = torch.tensor([6, 4, 5], dtype=torch.long)

        non_contiguous_calls = 0

        def _check_hx_contiguous(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            nonlocal non_contiguous_calls
            hx = inputs[1]
            if not (hx[0].is_contiguous() and hx[1].is_contiguous()):
                non_contiguous_calls += 1

        handle = model.decoder.rnn.register_forward_pre_hook(_check_hx_contiguous)
        try:
            res_cpu = beam_search_seq2seq(
                model, src_ids, src_lengths, beam_size=4, length_penalty=0.6, max_len=8
            )
        finally:
            handle.remove()

        self.assertEqual(non_contiguous_calls, 0)
        self.assertEqual(len(res_cpu.token_ids), 3)

        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            model_cuda = model.to(cuda_device)
            res_cuda = beam_search_seq2seq(
                model_cuda,
                src_ids.to(cuda_device),
                src_lengths.to(cuda_device),
                beam_size=4,
                length_penalty=0.6,
                max_len=8,
            )
            self.assertEqual(len(res_cuda.token_ids), 3)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Bo giai ma Greedy Search & Beam Search ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDecoding)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
