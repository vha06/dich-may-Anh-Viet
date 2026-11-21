"""Unit and integration tests for LSTMEncoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import DEFAULT_VOCAB_SIZE, PAD_ID, load_tokenizer
from src.models.seq2seq_lstm import LSTMEncoder


class TestLSTMEncoder(unittest.TestCase):
    """Test suite verifying shapes, padding invariance, gradients, and real batch integration."""

    def test_output_shapes_all_configs(self) -> None:
        bsz, s_max = 4, 15
        src_ids = torch.randint(4, 100, (bsz, s_max))
        src_lengths = torch.tensor([15, 11, 8, 5], dtype=torch.long)

        configs = [
            # Default: 2-layer BiLSTM (256 -> 512)
            dict(
                vocab_size=200,
                embed_dim=64,
                hidden_dim=256,
                dec_hidden_dim=None,
                num_layers=2,
                bidirectional=True,
                expected_out_dim=512,
                expected_dec_dim=512,
                expect_bridge=True,
            ),
            # 1-layer BiLSTM (128 -> 256)
            dict(
                vocab_size=200,
                embed_dim=64,
                hidden_dim=128,
                dec_hidden_dim=None,
                num_layers=1,
                bidirectional=True,
                expected_out_dim=256,
                expected_dec_dim=256,
                expect_bridge=True,
            ),
            # 2-layer UniLSTM without bridge (128 -> 128)
            dict(
                vocab_size=200,
                embed_dim=64,
                hidden_dim=128,
                dec_hidden_dim=None,
                num_layers=2,
                bidirectional=False,
                expected_out_dim=128,
                expected_dec_dim=128,
                expect_bridge=False,
            ),
            # 2-layer UniLSTM with bridge to mismatched dec_hidden_dim (128 -> 256)
            dict(
                vocab_size=200,
                embed_dim=64,
                hidden_dim=128,
                dec_hidden_dim=256,
                num_layers=2,
                bidirectional=False,
                expected_out_dim=128,
                expected_dec_dim=256,
                expect_bridge=True,
            ),
        ]

        for cfg in configs:
            encoder = LSTMEncoder(
                vocab_size=cfg["vocab_size"],
                embed_dim=cfg["embed_dim"],
                hidden_dim=cfg["hidden_dim"],
                dec_hidden_dim=cfg["dec_hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=0.2,
                bidirectional=cfg["bidirectional"],
            )
            has_bridge = encoder.fc_hidden is not None and encoder.fc_cell is not None
            self.assertEqual(has_bridge, cfg["expect_bridge"])

            outputs, (h_n, c_n) = encoder(src_ids, src_lengths)
            self.assertEqual(outputs.shape, (bsz, s_max, cfg["expected_out_dim"]))
            self.assertEqual(
                h_n.shape, (cfg["num_layers"], bsz, cfg["expected_dec_dim"])
            )
            self.assertEqual(
                c_n.shape, (cfg["num_layers"], bsz, cfg["expected_dec_dim"])
            )

    def test_padding_invariance_and_zero_pad_outputs(self) -> None:
        torch.manual_seed(42)
        encoder = LSTMEncoder(
            vocab_size=500,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
        )
        encoder.eval()

        tokens = torch.tensor([[2, 15, 27, 42, 88, 19, 3]], dtype=torch.long)
        len_single = torch.tensor([7], dtype=torch.long)

        with torch.no_grad():
            out_single, (h_single, c_single) = encoder(tokens, len_single)

            padded_batch = torch.tensor(
                [
                    [2, 15, 27, 42, 88, 19, 3, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
                    [2, 99, 101, 55, 77, 33, 21, 44, 66, 88, 11, 3],
                ],
                dtype=torch.long,
            )
            len_batch = torch.tensor([7, 12], dtype=torch.long)
            out_batch, (h_batch, c_batch) = encoder(padded_batch, len_batch)

        self.assertTrue(
            torch.allclose(out_single[0], out_batch[0, :7], atol=1e-6),
            "Dau ra encoder_outputs tai cac tu thuc bi anh huong boi padding.",
        )
        self.assertTrue(
            torch.allclose(h_single[:, 0], h_batch[:, 0], atol=1e-6),
            "Trang thai an h_n bi anh huong boi padding.",
        )
        self.assertTrue(
            torch.allclose(c_single[:, 0], c_batch[:, 0], atol=1e-6),
            "Trang thai o nho c_n bi anh huong boi padding.",
        )
        self.assertTrue(
            torch.equal(out_batch[0, 7:], torch.zeros_like(out_batch[0, 7:])),
            "Cac vi tri padding trong encoder_outputs khong bang 0.0.",
        )

    def test_gradient_flow_and_pad_embedding_zero_grad(self) -> None:
        torch.manual_seed(42)
        encoder = LSTMEncoder(
            vocab_size=300,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
        )
        encoder.train()

        src_ids = torch.tensor(
            [
                [2, 10, 20, 30, 3],
                [2, 40, 3, PAD_ID, PAD_ID],
            ],
            dtype=torch.long,
        )
        src_lengths = torch.tensor([5, 3], dtype=torch.long)

        outputs, (h_n, c_n) = encoder(src_ids, src_lengths)
        loss = outputs.sum() + h_n.sum() + c_n.sum()
        encoder.zero_grad()
        loss.backward()

        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Thieu gradient cho {name}")
            self.assertTrue(
                torch.isfinite(param.grad).all(),
                f"Gradient cua {name} chua NaN hoac Inf",
            )

        pad_grad = encoder.embedding.weight.grad[PAD_ID]
        self.assertTrue(
            torch.equal(pad_grad, torch.zeros_like(pad_grad)),
            "Gradient tai padding_idx=0 cua Embedding phai bang 0.",
        )

    def test_integration_with_real_val_dataloader(self) -> None:
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu hoac model BPE de chay integration test.")

        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, pre_tokenize=True)
        loader = create_dataloader(ds, batch_size=64, shuffle=False)
        batch = next(iter(loader))

        encoder = LSTMEncoder(vocab_size=src_tok.vocab_size)
        encoder.eval()

        src_ids = batch["src_ids"]
        src_lengths = batch["src_lengths"]
        with torch.no_grad():
            outputs, (h_n, c_n) = encoder(src_ids, src_lengths)

        bsz, s_max = src_ids.shape
        self.assertEqual(src_tok.vocab_size, DEFAULT_VOCAB_SIZE)
        self.assertEqual(outputs.shape, (bsz, s_max, 512))
        self.assertEqual(h_n.shape, (2, bsz, 512))
        self.assertEqual(c_n.shape, (2, bsz, 512))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(h_n).all())
        self.assertTrue(torch.isfinite(c_n).all())

        pad_mask = src_ids == PAD_ID
        self.assertTrue(
            torch.equal(outputs[pad_mask], torch.zeros_like(outputs[pad_mask]))
        )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu LSTMEncoder (Seq2Seq Baseline) ===")
    encoder = LSTMEncoder(vocab_size=DEFAULT_VOCAB_SIZE)
    breakdown = encoder.parameter_breakdown()
    print(
        f"Cau hinh mac dinh: vocab={encoder.vocab_size}, embed={encoder.embed_dim}, "
        f"enc_hidden={encoder.hidden_dim}, bidirectional={encoder.bidirectional}, "
        f"num_layers={encoder.num_layers}, output_dim={encoder.output_dim}, "
        f"dec_hidden_dim={encoder.dec_hidden_dim}"
    )
    print(
        f"Tham so: embedding={breakdown['embedding']:,} | "
        f"rnn={breakdown['rnn']:,} | "
        f"bridge={breakdown['bridge']:,} | "
        f"tong={breakdown['total']:,}"
    )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLSTMEncoder)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
