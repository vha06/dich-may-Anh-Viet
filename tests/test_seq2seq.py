"""Unit tests cho mô hình Seq2Seq LSTM + Attention."""

import unittest

import torch

from src.models.seq2seq_lstm import Encoder, BahdanauAttention, Decoder, Seq2Seq


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder(
            vocab_size=100, embed_dim=32, hidden_dim=64,
            num_layers=2, dropout=0.0,
        )
        self.batch_size = 4
        self.src_len = 10

    def test_output_shape(self):
        src = torch.randint(1, 100, (self.batch_size, self.src_len))
        outputs, (hidden, cell) = self.encoder(src)
        self.assertEqual(outputs.shape, (4, 10, 128))
        self.assertEqual(hidden.shape, (2, 4, 64))
        self.assertEqual(cell.shape, (2, 4, 64))


class TestAttention(unittest.TestCase):
    def test_output_shape_and_sum(self):
        attn = BahdanauAttention(encoder_dim=128, decoder_dim=64, attention_dim=32)
        decoder_hidden = torch.randn(4, 64)
        encoder_outputs = torch.randn(4, 10, 128)

        context, weights = attn(decoder_hidden, encoder_outputs)
        self.assertEqual(context.shape, (4, 128))
        self.assertEqual(weights.shape, (4, 10))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))

    def test_mask(self):
        attn = BahdanauAttention(encoder_dim=128, decoder_dim=64, attention_dim=32)
        decoder_hidden = torch.randn(2, 64)
        encoder_outputs = torch.randn(2, 5, 128)
        mask = torch.tensor([[False, False, False, True, True],
                             [False, False, True, True, True]])

        _, weights = attn(decoder_hidden, encoder_outputs, mask)
        self.assertTrue((weights[:, -1] < 1e-6).all())


class TestSeq2Seq(unittest.TestCase):
    def setUp(self):
        enc = Encoder(vocab_size=100, embed_dim=32, hidden_dim=64,
                      num_layers=2, dropout=0.0)
        dec = Decoder(vocab_size=100, embed_dim=32, encoder_dim=128,
                      hidden_dim=64, attention_dim=32, num_layers=2, dropout=0.0)
        self.model = Seq2Seq(enc, dec, pad_id=0)

    def test_forward_shape(self):
        src = torch.randint(1, 100, (3, 8))
        tgt = torch.randint(1, 100, (3, 6))
        outputs, attn = self.model(src, tgt, teacher_forcing_ratio=1.0)
        self.assertEqual(outputs.shape, (3, 5, 100))
        self.assertEqual(attn.shape, (3, 5, 8))

    def test_translate_shape(self):
        src = torch.randint(1, 100, (2, 8))
        tokens, attn = self.model.translate(src, bos_id=2, eos_id=3, max_len=15)
        self.assertEqual(tokens.shape[0], 2)
        self.assertLessEqual(tokens.shape[1], 15)


if __name__ == "__main__":
    unittest.main()
