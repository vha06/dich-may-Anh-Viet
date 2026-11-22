"""Integration tests for Trainer, checkpoint resume, Early Stopping, YAML config, and evaluate.py."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate import evaluate_checkpoint
from scripts.train import build_model_from_config, load_config
from src.data.dataset import TranslationDataset, create_dataloader
from src.data.tokenizer import DEFAULT_VOCAB_SIZE, PAD_ID, load_tokenizer
from src.models.seq2seq_lstm import build_seq2seq_lstm
from src.training import Trainer, build_loss_pair, build_scheduler


class TestTrainerAndScripts(unittest.TestCase):
    """Test suite verifying Trainer fit, safe AMP setup, resume state, Early Stopping, and eval."""

    def _build_small_trainer_components(
        self,
        tmp_dir: Path,
        save_best_by: str = "val_bleu",
        early_stopping_patience: int = 5,
        train_label_smoothing: float = 0.0,
    ) -> tuple[Trainer, Path, Path]:
        val_en = Path("data/processed/val.en")
        val_vi = Path("data/processed/val.vi")
        bpe_en = Path("data/tokenizer/bpe_en.json")
        bpe_vi = Path("data/tokenizer/bpe_vi.json")
        if not (val_en.exists() and val_vi.exists() and bpe_en.exists() and bpe_vi.exists()):
            self.skipTest("Thieu du lieu hoac model BPE de chay Trainer test.")

        torch.manual_seed(42)
        src_tok = load_tokenizer(bpe_en, tokenizer_type="bpe")
        tgt_tok = load_tokenizer(bpe_vi, tokenizer_type="bpe")
        ds = TranslationDataset(val_en, val_vi, src_tok, tgt_tok, max_len=32, pre_tokenize=True)
        train_loader = create_dataloader(ds, batch_size=8, shuffle=False)
        val_loader = create_dataloader(ds, batch_size=8, shuffle=False)

        cfg = {
            "experiment": {"name": "unit_test_seq2seq", "seed": 42},
            "data": {
                "processed_dir": "data/processed",
                "tokenizer_dir": "data/tokenizer",
                "tokenizer_type": "bpe",
                "max_len": 32,
                "batch_size": 8,
            },
            "model": {
                "arch": "seq2seq_lstm",
                "src_vocab_size": DEFAULT_VOCAB_SIZE,
                "tgt_vocab_size": DEFAULT_VOCAB_SIZE,
                "embed_dim": 32,
                "enc_hidden_dim": 32,
                "dec_hidden_dim": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "bidirectional": True,
                "attention_method": "general",
                "attn_dim": 32,
            },
            "training": {
                "test_beam_size": 2,
                "test_length_penalty": 0.6,
            },
        }
        model = build_model_from_config(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(
            optimizer, "plateau", factor=0.5, patience=1, min_lr=1e-5
        )
        train_loss_fn, eval_loss_fn = build_loss_pair(
            pad_id=PAD_ID, train_label_smoothing=train_label_smoothing
        )

        ckpt_dir = tmp_dir / "checkpoints"
        out_dir = tmp_dir / "outputs"
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss_fn=train_loss_fn,
            eval_loss_fn=eval_loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            tgt_tokenizer=tgt_tok,
            config=cfg,
            device="cpu",
            checkpoint_dir=ckpt_dir,
            output_dir=out_dir,
            use_amp=True,  # Verifies CPU gracefully disables AMP even if config enables it
            grad_clip=1.0,
            teacher_forcing_ratio=1.0,
            save_best_by=save_best_by,
            early_stopping_patience=early_stopping_patience,
            eval_decode_strategy="greedy",
            max_decode_len=16,
            wandb_run=None,
            show_progress=False,
        )
        return trainer, ckpt_dir, out_dir

    def test_baseline_yaml_config_loads_and_builds_15m_model(self) -> None:
        cfg_path = Path("configs/baseline_lstm.yaml")
        self.assertTrue(cfg_path.exists(), "Thieu configs/baseline_lstm.yaml")
        cfg = load_config(cfg_path)

        self.assertEqual(cfg["wandb"]["entity"], "nikoteen-iai-uet")
        self.assertEqual(cfg["wandb"]["project"], "dich-may-Anh-Viet")
        self.assertEqual(cfg["training"]["teacher_forcing_ratio"], 1.0)
        self.assertEqual(cfg["training"]["epochs"], 20)
        self.assertEqual(cfg["training"]["early_stopping_patience"], 5)

        model = build_model_from_config(cfg)
        self.assertEqual(model.parameter_breakdown()["total"], 15_820_096)

    def test_trainer_two_epochs_and_resume_preserves_best_metric_and_early_stopping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            trainer1, ckpt_dir, out_dir = self._build_small_trainer_components(
                tmp_dir,
                save_best_by="val_bleu",
                early_stopping_patience=5,
                train_label_smoothing=0.1,
            )
            self.assertFalse(trainer1.amp_enabled)
            self.assertEqual(trainer1.eval_loss_fn.label_smoothing, 0.0)

            hist1 = trainer1.fit(num_epochs=2, max_train_batches=2, max_val_batches=1)
            self.assertEqual(len(hist1), 2)
            self.assertTrue((ckpt_dir / "best_model.pt").exists())
            self.assertTrue((ckpt_dir / "last_model.pt").exists())
            self.assertTrue((out_dir / "history.json").exists())

            # Simulate high prior best_metric & non-zero patience counter in last_model.pt,
            # verifying --resume preserves both rather than resetting best_metric to -inf!
            last_payload = torch.load(
                ckpt_dir / "last_model.pt", map_location="cpu", weights_only=False
            )
            last_payload["best_metric"] = 99.0
            last_payload["best_epoch"] = 1
            last_payload["early_stopping_counter"] = 2
            torch.save(last_payload, ckpt_dir / "last_model.pt")
            torch.save({"sentinel": True}, ckpt_dir / "best_model.pt")

            trainer2, _, _ = self._build_small_trainer_components(
                tmp_dir,
                save_best_by="val_bleu",
                early_stopping_patience=5,
                train_label_smoothing=0.1,
            )
            resumed_epoch = trainer2.load_checkpoint(ckpt_dir / "last_model.pt")
            self.assertEqual(resumed_epoch, 3)
            self.assertEqual(trainer2.best_metric, 99.0)
            self.assertEqual(trainer2.best_epoch, 1)
            self.assertEqual(trainer2.early_stopping_counter, 2)
            self.assertEqual(len(trainer2.history), 2)

            hist2 = trainer2.fit(num_epochs=3, max_train_batches=2, max_val_batches=1)
            self.assertEqual(len(hist2), 3)
            self.assertEqual(hist2[-1]["epoch"], 3)
            self.assertFalse(hist2[-1]["is_best"])
            self.assertEqual(trainer2.best_metric, 99.0)
            self.assertEqual(trainer2.early_stopping_counter, 3)

            # Confirm inferior Epoch 3 did NOT overwrite best_model.pt
            best_check = torch.load(
                ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False
            )
            self.assertTrue(best_check.get("sentinel", False))

            disk_hist = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(disk_hist), 3)

    def test_early_stopping_triggers_on_stagnation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            trainer, _, _ = self._build_small_trainer_components(
                tmp_dir, save_best_by="val_bleu", early_stopping_patience=2
            )
            trainer.best_metric = 99.9
            trainer.best_epoch = 0
            hist = trainer.fit(num_epochs=6, max_train_batches=1, max_val_batches=1)
            self.assertEqual(len(hist), 2)
            self.assertEqual(trainer.early_stopping_counter, 2)

    def test_evaluate_checkpoint_greedy_and_beam(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            trainer, ckpt_dir, out_dir = self._build_small_trainer_components(tmp_dir)
            trainer.fit(num_epochs=1, max_train_batches=1, max_val_batches=1)

            best_ckpt = ckpt_dir / "best_model.pt"
            eval_json = out_dir / "eval_val_beam.json"
            report_greedy = evaluate_checkpoint(
                checkpoint_path=best_ckpt,
                split="val",
                strategy="greedy",
                batch_size=8,
                device="cpu",
                max_batches=1,
            )
            report_beam = evaluate_checkpoint(
                checkpoint_path=best_ckpt,
                split="val",
                strategy="beam",
                beam_size=2,
                length_penalty=0.6,
                batch_size=8,
                device="cpu",
                output_path=eval_json,
                max_batches=1,
            )

            self.assertEqual(report_greedy["num_sentences"], 8)
            self.assertEqual(report_beam["num_sentences"], 8)
            self.assertEqual(report_beam["strategy"], "beam")
            self.assertTrue(eval_json.exists())
            # Untouched gold reference line 0 ends with '?'
            self.assertTrue(report_beam["translations"][0]["reference"].endswith("?"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Trainer, Resume, Early Stopping, Config & Evaluate Script ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTrainerAndScripts)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
