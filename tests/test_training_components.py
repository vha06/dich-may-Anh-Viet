"""Unit tests for TranslationLoss, perplexity safety, NoamScheduler, and PlateauScheduler."""

from __future__ import annotations

import math
import sys
import unittest

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID
from src.training import (
    DEFAULT_MAX_PPL_LOSS,
    NoamScheduler,
    PlateauScheduler,
    TranslationLoss,
    build_loss_pair,
    build_scheduler,
    compute_perplexity,
)


class TestTrainingComponents(unittest.TestCase):
    """Test suite verifying loss masking, eval unsmoothed PPL, Noam, and Plateau contracts."""

    def test_translation_loss_pad_invariance_and_token_count(self) -> None:
        torch.manual_seed(42)
        bsz, t_max, vocab = 2, 5, 20
        targets = torch.tensor(
            [
                [5, 8, 3, PAD_ID, PAD_ID],
                [7, 9, 11, 13, 3],
            ],
            dtype=torch.long,
        )
        logits_a = torch.randn(bsz, t_max, vocab, requires_grad=True)
        logits_b = logits_a.detach().clone()
        logits_b[targets == PAD_ID] = 1e5
        logits_b.requires_grad_(True)

        loss_fn = TranslationLoss(pad_id=PAD_ID, label_smoothing=0.0)
        loss_a, tokens_a = loss_fn(logits_a, targets)
        loss_b, tokens_b = loss_fn(logits_b, targets)

        self.assertEqual(tokens_a, 8)
        self.assertEqual(tokens_b, 8)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6))

        loss_b.backward()
        self.assertIsNotNone(logits_b.grad)
        assert logits_b.grad is not None
        self.assertTrue(
            torch.equal(
                logits_b.grad[targets == PAD_ID],
                torch.zeros_like(logits_b.grad[targets == PAD_ID]),
            )
        )

    def test_loss_pair_enforces_unsmoothed_eval_and_ppl_clamp(self) -> None:
        train_fn, eval_fn = build_loss_pair(pad_id=PAD_ID, train_label_smoothing=0.1)
        self.assertEqual(train_fn.label_smoothing, 0.1)
        self.assertEqual(eval_fn.label_smoothing, 0.0)

        vocab = 10
        targets = torch.tensor([[2, 3]], dtype=torch.long)
        confident_logits = torch.full((1, 2, vocab), -10.0)
        confident_logits[0, 0, 2] = 10.0
        confident_logits[0, 1, 3] = 10.0

        train_loss, _ = train_fn(confident_logits, targets)
        eval_loss, _ = eval_fn(confident_logits, targets)

        # Label smoothing intentionally adds residual loss on confident targets
        self.assertLess(float(eval_loss.item()), 1e-4)
        self.assertGreater(float(train_loss.item()), float(eval_loss.item()) + 1.0)

        self.assertAlmostEqual(
            compute_perplexity(1000.0),
            math.exp(DEFAULT_MAX_PPL_LOSS),
            places=5,
        )
        self.assertAlmostEqual(compute_perplexity(2.0), math.exp(2.0), places=6)
        with self.assertRaises(ValueError):
            compute_perplexity(float("nan"))

    def test_noam_scheduler_no_zero_div_no_base_lr_double_mul(self) -> None:
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt_small_init = torch.optim.Adam([p1], lr=1e-4)
        opt_large_init = torch.optim.Adam([p2], lr=1.0)

        d_model, warmup_steps = 256, 10
        sched_1 = build_scheduler(
            opt_small_init, "noam", d_model=d_model, warmup_steps=warmup_steps
        )
        sched_2 = NoamScheduler(
            opt_large_init, d_model=d_model, warmup_steps=warmup_steps
        )

        self.assertEqual(sched_1.step_mode, "batch")
        self.assertFalse(sched_1.requires_metric)

        lrs_1 = [sched_1.get_last_lr()[0]]
        lrs_2 = [sched_2.get_last_lr()[0]]
        for _ in range(29):
            opt_small_init.step()
            opt_large_init.step()
            sched_1.step()
            sched_2.step()
            lrs_1.append(sched_1.get_last_lr()[0])
            lrs_2.append(sched_2.get_last_lr()[0])

        # Identical trajectories prove initial optimizer lr does not double-multiply
        for lr_a, lr_b in zip(lrs_1, lrs_2):
            self.assertAlmostEqual(lr_a, lr_b, places=12)

        # Warmup steps 1..10 strictly increase; peak occurs at step 10 (index 9)
        for i in range(warmup_steps - 1):
            self.assertLess(lrs_1[i], lrs_1[i + 1])

        expected_peak = (d_model ** -0.5) * (warmup_steps ** -0.5)
        self.assertAlmostEqual(lrs_1[warmup_steps - 1], expected_peak, places=12)

        # Decay steps 11..30 strictly decrease
        for i in range(warmup_steps - 1, len(lrs_1) - 1):
            self.assertGreater(lrs_1[i], lrs_1[i + 1])

        # Checkpoint state restore check
        saved_state = sched_1.state_dict()
        p3 = nn.Parameter(torch.randn(4, 4))
        opt_restore = torch.optim.Adam([p3], lr=0.5)
        sched_restore = NoamScheduler(
            opt_restore, d_model=d_model, warmup_steps=warmup_steps
        )
        sched_restore.load_state_dict(saved_state)
        self.assertAlmostEqual(
            sched_restore.get_last_lr()[0], sched_1.get_last_lr()[0], places=12
        )
        opt_small_init.step()
        opt_restore.step()
        sched_1.step()
        sched_restore.step()
        self.assertAlmostEqual(
            sched_restore.get_last_lr()[0], sched_1.get_last_lr()[0], places=12
        )

    def test_plateau_scheduler_epoch_mode_and_lr_decay(self) -> None:
        param = nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.Adam([param], lr=1e-3)
        scheduler = build_scheduler(
            optimizer,
            "plateau",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            threshold=1e-4,
        )
        self.assertIsInstance(scheduler, PlateauScheduler)
        self.assertEqual(scheduler.step_mode, "epoch")
        self.assertTrue(scheduler.requires_metric)

        with self.assertRaises(ValueError):
            scheduler.step(None)

        # Best = 5.0; next two epochs stagnant at 5.0 -> triggers 0.5x decay on 2nd bad epoch
        scheduler.step(5.0)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-3, places=10)
        scheduler.step(5.0)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-3, places=10)
        scheduler.step(5.0)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-4, places=10)

        # State dict roundtrip
        state = scheduler.state_dict()
        param2 = nn.Parameter(torch.randn(4, 4))
        opt2 = torch.optim.Adam([param2], lr=5e-4)
        sched2 = PlateauScheduler(opt2, factor=0.5, patience=1, min_lr=1e-5)
        sched2.load_state_dict(state)
        self.assertEqual(sched2.get_last_lr()[0], 5e-4)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiem thu Loss & Schedulers (Chang 1 Ha tang Huan luyen) ===")
    dummy_p = nn.Parameter(torch.zeros(2, 2))
    dummy_opt = torch.optim.Adam([dummy_p], lr=1e-3)
    noam = NoamScheduler(dummy_opt, d_model=256, warmup_steps=4000, factor=1.0)
    milestones = {1: noam.get_last_lr()[0]}
    for s in range(2, 20001):
        dummy_opt.step()
        noam.step()
        if s in (1000, 4000, 8000, 20000):
            milestones[s] = noam.get_last_lr()[0]

    print("[NoamScheduler (d_model=256, warmup_steps=4000, factor=1.0)]")
    for step_id, lr_val in milestones.items():
        print(f"  - step={step_id:<5}: lr={lr_val:.7f}")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTrainingComponents)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
