"""Hàm tính điểm BLEU — thước đo chất lượng bản dịch (dùng sacrebleu)."""

from __future__ import annotations

import sacrebleu
from sacrebleu.metrics import BLEU

_bleu_scorer = BLEU(effective_order=True)


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Tính điểm BLEU trên tập dữ liệu (0.0 – 100.0)."""
    if not predictions or not references:
        raise ValueError("predictions và references không được rỗng!")
    if len(predictions) != len(references):
        raise ValueError(
            f"Số lượng predictions ({len(predictions)}) ≠ references ({len(references)})"
        )
    result = sacrebleu.corpus_bleu(hypotheses=predictions, references=[references])
    return result.score


def compute_sentence_bleu(prediction: str, reference: str) -> float:
    """Tính điểm BLEU cho 1 câu đơn lẻ (0.0 – 100.0)."""
    result = _bleu_scorer.sentence_score(hypothesis=prediction, references=[reference])
    return result.score
