"""Evaluation metrics and decoding utilities for NMT."""

from src.evaluation.metrics import (
    DEFAULT_SMOOTH_METHOD,
    DEFAULT_TOKENIZE,
    compute_bleu,
    compute_corpus_bleu,
    compute_sentence_bleu,
    decode_and_compute_bleu,
)

# Decode exports (Greedy/Beam Search) will be added here when Mảng 2 is merged onto main.

__all__ = [
    "DEFAULT_SMOOTH_METHOD",
    "DEFAULT_TOKENIZE",
    "compute_bleu",
    "compute_corpus_bleu",
    "compute_sentence_bleu",
    "decode_and_compute_bleu",
]
