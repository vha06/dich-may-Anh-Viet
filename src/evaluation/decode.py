"""Greedy Search, Beam Search (with Google NMT length penalty), and DataLoader translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, RandomSampler

from src.data.dataset import TranslationDataset
from src.data.tokenizer import DEFAULT_MAX_SEQ_LEN, BaseTokenizer
from src.evaluation.metrics import DEFAULT_TOKENIZE, compute_corpus_bleu
from src.models.seq2seq_lstm import Seq2SeqLSTM

DEFAULT_BEAM_SIZE = 4
DEFAULT_LENGTH_PENALTY = 0.6


@dataclass
class DecodeResult:
    """Unified decoding output shared by Seq2Seq and future Transformer decoders.

    Attributes:
        token_ids: Generated token IDs per sentence, retaining `<eos>` at the end so
            `len(token_ids[i]) == attn_weights[i].shape[0]` matches 1-to-1.
        scores: Cumulative (or length-normalized) log-probability score per sentence.
        attn_weights: Optional per-sentence alignment tensor of shape `(L_gen_i, S_i)`
            trimmed to the exact generated and source lengths, or `None` without Attention.
    """

    token_ids: list[list[int]]
    scores: list[float]
    attn_weights: list[torch.Tensor] | None


def _length_penalty_denom(length: int, alpha: float) -> float:
    """Google NMT length normalization denominator ((5 + |Y|) / 6)^alpha (Wu et al., 2016)."""
    if alpha == 0.0:
        return 1.0
    return float(((5.0 + max(length, 1)) / 6.0) ** alpha)


def _mask_invalid_decode_logits(
    logits: torch.Tensor,
    pad_id: int,
    bos_id: int,
) -> torch.Tensor:
    """Mask <pad> and <bos> logits to -inf before log_softmax during decoding."""
    masked = logits.clone()
    masked[..., pad_id] = float("-inf")
    masked[..., bos_id] = float("-inf")
    return masked


def greedy_decode_seq2seq(
    model: Seq2SeqLSTM,
    src_ids: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
    length_penalty: float = 0.0,
) -> DecodeResult:
    """Decode a batch in parallel with Greedy Search while preserving model train/eval mode."""
    if max_len < 1:
        raise ValueError(f"max_len phai >= 1, nhan {max_len}")
    if length_penalty < 0.0:
        raise ValueError(f"length_penalty phai >= 0.0, nhan {length_penalty}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batch_size = src_ids.size(0)
            device = src_ids.device
            src_mask = src_ids != model.pad_id
            encoder_outputs, hidden = model.encoder(src_ids, src_lengths)

            input_token = torch.full(
                (batch_size,), model.bos_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            cum_log_probs = torch.zeros(batch_size, dtype=torch.float, device=device)

            step_tokens: list[torch.Tensor] = []
            step_attns: list[torch.Tensor] = []

            for _ in range(max_len):
                logits_t, hidden, attn_t = model.decoder(
                    input_token,
                    hidden,
                    encoder_outputs=encoder_outputs,
                    src_mask=src_mask,
                )
                masked_logits = _mask_invalid_decode_logits(
                    logits_t, model.pad_id, model.bos_id
                )
                log_probs = torch.log_softmax(masked_logits, dim=-1)
                top_logp, raw_next = log_probs.max(dim=-1)

                cum_log_probs = cum_log_probs + torch.where(
                    finished, torch.zeros_like(top_logp), top_logp
                )
                next_token = raw_next.masked_fill(finished, model.pad_id)
                step_tokens.append(next_token)
                if attn_t is not None:
                    step_attns.append(attn_t)

                finished = finished | (next_token == model.eos_id)
                if finished.all():
                    break
                input_token = next_token

            token_ids_list: list[list[int]] = []
            scores_list: list[float] = []
            attn_list: list[torch.Tensor] | None = [] if step_attns else None

            for i in range(batch_size):
                raw_seq = [int(t[i].item()) for t in step_tokens]
                if model.eos_id in raw_seq:
                    l_gen = raw_seq.index(model.eos_id) + 1
                else:
                    l_gen = len(raw_seq)

                token_ids_list.append(raw_seq[:l_gen])
                scores_list.append(
                    float(cum_log_probs[i].item())
                    / _length_penalty_denom(l_gen, length_penalty)
                )
                if attn_list is not None:
                    s_len_i = int(src_lengths[i].item())
                    per_sent_attn = torch.stack(
                        [a[i, :s_len_i] for a in step_attns[:l_gen]], dim=0
                    )
                    attn_list.append(per_sent_attn.detach().cpu())

            return DecodeResult(
                token_ids=token_ids_list,
                scores=scores_list,
                attn_weights=attn_list,
            )
    finally:
        model.train(was_training)


def beam_search_seq2seq(
    model: Seq2SeqLSTM,
    src_ids: torch.Tensor,
    src_lengths: torch.Tensor,
    beam_size: int = DEFAULT_BEAM_SIZE,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
) -> DecodeResult:
    """Decode a batch using batched Encoder + per-sentence Beam Search with length penalty."""
    if beam_size < 1:
        raise ValueError(f"beam_size phai >= 1, nhan {beam_size}")
    if length_penalty < 0.0:
        raise ValueError(f"length_penalty phai >= 0.0, nhan {length_penalty}")
    if max_len < 1:
        raise ValueError(f"max_len phai >= 1, nhan {max_len}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batch_size = src_ids.size(0)
            device = src_ids.device

            # Run Encoder once across the entire batch before decoding per sentence
            encoder_outputs, (h_enc, c_enc) = model.encoder(src_ids, src_lengths)

            token_ids_list: list[list[int]] = []
            scores_list: list[float] = []
            attn_results: list[torch.Tensor] = []
            has_attention = True

            for i in range(batch_size):
                s_len_i = int(src_lengths[i].item())
                enc_out_i = encoder_outputs[i : i + 1, :s_len_i, :]
                src_mask_i = torch.ones((1, s_len_i), dtype=torch.bool, device=device)

                h_active = h_enc[:, i : i + 1, :].contiguous()
                c_active = c_enc[:, i : i + 1, :].contiguous()
                input_tokens = torch.tensor(
                    [model.bos_id], dtype=torch.long, device=device
                )
                beam_scores = torch.zeros(1, dtype=torch.float, device=device)
                beam_seqs: list[list[int]] = [[]]
                beam_attns: list[list[torch.Tensor]] = [[]]

                completed: list[tuple[float, list[int], list[torch.Tensor]]] = []
                remaining_beams = beam_size

                for _ in range(max_len):
                    num_active = input_tokens.size(0)
                    logits_t, (h_next, c_next), attn_t = model.decoder(
                        input_tokens,
                        (h_active, c_active),
                        encoder_outputs=enc_out_i.expand(num_active, -1, -1),
                        src_mask=src_mask_i.expand(num_active, -1),
                    )
                    if attn_t is None:
                        has_attention = False

                    masked_logits = _mask_invalid_decode_logits(
                        logits_t, model.pad_id, model.bos_id
                    )
                    log_probs = torch.log_softmax(masked_logits, dim=-1)
                    vocab_size = log_probs.size(-1)

                    total_scores = beam_scores.unsqueeze(1) + log_probs
                    flat_scores = total_scores.reshape(-1)
                    k_cand = min(remaining_beams, flat_scores.numel())
                    top_scores, top_indices = torch.topk(flat_scores, k=k_cand)

                    next_parents: list[int] = []
                    next_tokens: list[int] = []
                    next_scores: list[float] = []
                    next_seqs: list[list[int]] = []
                    next_attns: list[list[torch.Tensor]] = []

                    for cand_score, flat_idx in zip(top_scores, top_indices):
                        flat_val = int(flat_idx.item())
                        p_idx = flat_val // vocab_size
                        tok_id = flat_val % vocab_size
                        score_val = float(cand_score.item())

                        seq = beam_seqs[p_idx] + [tok_id]
                        attn_hist = (
                            beam_attns[p_idx] + [attn_t[p_idx].detach().cpu()]
                            if attn_t is not None
                            else []
                        )

                        if tok_id == model.eos_id:
                            norm_score = score_val / _length_penalty_denom(
                                len(seq), length_penalty
                            )
                            completed.append((norm_score, seq, attn_hist))
                            remaining_beams -= 1
                        else:
                            next_parents.append(p_idx)
                            next_tokens.append(tok_id)
                            next_scores.append(score_val)
                            next_seqs.append(seq)
                            next_attns.append(attn_hist)

                    if remaining_beams <= 0 or not next_tokens:
                        break

                    parent_tensor = torch.tensor(
                        next_parents, dtype=torch.long, device=device
                    )
                    input_tokens = torch.tensor(
                        next_tokens, dtype=torch.long, device=device
                    )
                    beam_scores = torch.tensor(
                        next_scores, dtype=torch.float, device=device
                    )
                    h_active = h_next.index_select(1, parent_tensor).contiguous()
                    c_active = c_next.index_select(1, parent_tensor).contiguous()
                    beam_seqs = next_seqs
                    beam_attns = next_attns

                if completed:
                    best_score, best_seq, best_attn = max(
                        completed, key=lambda item: item[0]
                    )
                else:
                    best_idx = max(
                        range(len(beam_seqs)),
                        key=lambda idx: float(beam_scores[idx].item())
                        / _length_penalty_denom(len(beam_seqs[idx]), length_penalty),
                    )
                    best_seq = beam_seqs[best_idx]
                    best_score = float(beam_scores[best_idx].item()) / _length_penalty_denom(
                        len(best_seq), length_penalty
                    )
                    best_attn = beam_attns[best_idx]

                token_ids_list.append(best_seq)
                scores_list.append(best_score)
                if has_attention and best_attn:
                    attn_results.append(torch.stack(best_attn, dim=0))

            return DecodeResult(
                token_ids=token_ids_list,
                scores=scores_list,
                attn_weights=attn_results if has_attention and attn_results else None,
            )
    finally:
        model.train(was_training)


def translate_dataloader(
    model: Any,
    dataloader: DataLoader,
    tgt_tokenizer: BaseTokenizer,
    strategy: str = "greedy",
    beam_size: int = DEFAULT_BEAM_SIZE,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
    device: torch.device | str | None = None,
    references: Sequence[str] | None = None,
    decode_fn: Callable[..., DecodeResult] | None = None,
    tokenize: str = DEFAULT_TOKENIZE,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Translate an entire DataLoader and evaluate SacreBLEU against untouched gold references."""
    if references is None:
        if not isinstance(dataloader.dataset, TranslationDataset):
            raise ValueError(
                "Can truyen danh sach references goc khi dataset khong phai TranslationDataset."
            )
        if isinstance(dataloader.sampler, RandomSampler):
            raise ValueError(
                "Khong the tu doi chieu gold references tren DataLoader dang shuffle=True."
            )

    norm_strategy = strategy.strip().lower()
    if decode_fn is None and norm_strategy not in ("greedy", "beam"):
        raise ValueError(
            f"Khong ho tro strategy={strategy!r}; chon 'greedy', 'beam', hoac truyen decode_fn."
        )

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    was_training = model.training
    model.eval()
    predictions: list[str] = []
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                src_ids = batch["src_ids"].to(target_device)
                src_lengths = batch["src_lengths"].to(target_device)

                if decode_fn is not None:
                    result = decode_fn(model, src_ids, src_lengths, max_len=max_len)
                elif norm_strategy == "greedy":
                    result = greedy_decode_seq2seq(
                        model, src_ids, src_lengths, max_len=max_len
                    )
                else:
                    result = beam_search_seq2seq(
                        model,
                        src_ids,
                        src_lengths,
                        beam_size=beam_size,
                        length_penalty=length_penalty,
                        max_len=max_len,
                    )

                for seq_ids in result.token_ids:
                    predictions.append(
                        tgt_tokenizer.decode(seq_ids, skip_special_tokens=True)
                    )
    finally:
        model.train(was_training)

    num_sentences = len(predictions)
    if references is not None:
        gold_refs = list(references)[:num_sentences]
    else:
        assert isinstance(dataloader.dataset, TranslationDataset)
        gold_refs = dataloader.dataset.tgt_lines[:num_sentences]

    metrics = compute_corpus_bleu(predictions, gold_refs, tokenize=tokenize)
    return {
        "bleu": metrics["bleu"],
        "metrics": metrics,
        "predictions": predictions,
        "references": gold_refs,
        "num_sentences": num_sentences,
        "strategy": "custom" if decode_fn is not None else norm_strategy,
    }
