"""Giải mã Greedy Search và Beam Search (Google NMT length penalty) cho Seq2Seq."""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.tokenizer import BaseTokenizer


@torch.no_grad()
def greedy_decode(
    model,
    src: Tensor,
    tokenizer: BaseTokenizer,
    max_len: int = 128,
    device: torch.device | None = None,
    src_lengths: Tensor | None = None,
) -> tuple[list[str], Tensor]:
    """Greedy decoding: mỗi bước chọn token có xác suất cao nhất."""
    if device is None:
        device = next(model.parameters()).device

    src = src.to(device)
    if src_lengths is not None:
        src_lengths = src_lengths.to(device)

    tokens, attention_maps = model.translate(
        src,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_len=max_len,
        src_lengths=src_lengths,
    )

    translations = [
        tokenizer.decode(tokens[i], skip_special_tokens=True)
        for i in range(tokens.size(0))
    ]
    return translations, attention_maps


@torch.no_grad()
def beam_search_decode(
    model,
    src: Tensor,
    tokenizer: BaseTokenizer,
    beam_width: int = 5,
    max_len: int = 128,
    device: torch.device | None = None,
    src_lengths: Tensor | None = None,
    length_penalty: float = 0.6,
) -> list[str]:
    """Beam search decoding cho 1 câu (batch_size=1) kèm Google NMT length penalty."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    src = src.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)
    if src_lengths is not None:
        src_lengths = src_lengths.to(device)

    mask = model.create_src_mask(src)
    encoder_outputs, (hidden, cell) = model.encoder(src, src_lengths=src_lengths)

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id

    beams: list[tuple[list[int], float, Tensor, Tensor, Tensor | None]] = [
        ([bos_id], 0.0, hidden, cell, None)
    ]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        candidates = []
        for seq, score, h, c, prev_ah in beams:
            if seq[-1] == eos_id:
                completed.append((seq, score))
                continue

            token = torch.tensor([seq[-1]], device=device)
            prediction, new_h, new_c, _, new_ah = model.decoder(
                token,
                h,
                c,
                encoder_outputs,
                mask,
                prev_attn_hidden=prev_ah,
            )
            log_probs = torch.log_softmax(prediction, dim=-1).squeeze(0)
            top_probs, top_ids = log_probs.topk(beam_width)

            for k in range(beam_width):
                new_seq = seq + [top_ids[k].item()]
                new_score = score + top_probs[k].item()
                candidates.append((new_seq, new_score, new_h, new_c, new_ah))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    for seq, score, _, _, _ in beams:
        completed.append((seq, score))

    if not completed:
        return [""]

    best_seq = max(
        completed,
        key=lambda x: x[1] / (((5.0 + len(x[0])) / 6.0) ** length_penalty),
    )[0]
    return [tokenizer.decode(best_seq, skip_special_tokens=True)]
