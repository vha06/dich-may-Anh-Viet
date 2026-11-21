"""Giải mã Greedy Search và Beam Search cho mô hình Seq2Seq."""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.tokenizer import BPETokenizer


@torch.no_grad()
def greedy_decode(
    model,
    src: Tensor,
    tokenizer: BPETokenizer,
    max_len: int = 100,
    device: torch.device | None = None,
) -> tuple[list[str], Tensor]:
    """Greedy decoding: mỗi bước chọn token có xác suất cao nhất.

    Returns:
        translations: danh sách câu dịch (đã decode về text).
        attention_maps: [batch, gen_len, src_len]
    """
    if device is None:
        device = next(model.parameters()).device

    src = src.to(device)
    tokens, attention_maps = model.translate(
        src,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_len=max_len,
    )

    translations = []
    for i in range(tokens.size(0)):
        ids = tokens[i].cpu().tolist()
        if tokenizer.eos_id in ids:
            ids = ids[: ids.index(tokenizer.eos_id)]
        translations.append(tokenizer.decode(ids))

    return translations, attention_maps


@torch.no_grad()
def beam_search_decode(
    model,
    src: Tensor,
    tokenizer: BPETokenizer,
    beam_width: int = 5,
    max_len: int = 100,
    device: torch.device | None = None,
) -> list[str]:
    """Beam search decoding cho 1 câu (batch_size=1).

    Returns:
        translations: danh sách 1 câu dịch tốt nhất.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    src = src.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)

    mask = model.create_src_mask(src)
    encoder_outputs, (hidden, cell) = model.encoder(src)

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id

    beams = [(
        [bos_id],
        0.0,
        hidden,
        cell,
    )]

    completed = []

    for _ in range(max_len):
        candidates = []
        for seq, score, h, c in beams:
            if seq[-1] == eos_id:
                completed.append((seq, score))
                continue

            token = torch.tensor([seq[-1]], device=device)
            prediction, new_h, new_c, _ = model.decoder(
                token, h, c, encoder_outputs, mask
            )
            log_probs = torch.log_softmax(prediction, dim=-1).squeeze(0)
            top_probs, top_ids = log_probs.topk(beam_width)

            for k in range(beam_width):
                new_seq = seq + [top_ids[k].item()]
                new_score = score + top_probs[k].item()
                candidates.append((new_seq, new_score, new_h, new_c))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    for seq, score, _, _ in beams:
        completed.append((seq, score))

    if not completed:
        return [""]

    best_seq = max(completed, key=lambda x: x[1] / max(len(x[0]), 1))[0]

    if eos_id in best_seq:
        best_seq = best_seq[: best_seq.index(eos_id)]
    if best_seq and best_seq[0] == bos_id:
        best_seq = best_seq[1:]

    return [tokenizer.decode(best_seq)]
