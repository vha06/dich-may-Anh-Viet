"""Tính chỉ số SacreBLEU chuẩn cho đánh giá dịch máy."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sacrebleu


def compute_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    tokenize: str = "13a",
    lowercase: bool = False,
) -> dict[str, Any]:
    """Tính corpus SacreBLEU cho danh sách câu dịch và câu tham chiếu.

    Args:
        hypotheses: Danh sách N câu dịch của mô hình.
        references: Danh sách N câu tham chiếu (1-1), hoặc K bộ tham chiếu mỗi bộ N câu.
        tokenize: Bộ tách token của SacreBLEU (mặc định: "13a").
        lowercase: Có chuyển về chữ thường trước khi chấm điểm hay không.
    """
    hyps = list(hypotheses)
    if not hyps:
        raise ValueError("Danh sách hypotheses không được rỗng.")

    if references and isinstance(references[0], str):
        ref_streams: list[list[str]] = [[str(x) for x in references]]
    else:
        ref_streams = [[str(x) for x in r] for r in references]

    if not ref_streams or any(len(r) != len(hyps) for r in ref_streams):
        raise ValueError(
            f"Số câu hypotheses ({len(hyps)}) không khớp với references "
            f"({[len(r) for r in ref_streams]})."
        )

    metric = sacrebleu.BLEU(tokenize=tokenize, lowercase=lowercase, force=True)
    bleu = metric.corpus_score(hyps, ref_streams)
    return {
        "bleu": round(float(bleu.score), 2),
        "precisions": [round(float(p), 2) for p in bleu.precisions],
        "bp": round(float(bleu.bp), 4),
        "sys_len": int(bleu.sys_len),
        "ref_len": int(bleu.ref_len),
        "signature": str(metric.get_signature()),
    }


if __name__ == "__main__":
    sample_hyps = ["Tôi thích học dịch máy .", "Hôm nay trời đẹp ."]
    sample_refs = ["Tôi thích học dịch máy .", "Hôm nay trời rất đẹp ."]
    res = compute_bleu(sample_hyps, sample_refs)
    assert 0.0 <= res["bleu"] <= 100.0
    assert compute_bleu(sample_hyps, sample_hyps)["bleu"] == 100.0
    print("OK:", res)
