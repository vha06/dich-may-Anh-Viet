"""Padding and Causal Look-Ahead Attention Masks for Transformer models."""

from __future__ import annotations

import torch

from src.data.tokenizer import PAD_ID

_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _validate_int_scalar(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{name} phai la so nguyen int (khong phai bool/float), nhan duoc {value!r}"
        )


def _validate_token_ids(token_ids: torch.Tensor, name: str = "token_ids") -> None:
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError(f"{name} phai la torch.Tensor, nhan {type(token_ids).__name__}")
    if token_ids.ndim != 2:
        raise ValueError(
            f"{name} phai la tensor 2D (B, L), nhan shape {tuple(token_ids.shape)}"
        )
    if token_ids.size(0) == 0:
        raise ValueError(
            f"{name} khong duoc co batch rong B == 0, nhan shape {tuple(token_ids.shape)}"
        )
    if token_ids.size(1) == 0:
        raise ValueError(
            f"{name} khong duoc co chieu dai rong L == 0, nhan shape {tuple(token_ids.shape)}"
        )
    if token_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError(
            f"{name} phai co kieu so nguyen (khong chap nhan float/complex/bool), nhan {token_ids.dtype}"
        )


def create_padding_mask(
    token_ids: torch.Tensor,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    """Create 4D boolean padding mask `(B, 1, 1, L)` (`True` keeps token, `False` masks `<pad>`)."""
    _validate_token_ids(token_ids, name="token_ids")
    _validate_int_scalar(pad_id, name="pad_id")
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(
    seq_len: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create 4D boolean lower-triangular causal mask `(1, 1, T, T)` (`True` for `j <= i`)."""
    _validate_int_scalar(seq_len, name="seq_len")
    if seq_len <= 0:
        raise ValueError(f"seq_len phai > 0, nhan duoc {seq_len}")
    lower = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    return lower.unsqueeze(0).unsqueeze(0)


def create_target_mask(
    tgt_ids: torch.Tensor,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    """Combine target key padding mask and causal look-ahead mask into `(B, 1, T, T)`."""
    _validate_token_ids(tgt_ids, name="tgt_ids")
    _validate_int_scalar(pad_id, name="pad_id")
    tgt_pad_mask = create_padding_mask(tgt_ids, pad_id=pad_id)
    causal_mask = create_causal_mask(tgt_ids.size(1), device=tgt_ids.device)
    return tgt_pad_mask & causal_mask


def create_masks(
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `(src_mask, tgt_mask)` with shapes `(B, 1, 1, S)` and `(B, 1, T, T)`."""
    _validate_token_ids(src_ids, name="src_ids")
    _validate_token_ids(tgt_ids, name="tgt_ids")
    _validate_int_scalar(pad_id, name="pad_id")

    if src_ids.size(0) != tgt_ids.size(0):
        raise ValueError(
            f"Batch size cua src_ids ({src_ids.size(0)}) va tgt_ids ({tgt_ids.size(0)}) khong khop"
        )
    if src_ids.device != tgt_ids.device:
        raise ValueError(
            f"src_ids ({src_ids.device}) va tgt_ids ({tgt_ids.device}) phai nam tren cung device"
        )

    src_mask = create_padding_mask(src_ids, pad_id=pad_id)
    tgt_mask = create_target_mask(tgt_ids, pad_id=pad_id)
    return src_mask, tgt_mask
