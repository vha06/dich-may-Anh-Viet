"""PyTorch Dataset và DataLoader cho IWSLT'15 En–Vi (hỗ trợ 2 Tokenizer riêng + padding động)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import BaseTokenizer


class TranslationDataset(Dataset):
    """Dataset chứa các cặp câu (Anh, Việt) đã mã hóa thành ID số."""

    def __init__(
        self,
        en_path: str | Path,
        vi_path: str | Path,
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer | None = None,
        max_len: int = 100,
    ) -> None:
        with Path(en_path).open("r", encoding="utf-8") as f:
            en_lines = [line.rstrip("\r\n") for line in f]
        with Path(vi_path).open("r", encoding="utf-8") as f:
            vi_lines = [line.rstrip("\r\n") for line in f]

        if len(en_lines) != len(vi_lines):
            raise ValueError("Số dòng file Anh và Việt không khớp!")

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer if tgt_tokenizer is not None else src_tokenizer
        self.max_len = max_len
        self.data: list[tuple[list[int], list[int]]] = []

        for en, vi in zip(en_lines, vi_lines):
            src_ids = self.src_tokenizer.encode(en, add_special_tokens=True, max_len=max_len)
            tgt_ids = self.tgt_tokenizer.encode(vi, add_special_tokens=True, max_len=max_len)
            self.data.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        src_ids, tgt_ids = self.data[idx]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def make_collate_fn(src_pad_id: int = 0, tgt_pad_id: int | None = None):
    """Tạo hàm collate ghép các mẫu thành 1 batch có padding động."""
    if tgt_pad_id is None:
        tgt_pad_id = src_pad_id

    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        src_list, tgt_list = zip(*batch)
        src_padded = pad_sequence(src_list, batch_first=True, padding_value=src_pad_id)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_id)
        return src_padded, tgt_padded

    return collate_fn


def build_dataloader(
    en_path: str | Path,
    vi_path: str | Path,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_len: int = 100,
) -> DataLoader:
    """Tạo DataLoader hoàn chỉnh từ file dữ liệu."""
    tgt_tok = tgt_tokenizer if tgt_tokenizer is not None else src_tokenizer
    dataset = TranslationDataset(
        en_path,
        vi_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tok,
        max_len=max_len,
    )
    collate_fn = make_collate_fn(src_pad_id=src_tokenizer.pad_id, tgt_pad_id=tgt_tok.pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
