"""PyTorch Dataset và DataLoader cho IWSLT'15 En–Vi (kèm padding động)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import BPETokenizer


class TranslationDataset(Dataset):
    """Dataset chứa các cặp câu (Anh, Việt) đã mã hóa thành ID số."""

    def __init__(
        self,
        en_path: str | Path,
        vi_path: str | Path,
        tokenizer: BPETokenizer,
        max_len: int = 100,
    ) -> None:
        en_lines = Path(en_path).read_text(encoding="utf-8").splitlines()
        vi_lines = Path(vi_path).read_text(encoding="utf-8").splitlines()

        if len(en_lines) != len(vi_lines):
            raise ValueError("Số dòng file Anh và Việt không khớp!")

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data: list[tuple[list[int], list[int]]] = []

        for en, vi in zip(en_lines, vi_lines):
            src_ids = tokenizer.encode(en, add_bos=True, add_eos=True)[:max_len]
            tgt_ids = tokenizer.encode(vi, add_bos=True, add_eos=True)[:max_len]
            self.data.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        src_ids, tgt_ids = self.data[idx]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def make_collate_fn(pad_id: int):
    """Tạo hàm collate ghép các mẫu thành 1 batch có padding."""

    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        src_list, tgt_list = zip(*batch)
        src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)
        return src_padded, tgt_padded

    return collate_fn


def build_dataloader(
    en_path: str | Path,
    vi_path: str | Path,
    tokenizer: BPETokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_len: int = 100,
) -> DataLoader:
    """Tạo DataLoader hoàn chỉnh từ file dữ liệu."""
    dataset = TranslationDataset(en_path, vi_path, tokenizer, max_len=max_len)
    collate_fn = make_collate_fn(pad_id=tokenizer.pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
