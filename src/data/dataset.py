"""PyTorch Dataset, DataLoader, dynamic padding, target shifting, bucketing và caching."""
from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.tokenizer import BPETokenizer, PAD_ID


class TranslationDataset(Dataset):
    """Dataset đọc song ngữ đã làm sạch và token hóa sang tensor ID.

    Hỗ trợ caching tokenized (P2), kiểm tra đồng bộ số dòng (P0),
    và bảo đảm luôn giữ token <eos> sau truncation (P0).
    """

    def __init__(
        self,
        src_file: Path | str,
        tgt_file: Path | str,
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        max_len: int = 160,
        cache_tokenized: bool = True,
    ):
        """Khởi tạo TranslationDataset.

        Args:
            src_file: Đường dẫn file văn bản nguồn (English).
            tgt_file: Đường dẫn file văn bản đích (Vietnamese).
            src_tokenizer: Bộ tách từ cho ngôn ngữ nguồn.
            tgt_tokenizer: Bộ tách từ cho ngôn ngữ đích.
            max_len: Độ dài tối đa tính theo số BPE token (mặc định 160, bao quát 99.9% câu 128 từ).
            cache_tokenized: Lưu sẵn tensor token trong RAM để tăng tốc DataLoader (P2).
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.cache_tokenized = cache_tokenized

        with open(src_file, "r", encoding="utf-8") as f_src:
            src_lines = f_src.read().splitlines()
        with open(tgt_file, "r", encoding="utf-8") as f_tgt:
            tgt_lines = f_tgt.read().splitlines()

        # P0: Kiểm tra số dòng giữa Source và Target
        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"Số dòng không khớp: source={len(src_lines)} vs target={len(tgt_lines)} "
                f"({src_file} vs {tgt_file})"
            )

        # P0: Đồng bộ xử lý dòng rỗng - bảo toàn 100% dòng 1-1 để không lệch chỉ số BLEU
        self.pairs = list(zip(src_lines, tgt_lines))

        # P2: Caching tokenized dataset
        self.cached_tensors: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.cache_tokenized:
            self.cached_tensors = [self._tokenize_pair(s, t) for s, t in self.pairs]

    def _truncate_and_wrap(self, text: str, tok: BPETokenizer) -> list[int]:
        """Tokenize chuỗi và áp dụng truncation luôn giữ <sos> và <eos> (P0)."""
        raw_ids = tok.encode(text, add_special_tokens=False)
        # Giữ tối đa (max_len - 2) token nội dung để nhường chỗ cho <sos> và <eos>
        if len(raw_ids) > self.max_len - 2:
            raw_ids = raw_ids[: self.max_len - 2]
        return [tok.sos_id] + raw_ids + [tok.eos_id]

    def _tokenize_pair(self, src_text: str, tgt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids = self._truncate_and_wrap(src_text, self.src_tokenizer)
        tgt_ids = self._truncate_and_wrap(tgt_text, self.tgt_tokenizer)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_tensors is not None:
            return self.cached_tensors[index]
        src_text, tgt_text = self.pairs[index]
        return self._tokenize_pair(src_text, tgt_text)


def create_padding_mask(seq: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Tạo boolean padding mask [B, L] với True tại vị trí pad_id (P1, chuẩn PyTorch)."""
    return seq == pad_id


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = PAD_ID,
    return_masks: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Padding động, target shifting cho teacher forcing và sinh padding masks (P1).

    Args:
        batch: Danh sách cặp tensor (src_ids, tgt_ids).
        pad_id: ID token padding (mặc định: 0).
        return_masks: Nếu True, trả về thêm (src_mask, tgt_mask) (P1).

    Returns:
        src_padded: Tensor [B, L_src] - câu nguồn được pad động.
        tgt_input:  Tensor [B, L_tgt - 1] - đầu vào Decoder ([<sos>, y_1, ..., y_{T-1}]).
        tgt_output: Tensor [B, L_tgt - 1] - nhãn mục tiêu ([y_1, ..., y_T, <eos>]).
        src_mask (tùy chọn): Boolean Tensor [B, L_src] (True tại vị trí pad).
        tgt_mask (tùy chọn): Boolean Tensor [B, L_tgt - 1] (True tại vị trí pad).
    """
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(list(src_list), batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(list(tgt_list), batch_first=True, padding_value=pad_id)

    # Target Shifting:
    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]

    if return_masks:
        src_mask = create_padding_mask(src_padded, pad_id)
        tgt_mask = create_padding_mask(tgt_input, pad_id)
        return src_padded, tgt_input, tgt_output, src_mask, tgt_mask

    return src_padded, tgt_input, tgt_output


class BucketBatchSampler(Sampler[list[int]]):
    """Gom cụm các câu có độ dài tương tự vào cùng batch để giảm padding tokens (P2)."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Sắp xếp index theo độ dài
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [
            sorted_indices[i : i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return len(self.batches)


def get_dataloader(
    src_file: Path | str,
    tgt_file: Path | str,
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    max_len: int = 160,
    use_bucketing: bool = False,
    return_masks: bool = False,
    cache_tokenized: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Tạo DataLoader cấp phát sẵn (src_padded, tgt_input, tgt_output[, src_mask, tgt_mask])."""
    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len,
        cache_tokenized=cache_tokenized,
    )

    # partial thay vì lambda: pickle được khi num_workers > 0 với start method spawn/forkserver
    batch_collate = partial(collate_fn, pad_id=src_tokenizer.pad_id, return_masks=return_masks)

    if use_bucketing:
        lengths = [len(s.split()) for s, _ in dataset.pairs]
        sampler = BucketBatchSampler(lengths, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=batch_collate,
            num_workers=num_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=batch_collate,
        num_workers=num_workers,
    )
