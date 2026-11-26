"""PyTorch Dataset, DataLoader, dynamic padding, target shifting, bucketing và caching."""
from __future__ import annotations

import random
from collections.abc import Iterator, Sized
from functools import partial
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.tokenizer import BaseTokenizer, PAD_ID


class TranslationDataset(Dataset):
    """Dataset đọc song ngữ đã làm sạch, tiền mã hóa sang list[list[int]] và giữ nguyên văn bản gốc."""

    def __init__(
        self,
        src_file: Path | str,
        tgt_file: Path | str,
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer,
        max_len: int = 128,
        pre_tokenize: bool = True,
        cache_tokenized: bool | None = None,
    ) -> None:
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.pre_tokenize = pre_tokenize if cache_tokenized is None else cache_tokenized

        with open(src_file, "r", encoding="utf-8") as f_src:
            src_lines = f_src.read().splitlines()
        with open(tgt_file, "r", encoding="utf-8") as f_tgt:
            tgt_lines = f_tgt.read().splitlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"Số dòng không khớp: source={len(src_lines)} vs target={len(tgt_lines)} "
                f"({src_file} vs {tgt_file})"
            )

        self.pairs: list[tuple[str, str]] = list(zip(src_lines, tgt_lines))
        self.src_cache: list[list[int]] | None = None
        self.tgt_cache: list[list[int]] | None = None
        if self.pre_tokenize:
            self.src_cache = [
                src_tokenizer.encode(s, add_special_tokens=True, max_len=self.max_len)
                for s, _ in self.pairs
            ]
            self.tgt_cache = [
                tgt_tokenizer.encode(t, add_special_tokens=True, max_len=self.max_len)
                for _, t in self.pairs
            ]

    def get_raw_pair(self, index: int) -> tuple[str, str]:
        """Trả về cặp câu văn bản gốc (chưa cắt) để phục vụ chấm SacreBLEU."""
        return self.pairs[index]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.src_cache is not None and self.tgt_cache is not None:
            src_ids = self.src_cache[index]
            tgt_ids = self.tgt_cache[index]
        else:
            src_text, tgt_text = self.pairs[index]
            src_ids = self.src_tokenizer.encode(
                src_text, add_special_tokens=True, max_len=self.max_len
            )
            tgt_ids = self.tgt_tokenizer.encode(
                tgt_text, add_special_tokens=True, max_len=self.max_len
            )
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def create_padding_mask(seq: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Tạo boolean padding mask [B, L] với True tại vị trí pad_id (chuẩn PyTorch)."""
    return seq == pad_id


def collate_translation_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    """Padding động, target shifting và trả về Batch Contract chuẩn 5 khóa (dtype=torch.long)."""
    src_list, tgt_list = zip(*batch)
    src_lengths = torch.tensor([s.size(0) for s in src_list], dtype=torch.long)
    tgt_lengths = torch.tensor([max(0, t.size(0) - 1) for t in tgt_list], dtype=torch.long)

    src_ids = pad_sequence(list(src_list), batch_first=True, padding_value=pad_id)
    tgt_input_ids = pad_sequence([t[:-1] for t in tgt_list], batch_first=True, padding_value=pad_id)
    tgt_output_ids = pad_sequence([t[1:] for t in tgt_list], batch_first=True, padding_value=pad_id)

    return {
        "src_ids": src_ids,
        "src_lengths": src_lengths,
        "tgt_input_ids": tgt_input_ids,
        "tgt_output_ids": tgt_output_ids,
        "tgt_lengths": tgt_lengths,
    }


collate_fn = collate_translation_batch


class BucketBatchSampler(Sampler[list[int]]):
    """Gom cụm các câu độ dài tương tự theo từng cửa sổ ngẫu nhiên mỗi epoch (noisy bucketing)."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        pool_multiplier: int = 20,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = max(batch_size, batch_size * pool_multiplier)

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=self.lengths.__getitem__)
            for i in range(0, len(pool), self.batch_size):
                batches.append(pool[i : i + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def get_dataloader(
    src_file: Path | str,
    tgt_file: Path | str,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    batch_size: int = 64,
    shuffle: bool = True,
    max_len: int = 128,
    use_bucketing: bool = False,
    pre_tokenize: bool = True,
    cache_tokenized: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Tạo DataLoader trả về dict 5 khóa chuẩn (Batch Contract)."""
    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len,
        pre_tokenize=pre_tokenize,
        cache_tokenized=cache_tokenized,
    )

    batch_collate = partial(collate_translation_batch, pad_id=src_tokenizer.pad_id)

    if use_bucketing:
        if dataset.src_cache is not None and dataset.tgt_cache is not None:
            lengths = [max(len(s), len(t)) for s, t in zip(dataset.src_cache, dataset.tgt_cache)]
        else:
            lengths = [max(len(s.split()), len(t.split())) for s, t in dataset.pairs]
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


def build_dataloaders(
    train_src: Path | str,
    train_tgt: Path | str,
    val_src: Path | str,
    val_tgt: Path | str,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    test_src: Path | str | None = None,
    test_tgt: Path | str | None = None,
    batch_size: int = 64,
    max_len: int = 128,
    use_bucketing: bool = False,
    pre_tokenize: bool = True,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Tạo bộ DataLoader train/val(/test), khóa cứng shuffle=False và use_bucketing=False cho val/test."""
    loaders: dict[str, DataLoader] = {
        "train": get_dataloader(
            src_file=train_src,
            tgt_file=train_tgt,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=batch_size,
            shuffle=True,
            max_len=max_len,
            use_bucketing=use_bucketing,
            pre_tokenize=pre_tokenize,
            num_workers=num_workers,
        ),
        "val": get_dataloader(
            src_file=val_src,
            tgt_file=val_tgt,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_len=max_len,
            use_bucketing=False,
            pre_tokenize=pre_tokenize,
            num_workers=num_workers,
        ),
    }
    if test_src is not None and test_tgt is not None:
        loaders["test"] = get_dataloader(
            src_file=test_src,
            tgt_file=test_tgt,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_len=max_len,
            use_bucketing=False,
            pre_tokenize=pre_tokenize,
            num_workers=num_workers,
        )
    return loaders


def _verify_batch_invariants(batch: dict[str, torch.Tensor], bos_id: int = 2, eos_id: int = 3, pad_id: int = 0) -> None:
    """Kiểm tra 4 ràng buộc cấu trúc trên 1 batch chuẩn."""
    src_ids = batch["src_ids"]
    src_lengths = batch["src_lengths"]
    tgt_in = batch["tgt_input_ids"]
    tgt_out = batch["tgt_output_ids"]
    tgt_lengths = batch["tgt_lengths"]

    assert bool((src_ids[:, 0] == bos_id).all()), "src_ids[:, 0] phải là <bos>"
    assert bool((tgt_in[:, 0] == bos_id).all()), "tgt_input_ids[:, 0] phải là <bos>"

    row_idx = torch.arange(src_ids.size(0))
    assert bool((src_ids[row_idx, src_lengths - 1] == eos_id).all()), "cuối src phải là <eos>"
    assert bool((tgt_out[row_idx, tgt_lengths - 1] == eos_id).all()), "cuối tgt_out phải là <eos>"

    non_pad = tgt_in[:, 1:] != pad_id
    assert bool((tgt_in[:, 1:][non_pad] == tgt_out[:, :-1][non_pad]).all()), "lệch teacher forcing"

    assert int((src_ids != pad_id).sum()) == int(src_lengths.sum()), "tổng src_ids != pad không khớp src_lengths"
    assert int((tgt_in != pad_id).sum()) == int(tgt_lengths.sum()), "tổng tgt_in != pad không khớp tgt_lengths"
    assert int((tgt_out != pad_id).sum()) == int(tgt_lengths.sum()), "tổng tgt_out != pad không khớp tgt_lengths"


if __name__ == "__main__":
    import argparse
    from src.data.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(description="Kiểm tra chỉ số đối chiếu Dataset & DataLoader")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    src_tok = load_tokenizer(args.tokenizer_dir / "bpe_en.json")
    tgt_tok = load_tokenizer(args.tokenizer_dir / "bpe_vi.json")

    total_batches = 0
    for split in ("train", "val", "test"):
        loader = get_dataloader(
            src_file=args.processed_dir / f"{split}.en",
            tgt_file=args.processed_dir / f"{split}.vi",
            src_tokenizer=src_tok,
            tgt_tokenizer=tgt_tok,
            batch_size=args.batch_size,
            shuffle=False,
            max_len=args.max_len,
            use_bucketing=False,
            pre_tokenize=True,
            num_workers=0,
        )
        src_tok_sum = 0
        tgt_tok_sum = 0
        max_s = 0
        max_t = 0
        first_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        for batch in loader:
            _verify_batch_invariants(batch, bos_id=src_tok.bos_id, eos_id=src_tok.eos_id, pad_id=src_tok.pad_id)
            if first_shapes is None:
                first_shapes = (tuple(batch["src_ids"].shape), tuple(batch["tgt_input_ids"].shape))
            src_tok_sum += int(batch["src_lengths"].sum())
            tgt_tok_sum += int(batch["tgt_lengths"].sum())
            max_s = max(max_s, int(batch["src_ids"].size(1)))
            max_t = max(max_t, int(batch["tgt_input_ids"].size(1)))
            total_batches += 1

        assert isinstance(loader.dataset, Sized)
        print(
            f"[{split.upper():<5}] samples={len(loader.dataset):,} | batches={len(loader):,} | "
            f"src_tokens={src_tok_sum:,} | tgt_tokens={tgt_tok_sum:,} | "
            f"max=(S={max_s}, T={max_t}) | first_batch={first_shapes}"
        )
    print(f"ALL_INVARIANTS_PASS | total_batches={total_batches:,}")
