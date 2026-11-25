"""PyTorch Dataset và DataLoader chuẩn nhóm cho IWSLT'15 En–Vi."""

from __future__ import annotations

import functools
import json
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.tokenizer import PAD_ID, BaseTokenizer, load_tokenizer

DEFAULT_MAX_LEN = 128


class TranslationDataset(Dataset):
    """Dataset chứa các cặp câu (Anh, Việt) với tiền mã hóa và lưu văn bản gốc."""

    def __init__(
        self,
        en_path: str | Path,
        vi_path: str | Path,
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer | None = None,
        max_len: int = DEFAULT_MAX_LEN,
        pre_tokenize: bool = True,
    ) -> None:
        if max_len < 2:
            raise ValueError(f"max_len phải >= 2 để chứa <bos> và <eos>, nhận được {max_len}")

        with Path(en_path).open("r", encoding="utf-8") as f:
            self.src_lines = [line.rstrip("\r\n") for line in f]
        with Path(vi_path).open("r", encoding="utf-8") as f:
            self.tgt_lines = [line.rstrip("\r\n") for line in f]

        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError("Số dòng file Anh và Việt không khớp!")

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer if tgt_tokenizer is not None else src_tokenizer
        self.max_len = max_len
        self.pre_tokenize = pre_tokenize
        self.pad_id = self.src_tokenizer.pad_id

        self.data: list[tuple[list[int], list[int]]] = []
        if pre_tokenize:
            for en, vi in zip(self.src_lines, self.tgt_lines):
                src_ids = self.src_tokenizer.encode(en, add_special_tokens=True, max_len=self.max_len)
                tgt_ids = self.tgt_tokenizer.encode(vi, add_special_tokens=True, max_len=self.max_len)
                self.data.append((src_ids, tgt_ids))

    def get_raw_pair(self, idx: int) -> tuple[str, str]:
        """Lấy cặp câu văn bản gốc (chưa cắt) để chấm SacreBLEU chuẩn."""
        return self.src_lines[idx], self.tgt_lines[idx]

    @property
    def src_lengths(self) -> list[int]:
        if self.pre_tokenize:
            return [len(src_ids) for src_ids, _ in self.data]
        return [
            len(self.src_tokenizer.encode(s, add_special_tokens=True, max_len=self.max_len))
            for s in self.src_lines
        ]

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.pre_tokenize:
            src_ids, tgt_ids = self.data[idx]
        else:
            src_ids = self.src_tokenizer.encode(
                self.src_lines[idx], add_special_tokens=True, max_len=self.max_len
            )
            tgt_ids = self.tgt_tokenizer.encode(
                self.tgt_lines[idx], add_special_tokens=True, max_len=self.max_len
            )
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_translation_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_id: int = PAD_ID,
) -> dict[str, Tensor]:
    """Đệm động cấp module và dịch lệch Teacher Forcing, trả về dict 5 khóa cố định."""
    src_list: list[Tensor] = []
    tgt_in_list: list[Tensor] = []
    tgt_out_list: list[Tensor] = []
    src_lengths: list[int] = []
    tgt_lengths: list[int] = []

    for src, tgt in batch:
        src_list.append(src)
        src_lengths.append(src.size(0))
        tgt_in = tgt[:-1]
        tgt_out = tgt[1:]
        tgt_in_list.append(tgt_in)
        tgt_out_list.append(tgt_out)
        tgt_lengths.append(tgt_in.size(0))

    return {
        "src_ids": pad_sequence(src_list, batch_first=True, padding_value=pad_id),
        "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
        "tgt_input_ids": pad_sequence(tgt_in_list, batch_first=True, padding_value=pad_id),
        "tgt_output_ids": pad_sequence(tgt_out_list, batch_first=True, padding_value=pad_id),
        "tgt_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Gom các câu có độ dài gần nhau vào cùng batch để giảm tối đa token <pad>."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        mega_batch_mult: int = 50,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mega_batch_size = batch_size * mega_batch_mult

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        batches: list[list[int]] = []
        for i in range(0, len(indices), self.mega_batch_size):
            mega = indices[i : i + self.mega_batch_size]
            mega.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(mega), self.batch_size):
                batches.append(mega[j : j + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def create_dataloader(
    en_path: str | Path,
    vi_path: str | Path,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    use_bucketing: bool = False,
    num_workers: int = 0,
    max_len: int = DEFAULT_MAX_LEN,
    pin_memory: bool = False,
    pre_tokenize: bool = True,
) -> DataLoader:
    """Tạo DataLoader đơn lẻ hỗ trợ BucketBatchSampler và collate_fn cấp module."""
    dataset = TranslationDataset(
        en_path=en_path,
        vi_path=vi_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len,
        pre_tokenize=pre_tokenize,
    )
    collate_fn = functools.partial(collate_translation_batch, pad_id=dataset.pad_id)

    if use_bucketing:
        batch_sampler = BucketBatchSampler(
            lengths=dataset.src_lengths,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


build_dataloader = create_dataloader


def build_dataloaders(
    processed_dir: str | Path,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer | None = None,
    batch_size: int = 64,
    max_len: int = DEFAULT_MAX_LEN,
    use_bucketing: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    pre_tokenize: bool = True,
) -> dict[str, DataLoader]:
    """Tạo trọn bộ 3 DataLoader (train / val / test); val & test luôn khóa shuffle=False."""
    pdir = Path(processed_dir)
    loaders: dict[str, DataLoader] = {}
    configs = {
        "train": (True, use_bucketing),
        "val": (False, False),
        "test": (False, False),
    }
    for split, (shuffle, bucketing) in configs.items():
        en_p = pdir / f"{split}.en"
        vi_p = pdir / f"{split}.vi"
        if en_p.exists() and vi_p.exists():
            loaders[split] = create_dataloader(
                en_path=en_p,
                vi_path=vi_p,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                batch_size=batch_size,
                shuffle=shuffle,
                use_bucketing=bucketing,
                num_workers=num_workers,
                max_len=max_len,
                pin_memory=pin_memory,
                pre_tokenize=pre_tokenize,
            )
    return loaders


def _verify_batch_invariants(
    batch: dict[str, Tensor],
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> None:
    """Kiểm tra hợp đồng 5 khóa, dtype, độ dài và vị trí <bos>/<eos>."""
    expected_keys = {"src_ids", "src_lengths", "tgt_input_ids", "tgt_output_ids", "tgt_lengths"}
    assert set(batch.keys()) == expected_keys
    for val in batch.values():
        assert val.dtype == torch.long

    bsz = batch["src_ids"].size(0)
    assert batch["src_lengths"].shape == (bsz,)
    assert batch["tgt_lengths"].shape == (bsz,)
    assert batch["tgt_input_ids"].shape == batch["tgt_output_ids"].shape

    for i in range(bsz):
        slen = int(batch["src_lengths"][i].item())
        tlen = int(batch["tgt_lengths"][i].item())
        assert batch["src_ids"][i, 0].item() == bos_id
        assert batch["src_ids"][i, slen - 1].item() == eos_id
        if slen < batch["src_ids"].size(1):
            assert (batch["src_ids"][i, slen:] == pad_id).all()
        assert batch["tgt_input_ids"][i, 0].item() == bos_id
        assert batch["tgt_output_ids"][i, tlen - 1].item() == eos_id


def evaluate_dataloader_stats(
    processed_dir: Path = Path("data/processed"),
    tokenizer_dir: Path = Path("data/tokenizer"),
    batch_size: int = 64,
    max_len: int = DEFAULT_MAX_LEN,
) -> dict:
    """Kiểm định toàn bộ split và đo mật độ token thực (không bucketing vs. có bucketing)."""
    src_tok = load_tokenizer(tokenizer_dir / "bpe_en.json")
    tgt_tok = load_tokenizer(tokenizer_dir / "bpe_vi.json")

    stats: dict = {"batch_size": batch_size, "max_len": max_len, "modes": {}}

    for use_bucketing in (False, True):
        mode_name = "bucketed" if use_bucketing else "standard"
        loaders = build_dataloaders(
            processed_dir=processed_dir,
            src_tokenizer=src_tok,
            tgt_tokenizer=tgt_tok,
            batch_size=batch_size,
            max_len=max_len,
            use_bucketing=use_bucketing,
        )
        mode_stats: dict = {}
        for split_name, loader in loaders.items():
            total_src_real = 0
            total_src_cells = 0
            total_tgt_real = 0
            total_tgt_cells = 0
            num_batches = 0

            for batch in loader:
                if num_batches < 5:
                    _verify_batch_invariants(batch, src_tok.bos_id, src_tok.eos_id, src_tok.pad_id)
                total_src_real += int(batch["src_lengths"].sum().item())
                total_src_cells += batch["src_ids"].numel()
                total_tgt_real += int(batch["tgt_lengths"].sum().item())
                total_tgt_cells += batch["tgt_input_ids"].numel()
                num_batches += 1

            mode_stats[split_name] = {
                "batches": num_batches,
                "src_token_density": round(total_src_real / max(total_src_cells, 1), 4),
                "tgt_token_density": round(total_tgt_real / max(total_tgt_cells, 1), 4),
            }
        stats["modes"][mode_name] = mode_stats

    out_path = processed_dir / "dataloader_stats.json"
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã kiểm định và lưu thống kê DataLoader tại: {out_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    evaluate_dataloader_stats()
