"""PyTorch Dataset and dynamic-padding DataLoader for English-Vietnamese NMT."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import _length_summary
from src.data.tokenizer import (
    BOS_ID,
    DEFAULT_MAX_SEQ_LEN,
    EOS_ID,
    PAD_ID,
    BaseTokenizer,
    load_tokenizer,
)

DEFAULT_BATCH_SIZE = 64


class TranslationDataset(Dataset):
    """Parallel corpus dataset wrapping BaseTokenizer contracts for NMT training/eval."""

    def __init__(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer,
        max_len: int = DEFAULT_MAX_SEQ_LEN,
        pre_tokenize: bool = True,
    ) -> None:
        self.src_path = Path(src_path)
        self.tgt_path = Path(tgt_path)
        if not self.src_path.exists():
            raise FileNotFoundError(f"Khong tim thay file nguon: {self.src_path}")
        if not self.tgt_path.exists():
            raise FileNotFoundError(f"Khong tim thay file dich: {self.tgt_path}")
        if max_len < 2:
            raise ValueError(f"max_len phai >= 2 de chua <bos> va <eos>, nhan duoc {max_len}")
        if src_tokenizer.pad_id != tgt_tokenizer.pad_id:
            raise ValueError(
                f"pad_id khong khop giua src ({src_tokenizer.pad_id}) va tgt ({tgt_tokenizer.pad_id})"
            )

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.pre_tokenize = pre_tokenize
        self.pad_id = src_tokenizer.pad_id

        self.src_lines = self.src_path.read_text(encoding="utf-8").splitlines()
        self.tgt_lines = self.tgt_path.read_text(encoding="utf-8").splitlines()

        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError(
                f"So dong khong khop: {self.src_path}={len(self.src_lines)}, "
                f"{self.tgt_path}={len(self.tgt_lines)}"
            )
        if not self.src_lines:
            raise ValueError(f"Du lieu rong tai {self.src_path}")

        self._src_ids: list[list[int]] | None = None
        self._tgt_ids: list[list[int]] | None = None
        if self.pre_tokenize:
            self._src_ids = [
                self.src_tokenizer.encode(
                    line, add_special_tokens=True, max_len=self.max_len
                )
                for line in self.src_lines
            ]
            self._tgt_ids = [
                self.tgt_tokenizer.encode(
                    line, add_special_tokens=True, max_len=self.max_len
                )
                for line in self.tgt_lines
            ]

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if self._src_ids is not None and self._tgt_ids is not None:
            return {
                "src_ids": self._src_ids[idx],
                "tgt_ids": self._tgt_ids[idx],
            }
        return {
            "src_ids": self.src_tokenizer.encode(
                self.src_lines[idx], add_special_tokens=True, max_len=self.max_len
            ),
            "tgt_ids": self.tgt_tokenizer.encode(
                self.tgt_lines[idx], add_special_tokens=True, max_len=self.max_len
            ),
        }

    def get_raw_pair(self, idx: int) -> tuple[str, str]:
        """Return original (src_text, tgt_text) pair at index idx for inspection/eval."""
        return self.src_lines[idx], self.tgt_lines[idx]


def collate_translation_batch(
    batch: Sequence[dict[str, list[int]]],
    pad_id: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    """Pad a batch dynamically to (S_max, T_max) and apply 1-step Teacher Forcing shift."""
    if not batch:
        raise ValueError("Batch khong duoc rong.")

    batch_size = len(batch)
    src_seqs: list[list[int]] = []
    tgt_in_seqs: list[list[int]] = []
    tgt_out_seqs: list[list[int]] = []
    src_lengths: list[int] = []
    tgt_lengths: list[int] = []

    for item in batch:
        src = item["src_ids"]
        tgt = item["tgt_ids"]
        if len(tgt) < 2:
            raise ValueError(
                f"Chuoi dich phai co it nhat 2 token (<bos>, <eos>), nhan duoc do dai {len(tgt)}"
            )

        # Teacher Forcing shift: tgt_input = [<bos>, y1..ym], tgt_output = [y1..ym, <eos>]
        tgt_in = tgt[:-1]
        tgt_out = tgt[1:]

        src_seqs.append(src)
        tgt_in_seqs.append(tgt_in)
        tgt_out_seqs.append(tgt_out)
        src_lengths.append(len(src))
        tgt_lengths.append(len(tgt_in))

    s_max = max(src_lengths)
    t_max = max(tgt_lengths)

    src_tensor = torch.full((batch_size, s_max), pad_id, dtype=torch.long)
    tgt_input_tensor = torch.full((batch_size, t_max), pad_id, dtype=torch.long)
    tgt_output_tensor = torch.full((batch_size, t_max), pad_id, dtype=torch.long)

    for i in range(batch_size):
        s_len = src_lengths[i]
        t_len = tgt_lengths[i]
        src_tensor[i, :s_len] = torch.as_tensor(src_seqs[i], dtype=torch.long)
        tgt_input_tensor[i, :t_len] = torch.as_tensor(tgt_in_seqs[i], dtype=torch.long)
        tgt_output_tensor[i, :t_len] = torch.as_tensor(tgt_out_seqs[i], dtype=torch.long)

    return {
        "src_ids": src_tensor,
        "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
        "tgt_input_ids": tgt_input_tensor,
        "tgt_output_ids": tgt_output_tensor,
        "tgt_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
    }


def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap TranslationDataset in a PyTorch DataLoader with dynamic batch padding."""
    collate_fn = partial(collate_translation_batch, pad_id=dataset.pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def build_dataloaders(
    processed_dir: str | Path,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
    pre_tokenize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Construct train (shuffled) and val/test (deterministic) DataLoaders."""
    proc_dir = Path(processed_dir)
    split_configs = (
        ("train", True),
        ("val", False),
        ("test", False),
    )

    loaders: dict[str, DataLoader] = {}
    for split_name, shuffle in split_configs:
        dataset = TranslationDataset(
            src_path=proc_dir / f"{split_name}.en",
            tgt_path=proc_dir / f"{split_name}.vi",
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_len=max_len,
            pre_tokenize=pre_tokenize,
        )
        loaders[split_name] = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def _verify_batch_invariants(
    batch: dict[str, torch.Tensor],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> None:
    """Assert structural invariants of a collated batch tensor dictionary."""
    src_ids = batch["src_ids"]
    src_lengths = batch["src_lengths"]
    tgt_in = batch["tgt_input_ids"]
    tgt_out = batch["tgt_output_ids"]
    tgt_lengths = batch["tgt_lengths"]

    bsz = src_ids.size(0)
    row_idx = torch.arange(bsz)

    if not torch.all(src_ids[:, 0] == bos_id):
        raise AssertionError("Moi cau nguon phai bat dau bang <bos>.")
    if not torch.all(tgt_in[:, 0] == bos_id):
        raise AssertionError("Moi tgt_input_ids phai bat dau bang <bos>.")
    if not torch.all(src_ids[row_idx, src_lengths - 1] == eos_id):
        raise AssertionError("Vi tri cuoi cua src_ids phai la <eos>.")
    if not torch.all(tgt_out[row_idx, tgt_lengths - 1] == eos_id):
        raise AssertionError("Vi tri cuoi cua tgt_output_ids phai la <eos>.")

    # Verify Teacher Forcing 1-step shift over all content tokens
    shift_mask = tgt_in[:, 1:] != pad_id
    if not torch.equal(tgt_in[:, 1:][shift_mask], tgt_out[:, :-1][shift_mask]):
        raise AssertionError("tgt_input_ids va tgt_output_ids khong lech dung 1 buoc Teacher Forcing.")

    # Verify padding counts match length tensors exactly
    if int((src_ids != pad_id).sum().item()) != int(src_lengths.sum().item()):
        raise AssertionError("So token thuc cua src_ids khong khop src_lengths.")
    if int((tgt_in != pad_id).sum().item()) != int(tgt_lengths.sum().item()):
        raise AssertionError("So token thuc cua tgt_input_ids khong khop tgt_lengths.")
    if int((tgt_out != pad_id).sum().item()) != int(tgt_lengths.sum().item()):
        raise AssertionError("So token thuc cua tgt_output_ids khong khop tgt_lengths.")


def evaluate_dataloader_stats(
    dataloader: DataLoader,
    static_max_len: int = DEFAULT_MAX_SEQ_LEN,
) -> dict[str, Any]:
    """Iterate over a DataLoader, verify invariants, and compute dynamic-padding efficiency."""
    dataset = dataloader.dataset
    if not isinstance(dataset, TranslationDataset):
        raise TypeError("DataLoader phai boc ben trong TranslationDataset.")

    pad_id = dataset.pad_id
    bos_id = dataset.src_tokenizer.bos_id
    eos_id = dataset.src_tokenizer.eos_id

    batch_s_max_list: list[int] = []
    batch_t_max_list: list[int] = []
    num_samples = 0
    src_real_tokens = 0
    tgt_real_tokens = 0
    src_dynamic_total_cells = 0
    tgt_dynamic_total_cells = 0

    for batch in dataloader:
        _verify_batch_invariants(batch, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)

        bsz, s_max = batch["src_ids"].shape
        _, t_max = batch["tgt_input_ids"].shape

        num_samples += bsz
        batch_s_max_list.append(int(s_max))
        batch_t_max_list.append(int(t_max))

        src_real_tokens += int(batch["src_lengths"].sum().item())
        tgt_real_tokens += int(batch["tgt_lengths"].sum().item())
        src_dynamic_total_cells += bsz * s_max
        tgt_dynamic_total_cells += bsz * t_max

    src_static_total_cells = num_samples * static_max_len
    tgt_static_total_cells = num_samples * (static_max_len - 1)

    return {
        "num_samples": num_samples,
        "num_batches": len(batch_s_max_list),
        "batch_size": dataloader.batch_size,
        "batch_s_max": _length_summary(batch_s_max_list),
        "batch_t_max": _length_summary(batch_t_max_list),
        "src_real_tokens": src_real_tokens,
        "src_dynamic_total_cells": src_dynamic_total_cells,
        "src_static_total_cells": src_static_total_cells,
        "src_real_token_density_dynamic_pct": round(
            100.0 * src_real_tokens / src_dynamic_total_cells, 2
        ),
        "src_real_token_density_static_pct": round(
            100.0 * src_real_tokens / src_static_total_cells, 2
        ),
        "src_cells_saved_vs_static_pct": round(
            100.0 * (src_static_total_cells - src_dynamic_total_cells) / src_static_total_cells,
            2,
        ),
        "tgt_real_tokens": tgt_real_tokens,
        "tgt_dynamic_total_cells": tgt_dynamic_total_cells,
        "tgt_static_total_cells": tgt_static_total_cells,
        "tgt_real_token_density_dynamic_pct": round(
            100.0 * tgt_real_tokens / tgt_dynamic_total_cells, 2
        ),
        "tgt_real_token_density_static_pct": round(
            100.0 * tgt_real_tokens / tgt_static_total_cells, 2
        ),
        "tgt_cells_saved_vs_static_pct": round(
            100.0 * (tgt_static_total_cells - tgt_dynamic_total_cells) / tgt_static_total_cells,
            2,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiem thu TranslationDataset & DataLoader voi Dynamic Padding"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thu muc chua train/val/test (.en, .vi) da lam sach",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Thu muc chua bpe_en.json va bpe_vi.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Kich thuoc moi lo (mac dinh: 64)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Do dai toi da moi chuoi bao gom <bos> va <eos> (mac dinh: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed co dinh thu tu shuffle tap train khi thong ke (mac dinh: 42)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    torch.manual_seed(args.seed)

    src_tok = load_tokenizer(args.tokenizer_dir / "bpe_en.json", tokenizer_type="bpe")
    tgt_tok = load_tokenizer(args.tokenizer_dir / "bpe_vi.json", tokenizer_type="bpe")

    print("=== Khoi tao TranslationDataset & DataLoader (IWSLT'15 En-Vi) ===")
    loaders = build_dataloaders(
        processed_dir=args.processed_dir,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=args.batch_size,
        max_len=args.max_len,
        pre_tokenize=True,
    )

    sample_batch = next(iter(loaders["val"]))
    print("\n[Mau Batch dau tien tren tap val]")
    for key, tensor in sample_batch.items():
        print(f"  - {key:<15}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    first_src = sample_batch["src_ids"][0]
    first_tgt_in = sample_batch["tgt_input_ids"][0]
    first_tgt_out = sample_batch["tgt_output_ids"][0]
    t_len = int(sample_batch["tgt_lengths"][0].item())
    s_len = int(sample_batch["src_lengths"][0].item())

    sample_inspection = {
        "val_first_batch_shapes": {k: list(v.shape) for k, v in sample_batch.items()},
        "first_sample_src_len": s_len,
        "first_sample_tgt_shifted_len": t_len,
        "first_sample_src_ids": first_src[:s_len].tolist(),
        "first_sample_tgt_input_ids": first_tgt_in[:t_len].tolist(),
        "first_sample_tgt_output_ids": first_tgt_out[:t_len].tolist(),
        "first_sample_src_decoded": src_tok.decode(first_src, skip_special_tokens=True),
        "first_sample_tgt_decoded": tgt_tok.decode(first_tgt_out, skip_special_tokens=True),
    }

    print(f"  - Cau nguon [0] (len={s_len}): {sample_inspection['first_sample_src_decoded']}")
    print(f"  - Cau dich  [0] (len={t_len}): {sample_inspection['first_sample_tgt_decoded']}")

    report: dict[str, Any] = {
        "config": {
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "seed": args.seed,
            "pad_id": PAD_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "pre_tokenize": True,
        },
        "sample_inspection": sample_inspection,
        "splits": {},
    }

    print("\n[Thong ke hieu qua Dynamic Padding qua cac tap]")
    for split_name in ("train", "val", "test"):
        split_stats = evaluate_dataloader_stats(
            loaders[split_name], static_max_len=args.max_len
        )
        report["splits"][split_name] = split_stats
        s_sum = split_stats["batch_s_max"]
        t_sum = split_stats["batch_t_max"]
        print(
            f"  - {split_name:<5}: {split_stats['num_samples']:,} cau | "
            f"{split_stats['num_batches']:,} batches | "
            f"S_max(mean={s_sum['mean']}, median={s_sum['median']}, max={s_sum['max']}) | "
            f"T_max(mean={t_sum['mean']}, median={t_sum['median']}, max={t_sum['max']}) | "
            f"Tiet kiem cell: src={split_stats['src_cells_saved_vs_static_pct']}%, "
            f"tgt={split_stats['tgt_cells_saved_vs_static_pct']}%"
        )

    out_path = args.processed_dir / "dataloader_stats.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDa luu thong ke chi tiet tai: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
