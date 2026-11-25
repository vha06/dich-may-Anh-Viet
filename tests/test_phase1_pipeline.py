"""End-to-end environment & Phase 1 pipeline verification suite."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
import tokenizers
import torch
import torch.nn as nn

from src.data.dataset import (
    DEFAULT_BATCH_SIZE,
    TranslationDataset,
    _verify_batch_invariants,
    create_dataloader,
)
from src.data.tokenizer import (
    BOS_ID,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_VOCAB_SIZE,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    BaseTokenizer,
    load_tokenizer,
)
from src.evaluation.metrics import decode_and_compute_bleu


class TinyDummyNMT(nn.Module):
    """Minimal neural NMT proxy model to verify tensor shapes, gradients, and BLEU loop."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 32,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_id)
        self.proj = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        src_mask = (src_ids != self.pad_id).unsqueeze(-1).float()
        src_emb = self.src_embed(src_ids) * src_mask
        src_ctx = src_emb.sum(dim=1, keepdim=True) / src_lengths.clamp(min=1).view(
            -1, 1, 1
        ).float()
        tgt_emb = self.tgt_embed(tgt_input_ids)
        hidden = tgt_emb + src_ctx
        return self.proj(hidden)


def check_environment_and_seed(seed: int = 42) -> dict[str, Any]:
    """Verify Python/ML package versions, compute device, and deterministic seeding."""
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    device = "cuda" if cuda_available else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t1 = torch.rand(4, 4)

    torch.manual_seed(seed)
    t2 = torch.rand(4, 4)
    if not torch.equal(t1, t2):
        raise AssertionError("Seed ngau nhien cua PyTorch khong tai lap duoc.")

    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
        "sacrebleu_version": sacrebleu.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "device": device,
        "device_name": device_name,
        "seed": seed,
        "deterministic_seed_ok": True,
    }


def check_preprocessed_data(
    processed_dir: Path = Path("data/processed"),
) -> dict[str, Any]:
    """Verify presence, line counts, alignment, and non-emptiness of processed splits."""
    expected_counts = {"train": 131942, "val": 1553, "test": 1268}
    stats_file = processed_dir / "stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Khong tim thay file thong ke: {stats_file}")

    split_info: dict[str, Any] = {}
    for split_name, expected_n in expected_counts.items():
        en_path = processed_dir / f"{split_name}.en"
        vi_path = processed_dir / f"{split_name}.vi"
        if not en_path.exists() or not vi_path.exists():
            raise FileNotFoundError(f"Thieu file du lieu da xu ly cho tap {split_name}")

        en_lines = en_path.read_text(encoding="utf-8").splitlines()
        vi_lines = vi_path.read_text(encoding="utf-8").splitlines()
        if len(en_lines) != len(vi_lines):
            raise AssertionError(
                f"Lech so dong {split_name}: en={len(en_lines)}, vi={len(vi_lines)}"
            )
        if len(en_lines) != expected_n:
            raise AssertionError(
                f"So cau {split_name} la {len(en_lines)}, ky vong {expected_n}"
            )
        if any(not line.strip() for line in en_lines) or any(
            not line.strip() for line in vi_lines
        ):
            raise AssertionError(f"Phat hien dong rong trong tap {split_name}")

        split_info[split_name] = {
            "num_pairs": len(en_lines),
            "en_path": str(en_path),
            "vi_path": str(vi_path),
        }

    return {"stats_file_exists": True, "splits": split_info}


def check_tokenizers(
    tokenizer_dir: Path = Path("data/tokenizer"),
    max_len: int = DEFAULT_MAX_SEQ_LEN,
) -> tuple[BaseTokenizer, BaseTokenizer, dict[str, Any]]:
    """Verify BPE tokenizers, special token IDs, max_len truncation, and exact roundtrip."""
    src_tok = load_tokenizer(tokenizer_dir / "bpe_en.json", tokenizer_type="bpe")
    tgt_tok = load_tokenizer(tokenizer_dir / "bpe_vi.json", tokenizer_type="bpe")

    for name, tok in (("en", src_tok), ("vi", tgt_tok)):
        if tok.vocab_size != DEFAULT_VOCAB_SIZE:
            raise AssertionError(
                f"vocab_size ({name})={tok.vocab_size}, ky vong {DEFAULT_VOCAB_SIZE}"
            )
        if (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) != (
            PAD_ID,
            UNK_ID,
            BOS_ID,
            EOS_ID,
        ):
            raise AssertionError(f"Special token IDs khong khop chuan cho {name}.")

    sample_en = "This is not a finished story ."
    sample_vi = "Câu chuyện này chưa kết thúc ."
    en_ids = src_tok.encode(sample_en, add_special_tokens=True, max_len=max_len)
    vi_ids = tgt_tok.encode(sample_vi, add_special_tokens=True, max_len=max_len)

    if src_tok.decode(en_ids, skip_special_tokens=True) != sample_en:
        raise AssertionError("Roundtrip decode BPE tieng Anh that bai.")
    if tgt_tok.decode(vi_ids, skip_special_tokens=True) != sample_vi:
        raise AssertionError("Roundtrip decode BPE tieng Viet that bai.")

    long_en = " ".join(["science"] * 200)
    long_vi = " ".join(["ngôn ngữ"] * 200)
    long_en_ids = src_tok.encode(long_en, add_special_tokens=True, max_len=max_len)
    long_vi_ids = tgt_tok.encode(long_vi, add_special_tokens=True, max_len=max_len)

    for lang_label, ids in (("en", long_en_ids), ("vi", long_vi_ids)):
        if len(ids) != max_len:
            raise AssertionError(
                f"Cat chuoi dai ({lang_label}) cho do dai {len(ids)}, ky vong {max_len}"
            )
        if ids[0] != BOS_ID or ids[-1] != EOS_ID:
            raise AssertionError(
                f"Cat chuoi dai ({lang_label}) khong giu nguyen <bos>/<eos>."
            )

    summary = {
        "src_vocab_size": src_tok.vocab_size,
        "tgt_vocab_size": tgt_tok.vocab_size,
        "special_tokens": {
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
        },
        "roundtrip_ok": True,
        "truncation_preserves_eos_ok": True,
        "max_len_tested": max_len,
    }
    return src_tok, tgt_tok, summary


def check_dataloader(
    processed_dir: Path,
    src_tok: BaseTokenizer,
    tgt_tok: BaseTokenizer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
) -> tuple[TranslationDataset, dict[str, torch.Tensor], dict[str, Any]]:
    """Verify TranslationDataset and dynamic-padding DataLoader batch contract."""
    dataset = TranslationDataset(
        src_path=processed_dir / "val.en",
        tgt_path=processed_dir / "val.vi",
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        max_len=max_len,
        pre_tokenize=True,
    )
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    expected_keys = {
        "src_ids",
        "src_lengths",
        "tgt_input_ids",
        "tgt_output_ids",
        "tgt_lengths",
    }
    if set(batch.keys()) != expected_keys:
        raise AssertionError(f"Thieu khoa trong batch: {set(batch.keys())}")

    for k, tensor in batch.items():
        if tensor.dtype != torch.int64:
            raise AssertionError(f"{k} co dtype={tensor.dtype}, ky vong torch.int64")

    bsz, s_max = batch["src_ids"].shape
    _, t_max = batch["tgt_input_ids"].shape
    if bsz != batch_size:
        raise AssertionError(f"Batch size thuc te {bsz}, ky vong {batch_size}")
    if s_max > max_len or t_max > max_len - 1:
        raise AssertionError(f"Kich thuoc batch vuot nguong: S_max={s_max}, T_max={t_max}")

    _verify_batch_invariants(batch, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID)

    summary = {
        "val_dataset_len": len(dataset),
        "batch_size": bsz,
        "s_max": int(s_max),
        "t_max": int(t_max),
        "tensor_shapes": {k: list(v.shape) for k, v in batch.items()},
        "invariants_ok": True,
    }
    return dataset, batch, summary


def check_end_to_end_training_and_bleu_step(
    dataset: TranslationDataset,
    batch: dict[str, torch.Tensor],
    src_tok: BaseTokenizer,
    tgt_tok: BaseTokenizer,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Run forward -> loss -> backward -> step -> decode -> SacreBLEU on a real batch."""
    torch.manual_seed(seed)
    torch_device = torch.device(device)

    model = TinyDummyNMT(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        embed_dim=32,
        pad_id=PAD_ID,
    ).to(torch_device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    src_ids = batch["src_ids"].to(torch_device)
    src_lengths = batch["src_lengths"].to(torch_device)
    tgt_input_ids = batch["tgt_input_ids"].to(torch_device)
    tgt_output_ids = batch["tgt_output_ids"].to(torch_device)

    bsz, t_max = tgt_input_ids.shape
    num_params = sum(p.numel() for p in model.parameters())

    model.train()
    logits_step1 = model(src_ids, src_lengths, tgt_input_ids)
    expected_shape = (bsz, t_max, tgt_tok.vocab_size)
    if tuple(logits_step1.shape) != expected_shape:
        raise AssertionError(
            f"Logits shape {tuple(logits_step1.shape)} != {expected_shape}"
        )

    loss_step1 = criterion(
        logits_step1.reshape(-1, tgt_tok.vocab_size),
        tgt_output_ids.reshape(-1),
    )
    if not torch.isfinite(loss_step1):
        raise AssertionError(f"Loss khong hop le: {loss_step1.item()}")

    optimizer.zero_grad()
    loss_step1.backward()

    sq_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"Tham so {name} khong co gradient.")
        if not torch.isfinite(param.grad).all():
            raise AssertionError(f"Gradient cua {name} chua NaN hoac Inf.")
        sq_norm += float(param.grad.detach().pow(2).sum().item())
    grad_norm = round(math.sqrt(sq_norm), 4)

    optimizer.step()

    with torch.no_grad():
        logits_step2 = model(src_ids, src_lengths, tgt_input_ids)
        loss_step2 = criterion(
            logits_step2.reshape(-1, tgt_tok.vocab_size),
            tgt_output_ids.reshape(-1),
        )

    if not (loss_step2.item() < loss_step1.item()):
        raise AssertionError(
            f"Loss khong giam sau buoc optimizer.step(): "
            f"{loss_step1.item():.4f} -> {loss_step2.item():.4f}"
        )

    pred_ids = logits_step2.argmax(dim=-1).detach().cpu()
    raw_refs = [dataset.get_raw_pair(i)[1] for i in range(bsz)]

    dummy_bleu = decode_and_compute_bleu(pred_ids, raw_refs, tgt_tok)
    oracle_bleu = decode_and_compute_bleu(
        tgt_output_ids.detach().cpu(), raw_refs, tgt_tok
    )
    if oracle_bleu["bleu"] <= 99.0:
        raise AssertionError(
            f"Oracle batch BLEU thap bat thuong: {oracle_bleu['bleu']}"
        )

    return {
        "dummy_model_params": num_params,
        "logits_shape": list(logits_step1.shape),
        "expected_random_init_loss_lnV": round(math.log(tgt_tok.vocab_size), 4),
        "loss_step1": round(float(loss_step1.item()), 4),
        "loss_step2": round(float(loss_step2.item()), 4),
        "loss_decreased": True,
        "grad_l2_norm": grad_norm,
        "gradients_all_finite": True,
        "dummy_batch_bleu": dummy_bleu["bleu"],
        "oracle_batch_bleu": oracle_bleu["bleu"],
    }


def run_all_checks(
    processed_dir: Path = Path("data/processed"),
    tokenizer_dir: Path = Path("data/tokenizer"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute all 5 pipeline stages and return a combined verification report."""
    env_info = check_environment_and_seed(seed=seed)
    data_info = check_preprocessed_data(processed_dir=processed_dir)
    src_tok, tgt_tok, tok_info = check_tokenizers(
        tokenizer_dir=tokenizer_dir, max_len=max_len
    )
    dataset, batch, loader_info = check_dataloader(
        processed_dir=processed_dir,
        src_tok=src_tok,
        tgt_tok=tgt_tok,
        batch_size=batch_size,
        max_len=max_len,
    )
    e2e_info = check_end_to_end_training_and_bleu_step(
        dataset=dataset,
        batch=batch,
        src_tok=src_tok,
        tgt_tok=tgt_tok,
        device=env_info["device"],
        seed=seed,
    )
    return {
        "stage1_environment": env_info,
        "stage2_preprocessed_data": data_info,
        "stage3_tokenizers": tok_info,
        "stage4_dataloader": loader_info,
        "stage5_e2e_train_and_bleu": e2e_info,
        "all_passed": True,
    }


class TestPhase1Pipeline(unittest.TestCase):
    """Automated unit/integration tests compatible with both pytest and unittest."""

    def test_phase1_end_to_end_pipeline(self) -> None:
        report = run_all_checks()
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["stage5_e2e_train_and_bleu"]["loss_decreased"])
        self.assertGreater(report["stage5_e2e_train_and_bleu"]["oracle_batch_bleu"], 99.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chay thu moi truong & kiem dinh tich hop Pipeline Giai doan 1"
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    print("=== Chay thu Moi truong & Kiem dinh Tich hop Pipeline Giai doan 1 ===")

    report = run_all_checks(
        processed_dir=args.processed_dir,
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
    )

    s1 = report["stage1_environment"]
    s2 = report["stage2_preprocessed_data"]["splits"]
    s3 = report["stage3_tokenizers"]
    s4 = report["stage4_dataloader"]
    s5 = report["stage5_e2e_train_and_bleu"]

    print(
        f"[1/5] Moi truong & Seed : PASS | Python {s1['python_version']} | "
        f"PyTorch {s1['torch_version']} | tokenizers {s1['tokenizers_version']} | "
        f"sacrebleu {s1['sacrebleu_version']} | device={s1['device']}"
    )
    print(
        f"[2/5] Du lieu da xu ly  : PASS | train={s2['train']['num_pairs']:,} | "
        f"val={s2['val']['num_pairs']:,} | test={s2['test']['num_pairs']:,}"
    )
    print(
        f"[3/5] BPE Tokenizers    : PASS | vocab={s3['src_vocab_size']}/{s3['tgt_vocab_size']} | "
        f"roundtrip=OK | truncation@{s3['max_len_tested']}=OK"
    )
    print(
        f"[4/5] Dataset/DataLoader: PASS | batch_size={s4['batch_size']} | "
        f"S_max={s4['s_max']}, T_max={s4['t_max']} | invariants=OK"
    )
    print(
        f"[5/5] Forward->Loss->BLEU: PASS | logits={tuple(s5['logits_shape'])} | "
        f"loss: {s5['loss_step1']} -> {s5['loss_step2']} (ln(V)~{s5['expected_random_init_loss_lnV']}) | "
        f"grad_norm={s5['grad_l2_norm']} | oracle_bleu={s5['oracle_batch_bleu']}"
    )

    out_path = args.processed_dir / "phase1_smoke_test.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTOAN BO 5 MAT XICH DA VUOT QUA! Da luu bao cao tai: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
