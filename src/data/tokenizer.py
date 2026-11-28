"""Huấn luyện và sử dụng Tokenizer (HF Word-boundary BPE & SentencePiece) cho Anh–Việt."""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

try:
    import sentencepiece as spm
except ImportError:
    spm = None  # type: ignore[assignment]
from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2
DEFAULT_VOCAB_SIZE = VOCAB_SIZE
DEFAULT_MIN_FREQUENCY = MIN_FREQUENCY
DEFAULT_MAX_SEQ_LEN = 128


class BaseTokenizer(ABC):
    """Giao diện thống nhất cho mọi Tokenizer trong dự án."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def sos_id(self) -> int:
        return self.bos_id

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        ...

    @abstractmethod
    def encode_tokens(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_len: int | None = None,
    ) -> list[str]:
        ...

    @abstractmethod
    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        ...

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """Ánh xạ 1 token ID về chuỗi token tương ứng."""
        ...

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """Ánh xạ 1 chuỗi token về token ID số nguyên."""
        ...

    def _apply_special_and_truncate(
        self,
        items: list,
        bos_item,
        eos_item,
        add_special_tokens: bool,
        max_len: int | None,
    ) -> list:
        if add_special_tokens:
            if max_len is not None:
                if max_len <= 0:
                    return []
                if max_len == 1:
                    return [bos_item]
                items = items[: max_len - 2]
            return [bos_item, *items, eos_item]
        if max_len is not None:
            return items[:max_len]
        return items

    def _prepare_decode_ids(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool,
    ) -> list[int]:
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().tolist()
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()

        cleaned: list[int] = []
        for raw_id in ids:
            tid = int(raw_id)
            if tid == self.eos_id:
                if not skip_special_tokens:
                    cleaned.append(tid)
                break
            if skip_special_tokens and tid in (self.pad_id, self.bos_id):
                continue
            cleaned.append(tid)
        return cleaned


class BPETokenizer(BaseTokenizer):
    """Word-boundary BPE Tokenizer (WhitespaceSplit + end_of_word_suffix='</w>')."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(self.model_path))
        self._verify_special_token_ids()

    def _verify_special_token_ids(self) -> None:
        expected = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }
        for token, expected_id in expected.items():
            actual_id = self.tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token {token!r} có ID={actual_id}, kỳ vọng ID={expected_id}"
                )

    @classmethod
    def train(
        cls,
        input_files: list[str | Path],
        save_path: str | Path,
        vocab_size: int = VOCAB_SIZE,
        min_frequency: int = MIN_FREQUENCY,
    ) -> BPETokenizer:
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.decoder = BPEDecoder(suffix="</w>")

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="",
            end_of_word_suffix="</w>",
        )
        files = [str(Path(p)) for p in input_files]
        tokenizer.train(files=files, trainer=trainer)

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out_path))
        return cls(out_path)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        return self._apply_special_and_truncate(
            ids, self.bos_id, self.eos_id, add_special_tokens, max_len
        )

    def encode_tokens(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_len: int | None = None,
    ) -> list[str]:
        tokens = self.tokenizer.encode(text).tokens
        return self._apply_special_and_truncate(
            tokens, "<bos>", "<eos>", add_special_tokens, max_len
        )

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        cleaned = self._prepare_decode_ids(ids, skip_special_tokens=skip_special_tokens)
        return self.tokenizer.decode(cleaned, skip_special_tokens=skip_special_tokens).strip()

    def id_to_token(self, token_id: int) -> str:
        tok = self.tokenizer.id_to_token(int(token_id))
        return UNK_TOKEN if tok is None else tok

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return UNK_ID if token_id is None else int(token_id)

    def __repr__(self) -> str:
        return f"BPETokenizer(path={self.model_path.name}, vocab_size={self.vocab_size})"


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece BPE Tokenizer (mở rộng so sánh với HF BPE)."""

    def __init__(self, model_path: str | Path) -> None:
        if spm is None:
            raise ImportError(
                "Thiếu thư viện 'sentencepiece'. Cài đặt bằng: pip install sentencepiece"
            )
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        model_prefix: str | Path,
        vocab_size: int = VOCAB_SIZE,
    ) -> SentencePieceTokenizer:
        if spm is None:
            raise ImportError(
                "Thiếu thư viện 'sentencepiece'. Cài đặt bằng: pip install sentencepiece"
            )
        prefix_path = Path(model_prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)

        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(prefix_path),
            vocab_size=vocab_size,
            model_type="bpe",
            normalization_rule_name="identity",
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            character_coverage=1.0,
            input_sentence_size=5_000_000,
            shuffle_input_sentence=True,
        )
        return cls(f"{prefix_path}.model")

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        ids = self.sp.encode(text, add_bos=False, add_eos=False)
        return self._apply_special_and_truncate(
            ids, self.bos_id, self.eos_id, add_special_tokens, max_len
        )

    def encode_tokens(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_len: int | None = None,
    ) -> list[str]:
        tokens = self.sp.encode_as_pieces(text)
        return self._apply_special_and_truncate(
            tokens, "<bos>", "<eos>", add_special_tokens, max_len
        )

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        cleaned = self._prepare_decode_ids(ids, skip_special_tokens=skip_special_tokens)
        return self.sp.decode(cleaned)

    def id_to_token(self, token_id: int) -> str:
        tid = int(token_id)
        if tid < 0 or tid >= self.vocab_size:
            return UNK_TOKEN
        return str(self.sp.id_to_piece(tid))

    def token_to_id(self, token: str) -> int:
        return int(self.sp.piece_to_id(token))

    def __repr__(self) -> str:
        return f"SentencePieceTokenizer(path={self.model_path.name}, vocab_size={self.vocab_size})"


def create_tokenizer(
    model_path: str | Path,
    tokenizer_type: str = "bpe",
) -> BaseTokenizer:
    """Factory khởi tạo Tokenizer theo loại ('bpe' hoặc 'sentencepiece')."""
    kind = tokenizer_type.lower().strip()
    if kind == "bpe":
        return BPETokenizer(model_path)
    if kind in ("sentencepiece", "spm"):
        return SentencePieceTokenizer(model_path)
    raise ValueError(f"Tokenizer type '{tokenizer_type}' chưa được hỗ trợ.")


def load_tokenizer(
    path: str | Path,
    tokenizer_type: str | None = None,
) -> BaseTokenizer:
    """Tải Tokenizer từ file (tự nhận diện .json -> BPE, .model -> SentencePiece)."""
    if tokenizer_type is None:
        tokenizer_type = "sentencepiece" if Path(path).suffix == ".model" else "bpe"
    return create_tokenizer(path, tokenizer_type=tokenizer_type)


def compute_tokenizer_stats(
    tokenizer: BaseTokenizer,
    lines: list[str],
) -> dict[str, float | int]:
    """Đo subword fertility, unk_rate, roundtrip và độ bao phủ tại max_len=100, 128."""
    if not lines:
        return {}

    total_words = 0
    total_subwords = 0
    total_unks = 0
    exact_roundtrip = 0
    covered_100 = 0
    covered_128 = 0

    for line in lines:
        words = line.split()
        total_words += len(words)

        ids_no_spec = tokenizer.encode(line, add_special_tokens=False)
        total_subwords += len(ids_no_spec)
        total_unks += sum(1 for tid in ids_no_spec if tid == tokenizer.unk_id)

        full_len = len(ids_no_spec) + 2
        if full_len <= 100:
            covered_100 += 1
        if full_len <= 128:
            covered_128 += 1

        if tokenizer.decode(ids_no_spec) == line:
            exact_roundtrip += 1

    n = len(lines)
    return {
        "sentences": n,
        "vocab_size": tokenizer.vocab_size,
        "subword_fertility": round(total_subwords / max(total_words, 1), 4),
        "unk_rate": round(total_unks / max(total_subwords, 1), 6),
        "roundtrip_rate": round(exact_roundtrip / n, 4),
        "coverage_max_len_100": round(covered_100 / n, 4),
        "coverage_max_len_128": round(covered_128 / n, 4),
    }


def _read_clean_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def train_and_evaluate_bpe(
    processed_dir: Path = Path("data/processed"),
    tokenizer_dir: Path = Path("data/tokenizer"),
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = MIN_FREQUENCY,
) -> dict:
    """Huấn luyện 2 bộ BPE riêng (bpe_en.json, bpe_vi.json) và xuất tokenizer_stats.json."""
    train_en = processed_dir / "train.en"
    train_vi = processed_dir / "train.vi"

    en_tok = BPETokenizer.train(
        [train_en],
        tokenizer_dir / "bpe_en.json",
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    vi_tok = BPETokenizer.train(
        [train_vi],
        tokenizer_dir / "bpe_vi.json",
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    stats: dict = {"en": {}, "vi": {}}
    for split_name in ("train", "val", "test"):
        en_file = processed_dir / f"{split_name}.en"
        vi_file = processed_dir / f"{split_name}.vi"
        if en_file.exists():
            stats["en"][split_name] = compute_tokenizer_stats(
                en_tok, _read_clean_lines(en_file)
            )
        if vi_file.exists():
            stats["vi"][split_name] = compute_tokenizer_stats(
                vi_tok, _read_clean_lines(vi_file)
            )

    stats_path = tokenizer_dir / "tokenizer_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Đã lưu: {tokenizer_dir / 'bpe_en.json'} (vocab={en_tok.vocab_size})")
    print(f"Đã lưu: {tokenizer_dir / 'bpe_vi.json'} (vocab={vi_tok.vocab_size})")
    print(f"Đã lưu thống kê: {stats_path}")

    sample_tests = [
        (en_tok, "EN", "I'd like \"AI: 100% / $50;\" in 2025!"),
        (vi_tok, "VI", "Tiếng Việt có dấu: học máy, dịch máy Anh–Việt (100%)!"),
    ]
    print("\n=== Kiểm thử tách từ & roundtrip ===")
    for tok, lang, sent in sample_tests:
        tokens = tok.encode_tokens(sent)
        ids = tok.encode(sent, add_special_tokens=True)
        decoded = tok.decode(ids)
        print(f"[{lang}] Câu gốc: {sent}")
        print(f"     Tokens : {tokens}")
        print(f"     Decode : {decoded} (match={decoded == sent})")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện 2 bộ BPE Tokenizer riêng cho Anh–Việt.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=MIN_FREQUENCY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate_bpe(
        processed_dir=args.processed_dir,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
