"""Tokenizers với giao diện thống nhất cho dịch máy (hỗ trợ WhitespaceSplit BPE; mở rộng được cho SentencePiece/Word)."""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from src.data.preprocess import _length_summary

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
SOS_ID = BOS_ID
EOS_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_MAX_SEQ_LEN = 160

T = TypeVar("T")


class BaseTokenizer(ABC):
    """Giao diện thống nhất cho các tokenizer (BPE, SentencePiece, Word-level)."""

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
    def sos_id(self) -> int:
        """Alias cho bos_id để tương thích ngược với quy ước đặt tên <sos>."""
        return self.bos_id

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def special_token_ids(self) -> set[int]:
        return {self.pad_id, self.unk_id, self.bos_id, self.eos_id}

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Tổng kích thước từ vựng bao gồm special tokens."""

    @abstractmethod
    def train(
        self,
        files: Sequence[str | Path],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
        show_progress: bool = False,
    ) -> None:
        """Huấn luyện tokenizer trên danh sách file văn bản."""

    @abstractmethod
    def _encode_raw(self, text: str) -> tuple[list[int], list[str]]:
        """Mã hóa câu thành (token_ids, token_strings) không kèm special tokens."""

    @abstractmethod
    def _decode_raw(self, ids: list[int], skip_special_tokens: bool) -> str:
        """Giải mã danh sách token IDs đã làm sạch thành văn bản."""

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """Ánh xạ 1 token ID sang chuỗi token."""

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """Ánh xạ 1 chuỗi token sang integer ID."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Lưu trạng thái tokenizer xuống đĩa."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseTokenizer:
        """Nạp tokenizer đã huấn luyện từ file."""

    @classmethod
    def from_file(cls, path: str | Path) -> BaseTokenizer:
        """Alias tương thích ngược cho load()."""
        return cls.load(path)

    @staticmethod
    def _format_sequence(
        content: list[T],
        bos_item: T,
        eos_item: T,
        add_special_tokens: bool,
        max_len: int | None,
    ) -> list[T]:
        """Cắt ngắn theo max_len và bọc BOS/EOS nếu cần."""
        if max_len is not None:
            if add_special_tokens and max_len < 2:
                raise ValueError("max_len phải >= 2 khi add_special_tokens=True")
            if not add_special_tokens and max_len < 0:
                raise ValueError("max_len phải >= 0")

        if add_special_tokens:
            limit = None if max_len is None else max_len - 2
            trimmed = content if limit is None else content[:limit]
            return [bos_item, *trimmed, eos_item]

        return content if max_len is None else content[:max_len]

    def _clean_decode_ids(
        self, ids: Sequence[int] | Any, skip_special_tokens: bool
    ) -> list[int]:
        """Chuyển tensor/sequence về list[int] và dừng tại EOS đầu tiên khi skip_special_tokens=True."""
        raw_ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        int_ids = [int(token_id) for token_id in raw_ids]
        if not skip_special_tokens:
            return int_ids

        cleaned: list[int] = []
        for token_id in int_ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            cleaned.append(token_id)
        return cleaned

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        """Mã hóa một câu thành danh sách token IDs."""
        content_ids, _ = self._encode_raw(text)
        return self._format_sequence(
            content_ids,
            bos_item=self.bos_id,
            eos_item=self.eos_id,
            add_special_tokens=add_special_tokens,
            max_len=max_len,
        )

    def encode_tokens(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_len: int | None = None,
    ) -> list[str]:
        """Mã hóa một câu thành danh sách chuỗi subword/word (tiện vẽ Attention)."""
        _, content_tokens = self._encode_raw(text)
        return self._format_sequence(
            content_tokens,
            bos_item=BOS_TOKEN,
            eos_item=EOS_TOKEN,
            add_special_tokens=add_special_tokens,
            max_len=max_len,
        )

    def decode(
        self,
        ids: Sequence[int] | Any,
        skip_special_tokens: bool = True,
    ) -> str:
        """Giải mã danh sách hoặc tensor 1D các token IDs về chuỗi văn bản."""
        cleaned_ids = self._clean_decode_ids(ids, skip_special_tokens=skip_special_tokens)
        return self._decode_raw(cleaned_ids, skip_special_tokens=skip_special_tokens)


class BPETokenizer(BaseTokenizer):
    """WhitespaceSplit BPE (Sennrich et al., 2016) với hậu tố </w> và chuẩn hóa NFC."""

    END_OF_WORD_SUFFIX = "</w>"

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer
        if self._tokenizer is not None:
            self._verify_special_token_ids()

    def _require_ready(self) -> Tokenizer:
        if self._tokenizer is None:
            raise RuntimeError("BPETokenizer chưa được train hoặc load từ file.")
        return self._tokenizer

    def _verify_special_token_ids(self) -> None:
        tok = self._require_ready()
        if tok.token_to_id(PAD_TOKEN) != PAD_ID:
            raise ValueError(f"Special token {PAD_TOKEN!r} phải có ID={PAD_ID}")
        if tok.token_to_id(UNK_TOKEN) != UNK_ID:
            raise ValueError(f"Special token {UNK_TOKEN!r} phải có ID={UNK_ID}")
        bos_actual = tok.token_to_id(BOS_TOKEN)
        sos_actual = tok.token_to_id("<sos>")
        if bos_actual != BOS_ID and sos_actual != BOS_ID:
            raise ValueError(
                f"Special token {BOS_TOKEN!r}/'<sos>' phải có ID={BOS_ID} "
                f"(nhận được {bos_actual}/{sos_actual})"
            )
        if tok.token_to_id(EOS_TOKEN) != EOS_ID:
            raise ValueError(f"Special token {EOS_TOKEN!r} phải có ID={EOS_ID}")

    @property
    def vocab_size(self) -> int:
        return self._require_ready().get_vocab_size()

    def train(
        self,
        files: Sequence[str | Path],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
        show_progress: bool = False,
    ) -> None:
        str_files = [str(f) for f in files]
        if not str_files:
            raise ValueError("Cần ít nhất 1 file dữ liệu để huấn luyện tokenizer.")

        tok = Tokenizer(
            models.BPE(unk_token=UNK_TOKEN, end_of_word_suffix=self.END_OF_WORD_SUFFIX)
        )
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok.decoder = decoders.BPEDecoder(suffix=self.END_OF_WORD_SUFFIX)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            end_of_word_suffix=self.END_OF_WORD_SUFFIX,
            show_progress=show_progress,
        )
        tok.train(str_files, trainer=trainer)
        self._tokenizer = tok
        self._verify_special_token_ids()

    def _encode_raw(self, text: str) -> tuple[list[int], list[str]]:
        encoding = self._require_ready().encode(text)
        return list(encoding.ids), list(encoding.tokens)

    def _decode_raw(self, ids: list[int], skip_special_tokens: bool) -> str:
        return self._require_ready().decode(ids, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, token_id: int) -> str:
        tok = self._require_ready().id_to_token(int(token_id))
        return UNK_TOKEN if tok is None else tok

    def token_to_id(self, token: str) -> int:
        token_id = self._require_ready().token_to_id(token)
        return UNK_ID if token_id is None else int(token_id)

    def save(self, path: str | Path) -> None:
        tok = self._require_ready()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(out_path))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file tokenizer: {in_path}")
        tok = Tokenizer.from_file(str(in_path))
        if tok.decoder is None:
            tok.decoder = decoders.BPEDecoder(suffix=cls.END_OF_WORD_SUFFIX)
        return cls(tok)


def create_tokenizer(tokenizer_type: str = "bpe", **kwargs: Any) -> BaseTokenizer:
    """Factory khởi tạo tokenizer theo loại ('bpe', 'sentencepiece', 'word')."""
    kind = tokenizer_type.strip().lower()
    if kind == "bpe":
        return BPETokenizer(**kwargs)
    if kind in ("sentencepiece", "word"):
        raise NotImplementedError(
            f"Tokenizer '{kind}' dành cho phần mở rộng sau; hiện tại chỉ hỗ trợ 'bpe'."
        )
    raise ValueError(f"Không hỗ trợ tokenizer_type={tokenizer_type!r}")


def load_tokenizer(path: str | Path, tokenizer_type: str = "bpe") -> BaseTokenizer:
    """Factory nạp tokenizer đã lưu theo loại ('bpe', 'sentencepiece', 'word')."""
    kind = tokenizer_type.strip().lower()
    if kind == "bpe":
        return BPETokenizer.load(path)
    if kind in ("sentencepiece", "word"):
        raise NotImplementedError(
            f"Tokenizer '{kind}' dành cho phần mở rộng sau; hiện tại chỉ hỗ trợ 'bpe'."
        )
    raise ValueError(f"Không hỗ trợ tokenizer_type={tokenizer_type!r}")


def train_bpe(
    input_file: Path | str,
    output_file: Path | str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> BPETokenizer:
    """Huấn luyện và lưu WhitespaceSplit BPE Tokenizer (giữ API tương thích ngược)."""
    tok = BPETokenizer()
    tok.train([input_file], vocab_size=vocab_size, min_frequency=min_frequency)
    tok.save(output_file)
    return tok


def evaluate_tokenizer(
    tokenizer: BaseTokenizer,
    text_path: Path,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> dict[str, Any]:
    """Tính phân phối độ dài token, coverage, subword fertility, UNK rate và roundtrip."""
    lines = text_path.read_text(encoding="utf-8").splitlines()

    token_lengths_with_specials: list[int] = []
    total_words = 0
    total_content_tokens = 0
    total_unks = 0
    covered_count = 0
    exact_roundtrip = 0
    split_word_examples: list[dict[str, Any]] = []
    seen_split_words: set[str] = set()
    sentence_samples: list[dict[str, Any]] = []

    for idx, line in enumerate(lines):
        words = line.split()
        total_words += len(words)

        content_ids, content_tokens = tokenizer._encode_raw(line)
        total_content_tokens += len(content_ids)
        total_unks += sum(1 for tid in content_ids if tid == tokenizer.unk_id)

        full_ids = [tokenizer.bos_id, *content_ids, tokenizer.eos_id]
        token_lengths_with_specials.append(len(full_ids))
        if len(full_ids) <= max_seq_len:
            covered_count += 1

        decoded = tokenizer.decode(full_ids, skip_special_tokens=True)
        if decoded == line:
            exact_roundtrip += 1

        if len(sentence_samples) < 3 and idx in (0, 2, 4):
            sentence_samples.append(
                {
                    "text": line,
                    "tokens_with_specials": [BOS_TOKEN, *content_tokens, EOS_TOKEN],
                    "ids_with_specials": full_ids,
                    "decoded": decoded,
                    "exact_match": decoded == line,
                }
            )

        if len(split_word_examples) < 10:
            for w in words:
                if w in seen_split_words or len(w) < 5:
                    continue
                w_ids, w_tokens = tokenizer._encode_raw(w)
                if len(w_tokens) >= 2:
                    seen_split_words.add(w)
                    split_word_examples.append(
                        {"word": w, "subwords": w_tokens, "ids": w_ids}
                    )
                    if len(split_word_examples) >= 10:
                        break

    num_sentences = len(lines)
    fertility = round(total_content_tokens / total_words, 4) if total_words else 0.0
    coverage = round(100.0 * covered_count / num_sentences, 2) if num_sentences else 0.0
    unk_rate = (
        round(100.0 * total_unks / total_content_tokens, 4)
        if total_content_tokens
        else 0.0
    )
    roundtrip_rate = (
        round(100.0 * exact_roundtrip / num_sentences, 2) if num_sentences else 0.0
    )

    return {
        "num_sentences": num_sentences,
        "total_words": total_words,
        "total_content_tokens": total_content_tokens,
        "subword_fertility": fertility,
        f"coverage_at_{max_seq_len}_pct": coverage,
        "truncated_at_max_len": num_sentences - covered_count,
        "unk_count": total_unks,
        "unk_rate_pct": unk_rate,
        "exact_roundtrip_pct": roundtrip_rate,
        "token_len_with_specials": _length_summary(token_lengths_with_specials),
        "sentence_samples": sentence_samples,
        "split_word_examples": split_word_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá bộ tách từ WhitespaceSplit BPE cho IWSLT'15 En-Vi"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục chứa dữ liệu đã làm sạch (mặc định: data/processed)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Thư mục lưu model BPE (mặc định: data/tokenizer)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Kích thước từ vựng BPE cho mỗi ngôn ngữ (mặc định: 8000)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help="Tần suất tối thiểu để gộp cặp BPE (mặc định: 2)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Ngưỡng độ dài tối đa (token) để thống kê độ bao phủ (mặc định: 160)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    processed_dir: Path = args.processed_dir
    tokenizer_dir: Path = args.tokenizer_dir

    report: dict[str, Any] = {
        "config": {
            "tokenizer_type": "bpe",
            "pre_tokenizer": "WhitespaceSplit",
            "normalizer": "NFC",
            "end_of_word_suffix": BPETokenizer.END_OF_WORD_SUFFIX,
            "vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
            "max_seq_len": args.max_len,
            "special_tokens": {
                PAD_TOKEN: PAD_ID,
                UNK_TOKEN: UNK_ID,
                BOS_TOKEN: BOS_ID,
                EOS_TOKEN: EOS_ID,
            },
        },
        "languages": {},
    }

    print("=== Huấn luyện & Đánh giá Bộ tách từ WhitespaceSplit BPE (IWSLT'15 En-Vi) ===")
    for lang in ("en", "vi"):
        train_file = processed_dir / f"train.{lang}"
        out_file = tokenizer_dir / f"bpe_{lang}.json"

        tok = create_tokenizer("bpe")
        tok.train([train_file], vocab_size=args.vocab_size, min_frequency=args.min_frequency)
        tok.save(out_file)

        reloaded = load_tokenizer(out_file, tokenizer_type="bpe")
        lang_report: dict[str, Any] = {
            "model_path": str(out_file),
            "vocab_size": reloaded.vocab_size,
            "splits": {},
        }

        print(f"\n[{lang.upper()}] Đã lưu tại: {out_file} | vocab_size={reloaded.vocab_size}")
        for split in ("train", "val", "test"):
            split_path = processed_dir / f"{split}.{lang}"
            split_stats = evaluate_tokenizer(
                reloaded, split_path, max_seq_len=args.max_len
            )
            lang_report["splits"][split] = split_stats
            tlen = split_stats["token_len_with_specials"]
            cov_key = f"coverage_at_{args.max_len}_pct"
            print(
                f"  - {split:<5}: mean={tlen['mean']}, median={tlen['median']}, "
                f"p95={tlen['p95']}, p99={tlen['p99']}, max={tlen['max']} | "
                f"cov<={args.max_len}: {split_stats[cov_key]}% "
                f"(cắt {split_stats['truncated_at_max_len']}) | "
                f"fertility={split_stats['subword_fertility']} | "
                f"unk={split_stats['unk_count']} ({split_stats['unk_rate_pct']}%) | "
                f"roundtrip={split_stats['exact_roundtrip_pct']}%"
            )

        report["languages"][lang] = lang_report

    stats_path = tokenizer_dir / "tokenizer_stats.json"
    stats_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nĐã lưu thống kê chi tiết tại: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
