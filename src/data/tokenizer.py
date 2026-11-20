"""Huấn luyện và đóng gói bộ tách từ BPE cho dịch máy Anh - Việt."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None  # type: ignore
    decoders = models = normalizers = pre_tokenizers = trainers = None  # type: ignore

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


class BPETokenizer:
    """Lớp đóng gói BPE Tokenizer hỗ trợ encode/decode kèm dynamic special tokens."""

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        # Lấy special token ID động trực tiếp từ tokenizer (P1)
        self.pad_id = self._resolve_token_id("<pad>", PAD_ID)
        self.unk_id = self._resolve_token_id("<unk>", UNK_ID)
        self.sos_id = self._resolve_token_id("<sos>", SOS_ID)
        self.eos_id = self._resolve_token_id("<eos>", EOS_ID)
        self.special_token_ids = {self.pad_id, self.unk_id, self.sos_id, self.eos_id}

    def _resolve_token_id(self, token: str, fallback_id: int) -> int:
        tid = self.tok.token_to_id(token)
        return tid if tid is not None else fallback_id

    @classmethod
    def from_file(cls, path: str | Path) -> BPETokenizer:
        tok = Tokenizer.from_file(str(path))
        if tok.decoder is None:
            tok.decoder = decoders.ByteLevel()
        return cls(tok)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Chuyển chuỗi thành list id; thêm <sos> và <eos> nếu add_special_tokens=True."""
        ids = self.tok.encode(text).ids
        if add_special_tokens:
            return [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Giải mã list id thành chuỗi văn bản sạch (P0: hỗ trợ đúng skip_special_tokens)."""
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_token_ids]
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


def train_bpe(
    input_file: Path | str,
    output_file: Path | str,
    vocab_size: int = 8000,
) -> BPETokenizer:
    """Huấn luyện BPE Tokenizer Byte-Level với initial_alphabet và lưu metadata (P0, P2)."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    # Bổ sung initial_alphabet để bao phủ toàn bộ 256 bytes UTF-8 (P0)
    byte_alphabet = pre_tokenizers.ByteLevel.alphabet()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=byte_alphabet,
        show_progress=True,
    )
    tok.train(files=[str(input_file)], trainer=trainer)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    # Lưu metadata và version (P2)
    metadata = {
        "model_type": "BPE",
        "pre_tokenizer": "ByteLevel",
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": {
            "<pad>": tok.token_to_id("<pad>"),
            "<unk>": tok.token_to_id("<unk>"),
            "<sos>": tok.token_to_id("<sos>"),
            "<eos>": tok.token_to_id("<eos>"),
        },
        "initial_alphabet_size": len(byte_alphabet),
        "source_data": str(input_file),
    }
    meta_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return BPETokenizer(tok)


def validate_tokenizer(tok: BPETokenizer, test_cases: list[str], lang: str = "vi") -> None:
    """Kiểm tra tokenizer toàn diện trên nhiều hình thái câu (P1)."""
    print(f"\n--- Kiểm tra Tokenizer [{lang.upper()}] với các dạng câu ---")
    for text in test_cases:
        encoded = tok.encode(text, add_special_tokens=True)
        decoded = tok.decode(encoded, skip_special_tokens=True)
        has_sos = encoded[0] == tok.sos_id
        has_eos = encoded[-1] == tok.eos_id
        is_exact = decoded.strip() == text.strip()
        status = "✓" if (is_exact and has_sos and has_eos) else "✗"
        print(f"[{status}] '{text[:40]}' -> tokens={len(encoded)} | exact_match={is_exact}")
        assert has_sos and has_eos, f"Thiếu SOS/EOS: {encoded}"


def main():
    proc_dir = Path("data/processed")
    tok_dir = Path("data/tokenizer")

    train_en = proc_dir / "train.en"
    train_vi = proc_dir / "train.vi"

    print("Đang huấn luyện English tokenizer...")
    tok_en = train_bpe(train_en, tok_dir / "tokenizer_en.json")
    print(f"✓ English vocab size: {tok_en.vocab_size}")

    print("Đang huấn luyện Vietnamese tokenizer...")
    tok_vi = train_bpe(train_vi, tok_dir / "tokenizer_vi.json")
    print(f"✓ Vietnamese vocab size: {tok_vi.vocab_size}")

    # P1: Kiểm tra đa dạng mẫu câu
    en_samples = [
        ".",  # 1 từ / dấu câu
        "Yeah .",
        "How are you today ?",
        "Climate change costs $1,000,000 & affects 99.5% of people on 2024-05-12 .",
        "",  # Câu rỗng
    ]
    vi_samples = [
        ".",
        "Vâng .",
        "Khoa học đằng sau một tiêu đề về biến đổi khí hậu .",
        "Thử nghiệm dấu câu: 100%, $50, ngày 12/05/2024 & các từ có dấu: ồ, ế, ắ, ự, ỡ .",
        "",
    ]
    validate_tokenizer(tok_en, en_samples, lang="en")
    validate_tokenizer(tok_vi, vi_samples, lang="vi")
    print("\n✓ Hoàn tất huấn luyện và kiểm định Tokenizer!")


if __name__ == "__main__":
    main()
