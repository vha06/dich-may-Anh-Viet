"""Huấn luyện và sử dụng BPE Tokenizer (sentencepiece) cho cặp Anh–Việt."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

VOCAB_SIZE = 8000
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_tokenizer(
    train_en_path: Path,
    train_vi_path: Path,
    model_prefix: str = "data/tokenizer/bpe",
    vocab_size: int = VOCAB_SIZE,
) -> None:
    """Huấn luyện joint BPE trên dữ liệu train Anh + Việt gộp chung."""
    combined_path = Path(model_prefix).parent / "combined_train.txt"
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    en_text = train_en_path.read_text(encoding="utf-8")
    vi_text = train_vi_path.read_text(encoding="utf-8")
    combined_path.write_text(en_text + "\n" + vi_text, encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(combined_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        character_coverage=1.0,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )

    combined_path.unlink(missing_ok=True)


class BPETokenizer:
    """Wrapper quanh SentencePieceProcessor cho encode/decode."""

    def __init__(self, model_path: str | Path) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Câu → danh sách ID số (kèm BOS/EOS nếu cần)."""
        return self.sp.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int]) -> str:
        """Danh sách ID số → câu (tự bỏ token đặc biệt)."""
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        filtered = [i for i in ids if i not in special_ids]
        return self.sp.decode(filtered)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    train_tokenizer(
        train_en_path=Path("data/processed/train.en"),
        train_vi_path=Path("data/processed/train.vi"),
    )
