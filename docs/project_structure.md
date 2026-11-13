# Gợi ý Cấu trúc Thư mục Dự án — Đề tài 8: Dịch máy Anh–Việt (Seq2Seq Attention & Transformer)

Tài liệu này tổng hợp cấu trúc thư mục và các file gợi ý dựa theo **To-Do List Nền tảng cơ bản (6 tuần)**. Hiện tại mới khởi tạo sẵn các thư mục trống (kèm `.gitkeep` để Git ghi nhận cấu trúc); các file code (`.py`), cấu hình (`.yaml`) và notebook (`.ipynb`) sẽ được tạo dần khi triển khai từng giai đoạn.

---

## 1. Sơ đồ cấu trúc đầy đủ dự kiến

```text
dich-may-Anh-Viet/
├── configs/                        # Chứa cấu hình siêu tham số (phục vụ Thí nghiệm đối sánh & Ablation Tuần 4)
│   ├── baseline_lstm.yaml          # Cấu hình chuẩn cho Seq2Seq LSTM + Attention
│   ├── transformer_small.yaml      # Cấu hình chuẩn cho Transformer nhỏ
│   └── ablation/                   # Các cấu hình khảo sát (thay đổi số head, số layer, d_model, dropout...)
│
├── data/                           # [Nằm trong .gitignore] Chứa dữ liệu IWSLT'15 En-Vi / PhoMT
│   ├── raw/                        # Dữ liệu gốc tải về (train/tst2012/tst2013 .en, .vi)
│   ├── processed/                  # Dữ liệu sau khi làm sạch & tiền xử lý
│   └── tokenizer/                  # Lưu file model & vocab của bộ tách từ BPE
│
├── docs/                           # Lưu tài liệu tham khảo, bài báo gốc và ghi chú thiết kế
│   ├── project_structure.md        # Tài liệu gợi ý cấu trúc dự án (file này)
│   └── papers/                     # Lưu file PDF các bài báo gốc (Bahdanau, Luong, Attention Is All You Need...)
│
├── src/                            # Mã nguồn chính (tự cài đặt từ đầu - From Scratch)
│   ├── __init__.py
│   ├── data/                       # [Mảng 1 - Tuần 1] Dữ liệu & Tokenizer
│   │   ├── preprocess.py           # Làm sạch câu, chuẩn hóa Unicode tiếng Việt, lọc độ dài
│   │   ├── tokenizer.py            # Huấn luyện và đóng gói bộ tách từ BPE
│   │   └── dataset.py              # PyTorch Dataset, DataLoader, hàm collate_fn (padding động)
│   │
│   ├── models/                     # [Mảng 2 & 3 - Tuần 2, 3] Kiến trúc mô hình tự code
│   │   ├── seq2seq_lstm/           # Baseline Tuần 2
│   │   │   ├── encoder.py          # Encoder LSTM
│   │   │   ├── attention.py        # Tầng Attention (Bahdanau / Luong) trả về attention weights
│   │   │   ├── decoder.py          # Decoder LSTM kết hợp context vector
│   │   │   └── seq2seq.py          # Ghép nối Encoder-Attention-Decoder
│   │   │
│   │   └── transformer/            # Mô hình chính Tuần 3
│   │       ├── positional_encoding.py  # Mã hóa vị trí từ (Sinusoidal)
│   │       ├── attention.py        # Scaled Dot-Product & Multi-Head Attention (lưu attention map)
│   │       ├── layers.py           # Position-wise FFN, Residual Connection, LayerNorm
│   │       ├── masks.py            # Tạo padding_mask và causal_mask (look-ahead mask)
│   │       └── transformer.py      # EncoderLayer, DecoderLayer và toàn bộ Transformer
│   │
│   ├── training/                   # Quy trình huấn luyện & giải mã
│   │   ├── trainer.py              # Vòng lặp train/val, lưu checkpoint, log W&B / TensorBoard
│   │   ├── losses.py               # CrossEntropyLoss (bỏ qua pad_idx, hỗ trợ Label Smoothing)
│   │   └── schedulers.py           # NoamScheduler (warmup) cho Transformer & scheduler cho LSTM
│   │
│   ├── evaluation/                 # [Mảng 1 & 4] Suy luận & Đánh giá
│   │   ├── decode.py               # Giải mã Greedy Search & Beam Search (trích xuất kèm attention)
│   │   └── metrics.py              # Hàm tính chỉ số BLEU chuẩn (SacreBLEU)
│   │
│   └── visualization/              # [Tuần 5] Trực quan hóa & Phân tích
│       └── attention_plot.py       # Vẽ biểu đồ nhiệt (heatmap) gióng hàng từ Anh-Việt
│
├── scripts/                        # Các script chạy từ dòng lệnh (đảm bảo tính Reproducibility Tuần 4)
│   ├── download_data.py            # Script tự động tải bộ dữ liệu IWSLT'15 từ link ngoài
│   ├── train.py                    # Điểm chạy huấn luyện: python scripts/train.py --config ...
│   ├── evaluate.py                 # Chấm điểm BLEU trên tập test và xuất kết quả đối sánh
│   └── translate.py                # Dịch thử 1 câu bất kỳ và xuất hình ảnh attention heatmap
│
├── notebooks/                      # Phục vụ khảo sát dữ liệu & Phân tích định tính (Tuần 1 & Tuần 5)
│   ├── 01_eda_and_tokenizer.ipynb  # Thống kê độ dài câu, từ vựng IWSLT'15, kiểm thử BPE
│   └── 02_error_analysis.ipynb     # Đối sánh câu dịch thực tế (tốt/xấu) giữa LSTM & Transformer
│
├── reports/                        # Các mốc nộp báo cáo chuẩn ACL LaTeX (XeLaTeX + BibTeX)
│   ├── week1_proposal/             # Proposal (tối đa 2 trang)
│   ├── week3_milestone/            # Milestone Report (tối đa 4 trang)
│   └── week6_final_report/         # Báo cáo cuối (tối đa 8 trang + phụ lục, xuất NhomXX_DeTai08.pdf)
│
├── checkpoints/                    # [Nằm trong .gitignore] Lưu trọng số mô hình tốt nhất (.pt)
├── tests/                          # Unit test kiểm tra kích thước tensor, hàm tạo mask, hàm attention
├── app_demo.py                     # [Tuần 6] Giao diện Demo dịch trực tiếp + hiển thị Attention Map
├── .gitignore                      # Chặn đẩy data/, checkpoints/, wandb/, runs/, file tạm LaTeX
├── requirements.txt                # Danh sách thư viện kèm phiên bản cố định
└── README.md                       # Hướng dẫn cài đặt, tải data/checkpoint và lệnh tái tạo kết quả
```

---

## 2. Ánh xạ thư mục theo Mảng công việc & Lộ trình 6 Tuần

| Giai đoạn | Mảng công việc liên quan | Thư mục / Thành phần trọng tâm |
| :--- | :--- | :--- |
| **Tuần 1** | Mảng 1: Dữ liệu, Tokenizer & Đánh giá + Proposal | `docs/papers/`, `data/`, `src/data/`, `src/evaluation/`, `scripts/download_data.py`, `notebooks/`, `reports/week1_proposal/`, `requirements.txt` |
| **Tuần 2** | Mảng 2: Mô hình Baseline (Seq2Seq LSTM Attention) | `configs/`, `src/models/seq2seq_lstm/`, `src/training/`, `scripts/train.py`, `checkpoints/` |
| **Tuần 3** | Mảng 3: Transformer nhỏ & Milestone Report | `src/models/transformer/`, `configs/`, `tests/`, `reports/week3_milestone/` |
| **Tuần 4** | Đối sánh có kiểm soát & Ablation Study | `configs/ablation/`, `scripts/train.py`, `scripts/evaluate.py` |
| **Tuần 5** | Trực quan hóa Attention, Phân tích lỗi & Bản nháp | `src/visualization/`, `scripts/translate.py`, `notebooks/`, `reports/week6_final_report/` |
| **Tuần 6** | Báo cáo cuối, Đóng gói GitHub, Demo & Vấn đáp | `reports/week6_final_report/`, `README.md`, `app_demo.py`, rà soát code trong `src/models/` |

---

## 3. Lưu ý quản lý Dữ liệu & Trọng số trên GitHub
- Không đẩy dữ liệu lớn (`data/raw/`, `data/processed/`, `data/tokenizer/`), log thí nghiệm (`wandb/`, `runs/`) hay trọng số mô hình (`checkpoints/`) lên GitHub.
- Các thư mục này được giữ cấu trúc rỗng trên Git nhờ file `.gitkeep` và đã được cấu hình bỏ qua nội dung bên trong tại `.gitignore`.
- Khi hoàn thiện Tuần 1 và Tuần 6, cần cung cấp script tải dữ liệu (`scripts/download_data.py`) hoặc link tải ngoài trong `README.md` để giảng viên/người đánh giá có thể tái tạo kết quả.
