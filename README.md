# Dịch Máy Anh–Việt (Seq2Seq Attention & Transformer)

Dự án xây dựng **từ đầu (from scratch)** hai mô hình dịch máy Anh→Việt:
- **Seq2Seq LSTM + Attention** (Bahdanau/Luong)
- **Transformer nhỏ** (theo bài báo "Attention Is All You Need")

Dữ liệu: IWSLT'15 En–Vi (~133,000 cặp câu)  
Đánh giá: Điểm BLEU (sacrebleu)

---

## 1. Cài đặt môi trường

```bash
# Tạo môi trường conda (Python 3.10)
conda create -n mt-env python=3.10
conda activate mt-env

# Cài thư viện
pip install -r requirements.txt
```

---

## 2. Tải dữ liệu

```bash
# Tải IWSLT'15 En–Vi về data/raw/
python scripts/download_data.py

# Tiền xử lý dữ liệu (làm sạch + lọc độ dài)
python src/data/preprocess.py

# Huấn luyện BPE tokenizer (joint, vocab_size=8000)
python src/data/tokenizer.py
```

---

## 3. Huấn luyện mô hình

```bash
# Huấn luyện Seq2Seq LSTM + Attention (Tuần 2)
python scripts/train.py --config configs/baseline_lstm.yaml

# Huấn luyện Transformer nhỏ (Tuần 3)
python scripts/train.py --config configs/transformer_small.yaml
```

> Checkpoint lưu tại `checkpoints/` (không đẩy lên GitHub).  
> Tải checkpoint đã huấn luyện: **[link Google Drive — cập nhật sau]**

---

## 4. Đánh giá

```bash
# Tính điểm BLEU trên tập test
python scripts/evaluate.py --config configs/baseline_lstm.yaml
python scripts/evaluate.py --config configs/transformer_small.yaml
```

---

## 5. Dịch thử 1 câu + xem Attention Map

```bash
python scripts/translate.py --model seq2seq --text "I love Vietnam."
python scripts/translate.py --model transformer --text "I love Vietnam."
```

---

## 6. Cấu trúc thư mục

```
dich-may-Anh-Viet/
├── configs/           # Siêu tham số cho từng mô hình
├── data/              # Dữ liệu (gitignore — tải qua script)
├── docs/              # Tài liệu, bài báo tham khảo
├── notebooks/         # Jupyter notebook phân tích & demo
├── reports/           # Proposal, Milestone, Báo cáo cuối
├── scripts/           # Script chạy từ dòng lệnh
├── src/
│   ├── data/          # Tiền xử lý, Tokenizer, DataLoader
│   ├── models/        # Seq2Seq LSTM & Transformer (from scratch)
│   ├── training/      # Vòng lặp huấn luyện, scheduler
│   ├── evaluation/    # BLEU, Beam Search
│   └── visualization/ # Vẽ Attention heatmap
├── tests/             # Unit test
├── requirements.txt
└── README.md
```

---

## 7. Kết quả

| Mô hình | Số tham số | BLEU (val) | BLEU (test) | Thời gian train (GPU T4) |
|---|---|---|---|---|
| **Seq2Seq Bi-LSTM + Bahdanau Attention** | 33,064,000 | **19.31** (Epoch 7) | Đang cập nhật | ~3.8 giờ (Kaggle T4) |
| **Transformer nhỏ** | — | — | — | — |

### Ví dụ dịch thực tế (Seq2Seq LSTM + Bahdanau Attention - Epoch 7):

| Câu tiếng Anh (Input) | Dịch Greedy | Dịch Beam Search (beam=5) |
|---|---|---|
| `how are you today ?` | vậy ngày nay nay ? | **vậy bạn ngày hôm nay ?** |
| `i am learning machine translation with deep learning .` | tôi học học thuật dịch với với học tập sâu sâu sâu . | **tôi đang học học thuật dịch với với học tập sâu sâu sắc .** |
| `vietnam is a beautiful country in southeast asia .` | việtet là một đất nước đẹp đẹp ở nam á á . | **việtet là một đất nước đẹp đẹp ở nam á á .** |
| `thank you very much for your help .` | cảm ơn các bạn rất nhiều vì sự giúp đỡ của bạn . | **cảm ơn các bạn rất nhiều vì sự giúp đỡ của bạn .** |

> *Nhận xét sơ bộ:*
> - **Beam Search** cho kết quả ngữ nghĩa tự nhiên và mạch lạc hơn hẳn so với **Greedy Search**.
> - Với các câu giao tiếp thông dụng (`thank you very much for your help .`), mô hình dịch chuẩn xác 100% tiếng Việt tự nhiên.
> - Với các câu kỹ thuật (`machine translation`, `deep learning`), mô hình dịch trúng nghĩa gốc ("thuật dịch", "học tập sâu sắc") dù còn hiện tượng lặp từ — hiện tượng kinh điển của mạng RNN/LSTM sẽ được khắc phục triệt để bằng Transformer ở Tuần 3.

---

## 8. Tác giả

- **Hiếu** — Nhánh `hieu`
