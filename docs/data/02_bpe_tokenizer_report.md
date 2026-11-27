# Báo cáo Bộ tách từ BPE & Giao diện Chuẩn hóa — Đề tài 8: Dịch máy Anh–Việt

> **Tài liệu lưu trữ:** Bản nháp kỹ thuật phục vụ soạn thảo Proposal (Tuần 1), Báo cáo cuối (Tuần 6) và ôn tập Vấn đáp cá nhân.  
> **Mã nguồn liên quan:** [`src/data/tokenizer.py`](../../src/data/tokenizer.py), [`data/tokenizer/tokenizer_stats.json`](../../data/tokenizer/tokenizer_stats.json)

---

## 1. Kiến trúc Chuẩn hóa để Mở rộng (`BaseTokenizer`)

Nhằm phục vụ mục tiêu mở rộng ở Giai đoạn 5 (so sánh giữa **BPE**, **SentencePiece** và **Word-level Tokenizer**) mà không phải sửa lại code của `DataLoader`, kiến trúc mô hình (`Seq2Seq LSTM` / `Transformer`) hay hàm chấm điểm `BLEU`, module [`src/data/tokenizer.py`](../../src/data/tokenizer.py) được thiết kế theo mô hình hướng đối tượng thống nhất:

### 1.1. Quy ước 4 Token đặc biệt cố định (Special Tokens)
Mọi loại bộ tách từ kế thừa từ `BaseTokenizer` đều bắt buộc ánh xạ đúng 4 ID đầu tiên trong từ điển:

| Token | Hằng số | ID cố định | Vai trò trong mô hình Dịch máy |
| :--- | :--- | :---: | :--- |
| `<pad>` | `PAD_TOKEN` / `PAD_ID` | `0` | Đệm các câu ngắn trong cùng 1 batch có cùng độ dài để tính toán song song. |
| `<unk>` | `UNK_TOKEN` / `UNK_ID` | `1` | Đại diện cho ký tự/từ hiếm chưa từng xuất hiện trong tập từ vựng huấn luyện. |
| `<bos>` | `BOS_TOKEN` / `BOS_ID` | `2` | Đánh dấu điểm bắt đầu câu (*Beginning of Sentence*), làm tín hiệu khởi động cho Decoder. |
| `<eos>` | `EOS_TOKEN` / `EOS_ID` | `3` | Đánh dấu kết thúc câu (*End of Sentence*), báo cho mô hình dừng sinh từ. |

### 1.2. Hợp đồng giao tiếp chuẩn của `BaseTokenizer`
- **Thuộc tính chuẩn:** `vocab_size`, `pad_id`, `unk_id`, `bos_id`, `eos_id`, cùng `@property sos_id` (`return self.bos_id`) để tương thích ngược nếu dùng chung tiện ích gọi `<sos>`.
- **Mã hóa ra ID (`encode(text, add_special_tokens=True, max_len=None) -> list[int]`):**
  - Nếu `add_special_tokens=True`, tự động bọc `[<bos>, ..., <eos>]`.
  - Nếu truyền `max_len` (mặc định `128`), tự động cắt phần nội dung sao cho tổng độ dài $\le \text{max\_len}$ và **luôn giữ `<eos>` ở vị trí cuối cùng**.
- **Mã hóa ra chuỗi ký hiệu (`encode_tokens(text, add_special_tokens=False, max_len=None) -> list[str]`):**
  - Trả về danh sách các mảnh từ/subword dạng chữ (phục vụ gắn nhãn trục X, Y cho biểu đồ nhiệt Attention Map ở Tuần 5).
- **Giải mã (`decode(ids, skip_special_tokens=True) -> str`):**
  - Chấp nhận cả `list[int]` lẫn `torch.Tensor` 1 chiều.
  - Khi `skip_special_tokens=True`, tự động **dừng tại token `<eos>` đầu tiên** (loại bỏ triệt để các token rác hoặc `<pad>` phía sau `<eos>`), bỏ qua `<pad>`/`<bos>` và khôi phục lại câu văn hoàn chỉnh.
- **Ánh xạ đơn lẻ (`id_to_token` & `token_to_id`):** Thuộc hợp đồng `BaseTokenizer`, hỗ trợ chuyển đổi qua lại giữa 1 token ID và chuỗi ký hiệu trên cả BPE lẫn SentencePiece.
- **Factory mở rộng (`create_tokenizer` & `load_tokenizer`):**
  - Đã cài đặt đầy đủ cả **`"bpe"` (`BPETokenizer`)** và **`"sentencepiece"` / `"spm"` (`SentencePieceTokenizer`**, bọc import `sentencepiece` tùy chọn qua `try ... except ImportError`).
  - Hàm `load_tokenizer(path)` tự động nhận diện đuôi `.json` (BPE) hoặc `.model` (SentencePiece); các loại chưa hỗ trợ (như `"word"`) trả về `ValueError` rõ ràng.

---

## 2. Nguyên lý Thuật toán BPE & Thiết lập Huấn luyện (`BPETokenizer`)

- **Thuật toán:** Byte-Pair Encoding theo bài báo gốc của Sennrich et al. (ACL 2016) sử dụng thư viện `tokenizers`:
  - `Pre-tokenizer`: `WhitespaceSplit()` (tách thô theo khoảng trắng vì dữ liệu IWSLT'15 đã được tiền xử lý dấu câu rõ ràng ở bước trước).
  - `Model`: `BPE` với hậu tố kết thúc từ `end_of_word_suffix="</w>"`. Những mảnh kết thúc bằng `</w>` đánh dấu ranh giới cuối từ, giúp `BPEDecoder(suffix="</w>")` biết chính xác chỗ nào ghép liền thành 1 từ và chỗ nào chèn dấu cách.
- **Quy tắc chống rò rỉ dữ liệu:** Hai bộ tách từ Tiếng Anh (`bpe_en.json`) và Tiếng Việt (`bpe_vi.json`) chỉ được huấn luyện **duy nhất trên tập `train`** (`data/processed/train.en` và `train.vi`), tuyệt đối không dùng `val` hay `test`.
- **Siêu tham số đã chốt:**
  - `vocab_size = 8,000` cho mỗi ngôn ngữ (gồm 4 special tokens + 7,996 subwords/từ).
  - `min_frequency = 2` (chỉ gộp các cặp xuất hiện từ 2 lần trở lên).
  - `max_seq_len = 128` token (đã bao gồm `<bos>` và `<eos>`).

---

## 3. Thống kê Đánh giá BPE trên 3 tập `Train` / `Validation` / `Test`

Kết quả trích xuất trực tiếp từ [`data/tokenizer/tokenizer_stats.json`](../../data/tokenizer/tokenizer_stats.json):

### 3.1. Phân bố độ dài câu (tính theo số token BPE gồm `<bos>` và `<eos>`) và Độ bao phủ tại `max_seq_len = 128`

| Ngôn ngữ | Tập | Số câu | Mean | Median (p50) | p95 | p99 | Min | Max | Tỷ lệ $\le 128$ token | Số câu vượt 128 |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EN** | **Train** | 131,942 | 25.02 | 21.0 | 56.0 | 82.0 | 3 | 180 | **99.97%** | 33 |
| **EN** | **Val** (`tst2012`) | 1,553 | 22.23 | 19.0 | 47.0 | 64.48 | 4 | 103 | **100.00%** | 0 |
| **EN** | **Test** (`tst2013`) | 1,268 | 26.00 | 22.0 | 57.0 | 80.33 | 4 | 112 | **100.00%** | 0 |
| **VI** | **Train** | 131,942 | 27.33 | 22.0 | 62.0 | 91.0 | 3 | 165 | **99.95%** | 62 |
| **VI** | **Val** (`tst2012`) | 1,553 | 24.61 | 21.0 | 50.0 | 72.0 | 4 | 129 | **99.94%** | 1 |
| **VI** | **Test** (`tst2013`) | 1,268 | 29.23 | 25.0 | 65.65 | 95.33 | 4 | 127 | **100.00%** | 0 |

### 3.2. Chất lượng tách & khôi phục chuỗi (Subword Fertility, UNK Rate, Exact Roundtrip)

| Ngôn ngữ | Tập | Tổng số từ gốc | Tổng token BPE nội dung | Subword Fertility *(token/từ)* | Số lượng `<unk>` | Tỷ lệ `<unk>` | Tỷ lệ khôi phục 100% |
| :---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: |
| **EN** | **Train** | 2,670,454 | 3,036,879 | **1.1372** | 0 | 0.0000% | **100.00%** |
| **EN** | **Val** | 27,983 | 31,420 | **1.1228** | 0 | 0.0000% | **100.00%** |
| **EN** | **Test** | 26,728 | 30,438 | **1.1388** | 2 | 0.0066% | **99.84%** |
| **VI** | **Train** | 3,265,896 | 3,341,603 | **1.0232** | 0 | 0.0000% | **100.00%** |
| **VI** | **Val** | 34,297 | 35,111 | **1.0237** | 1 | 0.0028% | **99.94%** |
| **VI** | **Test** | 33,682 | 34,529 | **1.0251** | 1 | 0.0029% | **99.92%** |

**Phân tích các chỉ số:**
1. **Độ dài tối đa `128` token cực kỳ hợp lý:** Ngưỡng `128` bao phủ **100%** tập `test` (EN & VI), **100%** tập `val` (EN), **99.94%** tập `val` (VI — chỉ duy nhất 1 câu dài 129 token) và **99.95%–99.97%** tập `train` (sau khi thống nhất lọc `max_words = 128` từ ở bước tiền xử lý).
2. **Subword Fertility thấp & tối ưu:**
   - Tiếng Anh có trung bình **~1.137 token/từ** (từ gốc có hậu tố `-s`, `-ed`, `-ing` hoặc từ ghép hiếm được tách gọn thành 2–3 mảnh).
   - Tiếng Việt có hệ số **~1.023 token/từ** (gần như 1:1 với âm tiết tiếng Việt vì chỉ có `25,431` âm tiết/từ phân biệt ở tập `train`; các tên riêng nước ngoài hoặc âm tiết rất hiếm mới bị tách nhỏ).
3. **Tỷ lệ `<unk>` gần như bằng 0:** Trên toàn bộ hơn 130,000 token của tập `val` và `test`, chỉ xuất hiện đúng **4 trường hợp** `<unk>` do ký tự ngoại lai chưa từng có trong `train`:
   - `test.en` (2 lần): `'Ashê'` (dòng 934) và `'Ōfunato'` (dòng 1,075).
   - `val.vi` (1 lần): `'̉'` — 1 dấu hỏi tổ hợp đứng cô lập (dòng 1,417).
   - `test.vi` (1 lần): `'Ōfunato'` (dòng 1,075).

---

## 4. Ví dụ Thực tế Mã hóa & Giải mã (Encode $\rightarrow$ Decode)

### 4.1. Ví dụ tách từ hiếm / từ ghép thành Subwords (từ tập `val`)
- **Tiếng Anh (`bpe_en.json`):**
  - `bonds` $\rightarrow$ `['bon', 'ds</w>']` *(IDs: `[4421, 370]`)*
  - `four-year-old` $\rightarrow$ `['four', '-year-old</w>']` *(IDs: `[3274, 3182]`)*
  - `jigsaw` $\rightarrow$ `['j', 'ig', 'saw</w>']` *(IDs: `[72, 2743, 1306]`)*
  - `playwright` $\rightarrow$ `['play', 'w', 'right</w>']` *(IDs: `[5319, 85, 583]`)*
  - `communists` $\rightarrow$ `['communi', 'sts</w>']` *(IDs: `[1401, 702]`)*
- **Tiếng Việt (`bpe_vi.json`):**
  - `trườn` $\rightarrow$ `['tr', 'ườn</w>']` *(IDs: `[448, 2198]`)*
  - `quạnh` $\rightarrow$ `['qu', 'ạnh</w>']` *(IDs: `[475, 1001]`)*
  - `nguôi` $\rightarrow$ `['ngu', 'ôi</w>']` *(IDs: `[1142, 451]`)*
  - `Melbourne` $\rightarrow$ `['Mel', 'bo', 'ur', 'ne</w>']` *(IDs: `[3947, 1317, 1761, 1959]`)*

### 4.2. Ví dụ vòng lặp đầy đủ trên 1 cặp câu (từ tập `val`)
- **Câu Tiếng Anh:** `It is a jigsaw puzzle still being put together .`
  - *Subword tokens (có `<bos>`, `<eos>`):*  
    `['<bos>', 'It</w>', 'is</w>', 'a</w>', 'j', 'ig', 'saw</w>', 'puzzle</w>', 'still</w>', 'being</w>', 'put</w>', 'together</w>', '.</w>', '<eos>']`
  - *Token IDs:*  
    `[2, 389, 243, 130, 72, 2743, 1306, 5923, 908, 718, 752, 933, 137, 3]`
  - *Kết quả `decode(ids, skip_special_tokens=True)`:*  
    `It is a jigsaw puzzle still being put together .` *(Khớp gốc 100%)*
- **Câu Tiếng Việt:** `Nó là một trò chơi ghép hình vẫn đang được xếp .`
  - *Subword tokens (có `<bos>`, `<eos>`):*  
    `['<bos>', 'Nó</w>', 'là</w>', 'một</w>', 'trò</w>', 'chơi</w>', 'ghép</w>', 'hình</w>', 'vẫn</w>', 'đang</w>', 'được</w>', 'xếp</w>', '.</w>', '<eos>']`
  - *Token IDs:*  
    `[2, 767, 452, 457, 1162, 1025, 2072, 668, 886, 566, 496, 1638, 256, 3]`
  - *Kết quả `decode(ids, skip_special_tokens=True)`:*  
    `Nó là một trò chơi ghép hình vẫn đang được xếp .` *(Khớp gốc 100%)*

---

## 5. Tình trạng các tệp đầu ra & Cách chạy lại

| File đầu ra | Vị trí | Dung lượng | Trạng thái |
| :--- | :--- | ---: | :--- |
| Model BPE Tiếng Anh | `data/tokenizer/bpe_en.json` | 560,337 bytes (~547.2 KB) | `vocab_size = 8,000`, `pad=0, unk=1, bos=2, eos=3` (đồng bộ `131,942` câu train) |
| Model BPE Tiếng Việt | `data/tokenizer/bpe_vi.json` | 546,585 bytes (~533.8 KB) | `vocab_size = 8,000`, `pad=0, unk=1, bos=2, eos=3` (đồng bộ `131,942` câu train) |
| Thống kê đánh giá | `data/tokenizer/tokenizer_stats.json` | — | Lưu `subword_fertility`, `unk_rate`, `roundtrip_rate`, `coverage_max_len_100` và `coverage_max_len_128` trên `train`/`val`/`test` |

Lệnh huấn luyện lại và xuất thống kê từ thư mục gốc dự án *(hàm `compute_tokenizer_stats` tự động đánh giá đồng thời cả `max_len=100` và `max_len=128`)*:
```bash
python -m src.data.tokenizer --vocab-size 8000 --min-frequency 2
```
