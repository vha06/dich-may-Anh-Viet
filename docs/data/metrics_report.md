# Báo cáo Chỉ số Đánh giá (SacreBLEU) — Đề tài 8: Dịch máy Anh–Việt (IWSLT'15 En–Vi)

> **Tài liệu lưu trữ:** Bản nháp kỹ thuật phục vụ soạn thảo Proposal (Tuần 1), Báo cáo cuối (Tuần 6) và ôn tập Vấn đáp cá nhân.  
> **Mã nguồn liên quan:** [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py), [`src/data/tokenizer.py`](../../src/data/tokenizer.py), [`src/data/dataset.py`](../../src/data/dataset.py)

---

## 1. Lý do lựa chọn SacreBLEU

Trong đánh giá dịch máy, điểm BLEU tự cài đặt hoặc tính trực tiếp trên chuỗi đã tách từ thủ công dễ bị sai lệch giữa các thí nghiệm do khác cách tách dấu câu, chữ hoa/thường hoặc cách làm mịn (smoothing). Dự án sử dụng thư viện chuẩn **SacreBLEU** (`sacrebleu>=2.3.0` trong `requirements.txt`) vì:

- **Chuẩn hóa cách tách token khi chấm điểm:** mặc định dùng bộ tách `13a` (chuẩn WMT/IWSLT), đảm bảo điểm số giữa baseline Seq2Seq LSTM và Transformer có thể đối sánh công bằng.
- **Kèm chữ ký tái lập (`signature`):** tự động ghi nhận cấu hình chấm điểm (phiên bản, tokenizer, case, smoothing) để kiểm chứng tính nhất quán khi báo cáo kết quả.
- **Tránh phụ thuộc tiền xử lý riêng lẻ:** nhận trực tiếp chuỗi văn bản đã giải mã (`BPETokenizer.decode(..., skip_special_tokens=True)`), không chấm trên token ID BPE.

---

## 2. Thiết kế Kỹ thuật (`src/evaluation/metrics.py`)

Hàm cốt lõi `compute_bleu(hypotheses, references, tokenize="13a", lowercase=False)` thực hiện các bước:

1. **Kiểm tra thư viện an toàn:** bọc `import sacrebleu` trong `try ... except ImportError`; nếu môi trường chưa cài `sacrebleu`, hàm chỉ báo `ImportError` kèm hướng dẫn cài đặt khi được gọi thay vì làm lỗi khi import module.
2. **Xác thực đầu vào nghiêm ngặt:**
   - Từ chối danh sách `hypotheses` rỗng (`ValueError`).
   - Hỗ trợ cả **1 bộ tham chiếu** (`Sequence[str]`, trường hợp chuẩn của `val.vi` / `test.vi`) và **nhiều bộ tham chiếu** (`Sequence[Sequence[str]]`).
   - Kiểm tra độ dài `len(hypotheses) == len(ref_stream)` cho mọi bộ tham chiếu; nếu lệch sẽ lập tức báo `ValueError` thay vì âm thầm cắt bằng `zip()`.
3. **Tính toán & Kết quả trả về:** gọi `sacrebleu.BLEU(tokenize=tokenize, lowercase=lowercase).corpus_score(hyps, ref_streams)` và trả về `dict`:

| Trường | Kiểu | Ý nghĩa |
| :--- | :---: | :--- |
| `bleu` | `float` | Điểm corpus BLEU (thang `0.00–100.00`, làm tròn 2 chữ số thập phân) |
| `precisions` | `list[float]` | Độ chính xác theo 1-gram, 2-gram, 3-gram, 4-gram (%) |
| `bp` | `float` | Brevity Penalty — hệ số phạt khi câu dịch ngắn hơn tham chiếu |
| `sys_len` / `ref_len` | `int` | Tổng số token của hệ thống dịch và của tập tham chiếu |
| `signature` | `str` | Chữ ký cấu hình SacreBLEU dùng để tái lập phép đo |

---

## 3. Lưu ý Quan trọng khi Tích hợp với Pipeline Dữ liệu

1. **Bảo toàn thứ tự 1:1 với tập `val` và `test`:**
   - Khi tạo `DataLoader` để suy luận và chấm BLEU trên `val` (`1,553` câu) hoặc `test` (`1,268` câu), **bắt buộc đặt `use_bucketing=False` và `shuffle=False`**. Nếu bật `BucketBatchSampler`, thứ tự câu sẽ bị sắp lại theo độ dài câu nguồn, làm lệch cặp khi đối sánh với `data/processed/{val,test}.vi`.
2. **Đặc điểm Moses-tokenized và `NFKC` của tập tham chiếu:**
   - Tập `data/processed/{val,test}.vi` giữ nguyên định dạng dấu câu tách bởi khoảng trắng (Moses, ví dụ `Cảm ơn các bạn .`), khớp với đầu ra của `BPETokenizer.decode(...)` vì tokenizer được huấn luyện trên dữ liệu cùng định dạng.
   - Do `tokenizer.py` dùng normalizer `NFKC` trong khi `preprocess.py` dùng `NFC`, một số ký tự đặc biệt (nếu có) có thể biến đổi khi encode/decode và ảnh hưởng nhẹ tới khớp n-gram.

---

## 4. Tình trạng Kiểm định & Hướng dẫn Chạy

- **Trạng thái hiện tại:**
  - Khối self-check trong `src/evaluation/metrics.py` đạt cả 2 assertion (khớp điểm `100.0` khi giống hoàn toàn và `71.74` với mẫu thử 2 câu, chữ ký `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`).
- **Cách chạy kiểm tra nhanh:**

```bash
python3 src/evaluation/metrics.py
```
