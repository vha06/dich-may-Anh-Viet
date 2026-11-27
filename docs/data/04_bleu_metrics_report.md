# Báo cáo Module Đánh giá SacreBLEU — Đề tài 8: Dịch máy Anh–Việt

> **Tài liệu lưu trữ:** Bản nháp kỹ thuật phục vụ soạn thảo Proposal (Tuần 1), Báo cáo cuối (Tuần 6) và ôn tập Vấn đáp cá nhân.  
> **Mã nguồn liên quan:** [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py), [`src/data/tokenizer.py`](../../src/data/tokenizer.py), `data/processed/metrics_sanity_stats.json`

---

## 1. Chuẩn hóa Đánh giá bằng `sacrebleu` (Post, 2018)

Thay vì tự cài đặt BLEU thủ công hoặc dùng hàm tách token tự do dễ gây lệch điểm số (1–3 điểm BLEU giữa các nghiên cứu), dự án sử dụng và khóa cứng thư viện chuẩn **`sacrebleu==2.6.0`** trong [`requirements.txt`](../../requirements.txt) để chữ ký đánh giá đồng nhất tuyệt đối trên mọi máy thành viên. Mọi kết quả đánh giá đều gắn kèm **Chữ ký tái lập (`signature`)** chuẩn quốc tế để đưa trực tiếp vào báo cáo LaTeX:
- Cấu hình mặc định toàn dự án: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- Thiết lập `force=True` cho `BLEU`: Vì dữ liệu IWSLT'15 đã được tách sẵn khoảng trắng trước dấu câu (ví dụ `" ."`), cả câu dự đoán (sinh ra từ BPE) lẫn câu gốc (`val.vi`/`test.vi`) đều dùng chung quy ước tiền xử lý này; tham số `force=True` giữ nguyên điểm số và chữ ký chuẩn `tok:13a` đồng thời tắt cảnh báo lặp lại trong log huấn luyện.

---

## 2. Công thức Toán học của Corpus BLEU-4

Điểm BLEU trên toàn bộ tập dữ liệu (Corpus-level) được tính từ **độ chính xác n-gram gộp toàn tập ($p_1, p_2, p_3, p_4$)** và **hệ số phạt độ ngắn (Brevity Penalty – BP)**:

$$p_n = \frac{\sum_{i=1}^{N} \text{Count}_{\text{clip}}(\text{n-gram khớp ở câu } i)}{\sum_{i=1}^{N} \text{Count}(\text{tổng n-gram câu dự đoán } i)}, \qquad \text{BP} = \begin{cases} 1 & \text{nếu } C \ge R \\ \exp\!\left(1 - \frac{R}{C}\right) & \text{nếu } C < R \end{cases}$$

$$\text{BLEU} = \text{BP} \times \exp\!\left(\frac{1}{4}\sum_{n=1}^{4} \ln p_n\right) \times 100$$

*(Trong đó $C = \text{sys\_len}$ là tổng số từ máy dịch, $R = \text{ref\_len}$ là tổng số từ bản gốc trên toàn tập).*

---

## 3. Ba Hàm Giao tiếp Chính trong `src/evaluation/metrics.py`

| Hàm | Đầu vào (Input) | Vai trò & Ngữ cảnh sử dụng |
| :--- | :--- | :--- |
| **`compute_corpus_bleu`** *(alias: `compute_bleu`)* | Danh sách chuỗi dự đoán & bản gốc đơn hoặc đa tham chiếu (`Sequence[str] \| Sequence[Sequence[str]]`) | Tính **1 điểm Corpus BLEU-4 duy nhất** kèm `precisions` `[p1..p4]`, `brevity_penalty`, `length_ratio`, `sys_len`, `ref_len` và `signature` cho toàn bộ tập `val`/`test` (được export thống nhất qua `src/evaluation/__init__.py`). |
| **`compute_sentence_bleu`** | 1 câu dự đoán & 1 câu gốc (`str`) | Tính BLEU từng câu (bật `effective_order=True`, `smooth_method="exp"`), phục vụ **Phân tích lỗi Tuần 5** và **App Demo Tuần 6**. |
| **`decode_and_compute_bleu`** | Tensor/danh sách Token IDs từ mô hình + câu gốc (`val.vi`/`test.vi`) | Gọi `tgt_tokenizer.decode(..., skip_special_tokens=True)` (tự động dừng tại `<eos>` đầu tiên, bỏ `<bos>`/`<pad>`) rồi gọi `compute_corpus_bleu` so sánh với **văn bản gốc chuẩn trên đĩa** (*Gold Reference*). |

---

## 4. Kết quả Kiểm định Toán học & Đo Ngưỡng trần BPE (`IWSLT'15 En-Vi`)

Kết quả trích xuất trực tiếp từ `data/processed/metrics_sanity_stats.json` (chạy qua `python -m src.evaluation.metrics`):

### 4.1. Kiểm định trên tập `Validation` (`val.vi` – 1,553 câu) và `Test` (`test.vi` – 1,268 câu)

| Kịch bản kiểm định | Tập | BLEU-4 | BP | Độ dài hệ thống / Gốc | Độ chính xác `[p1, p2, p3, p4]` (%) | Ý nghĩa kiểm chứng |
| :--- | :---: | ---: | :---: | :---: | :---: | :--- |
| **1. Identity** *(Gốc vs Gốc)* | `Val` | **100.00** | 1.0000 | 34,327 / 34,327 | `[100.0, 100.0, 100.0, 100.0]` | Hàm chấm điểm đạt đúng chuẩn tuyệt đối `100.00`. |
| **1. Identity** *(Gốc vs Gốc)* | `Test` | **100.00** | 1.0000 | 33,722 / 33,722 | `[100.0, 100.0, 100.0, 100.0]` | Hàm chấm điểm đạt đúng chuẩn tuyệt đối `100.00`. |
| **2. Oracle BPE Ceiling** (`13a`) | `Val` | **99.99** | 0.9999 | 34,325 / 34,327 | `[100.0, 100.0, 99.99, 99.99]` | Ngưỡng trần lý thuyết tối đa với `BPE 8000` + `max_len=128`. |
| **2. Oracle BPE Ceiling** (`13a`) | `Test` | **99.99** | 1.0000 | 33,722 / 33,722 | `[100.0, 99.99, 99.99, 99.99]` | Chỉ 1 ký tự `<unk>` (`Ōfunato`) làm giảm đúng `0.01%` ở 2–4 gram. |
| **3. Cắt ngắn 50% độ dài** | `Val` | **35.15** | 0.3515 | 16,781 / 34,327 | `[100.0, 100.0, 100.0, 100.0]` | Chứng minh `BP` phạt nặng (`0.3515`) dù phần đầu khớp 100%. |
| **3. Cắt ngắn 50% độ dài** | `Test` | **35.33** | 0.3533 | 16,526 / 33,722 | `[100.0, 100.0, 100.0, 100.0]` | `length_ratio = 0.4901` kéo điểm BLEU xuống ~35.3. |
| **4. Xáo trộn trật tự từ** | `Val` | **1.65** | 1.0000 | 34,327 / 34,327 | `[100.0, 6.17, 0.4, 0.03]` | `p1=100%` nhưng `p2, p3, p4` sụt mạnh $\rightarrow$ BLEU nhạy trật tự từ. |
| **4. Xáo trộn trật tự từ** | `Test` | **1.81** | 1.0000 | 33,722 / 33,722 | `[100.0, 5.51, 0.34, 0.06]` | Chứng minh BLEU chống gian lận "túi từ" (*bag-of-words*). |

### 4.2. Ví dụ minh họa Sentence-level SacreBLEU (`effective_order=True`)
Trên câu gốc: *"Hôm nay bầu trời ở Hà Nội rất trong xanh và mát mẻ ."* (`14` token):

| Trường hợp thử | Câu dự đoán | BLEU | BP | `precisions` `[p1..p4]` (%) |
| :--- | :--- | ---: | :---: | :--- |
| `exact_match` | Khớp y nguyên 100% | **100.00** | 1.0000 | `[100.0, 100.0, 100.0, 100.0]` |
| `one_word_diff` | Đổi 1 từ (`"ở"` $\rightarrow$ `"tại"`) | **78.25** | 1.0000 | `[92.86, 84.62, 75.0, 63.64]` |
| `half_truncated` | Chỉ lấy 7 từ đầu (`"Hôm nay ... Hà Nội"`) | **36.79** | 0.3679 | `[100.0, 100.0, 100.0, 100.0]` |
| `word_scrambled` | Xáo trộn ngẫu nhiên trật tự từ (`seed=42`) | **10.99** | 1.0000 | `[100.0, 15.38, 4.17, 2.27]` |

---

## 5. Cách chạy lại Kiểm định

```bash
python -m src.evaluation.metrics --max-len 128 --seed 42
```
Lệnh trên kiểm định đầy đủ cả tập `val.vi`, `test.vi` cùng 4 ví dụ mức câu và xuất kết quả JSON tại `data/processed/metrics_sanity_stats.json`.
