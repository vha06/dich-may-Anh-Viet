# Báo cáo Kỹ thuật Mảng 1: Quy trình Đánh giá SacreBLEU Chuẩn hóa

## 1. Cấu hình SacreBLEU Chuẩn chung của Nhóm
Toàn bộ thực nghiệm (Seq2Seq LSTM Baseline, Transformer và Ablation Study) sử dụng chung một cấu hình `sacrebleu.metrics.BLEU` trong [src/evaluation/metrics.py](../../src/evaluation/metrics.py):

```python
BLEU(lowercase=False, tokenize="13a", smooth_method="exp", force=True)
```

* **Chữ ký chuẩn (Signature):** `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
* **Lý do lựa chọn:**
  * `lowercase=False` (`case:mixed`): Chấm phân biệt chữ hoa/thường để đánh giá đúng khả năng giữ tên riêng (`Vietnam`, `Cambridge`, `El Niño`) và viết hoa đầu câu.
  * `tokenize="13a"`: Chuẩn WMT/ACL quốc tế, đảm bảo tính tái lập (Reproducibility).
  * `smooth_method="exp"` + `effective_order=True` (riêng cho cấp câu): Tránh điểm BLEU bị rơi về 0 khi câu đơn ngắn chưa khớp đủ 4-gram.
  * `force=True`: Tắt cảnh báo nhiễu của `sacrebleu` trên dữ liệu IWSLT'15 đã tách sẵn dấu câu, giữ nguyên 100% điểm số và chữ ký.

---

## 2. Ba Nguyên tắc Vàng khi Đánh giá
1. **Đối chiếu Gold Reference gốc trên đĩa:** Luôn so sánh bản dịch của mô hình với câu tham chiếu nguyên văn (`data/processed/{val,test}.vi` hoặc `dataset.tgt_lines`), tuyệt đối **không** giải mã ngược từ tensor `tgt_output_ids` (vốn có thể bị cắt ở `max_len=128` hoặc biến từ hiếm thành `<unk>`).
2. **Suy luận độc lập không nhìn `tgt`:** Khi chấm Validation/Test BLEU, mô hình chỉ nhận `src_ids` và `src_lengths` qua hàm giải mã tự do (`greedy_decode` hoặc `beam_search_decode`), tự dừng khi gặp `<eos>` hoặc chạm `max_len=128`.
3. **Chuẩn hóa khoảng trắng đầu/cuối & liên tiếp:** Mọi câu trước khi đưa vào `sacrebleu` đều đi qua `" ".join(text.strip().split())`.

---

## 3. Giao diện Hàm & Dữ liệu Trả về trong `metrics.py`
* **`compute_corpus_bleu(hypotheses, references, ...)`** *(kèm alias `compute_bleu`)*: Chấm điểm cấp toàn bộ tập dữ liệu, hỗ trợ cả 1 bộ tham chiếu (`Sequence[str]`) lẫn nhiều bộ tham chiếu (`Sequence[Sequence[str]]`).
* **`compute_sentence_bleu(hypothesis, reference, ...)`**: Chấm điểm cấp câu đơn lẻ với `effective_order=True`.
* **`decode_and_compute_bleu(pred_ids, references, tokenizer, ...)`**: Giải mã trực tiếp Tensor 2D / danh sách ID và tính SacreBLEU.
* **Từ điển 8 khóa trả về bắt buộc:** `bleu`, `bleu_exact`, `precisions` (`[P1..P4]`), `brevity_penalty`, `length_ratio`, `sys_len`, `ref_len`, `signature`.

---

## 4. Hai Lỗi đã Khắc phục trong `Trainer.evaluate()` ([src/training/trainer.py](../../src/training/trainer.py))
* **Lỗi 1 — Suy luận phụ thuộc `tgt`:** Trước đây `evaluate()` gọi `model(src, tgt_in, teacher_forcing_ratio=0.0)` làm mô hình dùng ké chiều dài `T_max` của đáp án. Đã tách riêng: tính `val_loss` bằng Teacher Forcing và tính `val_bleu` bằng `greedy_decode(self.model, src, self.tokenizer, max_len=self.max_decode_len, src_lengths=src_lengths)`.
* **Lỗi 2 — Lấy tham chiếu từ `tgt_out`:** Đã chuyển sang đọc trực tiếp `dataloader.dataset.tgt_lines` (văn bản gốc nguyên vẹn 1,553 câu `val` và 1,268 câu `test`).

---

## 5. Kết quả Kiểm định Thực tế (`python -m src.evaluation.metrics`)
Dữ liệu chi tiết lưu tại `data/processed/metrics_sanity_stats.json`:

### Tập Validation (`val.vi` — 1,553 câu, `ref_len = 34,327`)
| Kịch bản kiểm định | BLEU | BP | Precisions `[P1, P2, P3, P4]` |
| :--- | ---: | ---: | :--- |
| `perfect_match` (khớp hoàn toàn) | **100.00** | 1.0000 | `[100.0, 100.0, 100.0, 100.0]` |
| `empty_predictions` (chuỗi rỗng) | **0.00** | 0.0000 | `[0.0, 0.0, 0.0, 0.0]` |
| `truncated_half` (cắt nửa câu) | **35.15** | 0.3515 | `[100.0, 100.0, 100.0, 100.0]` |
| `word_order_shuffled` (đảo trật tự từ) | **1.65** | 1.0000 | `[100.0, 6.17, 0.4, 0.03]` |
| `lowercased` (mất viết hoa) | **90.94** | 1.0000 | `[93.79, 91.79, 90.02, 88.23]` |

### Tập Test (`test.vi` — 1,268 câu, `ref_len = 33,722`)
| Kịch bản kiểm định | BLEU | BP | Precisions `[P1, P2, P3, P4]` |
| :--- | ---: | ---: | :--- |
| `perfect_match` (khớp hoàn toàn) | **100.00** | 1.0000 | `[100.0, 100.0, 100.0, 100.0]` |
| `empty_predictions` (chuỗi rỗng) | **0.00** | 0.0000 | `[0.0, 0.0, 0.0, 0.0]` |
| `truncated_half` (cắt nửa câu) | **35.33** | 0.3533 | `[100.0, 100.0, 100.0, 100.0]` |
| `word_order_shuffled` (đảo trật tự từ) | **1.58** | 1.0000 | `[100.0, 5.69, 0.37, 0.03]` |
| `lowercased` (mất viết hoa) | **89.74** | 1.0000 | `[93.31, 90.74, 88.57, 86.49]` |
