# Báo cáo Bộ giải mã Greedy Search & Beam Search — Đề tài 8: Dịch máy Anh–Việt

> **Tài liệu lưu trữ:** Bản nháp kỹ thuật phục vụ Chặng 2 của Hạ tầng Huấn luyện & Đánh giá (dùng cho Baseline Seq2Seq LSTM và thiết kế mở để cắm tiếp Transformer ở Tuần 3).  
> **Mã nguồn liên quan:** [`src/evaluation/decode.py`](../../src/evaluation/decode.py), [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py), [`tests/test_decode.py`](../../tests/test_decode.py)

---

## 1. Kiến trúc Giao tiếp Thống nhất (`DecodeResult` & `decode_fn` Hook)

Module [`src/evaluation/decode.py`](../../src/evaluation/decode.py) chuẩn hóa đầu ra giải mã thông qua lớp `DecodeResult`:

| Trường trong `DecodeResult` | Kiểu dữ liệu | Quy ước & Vai trò |
| :--- | :--- | :--- |
| **`token_ids`** | `list[list[int]]` | Chuỗi ID tiếng Việt sinh ra cho từng câu, **giữ nguyên `<eos>` ở cuối** (khi sinh ra `<eos>`) để độ dài `len(token_ids[i])` khớp 1–1 tuyệt đối với trục hàng $L_{\text{gen}, i}$ của `attn_weights[i]`. Việc loại bỏ `<eos>` chỉ thực hiện khi chuyển thành văn bản qua `tgt_tokenizer.decode(..., skip_special_tokens=True)`. |
| **`scores`** | `list[float]` | Điểm log-xác suất tích lũy (đã chia chuẩn hóa độ dài theo Google NMT Length Penalty khi dùng Beam Search). |
| **`attn_weights`** | `list[Tensor] \| None` | Với mỗi câu `i`, là tensor shape `(L_gen_i, S_i)` tương ứng đúng nhánh dịch chiến thắng (đã cắt bỏ phần `<pad>` nguồn lẫn `<pad>` đích); bằng `None` nếu mô hình tắt Attention. |

---

## 2. Xử lý 4 Điểm Kỹ thuật Cốt lõi trong `src/evaluation/decode.py`

1. **Che `<pad>` và `<bos>` trước `log_softmax` (`_mask_invalid_decode_logits`):**
   - Trong những Epoch đầu khi mô hình chưa hội tụ, `<pad> (0)` hoặc `<bos> (2)` vẫn có thể có logits cao ở `fc_out`. Gán giá trị `-inf` vào hai chỉ số này trước khi tính `log_softmax` đảm bảo cả Greedy Search lẫn Beam Search **không bao giờ sinh token đặc biệt bất hợp lệ**.
2. **Greedy theo Batch nhanh (`greedy_decode_seq2seq`) vs Encoder Batch + Beam từng câu (`beam_search_seq2seq`):**
   - `greedy_decode_seq2seq` giải mã song song toàn bộ Batch (`B=64`), theo dõi cờ `finished` từng câu và dừng sớm khi toàn batch gặp `<eos>` — phục vụ đánh giá nhanh sau mỗi Epoch.
   - `beam_search_seq2seq` (`beam_size = 4`, `length_penalty = 0.6` mặc định): chạy `LSTMEncoder` **1 lần duy nhất cho cả Batch**, sau đó tách phần không padding `encoder_outputs[i, :s_len_i]` để giải mã Beam từng câu một. Cách này vừa triệt tiêu hoàn toàn nguy cơ lệch chỉ số 3D `(B * k)` lẫn loại bỏ `<pad>` nguồn ngay từ đầu, đồng thời lưu chuẩn xác lịch sử trạng thái LSTM `(h, c)` và từng hàng `attn_weights` dọc theo nhánh cha `parent_idx`.
   - Áp dụng công thức chuẩn Google NMT Length Penalty (Wu et al., 2016):
     $$\text{score}(Y) = \frac{\sum_{t=1}^{|Y|} \log P(y_t \mid y_{<t}, X)}{\left(\frac{5 + |Y|}{6}\right)^{\alpha}}, \qquad (\alpha = 0.6)$$
3. **Đánh giá trên Gold Reference & Khôi phục chế độ `train/eval` (`translate_dataloader`):**
   - Luôn bọc toàn bộ quá trình dịch trong `torch.no_grad()` + `model.eval()` kèm khối `try ... finally: model.train(was_training)` để **khôi phục nguyên trạng chế độ `train()`/`eval()`** ban đầu sau khi kết thúc, tránh làm tắt Dropout ở Epoch huấn luyện kế tiếp.
   - Khi chấm điểm **SacreBLEU**, lấy **văn bản gốc nguyên vẹn (`dataset.tgt_lines`)** chưa từng bị cắt 128 token (và chặn nếu `DataLoader` đang bật `RandomSampler` mà không truyền `references`), đảm bảo chuẩn *Gold-Reference Evaluation*.
4. **Sẵn sàng cắm cho Transformer (Tuần 3):**
   - Hàm `translate_dataloader` hỗ trợ tham số tùy chọn `decode_fn(model, src_ids, src_lengths, max_len=...) -> DecodeResult`. Sang Tuần 3 chỉ cần truyền hàm giải mã của Transformer vào `decode_fn` là tái sử dụng trọn vẹn quy trình gom Batch, cắt `<eos>`, và chấm SacreBLEU.

---

## 3. Kết quả Kiểm thử Tự động (`tests/test_decode.py`)

Đã chạy kiểm định bằng cả `python -m tests.test_decode` (5/5 test pass trong `0.420s`) và `pytest tests/test_decode.py -v` (`5 passed in 2.11s`):

| Bài kiểm thử | Mục tiêu kiểm chứng | Kết quả |
| :--- | :--- | :---: |
| `test_invalid_pad_bos_masked_and_train_mode_restored` | Ép bias của `<pad>` và `<bos>` lên `100.0` vẫn không bao giờ bị chọn; kiểm tra `model.training` trước & sau Greedy/Beam đều giữ nguyên `True` | **PASS** |
| `test_beam_size_1_exact_equivalence_with_greedy` | Kiểm chứng `beam_search_seq2seq(beam_size=1, length_penalty=0.0)` khớp **100%** `token_ids`, `scores` và ma trận `attn_weights` với `greedy_decode_seq2seq`, đúng kích thước `(L_gen_i, S_i)` | **PASS** |
| `test_beam_search_beats_greedy_on_trap_graph_and_tracks_attention` | Trên đồ thị bẫy xác suất chuẩn hóa: Greedy chọn nhầm nhánh `[4, 6, 3]` còn `beam_size=4` tìm được nhánh tối ưu `[5, 7, 3]` có điểm log-prob cao hơn hẳn và giữ đúng ma trận Attention từng bước của nhánh thắng | **PASS** |
| `test_length_penalty_and_no_attention_mode` | Kiểm chứng hiệu ứng `length_penalty` (`alpha=0.0` chọn câu ngắn `[4, 3]`, `alpha=2.0` ưu tiên câu dài `[5, 7, 8, 3]`), tương thích tắt Attention (`attn_weights=None`) và cắt fallback tại `max_len` khi chưa gặp `<eos>` | **PASS** |
| `test_translate_dataloader_gold_refs_and_custom_decode_fn` | Dịch Batch thật của `val` DataLoader qua `"greedy"`, `"beam"` (`beam_size=4, alpha=0.6`) và `decode_fn` giả lập Transformer: khớp đúng 100% văn bản gốc `ds.tgt_lines[:8]` (giữ nguyên dấu `?` câu 0) và khôi phục `model.train()` | **PASS** |

---

## 4. Lệnh chạy lại Kiểm thử

```bash
python -m tests.test_decode
# Hoặc:
pytest tests/test_decode.py -v
```
