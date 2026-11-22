# Báo cáo Hàm mất mát & Bộ điều chỉnh Tốc độ học — Đề tài 8: Dịch máy Anh–Việt

> **Tài liệu lưu trữ:** Bản nháp kỹ thuật phục vụ Chặng 1 của Hạ tầng Huấn luyện (dùng chung cho **Seq2Seq LSTM** và **Transformer**), soạn thảo báo cáo và ôn tập Vấn đáp.  
> **Mã nguồn liên quan:** [`src/training/losses.py`](../../src/training/losses.py), [`src/training/schedulers.py`](../../src/training/schedulers.py), [`tests/test_training_components.py`](../../tests/test_training_components.py)

---

## 1. Hàm Mất mát & Đo lường Perplexity Chuẩn hóa (`src/training/losses.py`)

### 1.1. Thiết kế `TranslationLoss` & Quy tắc cặp Train/Eval (`build_loss_pair`)
- **`TranslationLoss(pad_id=0, label_smoothing=...)`**:
  - Bọc `nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=..., reduction="mean")`, làm phẳng `logits (B, T_max, V)` và `targets (B, T_max)`.
  - Trả về cặp `(loss, num_tokens)` với `num_tokens` là tổng số token thực (`targets != pad_id`). Trong `trainer.py`, ta cộng dồn theo trọng số token thực (`sum(loss_i * num_tokens_i) / sum(num_tokens_i)`) để batch cuối (thường ít hơn 64 câu) không làm lệch trung bình toàn Epoch.
- **Khắc phục méo Perplexity khi bật Label Smoothing (`build_loss_pair`):**
  - Khi huấn luyện Transformer hoặc thử nghiệm Ablation với `train_label_smoothing = 0.1`, Cross-Entropy bị cộng thêm phần dư phân tán xác suất, làm giá trị `train_loss` cao hơn thực tế và khiến chỉ số $\text{PPL} = \exp(\text{loss})$ không thể so sánh trực tiếp với mô hình tắt smoothing.
  - Hàm `build_loss_pair(pad_id=0, train_label_smoothing=...)` luôn trả về `(train_loss_fn, eval_loss_fn)` trong đó **`eval_loss_fn` bắt buộc `label_smoothing = 0.0`**. Nhờ vậy `val_loss` và `val_ppl` luôn đứng trên một thước đo chuẩn duy nhất qua mọi thí nghiệm.
- **Chống tràn số `OverflowError` (`compute_perplexity`):**
  - Kẹp giá trị đầu vào của hàm mũ tại ngưỡng `DEFAULT_MAX_PPL_LOSS = 20.0` (`math.exp(min(loss_value, 20.0))`), ngăn sập chương trình nếu `loss` vọt cao bất thường ở các bước đầu.

---

## 2. Hợp đồng Thống nhất cho Các Bộ Scheduler (`src/training/schedulers.py`)

Cả hai bộ điều chỉnh tốc độ học (`PlateauScheduler` dùng cho Seq2Seq LSTM và `NoamScheduler` dùng cho Transformer) đều tuân thủ một **hợp đồng giao diện chung** để `trainer.py` gọi tự động không cần viết logic rẽ nhánh phức tạp:

| Thuộc tính / Phương thức | `PlateauScheduler` *(Seq2Seq LSTM)* | `NoamScheduler` *(Transformer)* | Ý nghĩa với `trainer.py` |
| :--- | :---: | :---: | :--- |
| **`step_mode`** | `"epoch"` | `"batch"` | Quy định rõ gọi `.step()` **sau mỗi Epoch** hay **sau mỗi Batch** |
| **`requires_metric`** | `True` | `False` | `True` truyền `val_loss` vào `.step(metric)`; `False` gọi `.step()` trực tiếp |
| **`step(metric=None)`** | Cập nhật theo `val_loss` | Tiến 1 bước Batch (bỏ qua `metric`) | Chữ ký thống nhất cho cả 2 Scheduler |
| **`get_last_lr()`** | Lấy LR hiện tại của Optimizer | Lấy LR hiện tại theo công thức Noam | Ghi log trực tiếp lên **W&B (`wandb`)** & màn hình console |
| **`state_dict` / `load_state_dict`** | Hỗ trợ đầy đủ | Hỗ trợ đầy đủ | Lưu và khôi phục tiến độ khi **Resume Checkpoint** |

### 2.1. Hai Bẫy Kỹ thuật đã được Xử lý Triệt để trong `NoamScheduler`
Công thức chuẩn từ Vaswani et al. (NeurIPS 2017):
$$\text{lr}(\text{step}) = \text{factor} \cdot d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

1. **Chống nhân đúp `base_lr` của PyTorch:** Lớp `LambdaLR` chuẩn của PyTorch mặc định lấy $\text{lr} = \text{base\_lr} \times \lambda(\text{step})$. Nếu người dùng đặt `Adam(lr=1e-3)` và viết công thức Noam vào `LambdaLR`, tốc độ học sẽ bị nhỏ đi 1,000 lần ($10^{-3} \times \text{Noam}$)! `NoamScheduler` kế thừa `LRScheduler` và trả về **trực tiếp giá trị LR tuyệt đối**, hoàn toàn miễn nhiễm với `lr` khởi tạo của `Optimizer`.
2. **Chống lỗi chia cho `0` ở bước đầu:** Dùng bước đếm 1-based `step = self.last_epoch + 1` (tức bước đầu tiên trước Batch 1 là `step = 1`, kết hợp `max(step, 1)`), vừa tránh `ZeroDivisionError` tại $0^{-0.5}$, vừa giúp chạm đỉnh đúng tại `step == warmup_steps`.

*(Minh họa thực tế từ `python -m tests.test_training_components` với `d_model=256, warmup_steps=4000, factor=1.0`: `step=1 -> 0.0000002`, `step=1000 -> 0.0002471`, đạt đỉnh `step=4000 -> 0.0009882`, giảm dần `step=8000 -> 0.0006988`, `step=20000 -> 0.0004419`).*

---

## 3. Kết quả Kiểm thử Tự động (`tests/test_training_components.py`)

Đã chạy kiểm định bằng cả `python -m tests.test_training_components` (4/4 test pass trong `0.010s`, không có cảnh báo) và `pytest tests/test_training_components.py -v` (`4 passed in 3.19s`):

| Bài kiểm thử | Mục tiêu kiểm chứng | Kết quả |
| :--- | :--- | :---: |
| `test_translation_loss_pad_invariance_and_token_count` | Chứng minh thay đổi logits tại `<pad>` lên `1e5` không đổi Loss, đếm đúng số token thực (`8`) và gradient tại `<pad>` bằng `0.0` | **PASS** |
| `test_loss_pair_enforces_unsmoothed_eval_and_ppl_clamp` | Xác nhận `build_loss_pair(train_label_smoothing=0.1)` giữ `eval_fn.label_smoothing == 0.0`, kiểm tra `compute_perplexity(1000.0)` kẹp về `exp(20.0)` không tràn số và bắt lỗi `NaN` | **PASS** |
| `test_noam_scheduler_no_zero_div_no_base_lr_double_mul` | Khởi tạo 2 Optimizer khác hẳn nhau (`lr=1e-4` vs `lr=1.0`) $\rightarrow$ `NoamScheduler` xuất quỹ đạo LR **trùng khớp tuyệt đối** tới 12 chữ số thập phân, tăng tuyến tính 1..10, đạt đỉnh chính xác tại bước 10, suy giảm từ 11..30 và khôi phục đúng qua `state_dict` | **PASS** |
| `test_plateau_scheduler_epoch_mode_and_lr_decay` | Kiểm tra `step_mode == "epoch"`, `requires_metric is True`, báo lỗi nếu quên truyền `metric`, giảm LR một nửa (`1e-3 -> 5e-4`) khi `val_loss` chững lại và khôi phục chuẩn qua `state_dict` | **PASS** |

---

## 4. Lệnh chạy lại Kiểm thử

```bash
python -m tests.test_training_components
# Hoặc:
pytest tests/test_training_components.py -v
```
