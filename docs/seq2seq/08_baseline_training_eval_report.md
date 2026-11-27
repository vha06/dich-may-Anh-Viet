# Báo cáo Thực nghiệm Huấn luyện & Đánh giá Baseline Seq2Seq LSTM + Luong Attention — Đề tài 8: Dịch máy Anh–Việt

> **Tài liệu lưu trữ:** Tổng kết đầy đủ kết quả thực nghiệm **Tuần 2 (Baseline Model)** sau 20 Epoch huấn luyện và đánh giá chính thức trên hai tập kiểm chứng (`tst2012`) & kiểm thử (`tst2013`).  
> **Cấu hình & Mã nguồn:** [`configs/baseline_lstm.yaml`](../../configs/baseline_lstm.yaml), [`scripts/train.py`](../../scripts/train.py), [`scripts/evaluate.py`](../../scripts/evaluate.py), [`src/models/seq2seq_lstm/`](../../src/models/seq2seq_lstm/), [`src/evaluation/decode.py`](../../src/evaluation/decode.py)  
> **Artifacts cục bộ (gitignore):** `checkpoints/baseline_lstm/best_model.pt` (Epoch 15), `outputs/baseline_lstm/history.json`, `outputs/baseline_lstm/eval_{val,test}_{greedy,beam}.json`

---

## 1. Tổng quan & Cấu hình Thực nghiệm

| Thành phần | Thiết lập cụ thể |
| :--- | :--- |
| **Tên thí nghiệm (`experiment.name`)** | `ha_baseline_lstm` (Seed `42`) |
| **Kiến trúc mô hình (`Seq2SeqLSTM`)** | Encoder: 2-layer BiLSTM (`embed=256, hidden=256 -> 512`)<br/>Decoder: 2-layer UniLSTM (`embed=256, hidden=512`)<br/>Attention: Luong Global Attention (`method=general`, `attn_dim=256`)<br/>Dropout: `0.3` |
| **Tổng số tham số** | **15,820,096** tham số huấn luyện |
| **Dữ liệu (`IWSLT'15 En–Vi`)** | `train`: `132,160` cặp câu \| `val` (`tst2012`): `1,553` câu \| `test` (`tst2013`): `1,268` câu |
| **Tokenization** | BPE riêng biệt Anh–Việt (`vocab_size = 8,000`, `max_len = 128`, `batch_size = 64`) |
| **Chiến lược tối ưu** | Adam (`lr = 0.001`, `weight_decay = 0.0`), `grad_clip = 1.0`, `label_smoothing = 0.0`, `teacher_forcing_ratio = 1.0` |
| **Lịch LR & Early Stopping** | `ReduceLROnPlateau(mode=min, factor=0.5, patience=2, min_lr=1e-6)`; `early_stopping_patience = 5`; chọn `best_model.pt` theo **`val_bleu`** (Greedy mỗi cuối epoch) |
| **Phần cứng & AMP** | NVIDIA **RTX 4060 Laptop GPU** (8 GB VRAM), PyTorch `2.5.1+cu121`, AMP **`torch.bfloat16`** (`amp_dtype: auto`) |
| **Giám sát thực nghiệm** | Weights & Biases (`nikoteen-iai-uet/dich-may-Anh-Viet`, `group: phase2-baseline-lstm`) |

---

## 2. Diễn biến Quá trình Huấn luyện (20 Epochs)

Toàn bộ quá trình huấn luyện 20 Epoch (`41,300` bước cập nhật gradient) mất tổng cộng **`16,882.33 giây` (~4 giờ 41 phút)**, trong đó `16,705.83s` (`278.43 phút`) cho huấn luyện và `176.50s` (`2.94 phút`, TB `~8.8s/epoch`) cho bước kiểm chứng Greedy BLEU trên tập `val`.

### Bảng tiến trình qua các Epoch tiêu biểu (trích từ `outputs/baseline_lstm/history.json`)

| Epoch | `train_loss` | `train_ppl` | `val_loss` | `val_ppl` | `val_bleu` (Greedy) | `val_bp` | `grad_norm` | `lr` | `train_time` (s) | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 5.2381 | 188.31 | 3.8484 | 46.92 | 7.70 | 1.0000 | 0.6296 | 1.00e-3 | 1304.12 | *BEST* #1 (`step=2,065`) |
| **2** | 3.6168 | 37.22 | 3.1752 | 23.93 | 15.15 | 1.0000 | 0.5708 | 1.00e-3 | 1134.00 | *BEST* (`+7.45 BLEU`) |
| **3** | 3.1286 | 22.84 | 2.8995 | 18.17 | 18.28 | 0.9984 | 0.5611 | 1.00e-3 | 1237.58 | *BEST* |
| **5** | 2.7068 | 14.98 | 2.6706 | 14.45 | 20.82 | 0.9924 | 0.5362 | 1.00e-3 | 719.21 | *BEST* (Vượt mốc `20.0`) |
| **8** | 2.4295 | 11.35 | 2.5249 | 12.49 | 22.45 | 0.9927 | 0.5063 | 1.00e-3 | 801.37 | *BEST* |
| **9** | 2.3704 | 10.70 | 2.5009 | 12.19 | 22.80 | 0.9947 | 0.4979 | 1.00e-3 | 653.67 | *BEST* |
| **12** | 2.2404 | 9.40 | 2.4537 | 11.63 | 22.98 | 0.9638 | 0.4791 | 1.00e-3 | 648.97 | *BEST* (Sau 2 epoch đi ngang) |
| **14** | 2.1763 | 8.81 | 2.4329 | 11.39 | 23.60 | 0.9972 | 0.4706 | 1.00e-3 | 713.05 | *BEST* |
| **15** | **2.1489** | **8.58** | **2.4322** | **11.38** | **23.75** | **0.9810** | **0.4673** | 1.00e-3 | 904.09 | ***BEST* tổng thể (`best_model.pt`)** |
| **16** | 2.1237 | 8.36 | 2.4208 | 11.26 | 23.60 | 0.9784 | 0.4638 | 1.00e-3 | 774.29 | `early_stopping = 1` |
| **17** | 2.1014 | 8.18 | 2.4125 | 11.16 | 23.30 | 0.9734 | 0.4613 | 1.00e-3 | 672.38 | `early_stopping = 2` |
| **18** | 2.0795 | 8.00 | 2.4156 | 11.20 | 23.55 | 0.9703 | 0.4581 | 1.00e-3 | 763.51 | `early_stopping = 3` |
| **19** | 2.0602 | 7.85 | 2.4153 | 11.19 | 23.54 | 0.9682 | 0.4565 | 1.00e-3 | 800.05 | `early_stopping = 4` |
| **20** | **2.0410** | **7.70** | **2.4095** | **11.13** | 23.54 | 0.9740 | 0.4532 | 1.00e-3 | 1015.90 | Chạm `early_stopping = 5` (Kết thúc) |

### Phân tích các hiện tượng quan trọng trong quá trình học

1. **Ba pha hội tụ rõ rệt:**
   - **Pha 1 (Epoch 1–5 — Bứt phá nhanh):** `val_bleu` nhảy vọt từ `7.70` lên `20.82` (`+13.12` điểm), `val_ppl` rơi mạnh từ `46.92` xuống `14.45`. Mô hình nhanh chóng nắm bắt cấu trúc câu và ánh xạ từ vựng cơ bản Anh–Việt.
   - **Pha 2 (Epoch 6–15 — Tinh chỉnh ổn định):** `val_bleu` tăng đều từ `21.37` lên đỉnh **`23.75`** ở **Epoch 15** (`precisions = [56.47, 31.27, 18.13, 10.74]`).
   - **Pha 3 (Epoch 16–20 — Bão hòa & Overfit nhẹ):** `train_loss` tiếp tục giảm (`2.1489 → 2.0410`), nhưng `val_loss` hầu như đi ngang (`2.4322 → 2.4095`, chỉ giảm `0.0227`), làm khoảng cách `val_loss - train_loss` nới rộng từ `0.2833` (Epoch 15) lên `0.3685` (Epoch 20). Bộ đếm `early_stopping_counter` tăng liên tục từ `1` lên đúng ngưỡng `5` ở Epoch 20, chứng tỏ lựa chọn `20` epoch và lưu checkpoint tại **Epoch 15** là tối ưu.
2. **Động lực học Gradient cực kỳ lành mạnh:**
   - `grad_norm` giảm đều và mượt từ `0.6296` (Epoch 1) xuống `0.4532` (Epoch 20), luôn nằm dưới ngưỡng cắt `grad_clip = 1.0`. Không xuất hiện bùng nổ/triệt tiêu gradient hay tràn số khi bật `bfloat16` AMP.
3. **Vì sao Epoch 16–20 có `val_loss` giảm nhẹ nhưng `val_bleu` không vượt qua Epoch 15 (`23.75`)?**
   - Thứ nhất là hiệu ứng **Brevity Penalty (`val_bp`)**: so sánh Epoch 15 (`precisions = [56.47, 31.27, 18.13, 10.74]`, `BP = 0.9810`) với Epoch 19 (`precisions = [56.79, 31.42, 18.15, 10.78]`, `BP = 0.9682`), thực chất độ chính xác n-gram tại Epoch 19 đều cao hơn nhẹ nhưng bản dịch tự sinh lại ngắn hơn một chút khiến `BP` giảm từ `0.9810` xuống `0.9682` (và `0.9740` ở Epoch 20), kéo tổng điểm BLEU xuống `23.54`.
   - Thứ hai là sự lệch pha giữa mục tiêu tối ưu và cơ chế sinh câu: `val_loss` đo Cross-Entropy theo từng token dưới chế độ **Teacher Forcing** (biết trước từ chuẩn phía trước), trong khi `val_bleu` đo chất lượng toàn chuỗi khi Decoder **tự sinh liên hoàn** (**Exposure Bias** — càng về cuối huấn luyện, mô hình càng quen bám vào phân phối từ chuẩn nên dễ phát `<eos>` sớm hoặc lệch nhẹ khi tự suy luận).
4. **Hành vi của `ReduceLROnPlateau`:**
   - Trong toàn bộ 20 Epoch, `lr` giữ nguyên `0.001` chưa kích hoạt giảm lần nào. Nguyên nhân là `val_loss` cứ mỗi `1–2` epoch lại giảm nhích từ `0.003` đến `0.011` (`2.4322 → 2.4208 → 2.4125 → 2.4156 → 2.4153 → 2.4095`), vượt ngưỡng `threshold = 1e-4` mặc định của PyTorch nên bộ đếm `patience = 2` liên tục bị reset. Đây là phát hiện hữu ích để thử giảm LR chủ động hơn ở Tuần 4 (Ablation).

---

## 3. Kết quả Đánh giá Chính thức trên Tập Validation (`tst2012`) & Test (`tst2013`)

Checkpoint tốt nhất **`checkpoints/baseline_lstm/best_model.pt` (Epoch 15)** đã được đánh giá đầy đủ bằng [`scripts/evaluate.py`](../../scripts/evaluate.py) trên GPU RTX 4060 theo hai chiến lược giải mã: **Greedy Search** (`beam_size = 1`) và **Beam Search** (`beam_size = 4`, Google NMT `length_penalty = 0.6`).

> **Chuẩn SacreBLEU Signature cho mọi phép đo:**  
> `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`

| Tập đánh giá | Số câu | Chiến lược giải mã | SacreBLEU | N-gram Precisions (`P1 / P2 / P3 / P4`) | `BP` | `sys_len / ref_len` | Thời gian (Tốc độ) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Val (`tst2012`)** | 1,553 | **Greedy** (`k=1`) | **23.75** | `56.47 / 31.27 / 18.13 / 10.74` | `0.9810` | `33,682 / 34,327` (`0.9812`) | `6.08s` (`255.43 câu/s`) |
| **Val (`tst2012`)** | 1,553 | **Beam** (`k=4, α=0.6`) | **24.58** (`+0.83`) | `57.46 / 32.38 / 19.01 / 11.45` | `0.9743` | `33,457 / 34,327` (`0.9747`) | `61.15s` (`25.40 câu/s`) |
| **Test (`tst2013`)** | 1,268 | **Greedy** (`k=1`) | **25.78** | `60.13 / 33.91 / 20.32 / 12.42` | `0.9624` | `32,478 / 33,722` (`0.9631`) | `6.83s` (`185.65 câu/s`) |
| **Test (`tst2013`)** | 1,268 | **Beam** (`k=4, α=0.6`) | **26.90** (`+1.12`) | `60.95 / 35.12 / 21.39 / 13.35` | `0.9620` | `32,464 / 33,722` (`0.9627`) | `62.05s` (`20.44 câu/s`) |

### Nhận xét chi tiết về Kết quả Đánh giá

1. **Xác minh tính tái lập tuyệt đối:** Chạy độc lập [`scripts/evaluate.py`](../../scripts/evaluate.py) trên tập `val` với Greedy Search cho ra **`23.75` (`23.7536`)** — trùng khớp 100% đến từng chữ số của `precisions` và `BP` với log tại Epoch 15 trong `history.json`.
2. **Chất lượng Baseline rất vững trên chuẩn IWSLT'15 En–Vi:**
   - Trên tập chuẩn **`test` (`tst2013`)**, mô hình Baseline 15.82M tham số đạt **`25.78` SacreBLEU (Greedy)** và **`26.90` SacreBLEU (Beam Search k=4, α=0.6)**. So với mốc tham chiếu trong bài báo gốc của Luong & Manning (2015) trên bộ dữ liệu IWSLT'15 Anh–Việt (`~23.3 BLEU`), kết quả này khẳng định toàn bộ quy trình tiền xử lý BPE 8,000, kiến trúc BiLSTM-2L + Luong General Attention và lịch huấn luyện đã hoạt động rất hiệu quả.
3. **Đánh đổi giữa Beam Search và Greedy Search:**
   - **Lợi ích chất lượng (`+0.83 BLEU` trên `val`, `+1.12 BLEU` trên `test`):** Beam Search cải thiện đồng đều cả 4 bậc n-gram (`P1: +0.82`, `P2: +1.21`, `P3: +1.07`, `P4: +0.93` trên `test`), chứng tỏ tìm kiếm theo chùm tạo ra cụm từ và ngữ pháp trôi chảy hơn rõ rệt chứ không chỉ trúng từ đơn lẻ.
   - **Chi phí tốc độ:** Greedy Search giải mã song song cả batch nên đạt `185–255 câu/giây` (~`6–7 giây` toàn tập). Beam Search hiện xử lý tuần tự từng câu để quản lý kích thước nhánh động nên đạt `20.4–25.4 câu/giây` (~`61–62 giây` cho hơn 1,200–1,500 câu) — chậm hơn ~`9–10x` nhưng vẫn rất nhanh khi đánh giá ngoại tuyến.
4. **Ghi nhận kỹ thuật trong quá trình đánh giá GPU (cuDNN Contiguous Fix):**
   - Khi chạy Beam Search lần đầu trên CUDA, `nn.LSTM` (backend cuDNN) báo lỗi `RuntimeError: rnn: hx is not contiguous` do phép cắt lát trạng thái ẩn theo batch `h_enc[:, i : i + 1, :]` tạo tensor view không liên tục trên bộ nhớ GPU (mà CPU ATen kernel trước đó bỏ qua).
   - Đã khắc phục dứt điểm bằng `.contiguous()` trong [`beam_search_seq2seq`](../../src/evaluation/decode.py), bổ sung bảo vệ tương tự ở [`LSTMDecoder.forward`](../../src/models/seq2seq_lstm/decoder.py), và thêm bài test hồi quy CUDA `test_beam_search_contiguous_hidden_states_and_cuda_compat` trong [`tests/test_decode.py`](../../tests/test_decode.py) (**14/14 tests PASS**).

---

## 4. Phân tích Hiệu năng theo Độ dài Câu trên Tập Test (`tst2013`)

Để làm cơ sở đối chiếu với kiến trúc **Transformer** (Tuần 3) và phục vụ **Phân tích lỗi** (Tuần 5), 1,268 câu nguồn của `tst2013` được chia thành 4 nhóm theo số từ (phân tách khoảng trắng của `src`):

| Nhóm độ dài câu nguồn (`src`) | Số câu (Tỉ lệ) | Greedy SacreBLEU (`BP`) | Beam SacreBLEU (`BP`) | Chênh lệch (Beam − Greedy) |
| :--- | :---: | :---: | :---: | :---: |
| **Ngắn (`1–10` từ)** | 292 (`23.0%`) | **26.76** (`0.9605`) | 26.37 (`0.9330`) | **−0.39** |
| **Trung bình (`11–20` từ)** | 445 (`35.1%`) | 26.23 (`0.9514`) | **27.16** (`0.9482`) | **+0.93** |
| **Dài (`21–35` từ)** | 363 (`28.6%`) | 25.42 (`0.9662`) | **26.86** (`0.9718`) | **+1.44** |
| **Rất dài (`>35` từ)** | 168 (`13.2%`) | 25.63 (`0.9678`) | **26.89** (`0.9697`) | **+1.26** |

**Nhận xét theo độ dài:**
- **Hiệu quả của Luong Global Attention trên câu dài:** Nhờ cơ chế Global Attention cho phép Decoder truy vấn trực tiếp toàn bộ `encoder_outputs` ở mọi bước giải mã, điểm BLEU của Baseline **không bị sụp đổ** khi câu dài ra (vẫn giữ **`26.86`** ở nhóm `21–35` từ và **`26.89`** ở nhóm `>35` từ với Beam Search, chỉ kém nhẹ nhóm `11–20` từ khoảng `0.3` điểm).
- **Lợi thế của Beam Search tỷ lệ thuận với độ dài câu:** Ở câu rất ngắn (`1–10` từ), không gian tìm kiếm ít phân nhánh nên Greedy và Beam tương đương nhau, thậm chí Beam hơi có xu hướng chọn phương án cô đọng hơn (`BP` giảm từ `0.9605` xuống `0.9330`, làm BLEU thấp hơn `-0.39`). Ngược lại, từ mốc `11` từ trở lên — đặc biệt ở nhóm `21–35` từ (`+1.44`) và `>35` từ (`+1.26`) — Beam Search kết hợp `length_penalty = 0.6` vừa cải thiện `BP`, vừa giữ đúng cấu trúc mệnh đề dài tốt hơn rõ rệt.

---

## 5. Phân tích Định tính các Mẫu Dịch Tiêu biểu trên Tập Test (`tst2013`)

Tất cả các câu trích dẫn dưới đây lấy nguyên văn từ `outputs/baseline_lstm/eval_test_greedy.json` và `eval_test_beam.json`.

### 5.1. Những mẫu dịch tốt tiêu biểu

| Index | Nguồn (`src`) & Tham chiếu (`ref`) | Bản dịch của Baseline (Greedy / Beam) | Nhận xét |
| :---: | :--- | :--- | :--- |
| **#1036** | **Src:** *There was a group of experts who were invited to discuss the future of the city of New York in 1860 .*<br/>**Ref:** *Có một nhóm chuyên gia được mời đến để thảo luận về tương lai của thành phố New York năm 1860 .* | **Greedy & Beam (`87.47` BLEU):**<br/>*Có một nhóm chuyên gia được mời đến để thảo luận tương lai của thành phố New York năm 1860 .* | Câu 22 từ được dịch rất mượt, bảo toàn chính xác tên riêng *"New York"* và năm *"1860"*, chỉ lược nhẹ giới từ *"về"*. |
| **#681** | **Src:** *There are a lot of ways the people around us can help improve our lives .*<br/>**Ref:** *Có rất nhiều cách mà những người ở xung quanh chúng ta có thể giúp cải thiện cuộc sống của chúng ta* | **Greedy (`78.39`):** *Có rất nhiều cách mà **người ta** xung quanh chúng ta có thể giúp cải thiện cuộc sống của chúng ta .*<br/>**Beam (`87.25`):** *Có rất nhiều cách mà **những người** xung quanh chúng ta có thể giúp cải thiện cuộc sống của chúng ta .* | Beam chọn cụm *"những người"* tự nhiên và chính xác ngữ cảnh hơn hẳn *"người ta"* của Greedy. |
| **#406** | **Src:** *I 'm a child of 1984 , and I live in the city of Berlin .*<br/>**Ref:** *Tôi là một đứa trẻ của năm 1984 , và tôi sống ở thành phố Berlin .* | **Greedy & Beam (`83.20` BLEU):**<br/>*Tôi là một đứa trẻ năm 1984 , và tôi sống ở thành phố Berlin .* | Dịch đúng ngữ điệu tự nhiên và tên riêng địa danh *"Berlin"*. |

### 5.2. Trường hợp Beam Search cải thiện rõ rệt so với Greedy Search

- **Câu #762** (`Greedy = 21.86` → **`Beam = 52.83`**, **`+30.97`**):
  - **Src:** *We have the opportunity to start to search for the early biomarkers of the disease before it 's too late .*
  - **Ref:** *Chúng tôi có cơ hội để bắt đầu tìm kiếm những dấu ấn sinh học ban đầu của căn bệnh trước khi nó quá muộn .*
  - **Greedy:** *Chúng ta có cơ hội để tìm kiếm những người sinh sản xuất đầu của bệnh trước khi quá muộn .* *(đánh mất động từ "bắt đầu", vỡ nghĩa ở cụm "biomarkers")*
  - **Beam:** *Chúng ta có cơ hội **bắt đầu** tìm kiếm những nhà sinh học đầu tiên **của căn bệnh trước khi nó quá muộn** .* *(giữ trọn cấu trúc câu phức và vế điều kiện cuối, chỉ sai thuật ngữ chuyên ngành hiếm "biomarkers")*
- **Câu #607** (`Greedy = 23.76` → **`Beam = 47.79`**, **`+24.03`**):
  - **Src:** *I 'd also note that I 'm quite privileged to be able to transform what you think of me in a very brief 10 seconds .*
  - **Ref:** *Tôi nhận ra tôi rất là may mắn khi có thể thay đổi những gì bạn nghĩ về tôi trong 10 giây ngắn ngủi .*
  - **Greedy:** *Tôi cũng ghi nhớ rằng tôi rất vinh dự có thể biến đổi những gì bạn nghĩ trong một giây phút ngắn ngủi .* *(rơi mất tân ngữ "về tôi" — of me)*
  - **Beam:** *Tôi cũng ghi nhớ rằng tôi rất vinh dự có thể **thay đổi** những gì bạn nghĩ **về tôi** trong một giây phút ngắn ngủi .* *(chọn động từ tự nhiên và đầy đủ thành phần câu hơn; tuy cả hai vẫn làm rơi chi tiết số "10 giây")*
- **Câu #818** (`Greedy = 4.63` → **`Beam = 27.98`**, **`+23.35`**):
  - **Src:** *L.A. leads the United States in vacant lots that the city actually owns .*
  - **Ref:** *L.A. dẫn đầu nước Mỹ về diện tích đất hoang mà thành phố sở hữu .*
  - **Greedy:** *LA dẫn dắt nước Mỹ trong những nơi không có sự thật .* *(sụp đổ ở vế quan hệ cuối)*
  - **Beam:** *L.A. dẫn dắt Hoa Kỳ trong **những nơi trống rỗng mà thành phố thực sự sở hữu** .* *(giữ nguyên viết tắt "L.A." và dịch đủ nghĩa mệnh đề quan hệ)*

### 5.3. Các dạng lỗi điển hình của Baseline Seq2Seq LSTM (Cơ sở cho Tuần 3 & Tuần 5)

1. **Biến dạng tên riêng hiếm & sai lệch con số do tách mảnh BPE (8,000 subwords):**
   - **Câu #941:** `Richard Turere` bị Decoder ghép lại sai chính tả thành `Richard Tursit` (Greedy) / `Richard Turt` (Beam); thành ngữ `you are something else` bị dịch máy móc thành `bạn là một thứ khác`.
   - **Câu #437:** Tên nhà mạng `Deutsche Telekom` bị biến dạng thành `Deutene Tuskelom` (Greedy) / `Deutz Telekom` (Beam).
   - **Câu #1129:** Số liệu `135,000 photographs` bị sai thành `15.000 bức ảnh`, kèm cụm động từ bị hiểu nhầm nghĩa đen (`passed through All Hands` → `đã qua đời qua tất cả những gì`). Vì không có cơ chế Copy/Pointer Network và từ điển chỉ `8,000`, tên riêng ít gặp dễ bị tách thành nhiều mảnh subword rồi tái tạo sai chữ cái.
2. **Vòng lặp từ ở câu siêu dài (Repetition Loop) & Bỏ sót vế câu (Omission):**
   - **Câu #219 (`86` từ nguồn):** Greedy rơi vào vòng lặp vô hạn ở cuối câu: `... và đồng nghiệp và đồng nghiệp của họ và đồng nghiệp và đồng nghiệp của họ ...` cho tới khi đụng `max_len`. Beam Search (`33.81` BLEU so với `23.72` của Greedy) thoát được vòng lặp này và đóng câu chuẩn với `những người có thể nhìn thấy vết bầm . "`, nhưng cả hai vẫn làm biến dạng tên riêng `Conor` (`Condor` / `Conhay`) và `Romeo` (`Rooo`).
   - **Câu #1203:** *I had only one way out : the stairs from where I 'd come in .* → **Beam (`3.03`):** *Tôi chỉ có một cách ra ngoài .* *(chọn `<eos>` quá sớm, bỏ mất toàn bộ vế sau dấu hai chấm)*; **Greedy (`3.98`):** nhầm `stairs` thành *cái thang máy*.
3. **Dịch sai ngữ cảnh văn nói / thành ngữ TED Talks:**
   - **Câu #942:** *We 're going to cheer you on every step of the way* bị cả Greedy lẫn Beam dịch nhầm `cheer` thành *gian lận bạn* thay vì *cổ vũ cháu/bạn*.
   - **Câu #1239:** Tên tổ chức `Free the Slaves` bị mất nghĩa thành `người khác` / `Fe`, còn danh từ hiếm `moped` (xe gắn máy) bị suy đoán sai thành *một người phụ nữ trên một cái nhìn thoáng qua*.

---

## 6. Đối chiếu Thực nghiệm: Baseline `v1` vs Baseline `v2` (Dữ liệu thống nhất + `BucketBatchSampler`)

Sau khi toàn nhóm chốt chuẩn dữ liệu Mảng 1 (`131,942` cặp `train` lọc `max_words = 128`, bộ từ điển `bpe_{en,vi}.json` tái huấn luyện đồng bộ), nhóm chạy lại thực nghiệm **`v2` (`ha_baseline_lstm_v2_unified` trên W&B)** giữ nguyên hoàn toàn kiến trúc **15.82M tham số** và siêu tham số tối ưu trong [`configs/baseline_lstm.yaml`](../../configs/baseline_lstm.yaml), chỉ bật thêm **`data.use_bucketing: true`** (`BucketBatchSampler`) để tăng tốc huấn luyện.

### 6.1. Bảng so sánh Thời gian & Điểm SacreBLEU Tổng thể

> **Chuẩn SacreBLEU Signature:** `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`

| Hạng mục | `v1` *(Cũ: `shuffle=True`, `132,160` câu)* | `v2` *(Thống nhất + Bucketing, `131,942` câu)* | Chênh lệch (`v2 - v1`) |
| :--- | :---: | :---: | :---: |
| **Tổng thời gian train (20 epochs)** | `16,882.33s` (**4.69 giờ**) | `8,518.73s` (**2.37 giờ**, Ep 12–20 `~4.7 phút/ep`) | **Nhanh hơn 1.98x** *(tiết kiệm `~2.32 giờ`)* |
| **Epoch đạt Best (`val_bleu`)** | Epoch 15 (`val_loss = 2.4322`, `es = 5` ở Ep 20) | Epoch 18 (`val_loss = 2.4938`, Ep 20 `val_loss = 2.4874`, `es = 2`) | — |
| **Val (`tst2012`) – Greedy** | `23.75` (`BP=0.9810`, `sys_len=33,682`) | `23.14` (`BP=0.9692`, `sys_len=33,287`) | **`-0.61`** |
| **Val (`tst2012`) – Beam (`k=4, α=0.6`)** | `24.58` (`BP=0.9743`, `sys_len=33,457`) | `23.81` (`BP=0.9574`, `sys_len=32,896`) | **`-0.77`** |
| **Test (`tst2013`) – Greedy** | `25.78` (`BP=0.9624`, `sys_len=32,478`) | `25.27` (`BP=0.9620`, `sys_len=32,464`) | **`-0.51`** |
| **Test (`tst2013`) – Beam (`k=4, α=0.6`)** | **`26.90`** (`BP=0.9620`, `P=[60.95, 35.12, 21.39, 13.35]`) | **`26.65`** (`BP=0.9596`, `P=[60.94, 35.02, 21.20, 13.15]`) | **`-0.25`** |

### 6.2. Đối chiếu theo Độ dài Câu nguồn trên Tập Test (`tst2013`)

| Nhóm độ dài `src` | Số câu | `v2` Greedy | `v2` Beam (`v2` Beam − Greedy) | `v1` Beam | Chênh lệch Beam (`v2 - v1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1–10` từ** | 292 | `26.25` | `26.10` (`-0.15`) | `26.37` | **`-0.27`** |
| **`11–20` từ** | 445 | `26.04` | **`27.16`** (`+1.12`) | **`27.16`** | **`+0.00`** |
| **`21–35` từ** | 363 | `25.43` | **`26.80`** (`+1.37`) | **`26.86`** | **`-0.06`** |
| **`>35` từ** | 168 | `24.23` | **`26.20`** (`+1.97`) | **`26.89`** | **`-0.69`** |

### 6.3. Đúc kết Kỹ thuật từ So sánh `v1` vs `v2`

1. **Đánh đổi Tốc độ vs Tổng quát hóa của static `BucketBatchSampler`:** Nhờ giảm chiều dài đệm trung bình mỗi batch (`T_max` từ `90.41` xuống `27.19`), thời gian huấn luyện rút ngắn **gần một nửa (`4.69h -> 2.37h`)**, trong khi kết quả Beam Search trên `test` vẫn bám rất sát (`26.65` vs `26.90`, chỉ lệch `-0.25 BLEU`). Đặc biệt hai nhóm câu phổ biến nhất (`11–20` và `21–35` từ, chiếm `63.7%` tập test) đạt hiệu năng **tương đương tuyệt đối** (`27.16` vs `27.16`, `26.80` vs `26.86`).
2. **Vì sao nhóm `>35` từ giảm nhẹ (`-0.69`) và điểm dao động mạnh hơn ở `v2`?** Do `v2` đã lọc bỏ `218` cặp câu `>128` từ khỏi `train` và `BucketBatchSampler` gom các câu siêu dài vào chỉ một số ít batch cuối (làm ít đa dạng tổ hợp batch ngắn/dài khi cập nhật trọng số và hơi tăng `grad_norm`: `0.5102` vs `0.4532`). Dù vậy, trên `v2`, **Beam Search (`k=4, α=0.6`)** vẫn mang lại mức tăng rất ấn tượng **`+1.38 BLEU`** trên toàn `test` (`25.27 -> 26.65`) và **`+1.97 BLEU`** riêng ở câu `>35` từ (`24.23 -> 26.20`).
3. **Lý do dừng ở Epoch 20:** Ở `v2`, `val_loss` từ Epoch 15 (`2.4936`) đến Epoch 20 (`2.4874`) gần như đi ngang (chỉ giảm `0.0062`), trong khi `train_loss` giảm nhanh (`2.2412 -> 2.1429`) cho thấy mô hình đã chạm ngưỡng bão hòa của `lr = 0.001`. Vì vậy checkpoint Epoch 18 (`26.65` Test Beam) được chốt làm **mốc Baseline chính thức trên bộ dữ liệu thống nhất** để so sánh công bằng với **Transformer** ở Tuần 3.

---

## 7. Tổng kết & Đề xuất cho Bước Tiếp Theo

### Tổng kết Giai đoạn 2 (Baseline)
- Đã hoàn tất huấn luyện và kiểm chứng đầy đủ cho **Baseline Seq2Seq BiLSTM-2L + Luong General Attention (15.82M tham số)** trên IWSLT'15 Anh–Việt:
  - **Bản `v1` (`shuffle=True`, trước thống nhất):** `val = 23.75 / 24.58` \| `test = 25.78 / 26.90` (Greedy / Beam `k=4, α=0.6`, train `4.69h`).
  - **Bản `v2` (Mốc chính thức trên dữ liệu thống nhất `131,942` + BPE mới + `BucketBatchSampler`):** `val = 23.14 / 23.81` \| `test = 25.27 / 26.65` (Greedy / Beam `k=4, α=0.6`, train `2.37h`).
- Hạ tầng `Trainer`, `SacreBLEU`, `Greedy/Beam Decode`, checkpoint resume và W&B logging đã được kiểm chứng hoạt động ổn định trong cả hai lần chạy thực tế.

### Đề xuất cho Tuần 3 & Tuần 4
1. **Tuần 3 — Triển khai kiến trúc Transformer từ đầu (`src/models/transformer/`):**
   - Tận dụng lại hoàn toàn `DataLoader` (có thể cân nhắc giữ `BucketBatchSampler` vì cực kỳ có lợi cho Attention $\mathcal{O}(L^2)$), `BPETokenizer 8,000`, `Trainer`, `NoamScheduler`, `TranslationLoss (label_smoothing = 0.1)`, và `scripts/evaluate.py`.
   - Đối chiếu trực tiếp trên cùng bộ dữ liệu thống nhất `131,942` / `1,553` / `1,268` với mốc **`26.65` Test Beam** của `Baseline v2` (và tham chiếu thêm `26.90` của `v1`).
2. **Tuần 4 — Ablation Study đề xuất cho nhánh Seq2Seq LSTM (nếu thực nghiệm thêm):**
   - Chỉnh `ReduceLROnPlateau` (`patience=2`, `threshold=0.01` hoặc dùng `CosineAnnealingLR`) để kích hoạt giảm LR thực sự sau Epoch 10 thay vì giữ nguyên `1e-3`.
   - Thử nghiệm `label_smoothing: 0.1`, `teacher_forcing_ratio < 1.0` (giảm Exposure Bias & cải thiện `BP`), và so sánh `use_bucketing=true` vs `false` hay các biến thể Attention (`dot` vs `general` vs `concat`).
