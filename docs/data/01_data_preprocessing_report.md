# Báo cáo Tiền xử lý Dữ liệu — Đề tài 8: Dịch máy Anh–Việt (IWSLT'15 En–Vi)

> **Tài liệu lưu trữ:** Bản nháp kỹ thuật phục vụ soạn thảo Proposal (Tuần 1), Báo cáo cuối (Tuần 6) và ôn tập Vấn đáp cá nhân.
> **Mã nguồn liên quan:** [`scripts/download_data.py`](../../scripts/download_data.py), [`src/data/preprocess.py`](../../src/data/preprocess.py), [`data/processed/stats.json`](../../data/processed/stats.json)

---

## 1. Tổng quan & Lựa chọn Nguồn Dữ liệu

### 1.1. Giới thiệu bộ dữ liệu
- **Tên bộ dữ liệu:** IWSLT'15 English–Vietnamese (Stanford NMT benchmark — Luong & Manning, 2015).
- **Miền dữ liệu:** Bản dịch phụ đề các bài diễn thuyết TED Talks (văn phong nói tự nhiên, đa dạng chủ đề khoa học, giáo dục, xã hội).
- **Hướng dịch:** Tiếng Anh (`en`) $\rightarrow$ Tiếng Việt (`vi`).

### 1.2. Lý do lựa chọn nguồn tải dữ liệu
Trong quá trình kiểm tra các nguồn dữ liệu công khai:
1. Máy chủ gốc của Đại học Stanford (`nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/`) hiện không phản hồi (lỗi timeout khi kết nối).
2. Một số bản đóng gói dạng Parquet trên Hugging Face (ví dụ `IWSLT/mt_eng_vietnamese` hoặc `thainq107/iwslt2015-en-vi`) bị lỗi trùng lặp ở script cấu hình khiến tập `validation` và `test` giống hệt nhau (cùng trỏ về `tst2013`).

Vì vậy, dự án lựa chọn kho lưu trữ chuẩn nguyên bản của Stanford NMT trên Hugging Face tại:
- **Trang chủ bộ dữ liệu:** https://huggingface.co/datasets/dtruong46me/standford-nmt-iwslt15

Bộ dữ liệu gồm đầy đủ 6 tệp song ngữ tách biệt rõ ràng 3 tập:

| Tập dữ liệu | File Tiếng Anh | File Tiếng Việt | Dung lượng (KB) | Số dòng gốc |
| :--- | :--- | :--- | ---: | ---: |
| **Train** | `train.en` | `train.vi` | 13,284.8 / 17,651.0 | 133,317 |
| **Validation** (`tst2012`) | `tst2012.en` | `tst2012.vi` | 137.0 / 184.0 | 1,553 |
| **Test** (`tst2013`) | `tst2013.en` | `tst2013.vi` | 129.2 / 179.5 | 1,268 |

---

## 2. Quy trình Tiền xử lý Dữ liệu (`src/data/preprocess.py`)

Dữ liệu thô trong `data/raw/` đi qua quy trình làm sạch gồm 4 bước chính để sinh ra bộ dữ liệu sạch tại `data/processed/` (`train.{en,vi}`, `val.{en,vi}`, `test.{en,vi}`):

### Bước 1: Giải mã HTML entities (`html.unescape`)
- **Hiện trạng dữ liệu thô:** Bộ dữ liệu Stanford NMT đã qua tiền xử lý bằng Moses Tokenizer, do đó các ký tự dấu câu bị chuyển thành HTML entities như `&apos;`, `&quot;`, `&amp;`, `&#91;`, `&#93;`.
- **Xử lý:** Gọi `html.unescape(html.unescape(text))` (giải mã lồng 2 lần để xử lý triệt để cả các chuỗi mã hóa kép như `&amp;apos;`), đưa toàn bộ về ký tự chuẩn (`'`, `"`, `&`, `[`, `]`).
- **Tác dụng:** Ngăn bộ tách từ (Tokenizer) hiểu nhầm các chuỗi như `apos` hay `quot` thành từ vựng riêng biệt.

### Bước 2: Chuẩn hóa bảng mã Unicode tiếng Việt về `NFC`
- **Hiện trạng:** Trong Unicode, ký tự tiếng Việt có dấu có thể tồn tại ở dạng dựng sẵn (**NFC**) hoặc tổ hợp (**NFD** — chữ cái gốc đi kèm dấu rời).
- **Xử lý:** Ép toàn bộ văn bản qua `unicodedata.normalize("NFC", text)`.
- **Tác dụng:** Tránh tình trạng cùng một từ tiếng Việt nhưng bị mã hóa thành 2 token ID khác nhau, gây phân mảnh bộ từ vựng và làm sai lệch điểm BLEU khi đối sánh chuỗi.

### Bước 3: Loại bỏ ký tự điều khiển ẩn & Đọc/Chuẩn hóa khoảng trắng an toàn
- **Xử lý ký tự ẩn:** Loại bỏ các ký tự điều khiển/định dạng vô hình thuộc nhóm Unicode `"C"`, nhưng **giữ lại mọi ký tự trắng (`ch.isspace()`)** như `\t`, `\n`, `\r`, `\x0c`, `\u2028` để phép gộp khoảng trắng (`re.sub(r"\s+", " ", text).strip()`) biến chúng thành dấu cách chuẩn, tránh làm dính liền hai từ với nhau.
- **Đọc tệp an toàn:** Đọc từng dòng bằng `[line.rstrip("\r\n") for line in f]` thay vì `splitlines()` nhằm tránh bị tách dòng sai (gây lệch gióng hàng `.en`/`.vi`) nếu dữ liệu thô chứa ký tự ngắt dòng Unicode như `\x0b`, `\x0c`, `\u2028`.

### Bước 4: Chiến lược lọc phân biệt giữa Tập Huấn luyện và Tập Đánh giá
- **Đối với tập `train` (`is_train=True`):**
  - Loại bỏ các cặp câu có câu nguồn hoặc câu đích bị rỗng sau khi làm sạch (**151 cặp**).
  - Lọc bỏ các cặp câu quá dài vượt ngưỡng mặc định **`max_words = 128` từ** ở nguồn hoặc đích (**218 cặp**; cấu hình qua `--max-words`, hỗ trợ truyền `<= 0` nếu muốn tắt bước lọc độ dài).
  - Loại bỏ các cặp câu trùng lặp hoàn toàn `(en, vi)` bằng tập hợp `seen_pairs` (**1,006 cặp** — chủ yếu là các câu kết thúc bài nói như *"Thank you ."* — *"Cảm ơn ."*) để tránh mô hình bị thiên lệch (overfitting) vào mẫu lặp.
  - Tổng sau lọc: **`131,942` cặp câu** (`133,317 - 151 - 218 - 1,006`).
- **Đối với tập `val` (`tst2012`) và `test` (`tst2013`) (`is_train=False`):**
  - **Chỉ làm sạch chuỗi (`clean_text`), tuyệt đối giữ nguyên 100% số lượng câu** (`1,553` câu `val` và `1,268` câu `test`, được kiểm tra tự động bằng `assert` ở cuối pipeline).
  - **Lý do khoa học:** Tập `val` và `test` là bộ kiểm chuẩn cố định (benchmark); việc giữ nguyên toàn bộ mẫu giúp điểm BLEU của mô hình đối sánh công bằng với các công trình trước đây.

---

## 3. Thống kê Dữ liệu Sau Tiền xử lý

Số liệu trích xuất trực tiếp từ `data/processed/stats.json`:

### 3.1. Bảng thống kê quy mô, phân bố độ dài câu và tỷ lệ độ dài (tính theo số từ)

| Tập | Số cặp gốc | Sau làm sạch | Đã loại bỏ | Ngôn ngữ | Mean | Median (p50) | p95 | p99 | Min | Max | Tỷ lệ VI/EN (`mean_VI / mean_EN`) |
| :--- | ---: | ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Train** | 133,317 | **131,942** | 151 rỗng, 218 quá dài (`>128`), 1,006 trùng | EN | 20.24 | 17.0 | 47.0 | 69.0 | 1 | 122 | `1.22` (`1.2228`) |
| | | | | VI | 24.75 | 20.0 | 58.0 | 86.0 | 1 | 128 | |
| **Val** (`tst2012`) | 1,553 | **1,553** | 0 *(giữ 100%)* | EN | 18.02 | 15.0 | 39.0 | 57.0 | 2 | 93 | `1.23` (`1.2253`) |
| | | | | VI | 22.08 | 19.0 | 46.4 | 69.0 | 2 | 122 | |
| **Test** (`tst2013`) | 1,268 | **1,268** | 0 *(giữ 100%)* | EN | 21.08 | 18.0 | 48.65 | 67.0 | 2 | 102 | `1.26` (`1.2600`) |
| | | | | VI | 26.56 | 22.0 | 61.0 | 87.0 | 2 | 123 | |

### 3.2. Kiểm tra rò rỉ và tách biệt tập dữ liệu (Overlap Check 3 tầng)
Kiểm tra trùng lặp toàn diện ở cả 3 mức độ: cả cặp `(en, vi)`, chỉ câu nguồn (`source_only`), và chỉ câu đích (`target_only`):
- **Trùng cả cặp (`pairs`):** `val` vs `test` = **3** cặp \| `train` vs `val` = **12** cặp \| `train` vs `test` = **12** cặp *(xác nhận `val` và `test` hoàn toàn tách biệt, không bị lỗi trùng toàn bộ như các bản Parquet trên Hugging Face)*.
- **Trùng câu nguồn Tiếng Anh (`source_only`):** `val` vs `test` = **3** câu \| `train` vs `val` = **14** câu \| `train` vs `test` = **13** câu.
- **Trùng câu đích Tiếng Việt (`target_only`):** `val` vs `test` = **3** câu \| `train` vs `val` = **15** câu \| `train` vs `test` = **11** câu.
- **Kiểm tra HTML entities còn sót:** **0** entities (`&apos;`, `&quot;`, `&amp;`, `&#91;`, `&#93;`, `&lt;`, `&gt;`) trên cả 6 file trong `data/processed/`.

---

### 3.3. Ví dụ thực tế minh họa cho từng bước xử lý & thống kê

#### A. Ví dụ chuẩn hóa ký tự (Giải mã HTML entities & Unicode — *trích từ tập `val`*)
- **Mẫu 1:**
  - *Raw EN:* `Imagine the first piece : a man burning his life &apos;s work .`
  - *Clean EN:* `Imagine the first piece : a man burning his life 's work .`
  - *Clean VI:* `Hãy tưởng tượng mảnh đầu tiên : một người đàn ông đốt cháy sự nghiệp cả đời mình .`
- **Mẫu 2:**
  - *Raw EN:* `He is a poet , a playwright , a man whose whole life had been balanced on the single hope of his country &apos;s unity and freedom .`
  - *Clean EN:* `He is a poet , a playwright , a man whose whole life had been balanced on the single hope of his country 's unity and freedom .`
  - *Clean VI:* `Ông là nhà thơ , nhà viết kịch , một người mà cả cuộc đời chênh vênh trên tia hi vọng duy nhất rằng đất nước ông sẽ độc lập tự do .`
- **Mẫu 3:**
  - *Raw EN:* `I don &apos;t remember the pirates who came many times , but were bluffed by the bravado of the men on our boat , or the engine dying and failing to start for six hours .`
  - *Clean EN:* `I don 't remember the pirates who came many times , but were bluffed by the bravado of the men on our boat , or the engine dying and failing to start for six hours .`
  - *Clean VI:* `Tôi không nhớ lũ cướp biến đã đến rất nhiều lần , nhưng bị lừa bỏ đi bởi sự dũng càm của những người đàn ông trên thuyền , hay là khi máy tàu bị chết và không khởi động được trong suốt sáu tiếng đồng hồ .`

#### B. Ví dụ lọc phân biệt Tập Huấn luyện (`train`) và Tập Đánh giá (`val` / `test`)
- **Các cặp câu rỗng bị loại bỏ ở tập `train` (`151` cặp trong `data/raw/train.{en,vi}`):**
  - *Dòng 470:* `EN: ""` $\longleftrightarrow$ `VI: "̣"` *(tiếng Anh rỗng, tiếng Việt chỉ sót 1 dấu nặng rời rạc)*.
  - *Dòng 8,696:* `EN: ""` $\longleftrightarrow$ `VI: "Khán giả vỗ tay"` *(chú thích sân khấu chỉ có ở một ngôn ngữ)*.
  - *Dòng 9,763:* `EN: ""` $\longleftrightarrow$ `VI: "Vỗ tay ."` *(tương tự, không có câu tiếng Anh tương ứng)*.
- **Các cặp câu trùng lặp bị lọc bỏ ở tập `train` (`1,006` bản sao bị loại):**
  - `("Thank you .", "Cảm ơn .")` xuất hiện **164 lần** (dòng 420, 1005, 3645, 4067, 4848...) $\rightarrow$ giữ lại duy nhất 1 lần ở dòng 420, loại bỏ 163 bản sao.
  - `("Thank you .", "Xin cảm ơn .")` xuất hiện **155 lần** (dòng 1800, 2414, 3741...) $\rightarrow$ giữ lại 1 lần, loại bỏ 154 bản sao.
  - `("Thank you .", "Cảm ơn các bạn .")` xuất hiện **65 lần** (dòng 201, 758, 9081...) $\rightarrow$ giữ lại 1 lần, loại bỏ 64 bản sao.
  - `("Thank you very much .", "Cảm ơn các bạn rất nhiều .")` xuất hiện **44 lần** (dòng 1500, 2453, 2815...) $\rightarrow$ giữ lại 1 lần, loại bỏ 43 bản sao.
- **Đối chứng ở tập `val` (`tst2012`) và `test` (`tst2013`) — GIỮ NGUYÊN 100% (`is_train=False`):**
  - *Tập `val`:* Cặp `("Thank you .", "Cảm ơn .")` xuất hiện 2 lần (dòng 524, 1423) và `("Thank you .", "Cám ơn các bạn .")` xuất hiện 2 lần (dòng 1362, 1479) $\rightarrow$ **giữ nguyên tất cả**, bảo toàn đủ **1,553** cặp câu chuẩn.
  - *Tập `test`:* Cặp `("Thank you .", "Cảm ơn .")` xuất hiện 4 lần (dòng 490, 517, 721, 881) và `("Thank you .", "Cảm ơn các bạn .")` xuất hiện 3 lần (dòng 681, 768, 1144) $\rightarrow$ **giữ nguyên tất cả**, bảo toàn đủ **1,268** cặp câu chuẩn.

#### C. Ví dụ minh họa cho Thống kê Độ dài câu (trích từ `data/processed/train.{en,vi}` sau lọc `<= 128` từ)
- **Ngắn nhất (`min = 1` từ EN / `1` từ VI — dòng 39,253):**
  - *EN (1 từ):* `.`
  - *VI (1 từ):* `.`
- **Mức trung vị (`median = 17` từ EN — dòng 18):**
  - *EN (17 từ):* `It 's a huge amount of stuff . It 's equal to the weight of methane .`
  - *VI (16 từ):* `Đó là một lượng khí thải khổng lồ , bằng tổng trọng lượng của mêtan .`
- **Ngưỡng phân vị `p95` (`47` từ EN / `58` từ VI — dòng 13,169):**
  - *EN (47 từ):* `And as a result -- once we got over a little hurdle , and having to audition twice -- they accepted me . And not only that -- what had happened was that it changed the whole role of the music institutions throughout the United Kingdom .`
  - *VI (58 từ):* `Và cuối cùng -- Khi chúng tôi vượt qua chướng ngại bé nhỏ đó , để có buổi thử giọng thứ hai -- họ đã chấp nhận tôi . Và không chỉ như vậy -- điều đã xảy ra làm thay đổi tất cả vai trò của những học viện âm nhạc trên khắp Vương quốc Anh .`
- **Ngưỡng phân vị `p99` (`69` từ EN / `86` từ VI — dòng 9,673):**
  - *EN (69 từ):* `PR : I would like to say that I have learned so much from Aicha , starting with that day we had our very first meeting with other family members -- which was a very private meeting with security , because it was November 2002 , and , frankly , we were...`
  - *VI (86 từ):* `PR : Tôi xin được nói rằng tôi đã học được rất nhiều từ Aicha , ngay từ lần gặp gỡ đầu tiên cùng với những gia đình khác -- đó là một cuộc họp rất riêng tư với sự quản lý an ninh , vì lúc đấy là tháng 11 năm 2002 và , thẳng thắn mà nói , chúng tôi - ...`
- **Dài nhất sau lọc (`max = 122` từ EN / dòng 32,825 và `max = 128` từ VI / dòng 5,274):**
  - **218** cặp câu ngoại lai dài `> 128` từ trong dữ liệu thô (như dòng chứa trọn lời bài hát *"Thula Mama..."* dài `628` từ EN / `850` từ VI) đã được lọc bỏ ở bước tiền xử lý tập `train`, đưa biên độ dài thực tế về tối đa `122` từ (EN) và `128` từ (VI).

#### D. Chi tiết các cặp câu trong Kiểm tra Rò rỉ & Tách biệt Tập Dữ liệu (Overlap Check)
Toàn bộ các cặp câu trùng giữa 3 tập đều là các câu ngắn mang tính xã giao/cảm thán xuất hiện tự nhiên cuối các bài nói, hoàn toàn không có sự rò rỉ nội dung bài diễn thuyết:
- **`val` $\cap$ `test` (3 cặp):**
  1. `("Thank you .", "Cám ơn các bạn .")`
  2. `("Thank you .", "Cảm ơn .")`
  3. `("Thank you .", "Cảm ơn các bạn .")`
- **`train` $\cap$ `val` (12 cặp):**
  1. `("I don 't know .", "Tôi không biết .")`
  2. `("No .", "Không .")`
  3. `("Okay .", "Được rồi .")`
  4. `("Wooo !", "Wooo !")`
  5. `("Thank you .", "Cám ơn các bạn .")` / `("Cảm ơn .")` / `("Cảm ơn các bạn .")` / `("Xin cám ơn các bạn .")` / `("Xin cảm ơn .")` *(5 biến thể)*
  6. `("Thank you very much .", "Cám ơn rất nhiều")` / `("Cảm ơn các bạn rất nhiều")` / `("Cảm ơn các bạn rất nhiều .")` *(3 biến thể)*
- **`train` $\cap$ `test` (12 cặp):**
  1. `("So what happened ?", "Vậy điều gì đã xảy ra ?")`
  2. `("Thanks .", "Cảm ơn")`
  3. `("Thank you .", "Cám ơn .")` / `("Cám ơn các bạn .")` / `("Cảm ơn")` / `("Cảm ơn .")` / `("Cảm ơn bạn .")` / `("Cảm ơn các bạn .")` / `("Cảm ơn quý vị .")` / `("Xin cám ơn")` *(8 biến thể)*
  4. `("Thank you very much .", "Cám ơn rất nhiều .")` / `("Cảm ơn rất nhiều .")` *(2 biến thể)*

---

### 3.4. Tình trạng dữ liệu Train / Validation / Test sau khi qua xử lý (`data/processed/`)

Sau khi chạy hoàn tất `python -m src.data.preprocess`, toàn bộ dữ liệu sạch được lưu trữ cục bộ trong thư mục `data/processed/` với trạng thái chi tiết như sau:

| Tập dữ liệu | File đầu ra | Dung lượng (KB) | Số cặp câu | Tổng số từ (Tokens mức từ) | Số từ phân biệt (Unique words) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Train** | `train.en` / `train.vi` | 12,832.8 / 17,436.1 | **131,942** | 2,670,454 (EN) / 3,265,896 (VI) | 53,743 (EN) / 25,431 (VI) |
| **Validation** (`tst2012`) | `val.en` / `val.vi` | 134.1 / 184.2 | **1,553** | 27,983 (EN) / 34,297 (VI) | 3,727 (EN) / 2,487 (VI) |
| **Test** (`tst2013`) | `test.en` / `test.vi` | 126.9 / 179.7 | **1,268** | 26,728 (EN) / 33,682 (VI) | 3,917 (EN) / 2,660 (VI) |
| **Metadata thống kê** | `stats.json` | — | — | Lưu cấu hình, thống kê độ dài, `length_ratio` và overlap check 3 tầng | — |

**Checklist xác nhận chất lượng dữ liệu đầu ra (Quality Checklist):**
- **Gióng hàng câu (Sentence alignment):** Khớp chính xác 1:1 giữa `.en` và `.vi` ở cả 3 tập (`131,942` / `1,553` / `1,268`, có `assert` bảo toàn `val` & `test`).
- **Dòng rỗng (Empty lines):** `0` dòng rỗng trên cả 6 file (đã lọc bỏ 151 cặp rỗng trong tập `train`, `val` và `test` vốn không có dòng rỗng).
- **Chuẩn hóa bảng mã:** `100%` sử dụng mã hóa UTF-8 theo chuẩn Unicode dựng sẵn `NFC` (`True` trên cả 6 file).
- **HTML entities còn sót:** `0` trên cả 6 file (`&apos;`, `&quot;`, `&amp;`, `&#91;`, `&#93;`, `&lt;`, `&gt;`).
- **Tính sẵn sàng:** Đã sạch hoàn toàn và thống nhất chuẩn toàn nhóm để đưa vào bước tách từ BPE (`src/data/tokenizer.py`) và xây dựng PyTorch `DataLoader` (`src/data/dataset.py`).

---

## 4. Nhận xét & Định hướng cho Giai đoạn tiếp theo (Tokenizer & Model)

1. **Lựa chọn ngưỡng độ dài chuỗi (`max_seq_len`):**
   - Trong dữ liệu gốc của tập `train`, có `218` câu ngoại lai dài vượt `128` từ (lên tới `628` từ EN / `850` từ VI) do phụ đề bị ghép nhiều câu.
   - Tuy nhiên, **95%** số câu ngắn hơn hoặc bằng `47` từ (EN) / `58` từ (VI), và **99%** số câu nằm dưới `69` từ (EN) / `86` từ (VI).
   - Vì độ phức tạp bộ nhớ của cơ chế Self-Attention trong Transformer là $\mathcal{O}(L^2)$, nhóm thống nhất mặc định **`max_words = 128`** ở bước tiền xử lý `train` (kết hợp `max_seq_len = 128` ở mức token BPE) để vừa bao phủ $>99.8\%$ dữ liệu, vừa tránh tràn VRAM GPU.
2. **Chênh lệch độ dài giữa Tiếng Anh và Tiếng Việt:**
   - Số lượng từ/tiếng trung bình của tiếng Việt luôn cao hơn tiếng Anh khoảng **22%–26%** (`24.75` vs `20.24` ở `train`, `length_ratio = 1.22` [`1.2228`]; `22.08` vs `18.02` ở `val`, `length_ratio = 1.23` [`1.2253`]; `26.56` vs `21.08` ở `test`, `length_ratio = 1.26` [`1.2600`]) do tiếng Việt là ngôn ngữ đơn lập, mỗi âm tiết tách rời thành một khoảng trắng (ví dụ một từ tiếng Anh *"freedom"* tương ứng 2 tiếng *"tự do"*).
3. **Đặc điểm dấu câu đã tách sẵn:**
   - Dữ liệu gốc đã được Moses tách dấu câu bằng khoảng trắng (ví dụ `life 's work .`). Khi huấn luyện bộ tách từ BPE ở bước sau cần lưu ý thiết lập pre-tokenizer phù hợp và kiểm tra bước hậu xử lý (detokenize) trước khi tính SacreBLEU.

---

## 5. Hướng dẫn tái tạo (Reproducibility)

Chạy lần lượt hai lệnh sau từ thư mục gốc dự án để tái tạo toàn bộ `data/raw/` và bộ dữ liệu chuẩn thống nhất tại `data/processed/`:

```bash
# 1. Tải 6 file gốc về data/raw/ và kiểm tra toàn vẹn số dòng
python scripts/download_data.py

# 2. Tiền xử lý dữ liệu (mặc định lọc > 128 từ ở train) sang data/processed/ và sinh stats.json
python -m src.data.preprocess
```

*(Tùy chọn tắt lọc độ dài trên tập train: `python -m src.data.preprocess --max-words 0`)*

---
