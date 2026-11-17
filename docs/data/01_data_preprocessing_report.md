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

### Bước 3: Loại bỏ ký tự điều khiển ẩn & Chuẩn hóa khoảng trắng
- **Xử lý:** Loại bỏ các ký tự điều khiển/định dạng vô hình thuộc nhóm Unicode `"C"` (trừ các ký tự xuống dòng/tab thông thường), sau đó gộp các khoảng trắng liên tiếp về đúng 1 dấu cách và xóa khoảng trắng thừa ở hai đầu câu (`re.sub(r"\s+", " ", text).strip()`).

### Bước 4: Chiến lược lọc phân biệt giữa Tập Huấn luyện và Tập Đánh giá
- **Đối với tập `train` (`is_train=True`):**
  - Loại bỏ các cặp câu có câu nguồn hoặc câu đích bị rỗng sau khi làm sạch (**151 cặp**).
  - Loại bỏ các cặp câu trùng lặp hoàn toàn `(en, vi)` bằng tập hợp `seen_pairs` (**1,006 cặp** — chủ yếu là các câu kết thúc bài nói như *"Thank you ."* — *"Cảm ơn ."*) để tránh mô hình bị thiên lệch (overfitting) vào mẫu lặp.
  - Hỗ trợ tùy chọn cờ `--max-len` khi cần lọc câu quá dài trong quá trình huấn luyện.
- **Đối với tập `val` (`tst2012`) và `test` (`tst2013`) (`is_train=False`):**
  - **Chỉ làm sạch chuỗi (`clean_text`), tuyệt đối giữ nguyên 100% số lượng câu** (`1,553` câu `val` và `1,268` câu `test`).
  - **Lý do khoa học:** Tập `val` và `test` là bộ kiểm chuẩn cố định (benchmark); việc giữ nguyên toàn bộ mẫu giúp điểm BLEU của mô hình đối sánh công bằng với các công trình trước đây.

---

## 3. Thống kê Dữ liệu Sau Tiền xử lý

Số liệu trích xuất trực tiếp từ `data/processed/stats.json`:

### 3.1. Bảng thống kê quy mô và phân bố độ dài câu (tính theo số từ tách bởi khoảng trắng)

| Tập | Số cặp gốc | Sau làm sạch | Đã loại bỏ | Ngôn ngữ | Mean | Median (p50) | p95 | p99 | Min | Max |
| :--- | ---: | ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train** | 133,317 | **132,160** | 151 rỗng, 1,006 trùng | EN | 20.45 | 17.0 | 48.0 | 71.0 | 1 | 628 |
| | | | | VI | 25.02 | 20.0 | 59.0 | 90.0 | 1 | 850 |
| **Val** (`tst2012`) | 1,553 | **1,553** | 0 *(giữ 100%)* | EN | 18.02 | 15.0 | 39.0 | 57.0 | 2 | 93 |
| | | | | VI | 22.08 | 19.0 | 46.4 | 69.0 | 2 | 122 |
| **Test** (`tst2013`) | 1,268 | **1,268** | 0 *(giữ 100%)* | EN | 21.08 | 18.0 | 48.65 | 67.0 | 2 | 102 |
| | | | | VI | 26.56 | 22.0 | 61.0 | 87.0 | 2 | 123 |

### 3.2. Kiểm tra rò rỉ và tách biệt tập dữ liệu (Overlap Check)
- **Trùng lặp giữa `val` và `test`:** 3 cặp câu (xác nhận 2 tập hoàn toàn tách biệt, không bị lỗi trùng toàn bộ như các bản Parquet trên Hugging Face).
- **Trùng lặp giữa `train` và `val`:** 12 cặp câu.
- **Trùng lặp giữa `train` và `test`:** 12 cặp câu.
- **Kiểm tra HTML entities còn sót:** **0** entities (`&apos;`, `&quot;`, `&amp;`, `&#91;`, `&#93;`, `&lt;`, `&gt;`) trên cả 6 file trong `data/processed/`.

---

### 3.3. Ví dụ thực tế minh họa cho từng bước xử lý & thống kê

#### A. Ví dụ chuẩn hóa ký tự (Giải mã HTML entities & Unicode)
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

#### C. Ví dụ minh họa cho Thống kê Độ dài câu (trích từ `data/processed/train.{en,vi}`)
- **Ngắn nhất (`min = 1` từ EN / `1` từ VI — dòng 39,305):**
  - *EN (1 từ):* `.`
  - *VI (1 từ):* `.`
- **Mức trung vị (`median = 17` từ EN — dòng 18):**
  - *EN (17 từ):* `It 's a huge amount of stuff . It 's equal to the weight of methane .`
  - *VI (16 từ):* `Đó là một lượng khí thải khổng lồ , bằng tổng trọng lượng của mêtan .`
- **Ngưỡng phân vị `p95` (`48` từ EN / `59` từ VI — dòng 252):**
  - *EN (48 từ):* `Well , working with the safe organisms that we normally work with , the chance of an accident happening with somebody accidentally creating , like , some sort of superbug , that 's literally about as probable as a snowstorm in the middle of the Sahara Desert .`
  - *VI (59 từ):* `À , làm việc với những sinh vật an toàn như chúng tôi thường tiếp xúc thì khả năng xảy ra tai nạn , ví dụ như người nào đó vô tình tạo ra một loại siêu bọ , khả năng này cũng tương đương như khả năng một trận bão tuyết xảy ra giữa sa mạc Sahara .`
- **Ngưỡng phân vị `p99` (`71` từ EN / `88` từ VI — dòng 925):**
  - *EN (71 từ):* `Now , for someone who has never experienced depression or doesn 't really know what that means , that might surprise them to hear , because there 's this pretty popular misconception that depression is just being sad when something in your life goes wrong , when you break up with your girlfriend , when you lose a loved one , when you don 't get the job you wanted .`
  - *VI (88 từ):* `Một số người chưa bao giờ trải qua cảm giác trầm cảm hay chưa thực sự biết trầm cảm là gì , họ thực sự sẽ ngạc nhiên , vì thường có quan niệm sai lầm khá phổ biến , rằng trầm cảm chỉ đơn thuần là cảm giác buồn chán khi điều gì đó trong cuộc sống có vấn đề , khi bạn chia tay với bạn gái , khi bạn mất đi người thân , khi bạn không tìm được công việc bạn mong muốn .`
- **Dài nhất ngoại lai (`max = 628` từ EN / `850` từ VI — dòng 14,954):**
  - *EN (628 từ):* `Thula Mama , Thula Mama , Thula Mama , Thula Mama . Through the mist of the tears in your eyes on my childhood memory , I know the truth in your smile , I know the truth in your smile , piercing through the gloom of my i...`
  - *VI (850 từ):* `Thula Mama , Thula Mama , Thula Mama , Thula Mama . Trong kí ức tuổi thơ con , qua những giọt lệ nhoè mắt bà , con thấy chân lý trong nụ cười của bà , con thấy chân lý trong nụ cười của bà , xuyên thấu màn đêm u tối tron...`
  - *Giải thích:* Dòng này chứa trọn vẹn lời một bài hát trong TED Talks bị gộp liền thành 1 dòng phụ đề duy nhất — minh chứng trực quan cho thấy vì sao nên chọn `max_seq_len` dựa trên `p99` (~100–128) thay vì theo `max = 850`.

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
| **Train** | `train.en` / `train.vi` | 12,983.8 / 17,653.4 | **132,160** | 2,702,476 (EN) / 3,306,877 (VI) | 54,167 (EN) / 25,611 (VI) |
| **Validation** (`tst2012`) | `val.en` / `val.vi` | 134.1 / 184.2 | **1,553** | 27,983 (EN) / 34,297 (VI) | 3,727 (EN) / 2,487 (VI) |
| **Test** (`tst2013`) | `test.en` / `test.vi` | 126.9 / 179.7 | **1,268** | 26,728 (EN) / 33,682 (VI) | 3,917 (EN) / 2,660 (VI) |
| **Metadata thống kê** | `stats.json` | — | — | Lưu cấu hình, thống kê phân bố và kết quả overlap check | — |

**Checklist xác nhận chất lượng dữ liệu đầu ra (Quality Checklist):**
- **Gióng hàng câu (Sentence alignment):** Khớp chính xác 1:1 giữa `.en` và `.vi` ở cả 3 tập (`132,160` / `1,553` / `1,268`).
- **Dòng rỗng (Empty lines):** `0` dòng rỗng trên cả 6 file (đã lọc bỏ 151 cặp rỗng trong tập `train`, `val` và `test` vốn không có dòng rỗng).
- **Chuẩn hóa bảng mã:** `100%` sử dụng mã hóa UTF-8 theo chuẩn Unicode dựng sẵn `NFC` (`True` trên cả 6 file).
- **HTML entities còn sót:** `0` trên cả 6 file (`&apos;`, `&quot;`, `&amp;`, `&#91;`, `&#93;`, `&lt;`, `&gt;`).
- **Tính sẵn sàng:** Đã sạch hoàn toàn và sẵn sàng đưa trực tiếp vào bước huấn luyện bộ tách từ BPE (`src/data/tokenizer.py`) và xây dựng PyTorch `DataLoader` (`src/data/dataset.py`).

---

## 4. Nhận xét & Định hướng cho Giai đoạn tiếp theo (Tokenizer & Model)

1. **Lựa chọn ngưỡng độ dài chuỗi (`max_seq_len`):**
   - Trong tập `train`, có một tỷ lệ rất nhỏ câu ngoại lai dài bất thường lên tới `max = 628` từ (EN) và `850` từ (VI) do phụ đề bị ghép nhiều câu.
   - Tuy nhiên, **95%** số câu ngắn hơn hoặc bằng `48` từ (EN) / `59` từ (VI), và **99%** số câu nằm dưới `71` từ (EN) / `90` từ (VI).
   - Vì độ phức tạp bộ nhớ của cơ chế Self-Attention trong Transformer là $\mathcal{O}(L^2)$, khi huấn luyện mô hình nên thiết lập ngưỡng `max_len` khoảng **100–128** (tính theo token BPE hoặc số từ ở tập train) để vừa bao phủ $>99\%$ dữ liệu, vừa tránh tràn VRAM GPU.
2. **Chênh lệch độ dài giữa Tiếng Anh và Tiếng Việt:**
   - Số lượng từ/tiếng trung bình của tiếng Việt luôn cao hơn tiếng Anh khoảng **22%–26%** (`25.02` vs `20.45` ở `train`; `26.56` vs `21.08` ở `test`) do tiếng Việt là ngôn ngữ đơn lập, mỗi âm tiết tách rời thành một khoảng trắng (ví dụ một từ tiếng Anh *"freedom"* tương ứng 2 tiếng *"tự do"*).
3. **Đặc điểm dấu câu đã tách sẵn:**
   - Dữ liệu gốc đã được Moses tách dấu câu bằng khoảng trắng (ví dụ `life 's work .`). Khi huấn luyện bộ tách từ BPE ở bước sau cần lưu ý thiết lập pre-tokenizer phù hợp và kiểm tra bước hậu xử lý (detokenize) trước khi tính SacreBLEU.

---

## 5. Hướng dẫn tái tạo (Reproducibility)

Chạy lần lượt hai lệnh sau từ thư mục gốc dự án để tái tạo toàn bộ `data/raw/` và `data/processed/`:

```bash
# 1. Tải 6 file gốc về data/raw/ và kiểm tra toàn vẹn số dòng
python scripts/download_data.py

# 2. Tiền xử lý dữ liệu sang data/processed/ và sinh stats.json
python -m src.data.preprocess
```

*(Tùy chọn lọc câu quá dài trên tập train khi chạy thực nghiệm: `python -m src.data.preprocess --max-len 100`)*

---
