# Spec dự án MixNext — Transition-aware Playlist Continuation

Oct 30, 2026 · @Licht Nguyen

## 1. Tổng quan

MixNext gợi ý bài hát tiếp theo vừa hợp gu người nghe vừa chuyển mượt từ bài đang phát. Nhóm 3 người, làm trong 3 tuần (23/11–13/12/2026), demo bằng Streamlit.

> **Cập nhật 20/11/2026:** spec đã được làm rõ để chia ticket. Thuật ngữ chuẩn nằm ở `GLOSSARY.md`; các quyết định khó đảo ngược nằm ở `docs/adr/`. Thay đổi chính: còn 3 tuần, bỏ RecBole (SASRec viết bằng PyTorch độc lập), item-kNN là bản liền kề, seed là 5 bài đầu và bài đích là bài thứ 6, trọng số mix và α chọn chung bằng grid search.

- **Bài toán:** automatic playlist continuation. Cho vài bài đầu của một phiên nghe, gợi ý bài tiếp theo. Đây là cùng dạng bài với RecSys Challenge 2018.
- **Điểm mới:** re-ranking đa mục tiêu, kết hợp điểm gu với điểm mix (BPM, tông, năng lượng).
- **Câu hỏi nghiên cứu:** muốn chuyển bài mượt hơn thì mất bao nhiêu độ chính xác? Kết quả chính là đường đánh đổi khi thay đổi trọng số α.
- **Tên đề tài đề xuất:** Transition-aware Playlist Continuation.

## 2. Đáp ứng yêu cầu môn học IT4613 (slide L0)

MixNext đáp ứng đủ các yêu cầu trong slide L0 sau ba bổ sung so với bản spec trước: mô hình tuần tự SASRec, điểm mix học từ dữ liệu bằng học tương phản (cả hai chuyển từ mở rộng sang bắt buộc), và bản nháp đề xuất đề tài ở mục 14.

### Sản phẩm phải nộp

| Yêu cầu trong L0 | MixNext đáp ứng thế nào | Mục |
| --- | --- | --- |
| Nhóm 2–4 người, tự chọn bài toán thực tế, xây dựng hệ thống thử nghiệm | Nhóm 3 người; app demo chạy được trên laptop, không cần mạng | 8, 10 |
| Đề xuất PDF 1–2 trang: mục đích, yêu cầu, kịch bản, ý tưởng phương pháp, đầu vào/đầu ra | Bản nháp đủ các ý, chỉ còn điền thông tin thành viên | 14 |
| Mã nguồn trong một file nén | Repo `mixnext/`, nén thư mục trừ `data/` | 9 |
| README cài đặt, chạy, kèm gói phần mềm dùng | `requirements.txt`; một lệnh chạy app từ `artifacts/` đã lưu; cách chạy lại pipeline trên Kaggle | 9 |
| Báo cáo PDF 6 mục | Mỗi mục có nội dung cụ thể ở mục 13 | 13 |
| Trình bày khoảng 15 phút, mọi thành viên tham gia, khớp với báo cáo | Mỗi người 5 phút, trình bày phần mình làm | 13 |
| Khai báo rõ mã nguồn, gói phần mềm, công cụ, dữ liệu dùng lại (cả trong slide) | MPD, bộ đặc trưng âm thanh, Music4All-Onion (CIKM 2022, CC BY 4.0) và đặc trưng Essentia do tác giả bộ dữ liệu tính, implicit, gensim, SASRec.pytorch (pmixer), PyTorch, librosa, Streamlit; một bảng riêng trong báo cáo và một slide | 13 |

### Tiêu chí chấm điểm

| Tiêu chí trong L0 | Điểm tựa của MixNext | Mục |
| --- | --- | --- |
| Mức độ phức tạp của bài toán | Bài toán hai mục tiêu mâu thuẫn nhau (đúng gu và chuyển mượt); hai nguồn tín hiệu khác bản chất (hành vi và âm thanh); dữ liệu hàng trăm nghìn playlist | 1, 4 |
| Chất lượng phương pháp | So sánh có kiểm soát: 2 baseline, 3 mô hình gu, điểm mix theo luật so với điểm mix học từ dữ liệu; cùng tập chia, 3 seed | 6, 7 |
| Sự thỏa đáng của tranh luận | Đường đánh đổi theo α; ablation từng thành phần điểm mix; giải thích vì sao NDCG giảm; dữ liệu có "học lại" được luật DJ hay không | 7 |
| Chất lượng trình bày và báo cáo | Thanh trượt α cho thấy đánh đổi ngay trên màn hình; người không rành kỹ thuật vẫn hiểu | 8, 13 |
| Cài đặt hệ thống thử nghiệm (chức năng, dễ dùng) | Chọn seed, hàng đợi kèm lý do, thanh trượt α, so sánh với baseline, chạy không cần mạng | 8 |

### Phủ nội dung môn học

| Chương trong L0 | Dùng ở đâu trong MixNext |
| --- | --- |
| Lọc cộng tác (cơ bản và nâng cao) | Item-kNN liền kề; ALS phân rã ma trận playlist × bài |
| Dựa trên nội dung | Đặc trưng âm thanh của bài: BPM, tông, năng lượng |
| Dựa trên tri thức | Luật mix của DJ: vòng Camelot, ngưỡng lệch BPM, nhịp gấp đôi/một nửa |
| Dựa trên phiên | Playlist continuation; vector phiên; SASRec |
| Kết hợp các kỹ thuật | Hybrid dạng tầng: mô hình gu sinh ứng viên, điểm mix xếp lại theo α |
| Đánh giá | HitRate, NDCG, MRR; chỉ số độ mượt; khảo sát người nghe nếu kịp |
| Học sâu, học tương phản | SASRec; bộ mã hóa chuyển bài huấn luyện bằng InfoNCE trên các cặp bài liền nhau |

**Nếu cô hỏi về việc trùng nhóm gợi ý nhạc khác:** nhóm kia (cần hỏi lại đề tài cụ thể) nhiều khả năng gợi ý bài theo sở thích. MixNext giải một bài toán khác: bài tiếp theo phải nối được với bài đang phát. Phần riêng là điểm mix (luật và học từ dữ liệu) và đường đánh đổi độ chính xác với độ mượt.

## 3. Phạm vi

MVP phải chạy end-to-end trước cuối tuần 2 (6/12); phần mở rộng chỉ làm khi MVP đã xong.

**Bắt buộc (MVP)**

- Pipeline ghép và lọc dữ liệu playlist với đặc trưng bài hát.
- 2 baseline (Popularity, item-kNN) và 3 mô hình gu: ALS, item2vec, SASRec.
- Điểm mix theo luật và bước re-rank có trọng số α.
- Điểm mix học từ dữ liệu bằng học tương phản, so với điểm mix theo luật.
- Đánh giá offline cả độ chính xác lẫn độ mượt, kèm đường đánh đổi và ablation.
- Demo Streamlit: hàng đợi gợi ý, thanh trượt α, giải thích, biểu đồ so sánh.

**Mở rộng (nếu còn thời gian)**

- Khảo sát người nghe 10–15 người.
- Gợi ý theo đường cong năng lượng (khởi động, cao trào, hạ nhiệt).
- Phát chuyển bài thật bằng crossfade.
- Điểm mix học dùng đủ đặc trưng âm thanh (danceability, valence, loudness…), so với bản chỉ dùng BPM, tông, năng lượng.

**Ngoài phạm vi**

- Cá nhân hóa theo user ID dài hạn: MPD không có user ID, nên cá nhân hóa ở mức phiên.
- Mix chuyên nghiệp kiểu DJ (beatmatching, EQ).

## 4. Dữ liệu

Dữ liệu là playlist từ Spotify MPD, ghép với đặc trưng âm thanh theo track ID. Tỷ lệ ghép được là rủi ro lớn nhất, nên phải đo ngay ngày đầu.

| Nguồn | Dùng cho | Ghi chú |
| --- | --- | --- |
| Spotify Million Playlist Dataset, bản sao Kaggle `himanshuwagh/spotify-million` | Phiên nghe để học gu | 1 triệu playlist; chỉ dùng tập con. AIcrowd đã gỡ bản chính thức, đây là bản sao không chính thức (ADR 0001) |
| `krishsharma0413/2-million-songs-from-mpd-with-audio-features` (Kaggle) | tempo, key, mode, energy | Làm riêng cho 2,26 triệu bài của MPD, khóa `track_uri`; dự phòng `rodolfofigueroa/spotify-12m-songs` (~1,2 triệu bài) |
| FMA + librosa (dự phòng) | Tự trích BPM và tông | Chỉ dùng nếu ghép thất bại, hoặc cho phần phát nhạc |
| Music4All-Onion (Moscati và cộng sự, CIKM 2022), Zenodo record 15394646, CC BY 4.0 (mở rộng #54–#57) | Bộ thứ hai cho phần điểm mix: lịch sử nghe last.fm (253 triệu lượt, 119 nghìn user) và đặc trưng Essentia (BPM, HPCP, loudness) | Không có nhãn tông và energy kiểu Spotify: tông tự ước lượng từ HPCP, energy thay thế = hạng phần trăm loudness (#56). Phiên cắt theo khoảng nghỉ 30 phút (#55) |

Spotify đã hạn chế API audio features cho app mới, nên không dựa vào việc gọi API để lấy thêm đặc trưng.

**Tiền xử lý**

1. Lấy mẫu khoảng 100k playlist. Đây là con số khởi đầu, chỉnh theo RAM của Kaggle.
2. Ghép theo track ID; đo tỷ lệ bài và tỷ lệ lượt xuất hiện ghép được.
3. Lọc: bỏ bài xuất hiện dưới 5 lần, bỏ playlist còn dưới 10 bài sau khi lọc.
4. Chuẩn hóa: tempo thành BPM, key + mode thành mã Camelot (1A–12B), energy trong khoảng 0–1.
5. Chia theo playlist: 80/10/10 cho train/val/test. Với playlist val/test, **seed là 5 bài đầu, bài đích là bài thứ 6** (ADR 0003). Bài trong seed bị loại khỏi ứng viên.

**Đã chốt:** dùng bộ features `krishsharma0413`; ngày 1 đo độ phủ thật (tỉ lệ bài và tỉ lệ lượt xuất hiện), nếu thấp thì chuyển sang bộ dự phòng.

## 5. Kiến trúc hệ thống

Hệ thống có hai tầng: huấn luyện offline một lần, và gợi ý online theo phiên nghe.

- **Offline (Kaggle):** MPD + features → ghép, lọc, chia → mô hình gu (Popularity, item-kNN liền kề, ALS, item2vec, SASRec) → top-100 ứng viên mỗi phiên → re-rank theo α với điểm mix theo luật hoặc học → đánh giá (độ chính xác, độ mượt, đường đánh đổi). Mỗi bước xuất ra một Kaggle Dataset có phiên bản; notebook clone repo và gọi hàm trong `src/`.
- **Online (Streamlit, laptop, không cần mạng):** đọc `artifacts/` (bảng bài, embedding bài, tham số mix) → tính vector phiên từ seed → top-100 bằng numpy → re-rank theo α → hàng đợi kèm giải thích. App không cần torch.

Mô hình gu chỉ chọn ra 100 ứng viên; điểm mix chỉ được dùng để xếp lại, nên hệ thống không bao giờ gợi ý bài mượt nhưng lạc gu.

## 6. Mô hình và thuật toán

Mô hình gu lấy top-100 ứng viên, sau đó điểm mix xếp lại để chọn top-10.

**Baseline**

- Popularity: bài phổ biến nhất trong tập train.
- Item-kNN liền kề: đếm cặp có hướng a→b trong train, gợi ý những bài hay đứng ngay sau bài cuối của seed. Nếu chưa đủ 100 ứng viên thì lấy thêm theo đồng xuất hiện trên cả playlist, rồi đến Popularity.

**Mô hình gu**

- ALS (thư viện implicit) trên ma trận playlist × bài.
- item2vec: Word2Vec (gensim) trên chuỗi bài của mỗi playlist.
- Vector phiên = Σ decay^(khoảng cách tới bài cuối) · vector bài, dùng chung cho ALS (vector bài ALS) và item2vec; decay chọn trên val trong {0.5, 0.7, 0.9, 1.0}. Điểm gu = tích vô hướng giữa vector phiên và vector bài.

**Điểm mix (theo luật)**

- BPM: giống nhau trong khoảng ±6% thì gần 1, giảm dần khi lệch xa. Tính cả trường hợp nhịp gấp đôi hoặc một nửa.
- Tông theo vòng Camelot: cùng mã = 1; lệch ±1 số cùng chữ, hoặc cùng số khác chữ = 0.8; còn lại = 0.
- Năng lượng: 1 − |Δenergy|.

```latex
\text{mix} = w_b \, s_{bpm} + w_k \, s_{key} + w_e \, s_{energy}
```

```latex
\text{score} = \alpha \cdot \text{gu}_{norm} + (1 - \alpha) \cdot \text{mix}
```

- gu\_norm: chuẩn hóa min-max trong top-100 ứng viên để cùng thang 0–1 với điểm mix.
- Trọng số khởi đầu w\_b = 0.4, w\_k = 0.4, w\_e = 0.2.
- α quét từ 0 đến 1, bước 0.1.
- **Chọn (w, α) chung bằng grid search** trên một mẫu khoảng 2.000 playlist val: lưới trọng số bước 0.2 (21 bộ, tổng bằng 1) × 11 giá trị α. Chọn cặp có **tỉ lệ cặp mix được** cao nhất với điều kiện NDCG@10 giảm ≤ 10% (tương đối) so với α = 1. Cặp mix được: ΔBPM ≤ 6% (tính cả ×2/×0.5) và tông tương thích Camelot. Điểm mix học không có w, chỉ chọn α theo cùng quy tắc.

**Mô hình tuần tự (bắt buộc)**

- SASRec viết bằng PyTorch, chuyển từ `pmixer/SASRec.pytorch` (Apache-2.0), không dùng RecBole (ADR 0002). Huấn luyện trên chuỗi bài của mỗi playlist, độ dài tối đa 50.
- Dùng như mô hình gu thứ ba, cũng sinh top-100 ứng viên. So với ALS và item2vec để biết thứ tự bài trong phiên có giúp không.

**Điểm mix học từ dữ liệu (bắt buộc, tuần 3)**

- Ý tưởng: hai bài đứng liền nhau trong playlist thật (chỉ lấy từ train) là tín hiệu "chuyển được". Học hai bộ mã hóa nhỏ (MLP) từ đặc trưng âm thanh: f\_out cho bài đang phát, f\_in cho bài kế tiếp.
- Đầu vào x giống hệt luật: BPM, tông (Camelot, từ key + mode), energy. Nhờ vậy so sánh với luật là công bằng.
- Cặp liền nhau là mẫu dương; các bài khác trong cùng batch là mẫu âm (InfoNCE).
- Chỉ dùng đặc trưng âm thanh, không dùng ID bài, nên chấm được cả bài mới chưa xuất hiện trong playlist nào.
- Câu hỏi phụ cho phần thảo luận: luật của DJ (tri thức chuyên gia) và điểm học từ hành vi người dùng có đồng ý với nhau không?
- Giới hạn cần nêu: cặp liền nhau phản ánh cả gu lẫn độ chuyển bài, nên điểm học được không hoàn toàn là "độ mượt".

```latex
s_{learn}(a, b) = f_{out}(x_a)^\top f_{in}(x_b)
```

```latex
\mathcal{L} = -\log \frac{\exp\left(s_{learn}(a, b^{+}) / \tau\right)}{\sum_{j} \exp\left(s_{learn}(a, b_j) / \tau\right)}
```

## 7. Đánh giá

Đánh giá đo hai trục là độ chính xác và độ mượt, rồi vẽ đường đánh đổi giữa chúng theo α.

| Nhóm | Chỉ số | Cách đo |
| --- | --- | --- |
| Độ chính xác | HitRate@10, NDCG@10, MRR | Bài đích (bài thứ 6) của playlist test, seed là 5 bài đầu |
| Độ mượt | ΔBPM trung bình (%, tính cả ×2/×0.5), tỷ lệ chuyển tông hợp, \|Δenergy\| trung bình, tỉ lệ cặp mix được | Sinh tham lam 10 bài từ seed: chọn top-1 theo điểm cuối, nối vào phiên, tính lại ứng viên. Đo 10 cặp liền kề, kể cả (bài cuối seed → bài 1). Baseline và α = 1 sinh theo cùng cách |
| Đánh đổi | NDCG@10 theo độ mượt | Quét α từ 0 đến 1 |
| Người nghe (mở rộng) | Điểm liền mạch và hợp gu, thang 1–5 | 10–15 người nghe mù 2 playlist qua link Spotify |

NDCG gần như chắc chắn giảm khi α nhỏ, vì playlist thật không được xếp theo kiểu DJ. Báo cáo cần nói rõ đây là cái giá đo được, không phải lỗi mô hình.

Chỉ số ΔBPM và tông trùng công thức với điểm mix theo luật, nên luật "thắng" gần như theo định nghĩa. Báo cáo phải nêu điều này; điểm mix học không biết luật, nên nếu nó vẫn cải thiện độ mượt thì đó là bằng chứng không bị vòng lặp.

**Tiêu chí thành công:** có một cặp (w, α) mà tỉ lệ cặp mix được cải thiện rõ so với α = 1 trong khi NDCG@10 giảm ≤ 10% (tương đối). Ngưỡng 10% có thể chỉnh sau khi có số liệu baseline ở tuần 1.

**Bảng kết quả chính:** Popularity, item-kNN liền kề, ALS, item2vec, SASRec (chỉ gu); mô hình gu tốt nhất (theo NDCG@10 trên val) + mix theo luật ở (w, α) đã chọn; mô hình gu tốt nhất + mix học ở α đã chọn. Mỗi cấu hình có 3 lần chạy (`random_seed` khác nhau), báo trung bình ± độ lệch chuẩn trên test.

**Ablation:** bỏ lần lượt BPM, tông, năng lượng khỏi điểm mix để biết thành phần nào đóng góp nhiều nhất vào độ mượt.

## 8. Demo

Demo là một app Streamlit dạng trình phát nhạc. Phần ăn điểm nhất là thanh trượt α, vì nó cho thấy sự đánh đổi ngay khi kéo.

1. **Bài đang phát:** tên bài, BPM, mã Camelot, năng lượng, và vòng Camelot tô sáng các tông mix được.
2. **Hàng đợi gợi ý:** top-10, mỗi bài có điểm gu, điểm mix và một dòng giải thích, ví dụ "cùng tông 8A, BPM 124→126".
3. **Thanh trượt α:** kéo giữa "đúng gu" và "mix mượt", danh sách đổi ngay.
4. **So sánh:** playlist baseline đặt cạnh playlist MixNext, kèm biểu đồ BPM và năng lượng theo thứ tự bài.
5. **Chọn đầu vào:** chọn một playlist có sẵn trong tập test, hoặc tự chọn 3 bài để xử lý cold-start. App tính ứng viên trực tiếp từ embedding bài của mô hình tốt hơn giữa ALS và item2vec (vector phiên + numpy); SASRec chỉ dùng trong đánh giá offline.
6. **Phát chuyển bài (mở rộng):** crossfade đoạn cuối bài này sang đầu bài kế, chỉ với vài bài mẫu có audio.

## 9. Công nghệ và cấu trúc repo

Huấn luyện chạy trên Kaggle (CPU là đủ cho ALS và item2vec); app Streamlit chỉ đọc các file kết quả đã lưu.

- **Ngôn ngữ:** Python 3.10+
- **Dữ liệu:** pandas hoặc polars, lưu trung gian dạng parquet
- **Môi trường huấn luyện:** Kaggle (Python 3.13 tại thời điểm 11/2026); phiên bản gói chốt trong `requirements.txt` sau khi chạy thử
- **Mô hình:** implicit (ALS), gensim (item2vec), PyTorch (SASRec độc lập và mix học), numpy, scipy
- **Demo:** Streamlit, Plotly
- **Dự phòng:** librosa để tự trích BPM và tông
- **Quản lý:** GitHub (`yashita23/MusicRecSys`); ticket là GitHub Issues gắn milestone M1/M2/M3, label `owner:A/B/C` và quan hệ "blocked by" giữa các issue
- **Nhánh:** chỉ 3 nhánh cố định `a`, `b`, `c` (A → `a`, B → `b`, C → `c`), không tạo nhánh theo ticket. Xong một ticket thì mở PR từ nhánh của mình vào `main`, mô tả ghi `Closes #N`. Sau mỗi lần merge, hai người còn lại merge `main` về nhánh mình, nhờ vậy ticket đang bị chặn của họ được mở và họ có code mới nhất
- **Bàn giao dữ liệu:** mỗi bước pipeline xuất ra một Kaggle Dataset riêng tư có phiên bản; định dạng file chốt ở `docs/data-contract.md`

```text
mixnext/             # = thư mục gốc của repo MusicRecSys
├── data/            # raw, processed (parquet) — không commit
├── src/
│   ├── data/        # ghép, lọc, chia train/val/test
│   ├── features/    # BPM, Camelot, energy
│   ├── models/      # popularity, item_knn, als, item2vec, sasrec
│   ├── rerank/      # mix_score, combine theo α
│   └── eval/        # metrics độ chính xác + độ mượt, quét α
├── notebooks/       # EDA, thí nghiệm trên Kaggle
├── artifacts/       # embedding, bảng features cho app
├── app/             # Streamlit
└── reports/         # hình, bảng kết quả, báo cáo
```

## 10. Phân công

Mỗi người sở hữu một tầng của hệ thống và đều code; tên thành viên điền vào cột đầu.

| Thành viên | Phụ trách | Đầu ra chính |
| --- | --- | --- |
| A | Dữ liệu và điểm mix | Pipeline ghép/lọc, bảng features, mix theo luật, mix học bằng InfoNCE (PyTorch), chỉ số độ mượt |
| B | Mô hình gu và đánh giá | Baseline, ALS, item2vec, SASRec (PyTorch), HitRate/NDCG/MRR, đường đánh đổi, grid search (w, α), ablation, 3 lần chạy |
| C | Re-rank, demo, khảo sát | Module kết hợp theo α, giải thích gợi ý, app Streamlit, khảo sát người nghe nếu kịp |

Cả nhóm cùng viết báo cáo; mỗi người viết phần mình phụ trách. Họp ngắn 2 lần mỗi tuần để ghép code.

## 11. Kế hoạch 3 tuần

Chỉ còn 3 tuần, nên áp dụng phương án gộp tuần 3 và 4: bỏ khảo sát người nghe khỏi MVP, viết báo cáo song song từ tuần 2. Mỗi mốc là một milestone trên GitHub.

| Mốc | Hạn | Phải xong |
| --- | --- | --- |
| M1 | CN 29/11 | Đo độ phủ ghép; môi trường Kaggle và `requirements.txt`; định dạng file bàn giao; dữ liệu đã lọc và chia; metric độ chính xác; 2 baseline; đề xuất PDF; prototype app trên dữ liệu giả |
| M2 | CN 6/12 | Mix theo luật; chỉ số độ mượt; ALS, item2vec, SASRec; re-rank; sinh chuỗi tham lam; đường đánh đổi đầu tiên (**chạy end-to-end**); demo v1; khung báo cáo |
| M3 | CN 13/12 | Mix học InfoNCE và so với luật; grid search (w, α); bảng kết quả 3 lần chạy; ablation; demo v2; báo cáo, slide, gói nộp |

Nếu trễ: cắt demo v2 trước (giữ demo v1), rồi đến ablation; giữ SASRec và mix học vì cả hai là bắt buộc.

## 12. Rủi ro

Rủi ro lớn nhất là dữ liệu ghép không đủ, nên mốc kiểm tra đầu tiên đặt ngay cuối tuần 1.

| Rủi ro | Dấu hiệu | Dự phòng |
| --- | --- | --- |
| Tỷ lệ ghép thấp | Phần lớn bài trong MPD không có features | Đổi sang bộ features khác; nếu vẫn thấp, dùng FMA + librosa và tự tạo phiên nghe |
| Không truy cập được MPD | Không tải được bộ dữ liệu gốc | Dùng một bộ playlist Spotify có sẵn trên Kaggle |
| Hết RAM trên Kaggle | Notebook bị khởi động lại khi ghép hoặc train | Giảm số playlist, dùng polars, lưu parquet theo phần |
| NDCG giảm mạnh khi re-rank | Đường đánh đổi dốc ở mọi α | Tăng số ứng viên, chỉnh trọng số mix, trình bày như kết quả đánh đổi |
| Trễ tiến độ | Pipeline chưa chạy end-to-end cuối tuần 2 | Cắt toàn bộ phần mở rộng, giữ MVP |

## 13. Báo cáo và trình bày

Báo cáo PDF theo đúng 6 mục trong L0; bài trình bày 15 phút dùng cùng số liệu và hình với báo cáo.

| Mục báo cáo theo L0 | Nội dung của MixNext |
| --- | --- |
| Giới thiệu và mô tả bài toán thực tế | Phát nhạc liên tục (tự phát tiếp, nhạc nền quán, tiệc, tập gym): bài kế tiếp hợp gu nhưng lệch nhịp, lệch tông, tụt năng lượng; định nghĩa bài toán hai mục tiêu |
| Chi tiết phương pháp | Hai tầng: sinh ứng viên theo gu, xếp lại theo điểm mix; công thức điểm mix theo luật và học từ dữ liệu; cách chia tập và đánh giá |
| Chức năng hệ thống và cách dùng | Ảnh chụp từng màn hình demo, các bước dùng |
| Phương pháp, phần mềm, dữ liệu dùng lại | Bảng: MPD (bản sao Kaggle), bộ đặc trưng âm thanh, implicit, gensim, SASRec.pytorch (pmixer), PyTorch, librosa, Streamlit, Plotly; bài báo gốc ALS, item2vec, SASRec, InfoNCE; ghi rõ phần nào nhóm tự viết |
| Khó khăn và cách vượt qua (bổ sung) | MPD chính thức bị gỡ; RecBole không chạy trên Kaggle Python 3.13 |
| Khó khăn và cách vượt qua | Tỉ lệ ghép đặc trưng âm thanh; API Spotify bị khóa; RAM trên Kaggle; NDCG giảm khi re-rank |
| Tranh luận, kết luận, hướng phát triển | Giá của độ mượt đo được; luật DJ so với điểm học từ dữ liệu; vì sao NDCG không đo hết giá trị của chuyển bài mượt; hướng phát triển: đường cong năng lượng, cá nhân hóa theo người dùng dài hạn |

**Chia phần trình bày, mỗi người khoảng 5 phút:**

1. A: bài toán, dữ liệu, điểm mix theo luật và học từ dữ liệu.
2. B: mô hình gu, bảng kết quả, đường đánh đổi.
3. C: demo trực tiếp, kết luận, giới hạn, slide khai báo phần dùng lại.

## 14. Bản nháp đề xuất đề tài (nộp dạng PDF 1–2 trang)

**Tên đề tài:** MixNext – Gợi ý bài hát tiếp theo vừa hợp gu vừa chuyển bài mượt (Transition-aware Playlist Continuation).

**1. Bài toán thực tế.** Khi nghe nhạc liên tục, như tính năng tự phát tiếp của ứng dụng nghe nhạc, nhạc nền ở quán cà phê, buổi tiệc hay phòng tập, hệ thống chọn bài tiếp theo chủ yếu theo gu. Bài kế tiếp có thể hợp gu nhưng chênh nhịp, lệch tông hoặc tụt năng lượng đột ngột, làm mạch nghe bị đứt.

- **Mục đích:** gợi ý bài tiếp theo vừa đúng gu của phiên nghe, vừa nối mượt với bài đang phát.
- **Yêu cầu:** gợi ý ngay từ vài bài đầu của phiên; mỗi gợi ý có lý do; người dùng chỉnh được mức ưu tiên giữa "đúng gu" và "mượt".
- **Kịch bản:** người dùng chọn vài bài đang nghe; hệ thống trả hàng đợi 10 bài, mỗi bài kèm lý do như "cùng tông 8A, BPM 124 lên 126". Khi mở nhạc cho buổi tiệc, người dùng kéo thanh trượt về phía "mượt" và hàng đợi thay đổi ngay.

**2. Phương pháp dự kiến.** Hệ gợi ý lai hai tầng.

- Tầng 1 sinh 100 ứng viên theo gu bằng lọc cộng tác (item-kNN, ALS), item2vec và mô hình tuần tự SASRec, học từ dữ liệu playlist.
- Tầng 2 xếp lại theo điểm chuyển bài. Có hai cách tính: luật của DJ (độ lệch BPM, vòng tông Camelot, năng lượng), và mô hình học độ chuyển bài từ các cặp bài liền nhau bằng học tương phản.
- Điểm cuối = α × điểm gu + (1 − α) × điểm chuyển bài.
- Đánh giá bằng HitRate@10, NDCG@10, MRR và các chỉ số độ mượt (độ lệch BPM, tỉ lệ chuyển tông hợp, độ lệch năng lượng). Kết quả chính là đường đánh đổi giữa độ chính xác và độ mượt khi thay đổi α.

**3. Dữ liệu.** Spotify Million Playlist Dataset (RecSys Challenge 2018), ghép với đặc trưng âm thanh (tempo, tông, năng lượng) từ bộ dữ liệu công khai theo mã bài hát.

**4. Đầu vào và đầu ra.**

- Đầu vào: danh sách một hoặc vài bài đang nghe (seed); tham số α từ 0 đến 1; số bài cần gợi ý K.
- Đầu ra: danh sách K bài xếp hạng, mỗi bài có điểm gu, điểm chuyển bài và một dòng giải thích. Ở chế độ tạo playlist: chuỗi K bài liên tiếp kèm biểu đồ BPM và năng lượng theo thứ tự bài.

**5. Thành viên.**

| Họ tên | Mã số sinh viên | Email |
| --- | --- | --- |
| \[Thành viên A\] | \[MSSV\] | \[Email\] |
| \[Thành viên B\] | \[MSSV\] | \[Email\] |
| \[Thành viên C\] | \[MSSV\] | \[Email\] |
