# Định dạng file bàn giao

Tài liệu này mô tả schema của mọi file đi qua tay nhiều người trong pipeline MixNext. `src/data/schema.py` kiểm tra các schema này, `src/data/fake.py` sinh dữ liệu giả theo đúng schema. Thuật ngữ theo `GLOSSARY.md`.

Muốn đổi schema thì mở PR sửa cả ba nơi (tài liệu này, `schema.py`, `fake.py`), và cả ba thành viên phải đồng ý.

## Bố cục thư mục

**Dữ liệu thật trên Kaggle:** [`titeohihi/mixnext-splited`](https://www.kaggle.com/datasets/titeohihi/mixnext-splited) (output của notebook 03, #7). Bộ này có đủ ba bảng chính, `split_report.json` và các file trung gian của #5. Muốn dùng trong notebook thì Add Input → Datasets → `mixnext-splited`. Bộ `mixnext-processed` chỉ chứa output của #5, và Kaggle không cho thêm version mới, nên các ticket sau không dùng nó.

```text
<root>/                                  # Kaggle Dataset titeohihi/mixnext-splited, hoặc data/processed/ ở máy
├── tracks.parquet
├── sessions.parquet
├── eval_targets.parquet
├── split_report.json                    # báo cáo chia tập, xem mục split_report.json
└── sessions_raw.parquet, tracks_raw.parquet, build_log.json   # file trung gian của A, giữ lại từ v1

<root>/                                  # ở máy hoặc dữ liệu giả (python -m src.data.fake); trên Kaggle xem bên dưới
├── candidates/{model}_{split}_run{r}.parquet
└── item_embeddings/{model}.npy, {model}_tracks.parquet
```

- `tracks`, `sessions`, `eval_targets` được tạo bằng `python -m src.data.split --raw-dir <thư mục #5> --out <thư mục ra>` (#6, #7). Mỗi phiên được gán vào một split theo hoán vị ngẫu nhiên của `session_id` với `random_seed` = 42, tỉ lệ 80/10/10.
- `model` ∈ `popularity`, `itemknn`, `als`, `item2vec`, `sasrec` (dữ liệu giả dùng `fake`).
- `split` ∈ `val`, `test`.
- `r` là số thứ tự lần chạy: 0, 1, 2.

**Output của các notebook mô hình (04–07) trên Kaggle.** Mỗi notebook đăng một Kaggle Dataset riêng, vì Dataset tạo từ output của một notebook chỉ nhận version mới từ chính notebook đó. Notebook 04 chứa cả hai baseline. Mọi file trong các bộ này nằm dưới thư mục `out/`. Ứng viên của mô hình có embedding (ALS, item2vec) nằm chung bộ với embedding của nó.

```text
mixnext-candidates   (notebook 04, #9)
└── out/candidates/{popularity,itemknn}_{val,test}_run0.parquet     # hai baseline tất định, chỉ có lần chạy 0

mixnext-embeddings   (notebook 05, #14 — ALS), rồi được chép vào artifacts/ cho app (#20)
└── out/
    ├── candidates/als_{val,test}_run{r}.parquet
    ├── item_embeddings/als.npy, als_tracks.parquet                  # chỉ ghi ở lần chạy 0, tên không có số lần chạy
    └── als_params.json                                              # xem mục {model}_params.json

mixnext-item2vec     (notebook 06, #15)
└── out/
    ├── candidates/item2vec_{val,test}_run{r}.parquet
    ├── item_embeddings/item2vec.npy, item2vec_tracks.parquet        # chỉ ghi ở lần chạy 0
    └── item2vec_params.json

mixnext-sasrec       (notebook 07, #16)
└── out/
    ├── candidates/sasrec_{val,test}_run{r}.parquet
    └── sasrec/sasrec_run{r}.pt, sasrec_run{r}.json                  # checkpoint PyTorch và báo cáo huấn luyện

mixnext-mix-learned  (notebook 08, #23), rồi được chép vào artifacts/ cho app (#20)
└── out/
    ├── mix_learned.npz                                              # xem mục mix_learned.npz
    └── mix_learned.json                                             # báo cáo huấn luyện

mixnext-artifacts    (notebook 14, #20): tải về rồi giải nén vào artifacts/ của repo
└── artifacts/                                                       # xem mục artifacts/
```

- Lần chạy 1, 2 (#26): chưa chốt cách đăng. Version mới của một Dataset thay **toàn bộ** file của version cũ, nên hoặc notebook phải ghi lại cả file của lần chạy 0, hoặc mỗi lần chạy dùng một Dataset riêng. #26 quyết định và cập nhật mục này.
- Trong notebook, Add Input các bộ cần dùng, rồi tìm file theo tên bằng `Path("/kaggle/input").rglob("<tên file>")`. Đừng ghi cứng đường dẫn, vì thư mục mount và tiền tố `out/` có thể khác nhau giữa các bộ.
- Khi tạo Dataset từ output, chỉ giữ `out/`. Bỏ thư mục `MusicRecSys` mà bước bootstrap đã clone vào `/kaggle/working`.
- Tiền tố `out/` chỉ có trên Kaggle. Ở máy và với dữ liệu giả, `candidates/` và `item_embeddings/` nằm thẳng dưới thư mục gốc (cây ở trên). Schema các bảng giống hệt nhau.

## tracks.parquet

Danh mục bài. Mỗi bài một dòng.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `track_id` | string | Spotify ID 22 ký tự, không có tiền tố `spotify:track:`; duy nhất |
| `track_name` | string | |
| `artist_name` | string | |
| `tempo` | float | Giá trị thô từ bộ features, > 0 |
| `bpm` | float | BPM mà điểm mix dùng, > 0. Hiện bằng `tempo` (tempo của Spotify vốn đã là BPM); nếu #6 đổi cách tính thì phải cập nhật dòng này |
| `key` | int | 0–11 (0 = C, 1 = C♯/D♭, …) |
| `mode` | int | 0 = thứ, 1 = trưởng |
| `camelot` | string | `1A`–`12B`, phải bằng `camelot_code(key, mode)` trong `src/features/camelot.py` (C trưởng = 8B, A thứ = 8A) |
| `energy` | float | 0–1; giữ nguyên giá trị thô (energy của Spotify vốn đã trong 0–1) |
| `n_occurrences` | int | Số lượt xuất hiện **trong các phiên train**, ≥ 0 |

Không cột nào được null. Ở bước lọc (#5), những bài sau bị loại như thể không ghép được: thiếu features, tempo ≤ 0, không xác định được tông (`key` ngoài 0–11, gồm cả −1), `mode` ∉ {0, 1}, hoặc `energy` ngoài 0–1. Đây là `usable_mask` trong `src/data/coverage.py`. Các bài này chiếm khoảng 0,11% số bài và 0,03% số lượt (ADR 0001). Đến bước tạo `tracks.parquet` (`make_tracks` trong `src/features/audio.py`, #6), nếu còn bài nào như vậy thì đó là lỗi của #5: hàm **báo lỗi** (nêu tên vài bài) chứ không tự bỏ, vì tự bỏ sẽ làm phiên trỏ tới bài không còn trong `tracks`. Hàm cũng báo lỗi khi bài trong `sessions` không có trong `tracks_raw`, và kiểm tra kết quả bằng `validate_tracks` trước khi trả về. `fake.py` dựng bảng `tracks` qua cùng hàm này.

## sessions.parquet

Mỗi lượt xuất hiện của một bài trong một phiên là một dòng.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `session_id` | int | `pid` của MPD |
| `split` | string | `train`, `val` hoặc `test`; mỗi phiên chỉ thuộc một tập |
| `pos` | int | 0, 1, …, n−1, liền mạch trong từng phiên (đánh số lại sau khi lọc) |
| `track_id` | string | Phải có trong `tracks.parquet` |

- `(session_id, pos)` là duy nhất.
- Mỗi phiên có ≥ 10 bài.
- Một bài có thể xuất hiện nhiều lần trong cùng một phiên.

## eval_targets.parquet

Seed và bài đích của các phiên val/test (ADR 0003). Mỗi phiên một dòng.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `session_id` | int | Duy nhất; là phiên val/test trong `sessions.parquet` |
| `split` | string | `val` hoặc `test`; trùng với `split` của phiên đó |
| `seed_track_ids` | list[string] | Đúng 5 bài, chính là `pos` 0–4 của phiên |
| `target_track_id` | string | Chính là `pos` 5 của phiên; không nằm trong seed |

- Mọi phiên val/test đều phải có mặt, trừ phiên có bài đích trùng một bài trong seed. Những phiên đó bị **bỏ**, vì bài trong seed bị loại khỏi ứng viên nên không mô hình nào đoán được.
- Phiên có bài đích không có trong train thì được **giữ** và tính là trượt cho mọi mô hình. #7 báo cáo tỉ lệ của những phiên này.

## split_report.json

Báo cáo đi kèm ba bảng trên, do `src/data/split.py` (`SplitReport`) ghi ra. `python -m src.data.fake` cũng ghi file này. Đây là báo cáo chứ không phải dữ liệu, nên `schema.py` không có validator cho nó.

| khóa | ý nghĩa |
| --- | --- |
| `random_seed` | `random_seed` dùng để chia tập |
| `n_sessions` | `{train, val, test}` → số phiên |
| `n_eval_targets` | `{val, test}` → số phiên trong `eval_targets` |
| `n_dropped_target_in_seed` | `{val, test}` → số phiên bị bỏ vì bài đích nằm trong seed |
| `share_target_unseen_in_train` | `{val, test}` → tỉ lệ bài đích không có trong train (luôn tính là trượt) |
| `fingerprint` | sha256 của `tracks`, `sessions`, `eval_targets` theo đúng thứ tự dòng. Chạy lại ra cùng giá trị nghĩa là ra cùng dữ liệu |

## {model}_params.json

Tham số mà lần chạy 0 đã chọn trên val (`src/models/embedding_runner.py`). Các lần chạy sau đọc lại file này qua `--params` thay vì chọn lại.

| khóa | ý nghĩa |
| --- | --- |
| `params` | Bộ tham số đã chọn (NDCG@10 val cao nhất), gồm cả tham số học (`factors`, `window`, …) và tham số chấm (`decay`, `normalize`) |
| `grid` | Toàn bộ lưới đã thử: mỗi dòng là một bộ tham số kèm `hit_rate`, `ndcg`, `mrr` trên val |

`sasrec_run{r}.json` cũng có `params`, kèm thêm `run`, `device`, `history` (NDCG mẫu val theo epoch) và `val`. Đây là báo cáo chứ không phải dữ liệu, nên `schema.py` không có validator cho các file này.

## mix_learned.npz và mix_learned.json

Trọng số của điểm mix học (#23), do `python -m src.rerank.mix_learned_train` ghi ra. App đọc bằng `LearnedMix.load(tracks, path)` trong `src/rerank/mix_learned.py`, chỉ cần numpy.

- `mix_learned.npz`: `f_out.{i}.weight`, `f_out.{i}.bias`, `f_in.{i}.weight`, `f_in.{i}.bias` (float32, lớp thứ i; weight có shape `(out, in)` như `torch.nn.Linear`), `temperature` (τ) và `feature_names`. `feature_names` phải trùng `FEATURE_NAMES` trong code, nếu không `LearnedMix.load` báo lỗi.
- `mix_learned.json`: `params`, `run`, `feature_names`, `n_train_pairs`, `n_val_pairs`, `best_epoch`, `val` (loss và Recall@10 trong batch của epoch tốt nhất, kèm mức ngẫu nhiên), `history` (epoch 0 là bộ mã hóa chưa học) và `seconds`. Đây là báo cáo chứ không phải dữ liệu, nên `schema.py` không có validator cho file này.

## candidates/{model}_{split}_run{r}.parquet

Top-100 ứng viên mà một mô hình gu chọn ra cho mỗi phiên trong `eval_targets` của `split`.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `session_id` | int | Có trong `eval_targets` với đúng `split` của file |
| `rank` | int | 1–100, liền mạch trong mỗi phiên |
| `track_id` | string | Có trong `tracks.parquet`; không trùng nhau trong một phiên; không nằm trong seed |
| `taste_score` | float | Điểm gu, hữu hạn, không tăng theo `rank` (rank 1 có điểm cao nhất) |

Mọi phiên trong `eval_targets` của `split` đều phải có ứng viên, mỗi phiên đúng 100 dòng. Bước xếp lại tự chuẩn hóa `taste_score` thành `gu_norm`, nên mô hình không cần chuẩn hóa trước.

## item_embeddings/{model}.npy và {model}_tracks.parquet

Vector bài của các mô hình có embedding (ALS, item2vec). App và bước tính vector phiên dùng các file này.

- `{model}.npy`: mảng `float32` có shape `(n, d)`, mọi phần tử hữu hạn.
- `{model}_tracks.parquet`: cột `row` (int, 0…n−1, liền mạch) và cột `track_id` (string, duy nhất, có trong `tracks.parquet`). Hàng `row` = i ứng với vector thứ i.

Mỗi mô hình có index riêng và chỉ chứa những bài nó đã học được, tức là bài có xuất hiện trong train. Không có vector 0 lấp chỗ trống.

## artifacts/

Mọi file app đọc (#20), do `python -m src.export_artifacts` ghi ra; app đọc bằng `load_artifacts` trong `src/export_artifacts.py`, chỉ cần numpy và pandas. Bên ghi (#20) và bên đọc (app, #21, #28) đều là của C, nên luật "sửa cả ba nơi" ở đầu tài liệu không áp dụng: định dạng định nghĩa và kiểm tra trong `src/export_artifacts.py` (`load_artifacts` kiểm tra `tracks`, vector và index khớp nhau) và `tests/test_export_artifacts.py`. Riêng `mix_params.json` có bên ghi thứ hai là #25 (B), nên định dạng của nó ghi ở đây.

- `tracks.parquet`: schema `tracks`, gồm mọi bài có vector và mọi bài của playlist demo.
- `item_embeddings.npy`, `track_index.parquet`: như `item_embeddings/{model}.npy` và `{model}_tracks.parquet` của mô hình được chọn (ALS hoặc item2vec, NDCG@10 val cao nhất trong `{model}_params.json`), thứ tự hàng giữ như file gốc. Mọi bài trong `track_index` có trong `tracks.parquet`.
- `demo_playlists.parquet`: `session_id`, `seed_track_ids`, `target_track_id` của 100 phiên test chọn theo `random_seed` 42 (như `eval_targets`, bỏ cột `split`).
- `mix_params.json`: cấu hình mix app dùng (`MixParams`). Có file này trong `--input` (ví dụ output của #25) thì chép theo; không có thì ghi giá trị mặc định: trọng số của #12 và α = 0,7 của #19 (`reports/sweep_v1.csv`).

  ```json
  {"rule": {"weights": {"bpm": 0.4, "key": 0.4, "energy": 0.2}, "alpha": 0.7},
   "learned": {"alpha": null},
   "source": "mặc định: trọng số của #12, α của #19 (reports/sweep_v1.csv)"}
  ```

  `learned.alpha` là α của điểm mix học (#25 chọn; `null` thì app dùng `rule.alpha`). Mọi α trong 0–1.
- `mix_learned.npz`: chép từ `mixnext-mix-learned` nếu có, xem mục `mix_learned.npz`. Không có thì file cũ trong thư mục ra bị xóa.
- `manifest.json`: báo cáo, không phải cấu hình. `model`, `decay`, `mix_learned` (có `mix_learned.npz` hay không), `n_tracks`, `n_tracks_in_index`, `dim`, `n_demo`, `params` (tham số đã chọn của mô hình), `val_ndcg` (NDCG@10 val tốt nhất theo mô hình), `random_seed`, `n_tracks_with_vector`, `n_dropped` (số bài có vector bị cắt), `bytes` (tổng dung lượng các file dữ liệu).

Tổng dung lượng phải dưới 200 MB (200.000.000 byte). Script cắt danh mục theo dung lượng ước lượng (bỏ bài ít lượt nhất, giữ bài của playlist demo), rồi kiểm tra dung lượng thật sau khi ghi và báo lỗi nếu vẫn vượt.

Phiên không có bài nào có vector thì app lấy ứng viên từ Popularity dựng theo cột `n_occurrences` của `tracks.parquet`, cho cùng kết quả với Popularity học trên các phiên train (khi danh mục không bị cắt). Playlist baseline của demo (#28) là chuỗi với α = 1, tính được từ chính các file này.

## File trung gian của A

Chỉ dùng giữa các ticket của A (#5 → #6, #7). B và C không đọc các file này, nên luật "sửa cả ba nơi" ở đầu tài liệu không áp dụng: schema của chúng định nghĩa trong `src/data/build.py` (`RAW_TRACK_COLUMNS`) và được kiểm tra ở `tests/test_build.py`.

| file | do ai tạo | cột |
| --- | --- | --- |
| `sessions_raw.parquet` | #5 | `session_id`, `pos`, `track_id`. Như `sessions.parquet` nhưng chưa có `split`; đã lọc (mỗi bài ≥ 5 lượt, mỗi phiên ≥ 10 bài, lặp đến khi ổn định) và đánh số lại `pos` |
| `tracks_raw.parquet` | #5 | `track_id`, `track_name`, `artist_name`, `tempo`, `key`, `mode`, `energy`. Đúng các bài còn trong `sessions_raw`, mọi bài đều dùng được |
| `build_log.json` | #5 | Số phiên, số bài, số lượt sau từng bước lọc, kèm cấu hình (`n_slices`, `random_seed`) |

## Bộ phụ last.fm (mở rộng #54–#57)

Bộ dữ liệu thứ hai cho phần điểm mix, từ Music4All-Onion (Moscati và cộng sự, CIKM 2022; Zenodo record 15394646, CC BY 4.0). Tách hẳn khỏi pipeline MPD: schema của MPD ở trên không đổi, `validate_dataset` không dùng cho bộ này.

Chỉ dùng giữa các ticket #54–#57 (B). Không phải schema của MPD, không qua `src/data/schema.py` và không có trong `src/data/fake.py`, nên luật "sửa cả ba nơi" ở đầu tài liệu không áp dụng: schema định nghĩa và kiểm tra ở `src/data/lastfm.py` (`validate_lastfm_sessions`) và `tests/test_lastfm.py`.

```text
Kaggle Dataset mixnext-onion (notebook 09, #54)
└── out/onion/listening_events.parquet, essentia.parquet, import_log.json, README.md

Kaggle Dataset mixnext-lastfm (notebook 10, #55; #56 thêm tracks.parquet)
└── out/lastfm/sessions.parquet, lastfm_log.json
```

### lastfm/sessions.parquet

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `session_id` | int | 0…n−1, theo thứ tự (user, thời điểm bắt đầu phiên) |
| `user_id` | int | user của last.fm |
| `split` | string | `train`, `val` hoặc `test`, chia **theo user** 80/10/10, `random_seed` = 42: mọi phiên của một user cùng một split |
| `pos` | int | 0…n−1 liền mạch trong từng phiên |
| `track_id` | string | ID 16 ký tự của Music4All-Onion (không phải Spotify ID); có đặc trưng Essentia với `rhythm.bpm` > 0 |

- Phiên mới khi khoảng nghỉ giữa hai lượt nghe liên tiếp của một user > 30 phút; bỏ lượt có timestamp trước 2002-01-01 (dữ liệu có mốc epoch 0).
- Không có hai lượt liên tiếp cùng bài (đã gộp `A A A → A`); mỗi phiên ≥ 5 bài.
- Kiểm tra: `src.data.lastfm.validate_lastfm_sessions` (không dùng `validate_sessions` vì track_id 16 ký tự).
- `lastfm_log.json`: phễu (lượt, phiên, bài, user sau từng bước), độ nhạy theo khoảng nghỉ 15/30/60 phút (số phiên, số cặp liền nhau), tỉ lệ lấy mẫu user, thống kê từng split.

### lastfm/tracks.parquet (#56)

Mọi bài có trong `lastfm/sessions.parquet`, mỗi bài một dòng. Có đủ cột `track_id`, `bpm`, `camelot`, `energy` nên `RuleMix`, `smoothness` và điểm mix học chạy được không cần sửa.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `track_id` | string | ID 16 ký tự của Music4All-Onion; duy nhất |
| `bpm` | float | `rhythm.bpm` của Essentia; > 0 |
| `key` | int | 0–11, 0 = Đô; **ước lượng** bằng Krumhansl–Schmuckler trên HPCP 36 bin (bin 0 = La) |
| `mode` | int | 1 = trưởng, 0 = thứ (ước lượng cùng `key`) |
| `camelot` | string | `camelot_code(key, mode)` |
| `energy` | float | **thay thế**: hạng phần trăm của `lowlevel.loudness_ebu128.integrated` trong tập bài, trong (0, 1]; không cùng nghĩa với energy của Spotify |
| `key_strength` | float | tương quan Pearson với hồ sơ tông tốt nhất |
| `n_occurrences` | int | số lượt trong các phiên train của last.fm |

- `tracks_log.json`: tương quan và độ lệch trung vị của `key_strength` so với `tonal.key_krumhansl.strength`, tỉ lệ bài nghi lỗi nhịp gấp đôi/một nửa, số bài trưởng/thứ.

## Kiểm tra

```python
from src.data.schema import validate_dataset
validate_dataset(tracks, sessions, eval_targets, candidates={"val": cands_val})   # ném SchemaError và liệt kê mọi lỗi
```

Dữ liệu giả cho B và C khi chưa có dữ liệu thật:

```bash
python -m src.data.fake --out data/fake        # sinh mọi file theo bố cục ở trên
```

Dữ liệu giả cố ý có đủ các trường hợp khó: phiên lặp bài, bài đích không có trong train (luôn trượt), và phiên val/test có bài đích nằm trong seed (không có trong `eval_targets`). Candidates giả là ngẫu nhiên, nên chỉ dùng để kiểm tra code chạy đúng, không dùng để so điểm.
