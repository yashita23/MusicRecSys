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

## File trung gian của A

Chỉ dùng giữa các ticket của A (#5 → #6, #7). B và C không đọc các file này, nên luật "sửa cả ba nơi" ở đầu tài liệu không áp dụng: schema của chúng định nghĩa trong `src/data/build.py` (`RAW_TRACK_COLUMNS`) và được kiểm tra ở `tests/test_build.py`.

| file | do ai tạo | cột |
| --- | --- | --- |
| `sessions_raw.parquet` | #5 | `session_id`, `pos`, `track_id`. Như `sessions.parquet` nhưng chưa có `split`; đã lọc (mỗi bài ≥ 5 lượt, mỗi phiên ≥ 10 bài, lặp đến khi ổn định) và đánh số lại `pos` |
| `tracks_raw.parquet` | #5 | `track_id`, `track_name`, `artist_name`, `tempo`, `key`, `mode`, `energy`. Đúng các bài còn trong `sessions_raw`, mọi bài đều dùng được |
| `build_log.json` | #5 | Số phiên, số bài, số lượt sau từng bước lọc, kèm cấu hình (`n_slices`, `random_seed`) |

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
