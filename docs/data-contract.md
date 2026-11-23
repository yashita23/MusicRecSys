# Định dạng file bàn giao

Tài liệu này mô tả schema của mọi file đi qua tay nhiều người trong pipeline MixNext. `src/data/schema.py` kiểm tra các schema này, `src/data/fake.py` sinh dữ liệu giả theo đúng schema. Thuật ngữ theo `GLOSSARY.md`.

Muốn đổi schema thì mở PR sửa cả ba nơi (tài liệu này, `schema.py`, `fake.py`), và cả ba thành viên phải đồng ý.

## Bố cục thư mục

```text
<root>/                                  # Kaggle Dataset mixnext-processed, hoặc data/processed/ ở máy
├── tracks.parquet
├── sessions.parquet
└── eval_targets.parquet

<root>/                                  # Kaggle Dataset mixnext-candidates
└── candidates/{model}_{split}_run{r}.parquet

<root>/                                  # Kaggle Dataset mixnext-embeddings, rồi được chép vào artifacts/ cho app
└── item_embeddings/
    ├── {model}.npy
    └── {model}_tracks.parquet
```

- `model` ∈ `popularity`, `itemknn`, `als`, `item2vec`, `sasrec` (dữ liệu giả dùng `fake`).
- `split` ∈ `val`, `test`.
- `r` là số thứ tự lần chạy: 0, 1, 2.

## tracks.parquet

Danh mục bài. Mỗi bài một dòng.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `track_id` | string | Spotify ID 22 ký tự, không có tiền tố `spotify:track:`; duy nhất |
| `track_name` | string | |
| `artist_name` | string | |
| `tempo` | float | Giá trị thô từ bộ features, > 0 |
| `bpm` | float | BPM mà điểm mix dùng, > 0 |
| `key` | int | 0–11 (0 = C, 1 = C♯/D♭, …) |
| `mode` | int | 0 = thứ, 1 = trưởng |
| `camelot` | string | `1A`–`12B`; chữ `A` khi `mode` = 0, chữ `B` khi `mode` = 1 |
| `energy` | float | 0–1 |
| `n_occurrences` | int | Số lượt xuất hiện **trong các phiên train**, ≥ 0 |

Không cột nào được null. Bài có tempo ≤ 0, không xác định được tông (`key` = −1) hoặc thiếu features bị loại ngay ở bước lọc (#5), như thể không ghép được. Các bài này chiếm 0,11% số bài và 0,03% số lượt (ADR 0001).

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

- Phiên có bài đích trùng một bài trong seed thì bị **bỏ**, vì bài trong seed bị loại khỏi ứng viên nên không mô hình nào đoán được.
- Phiên có bài đích không có trong train thì được **giữ** và tính là trượt cho mọi mô hình. #7 báo cáo tỉ lệ của những phiên này.

## candidates/{model}_{split}_run{r}.parquet

Top-100 ứng viên mà một mô hình gu chọn ra cho mỗi phiên trong `eval_targets` của `split`.

| cột | kiểu | ràng buộc |
| --- | --- | --- |
| `session_id` | int | Có trong `eval_targets` với đúng `split` của file |
| `rank` | int | 1–100, liền mạch trong mỗi phiên |
| `track_id` | string | Có trong `tracks.parquet`; không trùng nhau trong một phiên; không nằm trong seed |
| `taste_score` | float | Điểm gu, hữu hạn, không tăng theo `rank` (rank 1 có điểm cao nhất) |

Mỗi phiên có đúng 100 dòng. Bước re-rank tự chuẩn hóa `taste_score` thành `gu_norm`, nên mô hình không cần chuẩn hóa trước.

## item_embeddings/{model}.npy và {model}_tracks.parquet

Vector bài của các mô hình có embedding (ALS, item2vec). App và bước tính vector phiên dùng các file này.

- `{model}.npy`: mảng `float32` có shape `(n, d)`.
- `{model}_tracks.parquet`: cột `row` (int, 0…n−1, liền mạch) và cột `track_id` (string, duy nhất, có trong `tracks.parquet`). Hàng `row` = i ứng với vector thứ i.

Mỗi mô hình có index riêng và chỉ chứa những bài nó đã học được, không có vector 0 lấp chỗ trống.

## Kiểm tra

```python
from src.data.schema import validate_dataset
validate_dataset(tracks, sessions, eval_targets, candidates={"val": cands_val})   # ném SchemaError và liệt kê mọi lỗi
```

Dữ liệu giả cho B và C khi chưa có dữ liệu thật:

```bash
python -m src.data.fake --out data/fake        # sinh mọi file theo bố cục ở trên
```
