# artifacts/

File app đọc để chạy không mạng, không torch (spec §5, §8). Tạo bằng `python -m src.export_artifacts` (#20), trên Kaggle là notebook `14_export_artifacts` → Dataset `mixnext-artifacts`. App đọc bằng `load_artifacts("artifacts")` trong `src/export_artifacts.py`.

| file | nội dung |
| --- | --- |
| `tracks.parquet` | schema `tracks` (có `track_name`, `artist_name`): bài có vector và bài của playlist demo |
| `item_embeddings.npy`, `track_index.parquet` | vector bài của mô hình gu được chọn (ALS hoặc item2vec, NDCG@10 val cao hơn) và cột `row`, `track_id` |
| `demo_playlists.parquet` | `session_id`, `seed_track_ids`, `target_track_id`: 100 phiên test mẫu cho app chọn |
| `manifest.json` | mô hình, `decay`, trọng số mix theo luật, NDCG@10 val của từng mô hình, số bài, số bài bị cắt |
| `mix_learned.npz` | trọng số điểm mix học (#23), nếu có |

Nội dung thư mục này **không commit vào git** (xem `.gitignore`); chỉ có file README này.

- Lấy về: tải Kaggle Dataset `mixnext-artifacts` và giải nén vào đây (các file nằm thẳng dưới `artifacts/`).
- Tổng dung lượng < 200 MB; nếu vượt, script bỏ bài ít lượt nhất (vẫn giữ bài của playlist demo) và ghi số bài bị cắt vào `manifest.json`.
- Khi nén bài nộp (#31): nén cả `artifacts/` đã tải về, để người chấm chạy app ngay.
