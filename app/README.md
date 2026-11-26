# app/

App Streamlit của MixNext (spec §8): hàng đợi gợi ý, thanh trượt α, giải thích, so sánh với baseline. Cách chạy ở README gốc, mục **Chạy app**.

- `main.py`: giao diện. Chỉ hiển thị, không tính toán.
- `player.py`: `Player` giữ danh sách playlist, bài đang phát (bài cuối seed) và hàng đợi top-10 xếp lại theo α bằng `src.rerank.combine.rerank` với `RuleMix`. Kiểm tra các bảng khớp nhau khi dựng.
- `config.py`, `config.toml`: mọi đường dẫn dữ liệu app đọc.
- `requirements.txt`: gói cho app (Python 3.11+, không có torch).
- `../.streamlit/config.toml`: cấu hình Streamlit để chạy offline (tắt thống kê sử dụng, không hỏi email, chỉ mở trên `localhost`). Streamlit chỉ đọc file này khi chạy từ thư mục gốc repo.

Prototype trên dữ liệu giả: #11. Demo v1 trên `artifacts/`: #21. Demo v2: #28. App chỉ đọc `artifacts/`, chạy không cần mạng và không cần torch (spec §5).
