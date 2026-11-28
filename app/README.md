# app/

App Streamlit của MixNext (spec §8): chọn playlist, bài đang phát, hàng đợi gợi ý kèm giải thích, thanh trượt α, người dùng chọn bài tiếp theo. Cách chạy ở README gốc, mục **Chạy app**.

- `main.py`: giao diện. Chỉ hiển thị và chuyển thao tác của người dùng cho `Player`.
- `player.py`: `Player` và `Session`. Bắt đầu từ seed của playlist demo, hàng đợi top-10 (vector phiên + top-100 ứng viên bằng numpy từ embedding trong `artifacts/`, xếp lại theo α bằng `rerank` với `RuleMix`, dòng giải thích từ `src.rerank.explain`), chọn bài (chỉ bài trong hàng đợi; bài đã có trong phiên không quay lại), quay lại, bắt đầu lại, lịch sử phiên. Ứng viên nhớ theo phiên nên kéo α chỉ phải xếp lại.
- `config.py`, `config.toml`: đường dẫn tới `artifacts/`.
- `requirements.txt`: gói cho app (Python 3.11+, không có torch).
- `../.streamlit/config.toml`: cấu hình Streamlit để chạy offline (tắt thống kê sử dụng, không hỏi email, chỉ mở trên `localhost`). Streamlit chỉ đọc file này khi chạy từ thư mục gốc repo.

Prototype trên dữ liệu giả: #11. Demo v1 trên `artifacts/`: #21. Demo v2 (so sánh baseline, biểu đồ, vòng Camelot, cold-start, mix học): #28. App chỉ đọc `artifacts/`, chạy không cần mạng và không cần torch (spec §5).
