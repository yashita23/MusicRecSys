# Dùng bản sao MPD trên Kaggle và bộ features làm riêng cho MPD

Bản MPD chính thức trên AIcrowd đã bị gỡ (tính đến 11/2026), còn API audio features của Spotify bị khóa với app mới từ 27/11/2024. Vì vậy nhóm dùng bản sao không chính thức `himanshuwagh/spotify-million` trên Kaggle (JSON gốc, có `track_uri` và `pos`), ghép với `krishsharma0413/2-million-songs-from-mpd-with-audio-features`, một bộ được làm riêng cho các bài của MPD. Dự phòng là `rodolfofigueroa/spotify-12m-songs`. Giấy phép MPD cấm phân phối lại, nên báo cáo phải khai báo nguồn là bản sao không chính thức, và dữ liệu không được commit vào repo.

## Consequences

- Độ phủ đo ngày 21/11/2026 (#1) trên 100 slice ngẫu nhiên (100.000 playlist, 6.624.113 lượt, 686.633 bài; `random_seed` = 42): 99,98% bài và 100,00% lượt có trong bộ features. Tính riêng bài dùng được (tempo > 0, key ≠ −1, không thiếu giá trị) là 99,89% bài và 99,97% lượt. 96,49% playlist còn ≥ 10 bài dùng được. Chỉ 0,09% bài ghép được có tempo ≤ 0. Bộ dự phòng không cần dùng.
- Trên Kaggle, dataset được mount ở `/kaggle/input/datasets/<owner>/<slug>/`, nên code tìm file bằng glob thay vì ghi cứng đường dẫn.
- Nếu muốn đổi nguồn dữ liệu sau tuần 1 thì phải chạy lại toàn bộ pipeline và mọi kết quả.
