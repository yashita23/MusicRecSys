# Dùng bản sao MPD trên Kaggle và bộ features làm riêng cho MPD

Bản MPD chính thức trên AIcrowd đã bị gỡ (tính đến 11/2026), còn API audio features của Spotify bị khóa với app mới từ 27/11/2024. Vì vậy nhóm dùng bản sao không chính thức `himanshuwagh/spotify-million` trên Kaggle (JSON gốc, có `track_uri` và `pos`), ghép với `krishsharma0413/2-million-songs-from-mpd-with-audio-features`, một bộ được làm riêng cho các bài của MPD. Dự phòng là `rodolfofigueroa/spotify-12m-songs`. Giấy phép MPD cấm phân phối lại, nên báo cáo phải khai báo nguồn là bản sao không chính thức, và dữ liệu không được commit vào repo.

## Consequences

- Chưa ai công bố độ phủ thật của bộ features này, nên việc đầu tiên của tuần 1 là đo tỉ lệ bài và tỉ lệ lượt xuất hiện ghép được.
- Nếu muốn đổi nguồn dữ liệu sau tuần 1 thì phải chạy lại toàn bộ pipeline và mọi kết quả.
