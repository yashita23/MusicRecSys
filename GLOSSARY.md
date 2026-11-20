# MixNext

MixNext gợi ý bài hát tiếp theo cho một phiên nghe sao cho vừa hợp gu vừa nối mượt với bài đang phát (transition-aware playlist continuation).

## Phiên và dữ liệu đánh giá

**Phiên** (session):
Chuỗi bài có thứ tự mà người dùng đang nghe; trong dữ liệu, mỗi playlist MPD là một phiên.
_Avoid_: user, người dùng (MPD không có user ID)

**Seed**:
Các bài đầu của phiên được đưa vào hệ thống làm đầu vào; khi đánh giá là 5 bài đầu của playlist.
_Avoid_: dùng "seed" cho random seed

**Bài đích** (target):
Bài thật đứng ngay sau seed trong playlist val/test (bài thứ 6), dùng để chấm độ chính xác.
_Avoid_: ground truth, nhãn

**Lần chạy** (run):
Một lần lặp lại toàn bộ thí nghiệm với một `random_seed`; mỗi cấu hình có 3 lần chạy.
_Avoid_: seed

## Điểm

**Điểm gu** (taste score):
Điểm mô hình gu chấm cho một ứng viên dựa trên cả phiên; `gu_norm` là điểm gu chuẩn hóa min-max trong tập ứng viên.
_Avoid_: relevance, điểm sở thích

**Điểm mix** (mix score):
Mức độ bài A nối được sang bài B, chỉ dựa vào đặc trưng âm thanh của hai bài.
_Avoid_: điểm chuyển bài, điểm mượt, transition score

**Điểm mix theo luật** (rule mix score):
Điểm mix tính bằng luật DJ: lệch BPM, vòng Camelot, chênh năng lượng.

**Điểm mix học** (learned mix score):
Điểm mix do hai bộ mã hóa học từ các cặp bài liền nhau trong playlist thật.
_Avoid_: điểm mix học từ dữ liệu, contrastive score

**Điểm cuối** (final score):
Điểm dùng để xếp lại ứng viên, bằng α · gu_norm + (1 − α) · điểm mix.

**α**:
Trọng số trong khoảng 0–1 cân giữa đúng gu (α = 1) và mix mượt (α = 0).

## Mô hình và đánh giá

**Mô hình gu** (taste model):
Mô hình học từ hành vi playlist, sinh tập ứng viên kèm điểm gu (ALS, item2vec, SASRec).

**Vector phiên** (session vector):
Tổng có trọng số các vector bài trong phiên, bài càng gần cuối phiên càng nặng; ALS và item2vec dùng cùng cách tính này.

**Ứng viên** (candidate):
Một trong top-100 bài mà mô hình gu chọn ra cho một phiên; chỉ ứng viên mới được xếp lại.

**Item-kNN liền kề**:
Baseline gợi ý những bài hay đứng ngay sau bài cuối của seed trong các playlist train.
_Avoid_: item-kNN đồng xuất hiện

**Độ mượt** (smoothness):
Nhóm chỉ số đánh giá trên các cặp bài liền kề của một chuỗi: ΔBPM trung bình (%), tỉ lệ chuyển tông hợp, |Δenergy| trung bình.
_Avoid_: dùng "độ mượt" để gọi một điểm

**Cặp mix được** (mixable pair):
Cặp bài liền kề có ΔBPM ≤ 6% (tính cả nhịp gấp đôi/một nửa) và tông tương thích theo vòng Camelot.

**Tỉ lệ cặp mix được**:
Phần trăm cặp mix được trong các cặp liền kề của chuỗi sinh ra; là mục tiêu khi chọn trọng số mix và α.

**Đường đánh đổi** (trade-off curve):
Đường NDCG@10 theo độ mượt khi quét α từ 0 đến 1.
