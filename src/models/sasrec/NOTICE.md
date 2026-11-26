# Nguồn và thay đổi của SASRec

Phần mạng (`network.py`) chuyển từ **SASRec.pytorch** của Zan Huang (pmixer), https://github.com/pmixer/SASRec.pytorch, commit `26a8fa880f312c3700f8dcca5512d5854b257f4e`, file `python/model.py`. License gốc là Apache License 2.0, chép nguyên văn ở `LICENSE` trong thư mục này. Mô hình do Kang & McAuley đề xuất: *Self-Attentive Sequential Recommendation*, ICDM 2018.

## Giữ nguyên từ bản gốc
- **Kiến trúc mạng:**
  - Embedding bài nhân √d, cộng embedding vị trí (vị trí đệm = 0).
  - Các khối `MultiheadAttention` có mặt nạ nhân quả, LayerNorm sau phép cộng (post-norm).
  - Feed-forward theo từng vị trí bằng `Conv1d` kernel 1, LayerNorm cuối.
- **Khởi tạo** (`python/main.py`): `xavier_normal_` cho tham số nhiều chiều, hàng đệm của hai embedding bằng 0.
- **Tối ưu:** Adam với `betas=(0.9, 0.98)`, lr 0,001, dropout 0,2.
- **Điểm của một bài:** tích vô hướng giữa đặc trưng ở vị trí cuối và embedding của bài.

## Thay đổi (MixNext, ticket #16)
- **Mạng** (`network.py`):
  - Lớp `SASRec` đổi tên thành `SASRecNetwork`, nhận tham số tường minh (`item_num, maxlen, hidden_units, num_blocks, num_heads, dropout_rate`) thay cho đối tượng `args`; bỏ `user_num`.
  - `log2feats` thành `forward`, nhận tensor thay vì mảng numpy.
  - Bỏ `forward(user_ids, log_seqs, pos_seqs, neg_seqs)` dùng khi huấn luyện (hàm mất mát viết lại ở `model.py`) và `predict` (điểm tính ở `model.py`).
  - Bỏ tùy chọn `norm_first`; chỉ giữ nhánh mặc định post-norm.
- **Bỏ số hạng `l2_emb`** của hàm mất mát: bản gốc mặc định bằng 0, tức không dùng.
- **Hàm mất mát:** cross-entropy trên (bài đúng + 256 bài âm ngẫu nhiên dùng chung cho cả batch) thay cho BCE với 1 bài âm mỗi vị trí. Theo Klenitskiy & Vasilev, *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?*, RecSys 2023. Bài âm trùng bài đúng bị loại khỏi softmax.
- **Lấy mẫu huấn luyện** (`data.py`, viết lại thay `utils.WarpSampler`):
  - Mỗi epoch mỗi playlist cho một đoạn liên tiếp ngẫu nhiên dài tối đa `maxlen + 1`, thay vì `maxlen` bài cuối.
  - Độ dài tối đa 50 theo spec §6.
- **Dừng sớm:** sau mỗi epoch đo NDCG@10 trên một mẫu 2.000 phiên val bằng `src.eval.run.evaluate`. Giữ trạng thái tốt nhất, dừng khi 3 epoch không tăng hoặc hết 40 phút.
- **Suy luận:**
  - Đầu vào là seed (5 bài đầu playlist), bỏ bài không có trong train, đệm trái đến 50.
  - Seed không còn bài nào thì lấy ứng viên theo Popularity, giống các mô hình gu khác.
  - Ứng viên không chứa bài đã có trong phiên.
- **Tham số:** hidden 128, 2 khối, 2 head, batch 256 (`SASRecParams`).
- **Chấm nhiều phiên một lượt** (`SASRec.candidates_batch`): che bài trong phiên rồi lấy top-k trên GPU.

## Tái lập
Cùng `random_seed` chưa chắc ra cùng mô hình: giới hạn thời gian huấn luyện (40 phút) làm điểm dừng phụ thuộc tốc độ máy, và một số phép tính trên GPU (attention, lan truyền ngược của embedding) không tất định. Khác biệt giữa các lần chạy đo ở #26 (3 lần chạy).
