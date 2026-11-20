# Đánh giá với seed 5 bài và một bài đích duy nhất

Trong mỗi playlist val/test, seed là 5 bài đầu và bài đích là bài thứ 6. Mỗi playlist chỉ có đúng một bài đích, và bài trong seed bị loại khỏi ứng viên. Nhóm chọn cách này thay cho seed bằng 50% playlist hay nhiều độ dài seed (1/5/10) như RecSys Challenge 2018, vì nó khớp thẳng với câu hỏi "bài kế tiếp", giúp HitRate/NDCG/MRR dễ hiểu, và đủ nhẹ cho 3 tuần. Độ mượt được đo riêng, trên chuỗi 10 bài sinh tham lam từ cùng seed đó. Mọi bảng kết quả đều dựa trên cách chia này, nên muốn đổi thì phải chạy lại toàn bộ.
