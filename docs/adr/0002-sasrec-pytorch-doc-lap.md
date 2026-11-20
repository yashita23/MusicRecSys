# SASRec viết bằng PyTorch độc lập, không dùng RecBole

Spec ban đầu chọn RecBole cho SASRec. Nhưng RecBole 1.2.1 cần `ray<=2.6.3`, bản này không có wheel cho Python 3.13 mà image Kaggle hiện tại đang dùng. RecBole còn lỗi với NumPy 2 (`np.float`) và PyTorch ≥ 2.6 (`torch.load` không có `weights_only=False`), và issue #2215 vẫn chưa được sửa. Nhóm chuyển SASRec từ `pmixer/SASRec.pytorch` (Apache-2.0) vào `src/models/sasrec`. Mô hình nạp dữ liệu theo đúng cách chia seed 5 → bài thứ 6 và xuất top-100 ứng viên như các mô hình gu khác. Dù có dùng RecBole thì vẫn phải tự viết bước xuất ứng viên theo cách chia riêng của nhóm, nên bỏ RecBole không mất nhiều.

## Considered Options

- Cài RecBole bằng `--no-deps` rồi tự vá lỗi: có thể mất 1–2 ngày chỉ để cài.
- Chạy RecBole trên Colab hoặc máy local với Python 3.10: phải giữ hai môi trường và chuyển file ứng viên qua lại.
