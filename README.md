# MusicRecSys
Hệ gợi ý bài hát cá nhân hóa

## Môi trường

Huấn luyện chạy trên Kaggle (Python 3.13, CPU). Phiên bản gói chốt trong `requirements.txt`, đã thử bằng `notebooks/00_env_check.ipynb`.

- **implicit (ALS) không có sẵn trên Kaggle.** Notebook dùng ALS phải bật Internet và chạy `pip install implicit==0.7.3` trước khi import. Các gói khác (numpy, pandas, pyarrow, scipy, gensim, torch, pytest) có sẵn đúng phiên bản trong `requirements.txt`.
- **torch:** Kaggle cài bản CPU (`2.11.0+cpu`); `requirements.txt` ghi `torch==2.11.0` vì PyPI không có bản có đuôi `+cpu`.
- **Không dùng RecBole:** RecBole không cài được trên Python 3.13 (ADR 0002), SASRec viết bằng PyTorch trong `src/models/sasrec`.

Trên máy: `pip install -r requirements.txt` rồi `python -m pytest -q`.
