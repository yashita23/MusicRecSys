# MixNext (MusicRecSys)

Gợi ý bài hát tiếp theo vừa hợp gu người nghe vừa chuyển mượt từ bài đang phát (transition-aware playlist continuation). Bài tập lớn môn IT4613. Đặc tả ở `Spec dự án MixNext.md`, thuật ngữ ở `GLOSSARY.md`, các quyết định khó đảo ngược ở `docs/adr/`, định dạng file bàn giao ở `docs/data-contract.md`.

## Cấu trúc repo

```text
src/
├── data/        # ghép, lọc, chia train/val/test; schema và dữ liệu giả
├── features/    # BPM, Camelot, energy
├── models/      # baseline (popularity, item_knn), mô hình gu (als, item2vec, sasrec/), vector phiên (session), embedding_runner (chọn tham số, CLI)
├── rerank/      # điểm mix theo luật (mix_rule), điểm mix học (mix_learned; huấn luyện ở mix_learned_train), xếp lại theo α (combine)
└── eval/        # độ chính xác (accuracy, run), độ mượt (smoothness)
notebooks/       # chạy trên Kaggle: clone repo rồi gọi hàm trong src/; template.ipynb là notebook mẫu
tests/           # pytest
app/             # app Streamlit: main.py (giao diện), player.py (hàng đợi theo α), config.toml (đường dẫn dữ liệu)
artifacts/       # file cho app, lấy từ Kaggle, không commit
reports/         # hình và bảng kết quả
data/            # dữ liệu ở máy (raw, processed), không commit
```

## Cài đặt

Python 3.10+ cho phần huấn luyện và test; riêng app cần Python 3.11+ (đọc config bằng `tomllib`).

```bash
pip install -r requirements.txt   # gói đã chốt trên Kaggle (#2)
pip install -e .                  # package src/ (pyproject.toml); không cài thêm gói phụ thuộc
python -m pytest -q               # chạy được cả khi chưa có dữ liệu thật
python -m src.data.fake --out data/fake   # dữ liệu giả theo data-contract để thử code
```

## Chạy app

App Streamlit chạy trên laptop, không cần mạng, cần Python 3.11+ và không cần torch. Bản hiện tại là prototype trên dữ liệu giả (#11); bản đọc `artifacts/` ở #21.

```bash
pip install -r app/requirements.txt
python -m src.data.fake --out data/fake   # một lần, ở thư mục gốc repo
streamlit run app/main.py                 # ở thư mục gốc repo, rồi mở http://localhost:8501
```

Mọi đường dẫn dữ liệu app đọc nằm trong `app/config.toml` (đường dẫn tương đối tính theo thư mục `app/`). Muốn dùng file config khác thì đặt biến môi trường `MIXNEXT_CONFIG`. Chạy từ thư mục gốc repo để Streamlit đọc `.streamlit/config.toml` (tắt thống kê sử dụng, không hỏi email): app không cần mạng.

## Chạy lại pipeline trên Kaggle

Mỗi notebook clone repo từ GitHub và gọi hàm trong `src/`. Từ notebook 02 trở đi, output được đăng thành một Kaggle Dataset có phiên bản, và bước sau Add Input bộ của bước trước. Khi tạo notebook mới, chép `notebooks/template.ipynb`: ô đầu tiên là đoạn bootstrap.

| notebook | ticket | input | output |
| --- | --- | --- | --- |
| `00_env_check` | #2 | không cần | không ghi file; kiểm tra môi trường và in khối `requirements.txt` |
| `01_coverage` | #1 | `himanshuwagh/spotify-million`, `krishsharma0413/2-million-songs-from-mpd-with-audio-features` | `coverage.json` (không đăng Dataset); đo độ phủ ghép |
| `02_build` | #5 | như notebook 01 | Dataset `mixnext-processed`: `sessions_raw.parquet`, `tracks_raw.parquet`, `build_log.json` |
| `03_split` | #6, #7 | `mixnext-processed` | Dataset `titeohihi/mixnext-splited`: `tracks.parquet`, `sessions.parquet`, `eval_targets.parquet`, `split_report.json` |
| `04_baselines` | #9 | `titeohihi/mixnext-splited` | Dataset `mixnext-candidates`: ứng viên của Popularity và item-kNN liền kề |
| `05_als` | #14 | `titeohihi/mixnext-splited` | Dataset `mixnext-embeddings`: ứng viên, embedding và `als_params.json` của ALS |
| `06_item2vec` | #15 | `titeohihi/mixnext-splited` | Dataset `mixnext-item2vec`: ứng viên, embedding và `item2vec_params.json` của item2vec |
| `07_sasrec` | #16 | `titeohihi/mixnext-splited` | Dataset `mixnext-sasrec`: ứng viên, checkpoint và báo cáo huấn luyện của SASRec. Cần **GPU** |
| `08_mix_learned` | #23 | `titeohihi/mixnext-splited` | Dataset `mixnext-mix-learned`: trọng số numpy của điểm mix học (`mix_learned.npz`) và báo cáo huấn luyện (`mix_learned.json`) |

Bật Internet ở mọi notebook (để clone repo và `pip install`). Chỉ cần CPU, trừ notebook 07 (SASRec) cần GPU. Các Dataset đều để **Private** vì sinh từ MPD.

Dataset tạo từ output của một notebook chỉ nhận version mới từ chính notebook đó. Vì vậy mỗi notebook 04–07 có một Dataset riêng; với mô hình có embedding, Dataset chứa cả ứng viên lẫn embedding của mô hình đó. Notebook nào cần ứng viên hay embedding thì Add Input các bộ cần dùng và tìm file theo tên bằng `Path("/kaggle/input").rglob(...)`; trong các bộ này file có tiền tố `out/`. Đường dẫn từng file trong các bộ 04–07 ghi ở mục Bố cục thư mục của [`docs/data-contract.md`](docs/data-contract.md).

## Môi trường

Huấn luyện chạy trên Kaggle (Python 3.13, CPU). Phiên bản gói chốt trong `requirements.txt`, đã thử bằng `notebooks/00_env_check.ipynb`.

- **implicit (ALS) không có sẵn trên Kaggle.** Notebook dùng ALS phải bật Internet và chạy `pip install implicit==0.7.3` trước khi import. Các gói khác (numpy, pandas, pyarrow, scipy, gensim, torch, pytest) có sẵn đúng phiên bản trong `requirements.txt`.
- **torch:** Kaggle cài bản CPU (`2.11.0+cpu`); `requirements.txt` ghi `torch==2.11.0` vì PyPI không có bản có đuôi `+cpu`. Trên Linux, `pip install torch==2.11.0` từ PyPI kéo theo bản CUDA (vài GB); máy không có GPU thì cài bản CPU trước: `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`.
- **Không dùng RecBole:** RecBole không cài được trên Python 3.13 (ADR 0002). SASRec sẽ được viết bằng PyTorch trong `src/models/sasrec` (#16).
- **scipy:** `requirements.txt` tự chọn 1.15.3 cho Python 3.10 và 1.16.3 cho 3.11 trở lên.
