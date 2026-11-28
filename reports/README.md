# reports/

Tài liệu nộp và hình, bảng kết quả cho báo cáo và slide (spec §13, §14): đề xuất đề tài (#10), đường đánh đổi (#19), grid search (#25), bảng kết quả chính (#26), ablation (#27).

- `proposal.tex` → `proposal.pdf`: đề xuất đề tài 1 trang (#10, spec §14). Biên dịch bằng XeLaTeX: `tectonic reports/proposal.tex` (Tectonic tự tải gói khi cần) hoặc upload lên Overleaf và chọn compiler XeLaTeX.
- `report/`: báo cáo bài tập lớn, 6 mục theo L0 (#22 khung, #29 nội dung). Ai viết mục nào và cách biên dịch: `report/README.md`.
- `grid.csv`: grid search (w, α) trên 2.000 phiên val với ALS (#25, notebook 15): 21 bộ w × 11 α cho mix theo luật, 11 α cho mix học; cột `chosen` đánh dấu cấu hình ghi vào `artifacts/mix_params.json`. Chép từ output Kaggle sau khi chạy.
