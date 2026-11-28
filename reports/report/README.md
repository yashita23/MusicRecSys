# reports/report/: báo cáo bài tập lớn

Khung báo cáo theo 6 mục của L0 (spec §13, ticket #22). Nội dung viết ở #29.

## Ai viết mục nào

Vai A, B, C theo spec §10. Đầu mỗi file cũng ghi vai phụ trách.

| File | Mục | Phụ trách |
| --- | --- | --- |
| `sections/1_gioi_thieu.tex` | 1. Giới thiệu và mô tả bài toán | A |
| `sections/2_phuong_phap.tex` | 2. Chi tiết phương pháp | A: 2.1–2.3 (hai tầng, dữ liệu, điểm mix) · B: 2.4–2.7 (mô hình gu, xếp lại, đánh giá, kết quả) |
| `sections/3_chuc_nang.tex` | 3. Chức năng hệ thống và cách dùng | C |
| `sections/4_dung_lai.tex` | 4. Phương pháp, phần mềm, dữ liệu dùng lại | C (đã điền bảng; ai thêm thư viện hay nguồn thì thêm dòng) |
| `sections/5_kho_khan.tex` | 5. Khó khăn và cách vượt qua | A, B, C: mỗi người một mục con |
| `sections/6_ket_luan.tex` | 6. Tranh luận, kết luận, hướng phát triển | A, B: 6.1–6.3 · C: 6.4–6.5 |
| `report.tex` | Trang bìa, mục lục, tài liệu tham khảo | C |

Phần trình bày 15 phút chia giống vậy: A bài toán, dữ liệu, điểm mix; B mô hình gu, kết quả, đường đánh đổi; C demo, kết luận, giới hạn, slide dùng lại.

## Quy ước

- Chỗ còn phải viết dùng `\canviet{...}` (hiện chữ đỏ). Tìm còn thiếu gì: `grep -rn canviet reports/report`.
- Hình chèn bằng `\hinh{file}{chú thích}{ticket}`, đường dẫn tính từ thư mục này: ảnh chụp app để ở `img/`, hình kết quả ở `../` (vd. `../fig_tradeoff_v1.png`). Chưa có file thì hiện khung chờ, nên báo cáo luôn biên dịch được.
- Trích bài báo bằng `\cite{als}`, `\cite{sasrec}`... (danh sách ở cuối `report.tex`).
- Số thập phân dùng dấu phẩy (0,0967), như app và notebook.

## Biên dịch

XeLaTeX, cùng font với `../proposal.tex`:

```
cd reports/report
tectonic report.tex
```

Hoặc upload cả thư mục `reports/` lên Overleaf, đặt main file là `report/report.tex`, compiler XeLaTeX.
