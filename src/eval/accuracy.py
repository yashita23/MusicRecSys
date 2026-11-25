"""Chỉ số độ chính xác với một bài đích mỗi phiên (ticket #8, spec §7, ADR 0003).

Mọi hàm nhận mảng hạng của bài đích, mỗi phiên một giá trị: 1 là đứng đầu, 0 là không có trong danh sách.
"""
from __future__ import annotations

import numpy as np

MISS = 0  # hạng của bài đích không có trong danh sách


def hit_rate(ranks: np.ndarray, k: int) -> float:
    """Tỉ lệ phiên có bài đích trong top-k."""
    r = np.asarray(ranks)
    return float(((r > MISS) & (r <= k)).mean())


def ndcg(ranks: np.ndarray, k: int) -> float:
    """Một bài đích nên IDCG = 1 và NDCG@k = 1/log2(hạng + 1) nếu hạng ≤ k, ngược lại 0."""
    r = np.asarray(ranks, dtype=float)
    hit = (r > MISS) & (r <= k)
    return float(np.where(hit, 1.0 / np.log2(np.where(hit, r, 1.0) + 1.0), 0.0).mean())


def mrr(ranks: np.ndarray) -> float:
    """1/hạng trên toàn bộ danh sách (không cắt ở k); 0 nếu bài đích không có trong danh sách."""
    r = np.asarray(ranks, dtype=float)
    return float(np.where(r > MISS, 1.0 / np.where(r > MISS, r, 1.0), 0.0).mean())
