"""Vector phiên và mô hình gu dựa trên embedding bài, dùng chung cho ALS (#14) và item2vec (#15) (spec §6).

Vector phiên = Σ decay^(khoảng cách tới bài cuối) · vector bài; điểm gu = tích vô hướng giữa vector phiên và vector bài.
Chỉ dùng numpy, để app (#21) tính lại được mà không cần implicit, gensim hay torch.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES
from src.models.base import TasteModel, top_k_excluding

MISSING = -1  # bài không có vector (không xuất hiện trong train)


def session_vector(vectors: np.ndarray, rows: np.ndarray, decay: float) -> np.ndarray:
    """rows: hàng của từng bài trong phiên theo thứ tự nghe, MISSING nếu bài không có vector.

    Bài không có vector không đóng góp nhưng vẫn giữ vị trí, nên khoảng cách tính theo vị trí thật trong phiên.
    Bài lặp được cộng ở mỗi lần xuất hiện.
    """
    rows = np.asarray(rows)
    distance = len(rows) - 1 - np.arange(len(rows))
    known = rows != MISSING
    weights = np.power(float(decay), distance[known])
    return weights @ vectors[rows[known]].astype(float)


class SessionVectorTaste:
    """Mô hình gu từ embedding bài: top-k theo tích vô hướng với vector phiên, không có bài đã trong phiên.

    Phiên không có bài nào có vector (vector phiên bằng 0) thì lấy ứng viên từ mô hình dự phòng (Popularity).
    """

    def __init__(self, name: str, vectors: np.ndarray, track_ids: Sequence[str], decay: float, fallback: TasteModel):
        self.name = name
        self.vectors = np.asarray(vectors, dtype=np.float32)
        # phần tử là str thường, không phải np.str_. Hòa điểm thì xếp theo thứ tự này: truyền track_ids đã sắp xếp
        self.index = pd.Index(np.asarray(track_ids, dtype=object))
        self.decay = decay
        self.fallback = fallback

    def with_decay(self, decay: float) -> SessionVectorTaste:
        """Cùng vector bài, decay khác; không học lại (chọn decay trên val)."""
        clone = copy.copy(self)  # bản sao nông: dùng chung mảng vector
        clone.decay = decay
        return clone

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        session = list(session_track_ids)
        rows = self.index.get_indexer(session)
        if (rows == MISSING).all():
            return self.fallback.candidates(session, k=k)
        # cùng kiểu float32 với vector bài: nhân với vector float64 thì numpy chép cả ma trận sang float64 (chậm ~15 lần)
        scores = self.vectors @ session_vector(self.vectors, rows, self.decay).astype(self.vectors.dtype)
        n_top = min(k + len(session), len(scores))  # đủ chỗ cho k ứng viên sau khi bỏ bài trong phiên
        top = np.argpartition(-scores, n_top - 1)[:n_top]
        top = top[np.lexsort((top, -scores[top]))]  # điểm giảm dần; hòa thì theo thứ tự bài trong index
        return top_k_excluding(self.index.to_numpy()[top], scores[top], session, k)

    def embeddings(self) -> tuple[np.ndarray, pd.DataFrame]:
        """(vectors, index) theo docs/data-contract.md: item_embeddings/{model}.npy và {model}_tracks.parquet."""
        index = pd.DataFrame({"row": np.arange(len(self.index)), "track_id": self.index.to_numpy()})
        return self.vectors, index
